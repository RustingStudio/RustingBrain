//! Adapters on the device UNet's attention projections, and what a forward
//! pass keeps of itself for the backward one.
//!
//! This is the device side of [`crate::unet_lora`]: the same adapters, in the
//! same places, matched to that host reference. The forward pass is
//! [`DeviceUnet`]'s own, with a [`Recorder`] threaded through it that adds
//! each adapter after its projection and keeps what the backward pass reads.

use super::*;
use crate::unet_lora::{PROJECTIONS, UnetLora, transformers};
use cudarc::driver::DevicePtrMut;

/// One adapter: `scale * up . (down . x)`, both factors FP16, and both
/// factors' gradients in FP32.
pub(crate) struct DeviceLora {
    name: String,
    down: Tensor,
    up: Tensor,
    scale: f32,
    grads: [CudaSlice<f32>; 2],
}

/// A [`UnetLora`] on the device. One entry a transformer, in the order a
/// pass meets them; in each, one entry a block, holding `attn1` and `attn2`,
/// each holding `to_q`, `to_k`, `to_v` and `to_out.0`.
pub struct DeviceAdapters {
    transformers: Vec<Vec<[[Option<DeviceLora>; 4]; 2]>>,
}

impl std::fmt::Debug for DeviceAdapters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceAdapters").finish_non_exhaustive()
    }
}

impl DeviceAdapters {
    pub fn upload(gpu: &ImageGpu, unet: &Unet, lora: &UnetLora) -> Result<Self, NetworkError> {
        let mut used = 0;
        let mut table = Vec::new();
        for (name, transformer) in transformers(unet) {
            let mut blocks = Vec::new();
            for index in 0..transformer.blocks.len() {
                let mut slots: [[Option<DeviceLora>; 4]; 2] = Default::default();
                for (attention, slot) in ["attn1", "attn2"].iter().zip(&mut slots) {
                    for (projection, target) in PROJECTIONS.iter().zip(slot.iter_mut()) {
                        let key =
                            format!("{name}.transformer_blocks.{index}.{attention}.{projection}");
                        if let Some(adapter) = lora.adapters.get(&key) {
                            used += 1;
                            let zeros = |len| {
                                let stream = &gpu.context.stream;
                                stream
                                    .alloc_zeros::<f32>(len)
                                    .map_err(cuda_err("device allocation"))
                            };
                            let (down, up) = (&adapter.down.value, &adapter.up.value);
                            *target = Some(DeviceLora {
                                grads: [zeros(down.data.len())?, zeros(up.data.len())?],
                                down: gpu.upload_matrix(down)?,
                                up: gpu.upload_matrix(up)?,
                                scale: adapter.scale,
                                name: key,
                            });
                        }
                    }
                }
                blocks.push(slots);
            }
            table.push(blocks);
        }
        if used != lora.adapters.len() {
            return Err(NetworkError::InvalidConfig(format!(
                "{} of {} adapters sit on no attention projection of this UNet",
                lora.adapters.len() - used,
                lora.adapters.len()
            )));
        }
        Ok(Self {
            transformers: table,
        })
    }

    /// Adds the gradients gathered on the device to `lora`'s, which is the
    /// file these adapters were uploaded from.
    pub fn add_gradients(&self, gpu: &ImageGpu, lora: &mut UnetLora) -> Result<(), NetworkError> {
        let all = self.transformers.iter().flatten().flatten().flatten();
        for adapter in all.flatten() {
            let target = lora.adapters.get_mut(&adapter.name).ok_or_else(|| {
                NetworkError::InvalidConfig(format!("no adapter {} in this file", adapter.name))
            })?;
            for (param, grad) in [&mut target.down, &mut target.up]
                .into_iter()
                .zip(&adapter.grads)
            {
                let grad = gpu
                    .context
                    .stream
                    .clone_dtoh(grad)
                    .map_err(cuda_err("device to host copy"))?;
                param
                    .grad
                    .data
                    .iter_mut()
                    .zip(grad)
                    .for_each(|(sum, value)| *sum += value);
            }
        }
        Ok(())
    }
}

/// Threaded through a forward pass: the adapters, which transformer is next,
/// and what has been kept so far.
pub(crate) struct Recorder<'a> {
    adapters: &'a DeviceAdapters,
    next: usize,
    pub(super) records: Records,
}

impl<'a> Recorder<'a> {
    /// The next transformer's adapters.
    pub(super) fn next(&mut self) -> Result<&'a [[[Option<DeviceLora>; 4]; 2]], NetworkError> {
        let adapters: &'a DeviceAdapters = self.adapters;
        let blocks = adapters.transformers.get(self.next).ok_or_else(|| {
            NetworkError::InvalidConfig("the adapters were uploaded for another UNet".into())
        })?;
        self.next += 1;
        Ok(blocks)
    }
}

/// Each list in the order the forward pass made it, which the backward pass
/// reads from the end.
#[derive(Default)]
pub(super) struct Records {
    /// A resnet's input and its hidden map before the second norm.
    pub(super) resnets: Vec<(Map, Map)>,
    /// A transformer's input and its blocks.
    pub(super) transformers: Vec<(Map, Vec<BlockRecord>)>,
    /// The size of every resampling convolution's input.
    pub(super) resampled: Vec<(usize, usize)>,
    /// On the way up, the channels that came from below rather than the skip.
    pub(super) joined: Vec<usize>,
}

/// The tokens before each of the three norms, the attentions, and the gated
/// feed-forward's projection before its bias.
pub(crate) struct BlockRecord {
    pub(super) before: [Tensor; 3],
    pub(super) own: AttentionRecord,
    pub(super) cross: AttentionRecord,
    pub(super) projected: Tensor,
}

/// Self-attention keeps its stacked `[q | k | v]`; cross-attention its queries
/// and its stacked `[k | v]` of the prompt.
pub(crate) struct AttentionRecord {
    pub(super) queries: Option<Tensor>,
    pub(super) projected: Tensor,
    pub(super) attended: Tensor,
}

/// What [`DeviceUnet::forward_adapted`] keeps for the backward pass.
pub struct DeviceTape {
    records: Records,
    /// The output norm's input.
    out: Map,
    text: Tensor,
}

impl std::fmt::Debug for DeviceTape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceTape").finish_non_exhaustive()
    }
}

impl ImageGpu {
    /// `out += scale * (input . down^T) . up^T`, `out` having rows `stride`
    /// apart.
    pub(super) fn lora_add<O: DevicePtrMut<f16>>(
        &self,
        lora: &DeviceLora,
        input: &Tensor,
        out: &mut O,
        stride: usize,
    ) -> Result<(), NetworkError> {
        let rank = lora.down.rows;
        let mut hidden = self.uninit(input.rows, rank)?;
        act_rhs_transposed::<f16, _>(
            &self.context,
            &input.bytes(),
            input.cols,
            &lora.down.bytes(),
            lora.down.cols,
            true,
            true,
            &mut hidden.data,
            rank,
            input.rows,
            rank,
            input.cols,
            1.0,
            0.0,
        )?;
        act_rhs_transposed::<f16, _>(
            &self.context,
            &hidden.bytes(),
            rank,
            &lora.up.bytes(),
            rank,
            true,
            true,
            out,
            stride,
            input.rows,
            lora.up.rows,
            rank,
            lora.scale,
            1.0,
        )
    }
}

impl ImageGpu {
    /// Adds both factors' gradients for `grad`, the gradient of the adapter's
    /// output, and with `grad_input` the adapter's share of `dL/d(input)`.
    fn lora_backward(
        &self,
        lora: &mut DeviceLora,
        input: &Tensor,
        grad: &Tensor,
        grad_input: Option<&mut Tensor>,
    ) -> Result<(), NetworkError> {
        let (rows, rank, inputs) = (input.rows, lora.down.rows, input.cols);
        let mut hidden = self.uninit(rows, rank)?;
        act_rhs_transposed::<f16, _>(
            &self.context,
            &input.bytes(),
            inputs,
            &lora.down.bytes(),
            inputs,
            true,
            true,
            &mut hidden.data,
            rank,
            rows,
            rank,
            inputs,
            lora.scale,
            0.0,
        )?;
        let [grad_down, grad_up] = &mut lora.grads;
        // `dL/d(up) += grad^T . (scale * down . x)`.
        self.sum_lhs_transposed(
            (&grad.data.slice(..), grad.cols, 0),
            (&hidden.data.slice(..), rank, 0),
            (grad_up, rank, 0),
            (rows, grad.cols, rank, 1),
        )?;
        let mut grad_hidden = self.uninit(rows, rank)?;
        act_plain::<f16, _>(
            &self.context,
            &grad.bytes(),
            grad.cols,
            &lora.up.bytes(),
            rank,
            true,
            true,
            &mut grad_hidden.data,
            rank,
            rows,
            rank,
            grad.cols,
            lora.scale,
            0.0,
        )?;
        self.sum_lhs_transposed(
            (&grad_hidden.data.slice(..), rank, 0),
            (&input.data.slice(..), inputs, 0),
            (grad_down, inputs, 0),
            (rows, rank, inputs, 1),
        )?;
        if let Some(grad_input) = grad_input {
            act_plain::<f16, _>(
                &self.context,
                &grad_hidden.bytes(),
                rank,
                &lora.down.bytes(),
                inputs,
                true,
                true,
                &mut grad_input.data,
                inputs,
                rows,
                inputs,
                rank,
                1.0,
                1.0,
            )?;
        }
        Ok(())
    }

    /// `dL/d(input)` of a projection by `weight`, `[outputs, inputs]`:
    /// `grad . weight`, added to `out` with `beta` one.
    fn dense_backward_into(
        &self,
        weight: &CudaView<'_, u8>,
        inputs: usize,
        grad: &Tensor,
        out: &mut Tensor,
        beta: f32,
    ) -> Result<(), NetworkError> {
        act_plain::<f16, _>(
            &self.context,
            &grad.bytes(),
            grad.cols,
            weight,
            inputs,
            true,
            true,
            &mut out.data,
            inputs,
            grad.rows,
            inputs,
            grad.cols,
            1.0,
            beta,
        )
    }

    fn dense_backward(&self, weight: &Tensor, grad: &Tensor) -> Result<Tensor, NetworkError> {
        let mut out = self.uninit(grad.rows, weight.cols)?;
        self.dense_backward_into(&weight.bytes(), weight.cols, grad, &mut out, 0.0)?;
        Ok(out)
    }

    /// `joined`'s first `channels` channels, and the rest.
    fn split(&self, joined: &Map, channels: usize) -> Result<(Map, Map), NetworkError> {
        let (pixels, split) = (joined.pixels(), channels * joined.pixels());
        let part = |range: std::ops::Range<usize>| {
            let mut part = self.uninit(range.len() / pixels, pixels)?;
            self.context
                .stream
                .memcpy_dtod(&joined.tensor.data.slice(range), &mut part.data)
                .map_err(cuda_err("device to device copy"))?;
            Ok::<_, NetworkError>(Map {
                tensor: part,
                height: joined.height,
                width: joined.width,
            })
        };
        Ok((part(0..split)?, part(split..joined.tensor.len())?))
    }
}

fn pop<T>(list: &mut Vec<T>) -> Result<T, NetworkError> {
    list.pop()
        .ok_or_else(|| NetworkError::InvalidConfig("the tape is not from this UNet".into()))
}

impl DeviceUnet {
    /// [`DeviceUnet::forward`] with `adapters` on the attention projections,
    /// keeping what the backward pass reads. `latent` and `time` are what
    /// [`crate::unet_lora::prepare`] makes of a latent and a noise level.
    pub fn forward_adapted(
        &self,
        adapters: &DeviceAdapters,
        latent: &FeatureMap,
        time: &[f32],
        text: &Matrix,
    ) -> Result<(FeatureMap, DeviceTape), NetworkError> {
        let gpu = &self.gpu;
        let time = self.time(time)?;
        let text = gpu.upload_matrix(text)?;
        let mut recorder = Recorder {
            adapters,
            next: 0,
            records: Records::default(),
        };
        let (sample, skips) = self.down(latent, &time, &text, None, Some(&mut recorder))?;
        let mut sample = self.up(sample, skips, &time, &text, Some(&mut recorder))?;
        if recorder.next != adapters.transformers.len() {
            return Err(NetworkError::InvalidConfig(
                "the adapters were uploaded for another UNet".into(),
            ));
        }
        let out = clone_map(gpu, &sample)?;
        let (norm_out, conv_out) = self.out.as_ref().ok_or_else(|| {
            NetworkError::InvalidConfig("a ControlNet has no output to train".into())
        })?;
        gpu.group_norm(norm_out, &mut sample, 0)?;
        let output = gpu.conv(conv_out, &sample)?;
        let tape = DeviceTape {
            records: recorder.records,
            out,
            text,
        };
        Ok((download_map(gpu, &output)?, tape))
    }

    /// Adds `dL/d(adapter)` to the gradients `adapters` keep on the device,
    /// given `dL/d(prediction)` of the pass that made `tape`. The way back
    /// reads the tape from its end: the output, the way up handing one
    /// gradient back to each skip, the middle, and the way down.
    pub fn backward_adapted(
        &self,
        adapters: &mut DeviceAdapters,
        tape: DeviceTape,
        grad: &FeatureMap,
    ) -> Result<(), NetworkError> {
        let gpu = &*self.gpu;
        let DeviceTape {
            mut records,
            out,
            text,
        } = tape;
        if records.transformers.len() != adapters.transformers.len() {
            return Err(NetworkError::InvalidConfig(
                "the tape is not from these adapters".into(),
            ));
        }
        let mut slots = adapters.transformers.iter_mut().rev();
        let mut transformer = |model: &DeviceTransformer, records: &mut Records, grad: Map| {
            let slot = slots
                .next()
                .expect("as many adapters as transformer records");
            self.transformer_backward(model, pop(&mut records.transformers)?, slot, &text, grad)
        };
        let (norm_out, conv_out) = self.out.as_ref().ok_or_else(|| {
            NetworkError::InvalidConfig("a ControlNet has no output to train".into())
        })?;
        let mut normed = clone_map(gpu, &out)?;
        gpu.group_norm(norm_out, &mut normed, -1)?;
        let (height, width) = (out.height, out.width);
        let mut grad = gpu.conv_backward(
            conv_out,
            &upload_map(gpu, grad)?,
            height,
            width,
            COLUMN_BUDGET,
        )?;
        gpu.silu_backward(&mut grad.tensor, &normed.tensor)?;
        gpu.group_norm_backward(norm_out, &out, &mut grad)?;
        drop((normed, out));

        let mut skips = Vec::new();
        for block in self.up.iter().rev() {
            if let Some(resampler) = &block.resampler {
                let (height, width) = pop(&mut records.resampled)?;
                let small = gpu.conv_backward(resampler, &grad, height, width, COLUMN_BUDGET)?;
                grad = gpu.upsample_backward(&small, 2)?;
            }
            for (index, resnet) in block.resnets.iter().enumerate().rev() {
                if let Some(model) = block.attentions.get(index) {
                    grad = transformer(model, &mut records, grad)?;
                }
                let joined = self.resnet_backward(resnet, pop(&mut records.resnets)?, &grad)?;
                let (below, skip) = gpu.split(&joined, pop(&mut records.joined)?)?;
                skips.push(skip);
                grad = below;
            }
        }

        let (first, middle, second) = &self.middle;
        grad = self.resnet_backward(second, pop(&mut records.resnets)?, &grad)?;
        grad = transformer(middle, &mut records, grad)?;
        grad = self.resnet_backward(first, pop(&mut records.resnets)?, &grad)?;

        for block in self.down.iter().rev() {
            if let Some(resampler) = &block.resampler {
                gpu.add(&mut grad.tensor, &pop(&mut skips)?.tensor)?;
                let (height, width) = pop(&mut records.resampled)?;
                grad = gpu.conv_backward(resampler, &grad, height, width, COLUMN_BUDGET)?;
            }
            for (index, resnet) in block.resnets.iter().enumerate().rev() {
                gpu.add(&mut grad.tensor, &pop(&mut skips)?.tensor)?;
                if let Some(model) = block.attentions.get(index) {
                    grad = transformer(model, &mut records, grad)?;
                }
                grad = self.resnet_backward(resnet, pop(&mut records.resnets)?, &grad)?;
            }
        }
        // What is left is the gradient of `conv_in`'s output and the first
        // skip's, which reach only frozen weights and the latent.
        Ok(())
    }

    /// `dL/d(input)` of [`DeviceUnet::resnet`], given its input and the
    /// hidden map its second norm read.
    fn resnet_backward(
        &self,
        resnet: &DeviceResnet,
        (input, hidden): (Map, Map),
        grad: &Map,
    ) -> Result<Map, NetworkError> {
        let gpu = &*self.gpu;
        let mut normed = clone_map(gpu, &hidden)?;
        gpu.group_norm(&resnet.norm2, &mut normed, -1)?;
        let (height, width) = (hidden.height, hidden.width);
        let mut inner = gpu.conv_backward(&resnet.conv2, grad, height, width, COLUMN_BUDGET)?;
        gpu.silu_backward(&mut inner.tensor, &normed.tensor)?;
        // The noise level's offset adds to `hidden` and passes the gradient on.
        gpu.group_norm_backward(&resnet.norm2, &hidden, &mut inner)?;
        drop((normed, hidden));

        let mut normed = clone_map(gpu, &input)?;
        gpu.group_norm(&resnet.norm1, &mut normed, -1)?;
        let (height, width) = (input.height, input.width);
        let mut output = gpu.conv_backward(&resnet.conv1, &inner, height, width, COLUMN_BUDGET)?;
        gpu.silu_backward(&mut output.tensor, &normed.tensor)?;
        gpu.group_norm_backward(&resnet.norm1, &input, &mut output)?;
        match &resnet.shortcut {
            Some(shortcut) => {
                let skip = gpu.conv_backward(shortcut, grad, height, width, COLUMN_BUDGET)?;
                gpu.add(&mut output.tensor, &skip.tensor)?;
            }
            None => gpu.add(&mut output.tensor, &grad.tensor)?,
        }
        Ok(output)
    }

    fn transformer_backward(
        &self,
        transformer: &DeviceTransformer,
        (input, blocks): (Map, Vec<BlockRecord>),
        adapters: &mut [[[Option<DeviceLora>; 4]; 2]],
        text: &Tensor,
        grad: Map,
    ) -> Result<Map, NetworkError> {
        let (gpu, eps, heads) = (&*self.gpu, self.eps, transformer.heads);
        let (height, width) = (grad.height, grad.width);
        let mut grad_tokens = match &transformer.output {
            DeviceProjection::Convolution(conv) => {
                gpu.to_tokens(&gpu.conv_backward(conv, &grad, height, width, COLUMN_BUDGET)?)?
            }
            DeviceProjection::Linear(dense) => {
                gpu.dense_backward(&dense.weight, &gpu.to_tokens(&grad)?)?
            }
        };
        for ((block, record), lora) in transformer
            .blocks
            .iter()
            .zip(blocks)
            .zip(adapters.iter_mut())
            .rev()
        {
            let BlockRecord {
                before: [before_self, before_cross, before_feed],
                own,
                cross,
                projected,
            } = record;
            let grad_hidden = gpu.dense_backward(&block.output.weight, &grad_tokens)?;
            let bias = block.gate.bias.as_ref();
            let grad_projected = gpu.gelu_gate_backward(&projected, bias, &grad_hidden)?;
            let mut grad_normed = gpu.dense_backward(&block.gate.weight, &grad_projected)?;
            gpu.layer_norm_backward(&block.norm3, &before_feed, eps, &mut grad_normed)?;
            gpu.add(&mut grad_tokens, &grad_normed)?;

            let normed = gpu.layer_norm(&block.norm2, &before_cross, eps)?;
            let (attention, lora_cross) = (&block.cross, &mut lora[1]);
            let mut grad_normed = self.attend_backward(
                attention,
                cross,
                &normed,
                Some(text),
                heads,
                lora_cross,
                &grad_tokens,
            )?;
            gpu.layer_norm_backward(&block.norm2, &before_cross, eps, &mut grad_normed)?;
            gpu.add(&mut grad_tokens, &grad_normed)?;

            let normed = gpu.layer_norm(&block.norm1, &before_self, eps)?;
            let (attention, lora_own) = (&block.attention, &mut lora[0]);
            let mut grad_normed =
                self.attend_backward(attention, own, &normed, None, heads, lora_own, &grad_tokens)?;
            gpu.layer_norm_backward(&block.norm1, &before_self, eps, &mut grad_normed)?;
            gpu.add(&mut grad_tokens, &grad_normed)?;
        }
        let mut output = match &transformer.input {
            DeviceProjection::Convolution(conv) => {
                let grad = gpu.to_map(&grad_tokens, height, width)?;
                gpu.conv_backward(conv, &grad, height, width, COLUMN_BUDGET)?
            }
            DeviceProjection::Linear(dense) => gpu.to_map(
                &gpu.dense_backward(&dense.weight, &grad_tokens)?,
                height,
                width,
            )?,
        };
        gpu.group_norm_backward(&transformer.norm, &input, &mut output)?;
        gpu.add(&mut output.tensor, &grad.tensor)?;
        Ok(output)
    }

    /// `dL/d(tokens)` of [`DeviceUnet::attend`], given the gradient of what it
    /// added to the residual, adding every adapter's gradients. The prompt is
    /// frozen, so cross-attention hands nothing back to it.
    #[allow(clippy::too_many_arguments)]
    fn attend_backward(
        &self,
        attention: &DeviceAttention,
        record: AttentionRecord,
        tokens: &Tensor,
        context: Option<&Tensor>,
        heads: usize,
        lora: &mut [Option<DeviceLora>; 4],
        grad: &Tensor,
    ) -> Result<Tensor, NetworkError> {
        let gpu = &*self.gpu;
        let units = attention.stacked.units();
        let AttentionRecord {
            queries,
            projected,
            attended,
        } = record;
        let mut grad_attended = gpu.dense_backward(&attention.output.weight, grad)?;
        if let Some(adapter) = &mut lora[3] {
            gpu.lora_backward(adapter, &attended, grad, Some(&mut grad_attended))?;
        }
        let (q, k, v) = match &queries {
            None => (
                projected.part(0, units),
                projected.part(1, units),
                projected.part(2, units),
            ),
            Some(queries) => (
                queries.view(),
                projected.part(0, units),
                projected.part(1, units),
            ),
        };
        let grads = gpu.attention_backward(q, k, v, heads, &grad_attended, SCORE_BUDGET)?;
        drop(grad_attended);
        let stack = &attention.stacked.dense.weight;
        let (stack_bytes, inputs) = (stack.bytes(), stack.cols);
        let rows = |slot: usize| {
            stack_bytes.slice(slot * units * inputs * 2..(slot + 1) * units * inputs * 2)
        };
        let mut grad_tokens = match &attention.query {
            Some(query) => gpu.dense_backward(&query.weight, &grads[0])?,
            None => {
                let mut out = gpu.uninit(tokens.rows, inputs)?;
                gpu.dense_backward_into(&rows(0), inputs, &grads[0], &mut out, 0.0)?;
                out
            }
        };
        if let Some(adapter) = &mut lora[0] {
            gpu.lora_backward(adapter, tokens, &grads[0], Some(&mut grad_tokens))?;
        }
        for slot in 1..3 {
            match context {
                None => {
                    gpu.dense_backward_into(
                        &rows(slot),
                        inputs,
                        &grads[slot],
                        &mut grad_tokens,
                        1.0,
                    )?;
                    if let Some(adapter) = &mut lora[slot] {
                        gpu.lora_backward(adapter, tokens, &grads[slot], Some(&mut grad_tokens))?;
                    }
                }
                Some(context) => {
                    if let Some(adapter) = &mut lora[slot] {
                        gpu.lora_backward(adapter, context, &grads[slot], None)?;
                    }
                }
            }
        }
        Ok(grad_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{assert_close, device};
    use super::*;
    use crate::unet::tests::{conditioning, latent, random, tiny};
    use crate::unet_lora::{prepare, tests::adapters};

    #[test]
    fn an_adapted_device_pass_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        let config = tiny();
        let unet = random(&config, "rusting_brain_cuda_unet_lora.safetensors");
        // Ten times the host test's adapters, so their share of the
        // prediction stands well clear of FP16 rounding.
        let mut lora = adapters(&unet);
        lora.adapters
            .values_mut()
            .for_each(|adapter| adapter.scale *= 10.0);
        let (latent, conditioning) = (latent(&config, 8), conditioning(&config, 8, 1.0));
        let sigma = 1.5;
        let (expected, _) = lora.forward(&unet, &latent, sigma, &conditioning).unwrap();
        let (plain, _) = UnetLora::default()
            .forward(&unet, &latent, sigma, &conditioning)
            .unwrap();

        let model = DeviceUnet::upload(&gpu, &unet).unwrap();
        let adapters = DeviceAdapters::upload(&gpu, &unet, &lora).unwrap();
        let (scaled, time) = prepare(&unet, &latent, sigma, &conditioning).unwrap();
        let (actual, tape) = model
            .forward_adapted(&adapters, &scaled, &time, &conditioning.text)
            .unwrap();
        assert_close("adapted UNet", &actual.data, &expected.data, 0.05);

        // A pass that dropped an adapter could still pass the bar above, so
        // the adapters' own share is compared too: device minus unadapted device against host minus unadapted host.
        let bare = DeviceAdapters::upload(&gpu, &unet, &UnetLora::default()).unwrap();
        let (unadapted, _) = model
            .forward_adapted(&bare, &scaled, &time, &conditioning.text)
            .unwrap();
        let delta = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a - b).collect::<Vec<_>>();
        assert_close(
            "the adapters' share",
            &delta(&actual.data, &unadapted.data),
            &delta(&expected.data, &plain.data),
            0.05,
        );

        let records = &tape.records;
        assert_eq!(records.transformers.len(), adapters.transformers.len());
        let resnets = unet.encoder.down.iter().chain(&unet.up);
        let resnets = resnets.map(|block| block.resnets.len()).sum::<usize>() + 2;
        assert_eq!(records.resnets.len(), resnets);
        assert_eq!(
            records.joined.len(),
            unet.up.iter().map(|b| b.resnets.len()).sum::<usize>()
        );
    }

    #[test]
    fn adapted_device_gradients_match_the_host_or_skip_without_a_device() {
        let Some(gpu) = device() else { return };
        let config = tiny();
        let unet = random(&config, "rusting_brain_cuda_unet_lora_grad.safetensors");
        let mut lora = adapters(&unet);
        lora.adapters
            .values_mut()
            .for_each(|adapter| adapter.scale *= 10.0);
        let (latent, conditioning) = (latent(&config, 8), conditioning(&config, 8, 1.0));
        let sigma = 1.5;
        let mut host = lora.clone();
        let (expected, tape) = host.forward(&unet, &latent, sigma, &conditioning).unwrap();
        let mut grad = expected.clone();
        grad.data
            .iter_mut()
            .enumerate()
            .for_each(|(i, value)| *value = ((i % 13) as f32 - 6.0) * 0.1);
        host.backward(&unet, &tape, &grad).unwrap();

        let model = DeviceUnet::upload(&gpu, &unet).unwrap();
        let mut adapters = DeviceAdapters::upload(&gpu, &unet, &lora).unwrap();
        let (scaled, time) = prepare(&unet, &latent, sigma, &conditioning).unwrap();
        let (_, tape) = model
            .forward_adapted(&adapters, &scaled, &time, &conditioning.text)
            .unwrap();
        model.backward_adapted(&mut adapters, tape, &grad).unwrap();
        let mut device = lora.clone();
        adapters.add_gradients(&gpu, &mut device).unwrap();

        // A missing or misplaced term puts its tensor near 100%; FP16 noise
        // stays at a few percent.
        let (mut total_norm, mut total_error) = (0.0f32, 0.0f32);
        let mut worst = (0.0f32, String::new());
        for (key, expected) in &host.adapters {
            let actual = &device.adapters[key];
            let pairs = [
                ("down", &expected.down.grad, &actual.down.grad),
                ("up", &expected.up.grad, &actual.up.grad),
            ];
            for (name, expected, actual) in pairs {
                let norm = expected.data.iter().map(|v| v * v).sum::<f32>().sqrt();
                let error = expected
                    .data
                    .iter()
                    .zip(&actual.data)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                assert!(norm > 0.0, "{key} {name} has no gradient");
                total_norm += norm * norm;
                total_error += error * error;
                if error / norm > worst.0 {
                    worst = (error / norm, format!("{key} {name}"));
                }
            }
        }
        let overall = (total_error / total_norm).sqrt();
        eprintln!(
            "relative gradient error {overall} overall, worst {} at {}",
            worst.0, worst.1
        );
        assert!(overall < 2e-2, "gradients off by {overall} overall");
        assert!(worst.0 < 1e-1, "{} off by {}", worst.1, worst.0);
    }

    /// G10 b-iii-c: one adapted forward and backward of Stable Diffusion
    /// 1.5's UNet at 512x512 (a 64x64 latent, 77 prompt tokens), timed for
    /// `docs/baseline.md`. The weights are random: the step's cost is its
    /// shapes, and the 1.5 checkpoint is not on this machine.
    ///
    /// ```bash
    /// cargo test --release --features cuda --lib an_sd15_step -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "builds a 3.4 GB random UNet; run by hand to time a step"]
    fn an_sd15_step_at_512_is_timed() {
        let Some(gpu) = device() else { return };
        let config = crate::unet::UnetConfig::default();
        let unet = random(&config, "rusting_brain_sd15_step.safetensors");
        let lora = adapters(&unet);
        let mut conditioning = conditioning(&config, 64, 1.0);
        conditioning.text = Matrix::new(77, config.cross_dim);
        let latent = latent(&config, 64);
        let model = DeviceUnet::upload(&gpu, &unet).unwrap();
        let mut adapters = DeviceAdapters::upload(&gpu, &unet, &lora).unwrap();
        let (scaled, time) = prepare(&unet, &latent, 1.5, &conditioning).unwrap();
        let mut grad = FeatureMap::new(config.out_channels, 64, 64);
        grad.data
            .iter_mut()
            .enumerate()
            .for_each(|(i, value)| *value = ((i % 7) as f32 - 3.0) * 1e-3);

        let mut step = || {
            let started = std::time::Instant::now();
            let (_, tape) = model
                .forward_adapted(&adapters, &scaled, &time, &conditioning.text)
                .unwrap();
            gpu.synchronize().unwrap();
            let forward = started.elapsed();
            model.backward_adapted(&mut adapters, tape, &grad).unwrap();
            gpu.synchronize().unwrap();
            (forward, started.elapsed())
        };
        step();
        let runs: Vec<_> = (0..5).map(|_| step()).collect();
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        let best = runs.iter().min_by_key(|run| run.1).unwrap();
        eprintln!(
            "SD 1.5 adapted step at 512x512: {:.1} ms (forward {:.1} ms, backward {:.1} ms), best of 5; all {:?}",
            ms(best.1),
            ms(best.0),
            ms(best.1 - best.0),
            runs.iter().map(|run| ms(run.1).round()).collect::<Vec<_>>()
        );
    }
}
