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

/// One adapter: `scale * up . (down . x)`, both factors FP16.
pub(crate) struct DeviceLora {
    down: Tensor,
    up: Tensor,
    scale: f32,
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
                            *target = Some(DeviceLora {
                                down: gpu.upload_matrix(&adapter.down.value)?,
                                up: gpu.upload_matrix(&adapter.up.value)?,
                                scale: adapter.scale,
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
#[allow(dead_code)] // NOTE: read by the device backward, roadmap G10 b-iii.
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
#[allow(dead_code)]
pub(crate) struct BlockRecord {
    pub(super) before: [Tensor; 3],
    pub(super) own: AttentionRecord,
    pub(super) cross: AttentionRecord,
    pub(super) projected: Tensor,
}

/// Self-attention keeps its stacked `[q | k | v]`; cross-attention its queries
/// and its stacked `[k | v]` of the prompt.
#[allow(dead_code)]
pub(crate) struct AttentionRecord {
    pub(super) queries: Option<Tensor>,
    pub(super) projected: Tensor,
    pub(super) attended: Tensor,
}

/// What [`DeviceUnet::forward_adapted`] keeps for the backward pass.
#[allow(dead_code)]
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
}
