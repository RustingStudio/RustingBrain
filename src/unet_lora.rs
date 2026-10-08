//! Low-rank adapters on a UNet's attention projections, and the backward pass
//! that trains them.
//!
//! This is the host reference the device path is checked against, as
//! [`crate::dit_lora`] is for the DiT. Adapters sit where diffusers' own
//! `train_text_to_image_lora` puts them: `to_q`, `to_k`, `to_v` and
//! `to_out.0` of both attentions in every transformer block. The base weights
//! and the conditioning are frozen, so the backward keeps only adapter
//! gradients, but it still runs back through every resnet, sampler and skip,
//! because an adapter on the way down changes everything after it.
//!
//! NOTE: no ControlNet, no inpainting channels; they are frozen inputs a
//! training step can add once someone trains with them.

use crate::conv::{Conv2d, Dense, FeatureMap, GroupNorm, silu, upsample_nearest};
use crate::dit_lora::{attention_backward, layer_norm_backward};
use crate::ffn::{gelu, gelu_derivative, silu_derivative};
use crate::matrix::Matrix;
use crate::mmdit::{Conditioning, attention};
use crate::network::NetworkError;
use crate::param::{Lora, Param};
use crate::unet::{Attention, Projection, Resnet, Transformer, Unet, concatenate};
use std::collections::BTreeMap;

/// Adapters by the diffusers name of the projection they add to, such as
/// `down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k`.
#[derive(Clone, Debug, Default)]
pub struct UnetLora {
    pub adapters: BTreeMap<String, Lora>,
}

pub(crate) const PROJECTIONS: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// Every transformer in the model, under its diffusers name.
pub(crate) fn transformers(unet: &Unet) -> Vec<(String, &Transformer)> {
    let mut list = Vec::new();
    for (level, block) in unet.encoder.down.iter().enumerate() {
        for (index, transformer) in block.attentions.iter().enumerate() {
            list.push((
                format!("down_blocks.{level}.attentions.{index}"),
                transformer,
            ));
        }
    }
    list.push(("mid_block.attentions.0".into(), &unet.encoder.middle.1));
    for (level, block) in unet.up.iter().enumerate() {
        for (index, transformer) in block.attentions.iter().enumerate() {
            list.push((format!("up_blocks.{level}.attentions.{index}"), transformer));
        }
    }
    list
}

/// Every projection an adapter can sit on, by name.
fn projections(unet: &Unet) -> BTreeMap<String, &Dense> {
    let mut map = BTreeMap::new();
    for (prefix, transformer) in transformers(unet) {
        for (index, block) in transformer.blocks.iter().enumerate() {
            for (name, attention) in [("attn1", &block.attention), ("attn2", &block.cross)] {
                let denses = [
                    &attention.query,
                    &attention.key,
                    &attention.value,
                    &attention.output,
                ];
                for (projection, dense) in PROJECTIONS.iter().zip(denses) {
                    let key = format!("{prefix}.transformer_blocks.{index}.{name}.{projection}");
                    map.insert(key, dense);
                }
            }
        }
    }
    map
}

impl UnetLora {
    /// Reads a PEFT file trained against a diffusers UNet (`{module}.lora_A`
    /// and `lora_B`, optionally under `unet.`), with `scale` the
    /// `alpha / rank` it was trained at.
    pub fn read_peft(
        path: impl AsRef<std::path::Path>,
        unet: &Unet,
        scale: f32,
    ) -> Result<Self, NetworkError> {
        let targets = projections(unet);
        let mut file = crate::safetensors::SafeTensors::open(path)?;
        let names: Vec<String> = file.names().map(str::to_string).collect();
        let mut result = Self::default();
        for name in &names {
            let Some(module) = name.strip_suffix(".lora_A.weight") else {
                continue;
            };
            let module = module.strip_prefix("unet.").unwrap_or(module);
            let dense = targets.get(module).ok_or_else(|| {
                NetworkError::InvalidConfig(format!("{name} adapts no attention projection"))
            })?;
            let down = file.matrix(name)?;
            let up = file.matrix(&name.replace(".lora_A.", ".lora_B."))?;
            if down.cols != dense.in_dim() || up.rows != dense.out_dim() || up.cols != down.rows {
                return Err(NetworkError::InvalidConfig(format!(
                    "the adapter on {module} does not fit its {}x{} projection",
                    dense.out_dim(),
                    dense.in_dim()
                )));
            }
            let lora = Lora {
                down: Param::new(down),
                up: Param::new(up),
                scale,
            };
            result.adapters.insert(module.to_string(), lora);
        }
        Ok(result)
    }

    /// Every adapter factor, for an optimizer.
    pub fn params_mut(&mut self) -> Vec<&mut Param> {
        self.adapters
            .values_mut()
            .flat_map(Lora::params_mut)
            .collect()
    }

    /// [`Unet::forward`] with the adapters added, and what the backward pass
    /// needs of it.
    pub fn forward(
        &self,
        unet: &Unet,
        latent: &FeatureMap,
        sigma: f32,
        conditioning: &Conditioning,
    ) -> Result<(FeatureMap, Tape), NetworkError> {
        let config = unet.config();
        let (scaled, time) = prepare(unet, latent, sigma, conditioning)?;
        let text = &conditioning.text;
        let eps = config.eps;
        let names: Vec<String> = transformers(unet)
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        let mut names = names.iter();
        let mut run = |transformer: &Transformer, sample: &FeatureMap| {
            let name = names.next().expect("one name a transformer");
            self.transformer(transformer, name, sample, text, eps)
        };

        let mut sample = unet.encoder.conv_in.forward(&scaled)?;
        let mut skips = vec![sample.clone()];
        let mut down = Vec::new();
        for block in &unet.encoder.down {
            let mut rungs = Vec::new();
            for (index, resnet) in block.resnets.iter().enumerate() {
                let (output, resnet_tape) = resnet_forward(resnet, &sample, &time)?;
                sample = output;
                let mut transformer_tape = None;
                if let Some(transformer) = block.attentions.get(index) {
                    let (output, tape) = run(transformer, &sample)?;
                    sample = output;
                    transformer_tape = Some(tape);
                }
                rungs.push((0, resnet_tape, transformer_tape));
                skips.push(sample.clone());
            }
            let resampled = match &block.resampler {
                Some(resampler) => {
                    let input = sample.clone();
                    sample = resampler.forward(&sample)?;
                    skips.push(sample.clone());
                    Some(input)
                }
                None => None,
            };
            down.push((rungs, resampled));
        }

        let (first, middle, second) = &unet.encoder.middle;
        let (output, first) = resnet_forward(first, &sample, &time)?;
        let (output, middle) = run(middle, &output)?;
        let (output, second) = resnet_forward(second, &output, &time)?;
        sample = output;
        let middle = (first, middle, second);

        let mut up = Vec::new();
        for block in &unet.up {
            let mut rungs = Vec::new();
            for (index, resnet) in block.resnets.iter().enumerate() {
                let skip = skips.pop().ok_or_else(|| {
                    NetworkError::InvalidConfig(
                        "the way up has more rungs than the way down".into(),
                    )
                })?;
                let channels = sample.channels;
                let (output, resnet_tape) =
                    resnet_forward(resnet, &concatenate(&sample, &skip), &time)?;
                sample = output;
                let mut transformer_tape = None;
                if let Some(transformer) = block.attentions.get(index) {
                    let (output, tape) = run(transformer, &sample)?;
                    sample = output;
                    transformer_tape = Some(tape);
                }
                rungs.push((channels, resnet_tape, transformer_tape));
            }
            let resampled = match &block.resampler {
                Some(resampler) => {
                    let input = upsample_nearest(&sample, 2);
                    sample = resampler.forward(&input)?;
                    Some(input)
                }
                None => None,
            };
            up.push((rungs, resampled));
        }
        if !skips.is_empty() {
            return Err(NetworkError::InvalidConfig(
                "the way down has rungs the way up never reads".into(),
            ));
        }

        let norm_input = sample.clone();
        unet.norm_out.forward(&mut sample)?;
        let normed = sample.clone();
        silu(&mut sample);
        let output = unet.conv_out.forward(&sample)?;
        let tape = Tape {
            down,
            middle,
            up,
            norm_input,
            normed,
        };
        Ok((output, tape))
    }

    /// Adds `dL/d(adapter)` to every adapter's gradients, given
    /// `dL/d(prediction)`.
    pub fn backward(
        &mut self,
        unet: &Unet,
        tape: &Tape,
        grad: &FeatureMap,
    ) -> Result<(), NetworkError> {
        let eps = unet.config().eps;
        let mut list = transformers(unet);
        let mut next = || list.pop().expect("one transformer a tape");

        let grad = conv_backward(&unet.conv_out, grad, &tape.normed);
        let grad = silu_backward(&grad, &tape.normed);
        let mut grad = group_norm_backward(&unet.norm_out, &tape.norm_input, &grad);

        // The way up hands back one gradient a skip, in the order the way down
        // pushed them.
        let mut skips = Vec::new();
        for (block, (rungs, resampled)) in unet.up.iter().zip(&tape.up).rev() {
            if let (Some(resampler), Some(input)) = (&block.resampler, resampled) {
                grad = upsample_backward(&conv_backward(resampler, &grad, input));
            }
            for (resnet, (channels, resnet_tape, transformer_tape)) in
                block.resnets.iter().zip(rungs).rev()
            {
                if let Some(transformer_tape) = transformer_tape {
                    let (name, transformer) = next();
                    grad =
                        self.transformer_backward(transformer, &name, transformer_tape, &grad, eps);
                }
                let joined = resnet_backward(resnet, resnet_tape, &grad);
                let split = channels * joined.pixels();
                skips.push(FeatureMap {
                    channels: joined.channels - channels,
                    height: joined.height,
                    width: joined.width,
                    data: joined.data[split..].to_vec(),
                });
                grad = FeatureMap {
                    channels: *channels,
                    height: joined.height,
                    width: joined.width,
                    data: joined.data[..split].to_vec(),
                };
            }
        }

        let (first, middle, second) = &unet.encoder.middle;
        grad = resnet_backward(second, &tape.middle.2, &grad);
        let (name, _) = next();
        grad = self.transformer_backward(middle, &name, &tape.middle.1, &grad, eps);
        grad = resnet_backward(first, &tape.middle.0, &grad);

        let mut add_skip = |grad: &mut FeatureMap| {
            let skip = skips.pop().expect("a skip gradient for every rung down");
            add_maps(grad, &skip);
        };
        for (block, (rungs, resampled)) in unet.encoder.down.iter().zip(&tape.down).rev() {
            if let (Some(resampler), Some(input)) = (&block.resampler, resampled) {
                add_skip(&mut grad);
                grad = conv_backward(resampler, &grad, input);
            }
            for (resnet, (_, resnet_tape, transformer_tape)) in
                block.resnets.iter().zip(rungs).rev()
            {
                add_skip(&mut grad);
                if let Some(transformer_tape) = transformer_tape {
                    let (name, transformer) = next();
                    grad =
                        self.transformer_backward(transformer, &name, transformer_tape, &grad, eps);
                }
                grad = resnet_backward(resnet, resnet_tape, &grad);
            }
        }
        // What is left is the gradient of `conv_in`'s output and the first
        // skip's, which reach only frozen weights and the input.
        Ok(())
    }

    fn transformer(
        &self,
        transformer: &Transformer,
        name: &str,
        input: &FeatureMap,
        text: &Matrix,
        eps: f32,
    ) -> Result<(FeatureMap, TransformerTape), NetworkError> {
        let (height, width) = (input.height, input.width);
        let mut normed = input.clone();
        transformer.norm.forward(&mut normed)?;
        let mut tokens = match &transformer.input {
            Projection::Convolution(conv) => conv.forward(&normed)?.to_tokens(),
            Projection::Linear(dense) => dense.forward(&normed.to_tokens())?,
        };
        let mut blocks = Vec::new();
        for (index, block) in transformer.blocks.iter().enumerate() {
            let prefix = format!("{name}.transformer_blocks.{index}");
            let before_self = tokens.clone();
            let normed1 = block.norm1.forward(&tokens, eps);
            let (attended, own) = self.attention(
                &block.attention,
                &format!("{prefix}.attn1"),
                &normed1,
                &normed1,
                transformer.heads,
            )?;
            add(&mut tokens, &attended);

            let before_cross = tokens.clone();
            let normed2 = block.norm2.forward(&tokens, eps);
            let (attended, cross) = self.attention(
                &block.cross,
                &format!("{prefix}.attn2"),
                &normed2,
                text,
                transformer.heads,
            )?;
            add(&mut tokens, &attended);

            let before_feed = tokens.clone();
            let normed3 = block.norm3.forward(&tokens, eps);
            let projected = block.gate.forward(&normed3)?;
            let hidden = geglu(&projected);
            add(&mut tokens, &block.output.forward(&hidden)?);
            blocks.push(BlockTape {
                before: [before_self, before_cross, before_feed],
                own,
                cross,
                projected,
            });
        }
        let mut output = match &transformer.output {
            Projection::Convolution(conv) => {
                conv.forward(&FeatureMap::from_tokens(&tokens, height, width)?)?
            }
            Projection::Linear(dense) => {
                FeatureMap::from_tokens(&dense.forward(&tokens)?, height, width)?
            }
        };
        add_maps(&mut output, input);
        let tape = TransformerTape {
            input: input.clone(),
            normed,
            blocks,
            tokens,
        };
        Ok((output, tape))
    }

    fn transformer_backward(
        &mut self,
        transformer: &Transformer,
        name: &str,
        tape: &TransformerTape,
        grad: &FeatureMap,
        eps: f32,
    ) -> FeatureMap {
        let (height, width) = (grad.height, grad.width);
        let mut grad_tokens = match &transformer.output {
            Projection::Convolution(conv) => {
                let input = FeatureMap::from_tokens(&tape.tokens, height, width).unwrap();
                conv_backward(conv, grad, &input).to_tokens()
            }
            Projection::Linear(dense) => dense_backward(dense, &grad.to_tokens()),
        };
        for (index, (block, block_tape)) in transformer
            .blocks
            .iter()
            .zip(&tape.blocks)
            .enumerate()
            .rev()
        {
            let prefix = format!("{name}.transformer_blocks.{index}");
            let [before_self, before_cross, before_feed] = &block_tape.before;

            let grad_hidden = dense_backward(&block.output, &grad_tokens);
            let grad_projected = geglu_backward(&block_tape.projected, &grad_hidden);
            let grad_normed = dense_backward(&block.gate, &grad_projected);
            add(
                &mut grad_tokens,
                &norm_backward(&block.norm3, before_feed, eps, &grad_normed),
            );

            let (grad_normed, _) = self.attention_backward(
                &block.cross,
                &format!("{prefix}.attn2"),
                &block_tape.cross,
                transformer.heads,
                &grad_tokens,
            );
            add(
                &mut grad_tokens,
                &norm_backward(&block.norm2, before_cross, eps, &grad_normed),
            );

            let (mut grad_normed, grad_source) = self.attention_backward(
                &block.attention,
                &format!("{prefix}.attn1"),
                &block_tape.own,
                transformer.heads,
                &grad_tokens,
            );
            add(&mut grad_normed, &grad_source);
            add(
                &mut grad_tokens,
                &norm_backward(&block.norm1, before_self, eps, &grad_normed),
            );
        }
        let grad_normed = match &transformer.input {
            Projection::Convolution(conv) => {
                let grad = FeatureMap::from_tokens(&grad_tokens, height, width).unwrap();
                conv_backward(conv, &grad, &tape.normed)
            }
            Projection::Linear(dense) => {
                FeatureMap::from_tokens(&dense_backward(dense, &grad_tokens), height, width)
                    .unwrap()
            }
        };
        let mut output = group_norm_backward(&transformer.norm, &tape.input, &grad_normed);
        add_maps(&mut output, grad);
        output
    }

    fn attention(
        &self,
        attention_layer: &Attention,
        name: &str,
        input: &Matrix,
        source: &Matrix,
        heads: usize,
    ) -> Result<(Matrix, AttentionTape), NetworkError> {
        let queries = self.project(&attention_layer.query, &format!("{name}.to_q"), input)?;
        let keys = self.project(&attention_layer.key, &format!("{name}.to_k"), source)?;
        let values = self.project(&attention_layer.value, &format!("{name}.to_v"), source)?;
        let attended = attention(&queries, &keys, &values, heads, heads, false);
        let output = self.project(
            &attention_layer.output,
            &format!("{name}.to_out.0"),
            &attended,
        )?;
        let tape = AttentionTape {
            input: input.clone(),
            source: source.clone(),
            queries,
            keys,
            values,
            attended,
        };
        Ok((output, tape))
    }

    /// `dL/d(input)` and `dL/d(source)`, adding the adapters' gradients.
    fn attention_backward(
        &mut self,
        attention_layer: &Attention,
        name: &str,
        tape: &AttentionTape,
        heads: usize,
        grad: &Matrix,
    ) -> (Matrix, Matrix) {
        let grad_attended = self.project_backward(
            &attention_layer.output,
            &format!("{name}.to_out.0"),
            &tape.attended,
            grad,
        );
        let (grad_queries, grad_keys, grad_values) = attention_backward(
            &tape.queries,
            &tape.keys,
            &tape.values,
            heads,
            &grad_attended,
        );
        let grad_input = self.project_backward(
            &attention_layer.query,
            &format!("{name}.to_q"),
            &tape.input,
            &grad_queries,
        );
        let mut grad_source = self.project_backward(
            &attention_layer.key,
            &format!("{name}.to_k"),
            &tape.source,
            &grad_keys,
        );
        add(
            &mut grad_source,
            &self.project_backward(
                &attention_layer.value,
                &format!("{name}.to_v"),
                &tape.source,
                &grad_values,
            ),
        );
        (grad_input, grad_source)
    }

    fn project(&self, dense: &Dense, name: &str, input: &Matrix) -> Result<Matrix, NetworkError> {
        let mut output = dense.forward(input)?;
        if let Some(lora) = self.adapters.get(name) {
            lora.add_forward(input, &mut output);
        }
        Ok(output)
    }

    fn project_backward(
        &mut self,
        dense: &Dense,
        name: &str,
        input: &Matrix,
        grad: &Matrix,
    ) -> Matrix {
        let mut grad_input = dense_backward(dense, grad);
        if let Some(lora) = self.adapters.get_mut(name) {
            lora.backward(input, grad, &mut grad_input);
        }
        grad_input
    }
}

/// The latent scaled for its noise level and the embedded time, which is what
/// [`Unet::forward`] works out on the host before its device pass too.
pub(crate) fn prepare(
    unet: &Unet,
    latent: &FeatureMap,
    sigma: f32,
    conditioning: &Conditioning,
) -> Result<(FeatureMap, Vec<f32>), NetworkError> {
    let config = unet.config();
    let mut scaled = latent.clone();
    let scale = (sigma * sigma + 1.0).sqrt().recip();
    let noisy = config.out_channels.min(latent.channels) * latent.pixels();
    scaled.data[..noisy]
        .iter_mut()
        .for_each(|value| *value *= scale);
    let time = unet
        .encoder
        .time(config, unet.timestep(sigma), latent, &conditioning.pooled)?;
    Ok((scaled, time))
}

/// What [`UnetLora::backward`] reads of a forward pass.
pub struct Tape {
    down: Vec<(Vec<Rung>, Option<FeatureMap>)>,
    middle: (ResnetTape, TransformerTape, ResnetTape),
    up: Vec<(Vec<Rung>, Option<FeatureMap>)>,
    norm_input: FeatureMap,
    normed: FeatureMap,
}

/// A resnet and the transformer after it, if there is one; on the way up, the
/// channels that came from below rather than from the skip.
type Rung = (usize, ResnetTape, Option<TransformerTape>);

struct ResnetTape {
    input: FeatureMap,
    normed1: FeatureMap,
    hidden: FeatureMap,
    normed2: FeatureMap,
}

struct TransformerTape {
    input: FeatureMap,
    normed: FeatureMap,
    blocks: Vec<BlockTape>,
    tokens: Matrix,
}

struct BlockTape {
    before: [Matrix; 3],
    own: AttentionTape,
    cross: AttentionTape,
    projected: Matrix,
}

struct AttentionTape {
    input: Matrix,
    source: Matrix,
    queries: Matrix,
    keys: Matrix,
    values: Matrix,
    attended: Matrix,
}

fn resnet_forward(
    resnet: &Resnet,
    input: &FeatureMap,
    time: &[f32],
) -> Result<(FeatureMap, ResnetTape), NetworkError> {
    let mut normed1 = input.clone();
    resnet.norm1.forward(&mut normed1)?;
    let mut activated = normed1.clone();
    silu(&mut activated);
    let mut hidden = resnet.conv1.forward(&activated)?;
    let time: Vec<f32> = time.iter().map(|value| crate::ffn::silu(*value)).collect();
    let per_channel = resnet.time.apply(&time)?;
    let pixels = hidden.pixels();
    for (plane, offset) in hidden.data.chunks_mut(pixels).zip(&per_channel) {
        plane.iter_mut().for_each(|value| *value += offset);
    }
    let mut normed2 = hidden.clone();
    resnet.norm2.forward(&mut normed2)?;
    let mut activated = normed2.clone();
    silu(&mut activated);
    let mut output = match &resnet.shortcut {
        Some(shortcut) => shortcut.forward(input)?,
        None => input.clone(),
    };
    add_maps(&mut output, &resnet.conv2.forward(&activated)?);
    let tape = ResnetTape {
        input: input.clone(),
        normed1,
        hidden,
        normed2,
    };
    Ok((output, tape))
}

fn resnet_backward(resnet: &Resnet, tape: &ResnetTape, grad: &FeatureMap) -> FeatureMap {
    let grad_hidden = conv_backward(&resnet.conv2, grad, &tape.normed2);
    let grad_hidden = silu_backward(&grad_hidden, &tape.normed2);
    // The noise level's offset adds to `hidden` and passes the gradient on.
    let grad_hidden = group_norm_backward(&resnet.norm2, &tape.hidden, &grad_hidden);
    let grad_hidden = conv_backward(&resnet.conv1, &grad_hidden, &tape.normed1);
    let grad_hidden = silu_backward(&grad_hidden, &tape.normed1);
    let mut output = group_norm_backward(&resnet.norm1, &tape.input, &grad_hidden);
    match &resnet.shortcut {
        Some(shortcut) => add_maps(&mut output, &conv_backward(shortcut, grad, &tape.input)),
        None => add_maps(&mut output, grad),
    }
    output
}

/// `dL/d(input)` of `conv.forward(input)`; only `input`'s shape is read.
pub(crate) fn conv_backward(conv: &Conv2d, grad: &FeatureMap, input: &FeatureMap) -> FeatureMap {
    let (kernel, stride, padding) = (conv.kernel, conv.stride, conv.padding);
    let patch = conv.in_channels * kernel * kernel;
    let mut columns = Matrix::new(grad.pixels(), patch);
    grad.to_tokens().dot(&conv.weight, &mut columns);
    // The reverse of `im2col`: every column entry goes back to the input
    // value it was read from.
    let mut output = FeatureMap::new(input.channels, input.height, input.width);
    for pixel in 0..grad.pixels() {
        let (out_y, out_x) = (pixel / grad.width, pixel % grad.width);
        let top = (out_y * stride) as isize - padding as isize;
        let left = (out_x * stride) as isize - padding as isize;
        let row = columns.row(pixel);
        for channel in 0..conv.in_channels {
            for dy in 0..kernel {
                let y = top + dy as isize;
                if y < 0 || y >= input.height as isize {
                    continue;
                }
                for dx in 0..kernel {
                    let x = left + dx as isize;
                    if x >= 0 && x < input.width as isize {
                        let target =
                            (channel * input.height + y as usize) * input.width + x as usize;
                        output.data[target] += row[(channel * kernel + dy) * kernel + dx];
                    }
                }
            }
        }
    }
    output
}

/// `dL/dx` of [`GroupNorm::forward`], given `x`.
pub(crate) fn group_norm_backward(
    norm: &GroupNorm,
    input: &FeatureMap,
    grad: &FeatureMap,
) -> FeatureMap {
    let pixels = input.pixels();
    let per_group = input.channels / norm.groups;
    let span = per_group * pixels;
    let mut output = FeatureMap::new(input.channels, input.height, input.width);
    for (group, ((values, grads), target)) in input
        .data
        .chunks(span)
        .zip(grad.data.chunks(span))
        .zip(output.data.chunks_mut(span))
        .enumerate()
    {
        let mean = values.iter().sum::<f32>() / span as f32;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / span as f32;
        let inverse = (variance + norm.eps).sqrt().recip();
        let scaled: Vec<f32> = grads
            .iter()
            .enumerate()
            .map(|(index, grad)| grad * norm.weight[group * per_group + index / pixels])
            .collect();
        let grad_mean = scaled.iter().sum::<f32>() / span as f32;
        let projection = values
            .iter()
            .zip(&scaled)
            .map(|(value, grad)| (value - mean) * inverse * grad)
            .sum::<f32>()
            / span as f32;
        for ((target, value), grad) in target.iter_mut().zip(values).zip(&scaled) {
            *target = inverse * (grad - grad_mean - (value - mean) * inverse * projection);
        }
    }
    output
}

/// `dL/dx` of the affine layer norm [`crate::clip::Norm`].
pub(crate) fn norm_backward(
    norm: &crate::clip::Norm,
    input: &Matrix,
    eps: f32,
    grad: &Matrix,
) -> Matrix {
    let mut scaled = grad.clone();
    scaled.data.chunks_mut(grad.cols).for_each(|row| {
        row.iter_mut()
            .zip(&norm.weight)
            .for_each(|(value, weight)| *value *= weight)
    });
    layer_norm_backward(input, eps, &scaled)
}

/// Half the projection is the value and half its gate.
fn geglu(projected: &Matrix) -> Matrix {
    let inner = projected.cols / 2;
    let mut hidden = Matrix::new(projected.rows, inner);
    for (target, row) in hidden
        .data
        .chunks_mut(inner)
        .zip(projected.data.chunks(projected.cols))
    {
        for (index, value) in target.iter_mut().enumerate() {
            *value = row[index] * gelu(row[inner + index]);
        }
    }
    hidden
}

pub(crate) fn geglu_backward(projected: &Matrix, grad: &Matrix) -> Matrix {
    let inner = grad.cols;
    let mut output = Matrix::new(projected.rows, projected.cols);
    for ((target, row), grad) in output
        .data
        .chunks_mut(projected.cols)
        .zip(projected.data.chunks(projected.cols))
        .zip(grad.data.chunks(inner))
    {
        for index in 0..inner {
            let (value, gate) = (row[index], row[inner + index]);
            target[index] = grad[index] * gelu(gate);
            target[inner + index] = grad[index] * value * gelu_derivative(gate);
        }
    }
    output
}

/// `dL/d(input)` of a frozen [`Dense`].
fn dense_backward(dense: &Dense, grad: &Matrix) -> Matrix {
    let weight = dense.weight_f32();
    let mut output = Matrix::new(grad.rows, weight.cols);
    grad.dot(&weight, &mut output);
    output
}

pub(crate) fn silu_backward(grad: &FeatureMap, pre: &FeatureMap) -> FeatureMap {
    let mut output = grad.clone();
    output
        .data
        .iter_mut()
        .zip(&pre.data)
        .for_each(|(grad, pre)| *grad *= silu_derivative(*pre));
    output
}

/// Each output pixel's gradient goes back to the one it was copied from.
pub(crate) fn upsample_backward(grad: &FeatureMap) -> FeatureMap {
    let (height, width) = (grad.height / 2, grad.width / 2);
    let mut output = FeatureMap::new(grad.channels, height, width);
    for channel in 0..grad.channels {
        for y in 0..grad.height {
            for x in 0..grad.width {
                output.data[(channel * height + y / 2) * width + x / 2] +=
                    grad.data[(channel * grad.height + y) * grad.width + x];
            }
        }
    }
    output
}

fn add(target: &mut Matrix, source: &Matrix) {
    target
        .data
        .iter_mut()
        .zip(&source.data)
        .for_each(|(target, source)| *target += source);
}

fn add_maps(target: &mut FeatureMap, source: &FeatureMap) {
    target
        .data
        .iter_mut()
        .zip(&source.data)
        .for_each(|(target, source)| *target += source);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::unet::tests::{conditioning, latent, random, tiny};
    use rand::{Rng, SeedableRng};

    pub(crate) fn adapters(unet: &Unet) -> UnetLora {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut random = |rows, cols| {
            let data = (0..rows * cols).map(|_| rng.gen_range(-0.3..0.3)).collect();
            Param::new(Matrix::from_vec(rows, cols, data))
        };
        let adapters = projections(unet)
            .into_iter()
            .map(|(name, dense)| {
                let lora = Lora {
                    down: random(2, dense.in_dim()),
                    up: random(dense.out_dim(), 2),
                    scale: 0.5,
                };
                (name, lora)
            })
            .collect();
        UnetLora { adapters }
    }

    #[test]
    fn adapter_gradients_match_finite_differences() {
        let config = tiny();
        let unet = random(&config, "rusting_brain_unet_lora.safetensors");
        let (latent, conditioning) = (latent(&config, 8), conditioning(&config, 8, 1.0));
        let sigma = 1.5;

        // Without adapters the pass is the model's own.
        let (plain, _) = UnetLora::default()
            .forward(&unet, &latent, sigma, &conditioning)
            .unwrap();
        let own = unet.forward(&latent, sigma, &conditioning).unwrap();
        let error = plain
            .data
            .iter()
            .zip(&own.data)
            .fold(0.0f32, |max, (a, b)| max.max((a - b).abs()));
        assert!(error < 1e-5, "the taped pass is off by {error}");

        let mut lora = adapters(&unet);
        let (output, tape) = lora.forward(&unet, &latent, sigma, &conditioning).unwrap();
        let mut weights = output.clone();
        for (index, value) in weights.data.iter_mut().enumerate() {
            *value = ((index * 7 % 11) as f32 - 5.0) * 0.1;
        }
        lora.backward(&unet, &tape, &weights).unwrap();
        let loss = |lora: &UnetLora| -> f64 {
            let (output, _) = lora.forward(&unet, &latent, sigma, &conditioning).unwrap();
            let pairs = output.data.iter().zip(&weights.data);
            pairs.map(|(a, b)| (a * b) as f64).sum()
        };

        // Each adapter on its own: the loss along its own gradient changes by
        // the gradient's length.
        let names: Vec<String> = lora.adapters.keys().cloned().collect();
        assert_eq!(names.len(), 88);
        let step = 0.5;
        for name in names {
            for factor in 0..2 {
                fn pick<'a>(lora: &'a mut UnetLora, name: &str, factor: usize) -> &'a mut Param {
                    let adapter = lora.adapters.get_mut(name).unwrap();
                    if factor == 0 {
                        &mut adapter.down
                    } else {
                        &mut adapter.up
                    }
                }
                let grad = pick(&mut lora, &name, factor).grad.data.clone();
                let length = grad.iter().map(|g| g * g).sum::<f32>().sqrt();
                let mut moved = [lora.clone(), lora.clone()];
                for (copy, sign) in moved.iter_mut().zip([1.0, -1.0]) {
                    let param = pick(copy, &name, factor);
                    for (value, grad) in param.value.data.iter_mut().zip(&grad) {
                        *value += sign * step * grad / length;
                    }
                }
                let slope = (loss(&moved[0]) - loss(&moved[1])) / (2.0 * step as f64);
                // f32 rounding in the loss leaves about 5e-7 of noise in the slope.
                assert!(
                    (slope as f32 - length).abs() < 1e-2 * length + 1e-6,
                    "{name} factor {factor}: slope {slope}, gradient length {length}"
                );
                assert!(
                    error < 2e-2,
                    "{name} factor {factor}: slope {slope}, gradient length {length}"
                );
            }
        }
    }

    /// Needs `RB_UNET_LORA_REF` from `scripts/unet_lora_gradients.py`.
    #[test]
    #[ignore]
    fn unet_lora_gradients_match_pytorch() {
        let dir = std::path::PathBuf::from(std::env::var("RB_UNET_LORA_REF").unwrap());
        let reference: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
                .unwrap();
        let floats = |value: &serde_json::Value| -> Vec<f32> {
            let values = value.as_array().unwrap().iter();
            values.map(|x| x.as_f64().unwrap() as f32).collect()
        };
        let config = crate::unet::UnetConfig::from_file(dir.join("unet/config.json")).unwrap();
        let unet = Unet::load_at(
            dir.join("unet/diffusion_pytorch_model.safetensors"),
            config.clone(),
            crate::transformer::Precision::F32,
        )
        .unwrap();
        let scale = reference["scale"].as_f64().unwrap() as f32;
        let mut lora = UnetLora::read_peft(dir.join("adapters.safetensors"), &unet, scale).unwrap();
        assert_eq!(lora.adapters.len(), 48);
        let latent = FeatureMap::from_vec(4, 8, 8, floats(&reference["latent"])).unwrap();
        let conditioning = Conditioning {
            text: Matrix::from_vec(5, config.cross_dim, floats(&reference["text"])),
            pooled: vec![],
            height: 8,
            width: 8,
            guidance: 0.0,
        };
        let sigma = reference["sigma"].as_f64().unwrap() as f32;
        let (output, tape) = lora.forward(&unet, &latent, sigma, &conditioning).unwrap();
        let worst = |got: &[f32], want: &[f32]| {
            assert_eq!(got.len(), want.len());
            let largest = want.iter().fold(0.0f32, |max, value| max.max(value.abs()));
            let error = got
                .iter()
                .zip(want)
                .fold(0.0f32, |max, (got, want)| max.max((got - want).abs()));
            error / largest
        };
        let error = worst(&output.data, &floats(&reference["output"]));
        assert!(error < 1e-4, "the prediction is off by {error} relative");

        let grad = FeatureMap::from_vec(4, 8, 8, floats(&reference["weights"])).unwrap();
        lora.backward(&unet, &tape, &grad).unwrap();
        let mut largest = 0.0f32;
        for (name, want) in reference["gradients"].as_object().unwrap() {
            let (module, kind) = name.split_once(".lora_").unwrap();
            let adapter = &lora.adapters[module];
            let got = match kind.starts_with('A') {
                true => &adapter.down.grad.data,
                false => &adapter.up.grad.data,
            };
            let error = worst(got, &floats(want));
            assert!(error < 1e-4, "{name} is off by {error} relative");
            largest = largest.max(error);
        }
        println!("prediction off by {error}, worst gradient off by {largest}");
    }
}
