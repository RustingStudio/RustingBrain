//! The UNet denoiser, which is what Stable Diffusion 1.x, 2.x and XL are.
//!
//! Where [`crate::mmdit`] treats a latent as a sequence of patches, this treats
//! it as an image: a ladder of residual convolution blocks halves the
//! resolution and doubles the channels on the way down, the same ladder climbs
//! back up, and every rung on the way up is handed the matching rung from the
//! way down. The prompt enters through cross-attention inside the blocks that
//! have it, and the noise level enters through a vector added to every residual
//! block.
//!
//! Almost nothing about the shape is read from the configuration: how many
//! blocks there are, which of them attend, whether the projections into
//! attention are linear or one-by-one convolutions — all of that is read from
//! which keys the checkpoint holds. What the configuration is needed for is the
//! number of attention heads, which is not recoverable from a weight's shape.
//!
//! The one place this model disagrees with the sampler's view of the world is
//! time: [`crate::diffusion::Scheduler::Ddim`] hands out a continuous noise
//! level, and the network was trained on a step index between zero and a
//! thousand. The level is turned back into that index here, and the latent is
//! scaled the way the training distribution expects.
//!
//! An inpainting checkpoint reads the mask and the masked picture's latent as
//! extra input channels; [`Unet::set_inpaint`] hands them over. A
//! [`ControlNet`] is a second copy of the way down that reads another picture
//! and corrects every rung; [`Unet::set_controls`] attaches them.
//!
//! NOTE: inference only.

use crate::clip::{Norm, norm};
use crate::conv::{Conv2d, Dense, FeatureMap, GroupNorm, silu, upsample_nearest};
use crate::diffusion::{Denoiser, Scheduler, alphas_cumprod};
use crate::matrix::Matrix;
use crate::mmdit::{Conditioning, attention, sinusoid};
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::transformer::Precision;
use crate::vae::read_conv;
use rayon::prelude::*;

/// The shape of a UNet, as its `config.json` states it.
#[derive(Clone, Debug, PartialEq)]
pub struct UnetConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_channels: Vec<usize>,
    /// How many attention heads each level of the ladder uses. Published
    /// configurations write this as `attention_head_dim`, which counts heads
    /// rather than dimensions whatever its name says.
    pub heads: Vec<usize>,
    pub cross_dim: usize,
    pub norm_groups: usize,
    pub eps: f32,
    /// The width of the vector the noise level becomes.
    pub time_dim: usize,
    /// The width of the sinusoidal table it is read from.
    pub freq_dim: usize,
    /// The width of the pooled prompt vector, for the models that take one.
    /// Stable Diffusion XL does; 1.x does not.
    pub pooled_dim: Option<usize>,
    /// The width each of the six size numbers XL conditions on is embedded to.
    pub addition_time_dim: usize,
}

impl Default for UnetConfig {
    /// Stable Diffusion 1.5.
    fn default() -> Self {
        Self {
            in_channels: 4,
            out_channels: 4,
            block_channels: vec![320, 640, 1280, 1280],
            heads: vec![8; 4],
            cross_dim: 768,
            norm_groups: 32,
            eps: 1e-5,
            time_dim: 1280,
            freq_dim: 320,
            pooled_dim: None,
            addition_time_dim: 256,
        }
    }
}

impl UnetConfig {
    /// Reads a diffusers `unet/config.json`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let text = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| NetworkError::InvalidDataset(format!("unet config.json: {error}")))?;

        let default = Self::default();
        let number = |name: &str, fallback: usize| -> usize {
            json.get(name)
                .and_then(|value| value.as_u64())
                .map_or(fallback, |value| value as usize)
        };
        let list = |name: &str| -> Option<Vec<usize>> {
            let value = json.get(name)?;
            match value.as_array() {
                Some(values) => values
                    .iter()
                    .map(|value| value.as_u64().map(|value| value as usize))
                    .collect(),
                // A single number means the same one at every level.
                None => value.as_u64().map(|value| vec![value as usize]),
            }
        };

        let block_channels = list("block_out_channels").unwrap_or(default.block_channels);
        let levels = block_channels.len();
        let heads = match list("attention_head_dim") {
            Some(heads) if heads.len() == levels => heads,
            // One number covers every level.
            Some(heads) if !heads.is_empty() => vec![heads[0]; levels],
            _ => vec![default.heads[0]; levels],
        };
        let freq_dim = *block_channels.first().unwrap_or(&default.freq_dim);
        let addition_time_dim = number("addition_time_embed_dim", default.addition_time_dim);

        Ok(Self {
            in_channels: number("in_channels", default.in_channels),
            out_channels: number("out_channels", default.out_channels),
            // A list here is one width per level, and they are all the same.
            cross_dim: list("cross_attention_dim")
                .and_then(|widths| widths.first().copied())
                .unwrap_or(default.cross_dim),
            norm_groups: number("norm_num_groups", default.norm_groups),
            eps: json
                .get("norm_eps")
                .and_then(|value| value.as_f64())
                .unwrap_or(default.eps as f64) as f32,
            time_dim: freq_dim * 4,
            freq_dim,
            // XL conditions on the pooled prompt vector and on six numbers
            // describing the crop, all through one projection, so the pooled
            // width is what is left of that projection's input.
            pooled_dim: json
                .get("projection_class_embeddings_input_dim")
                .and_then(|value| value.as_u64())
                .map(|value| (value as usize).saturating_sub(6 * addition_time_dim)),
            addition_time_dim,
            block_channels,
            heads,
        })
    }
}

/// A residual convolution block, carrying the noise level in.
pub(crate) struct Resnet {
    pub(crate) norm1: GroupNorm,
    pub(crate) conv1: Conv2d,
    pub(crate) time: Dense,
    pub(crate) norm2: GroupNorm,
    pub(crate) conv2: Conv2d,
    pub(crate) shortcut: Option<Conv2d>,
}

impl Resnet {
    fn forward(&self, input: &FeatureMap, time: &[f32]) -> Result<FeatureMap, NetworkError> {
        let mut hidden = input.clone();
        self.norm1.forward(&mut hidden)?;
        silu(&mut hidden);
        let mut hidden = self.conv1.forward(&hidden)?;

        // The noise level arrives as one number per channel.
        let activated: Vec<f32> = time.iter().map(|value| crate::ffn::silu(*value)).collect();
        let per_channel = self.time.apply(&activated)?;
        let pixels = hidden.pixels();
        hidden
            .data
            .par_chunks_mut(pixels)
            .zip(per_channel.par_iter())
            .for_each(|(plane, offset)| plane.iter_mut().for_each(|value| *value += offset));

        self.norm2.forward(&mut hidden)?;
        silu(&mut hidden);
        let hidden = self.conv2.forward(&hidden)?;

        let mut output = match &self.shortcut {
            Some(shortcut) => shortcut.forward(input)?,
            None => input.clone(),
        };
        for (output, hidden) in output.data.iter_mut().zip(&hidden.data) {
            *output += hidden;
        }
        Ok(output)
    }
}

/// One attention, used both for the latent attending to itself and for it
/// attending to the prompt.
pub(crate) struct Attention {
    pub(crate) query: Dense,
    pub(crate) key: Dense,
    pub(crate) value: Dense,
    pub(crate) output: Dense,
}

impl Attention {
    fn forward(
        &self,
        tokens: &Matrix,
        context: Option<&Matrix>,
        heads: usize,
    ) -> Result<Matrix, NetworkError> {
        let source = context.unwrap_or(tokens);
        let attended = attention(
            &self.query.forward(tokens)?,
            &self.key.forward(source)?,
            &self.value.forward(source)?,
            heads,
            heads,
            false,
        );
        self.output.forward(&attended)
    }
}

/// A transformer block: attend to the latent, attend to the prompt, then a
/// gated feed-forward.
pub(crate) struct TransformerBlock {
    pub(crate) norm1: Norm,
    pub(crate) attention: Attention,
    pub(crate) norm2: Norm,
    pub(crate) cross: Attention,
    pub(crate) norm3: Norm,
    pub(crate) gate: Dense,
    pub(crate) output: Dense,
}

/// The stack of those blocks that sits between two convolutions at one
/// resolution.
pub(crate) struct Transformer {
    pub(crate) norm: GroupNorm,
    pub(crate) input: Projection,
    pub(crate) blocks: Vec<TransformerBlock>,
    pub(crate) output: Projection,
    pub(crate) heads: usize,
}

/// How a transformer enters and leaves the feature map. The 1.x checkpoints
/// write these as one-by-one convolutions and the XL ones as linear layers,
/// which the weight's own rank says.
pub(crate) enum Projection {
    Convolution(Conv2d),
    Linear(Dense),
}

impl Transformer {
    fn forward(
        &self,
        input: &FeatureMap,
        time_free: &Matrix,
        eps: f32,
    ) -> Result<FeatureMap, NetworkError> {
        let mut normed = input.clone();
        self.norm.forward(&mut normed)?;
        let (height, width) = (input.height, input.width);
        let mut tokens = match &self.input {
            Projection::Convolution(conv) => conv.forward(&normed)?.to_tokens(),
            Projection::Linear(dense) => dense.forward(&normed.to_tokens())?,
        };

        for block in &self.blocks {
            let attended =
                block
                    .attention
                    .forward(&block.norm1.forward(&tokens, eps), None, self.heads)?;
            add(&mut tokens, &attended);

            let attended = block.cross.forward(
                &block.norm2.forward(&tokens, eps),
                Some(time_free),
                self.heads,
            )?;
            add(&mut tokens, &attended);

            // The gated feed-forward: half the projection is the value and
            // half is its gate.
            let projected = block.gate.forward(&block.norm3.forward(&tokens, eps))?;
            let inner = projected.cols / 2;
            let mut hidden = Matrix::new(projected.rows, inner);
            hidden
                .data
                .par_chunks_mut(inner)
                .zip(projected.data.par_chunks(projected.cols))
                .for_each(|(target, row)| {
                    for (index, value) in target.iter_mut().enumerate() {
                        *value = row[index] * crate::ffn::gelu(row[inner + index]);
                    }
                });
            add(&mut tokens, &block.output.forward(&hidden)?);
        }

        let mut output = match &self.output {
            Projection::Convolution(conv) => {
                conv.forward(&FeatureMap::from_tokens(&tokens, height, width)?)?
            }
            Projection::Linear(dense) => {
                FeatureMap::from_tokens(&dense.forward(&tokens)?, height, width)?
            }
        };
        for (output, input) in output.data.iter_mut().zip(&input.data) {
            *output += input;
        }
        Ok(output)
    }
}

/// One rung of the ladder, going either way. The resampler is the convolution
/// that halves or doubles the resolution, where the rung has one.
pub(crate) struct Block {
    pub(crate) resnets: Vec<Resnet>,
    pub(crate) attentions: Vec<Transformer>,
    pub(crate) resampler: Option<Conv2d>,
}

/// The half of a UNet that climbs down: the input convolution, the noise
/// level's embedding, the down blocks and the middle. A ControlNet is a
/// trained copy of it.
pub(crate) struct Encoder {
    pub(crate) conv_in: Conv2d,
    time_in: Dense,
    time_out: Dense,
    /// The projection Stable Diffusion XL conditions its crop and its pooled
    /// prompt through.
    add_in: Option<Dense>,
    add_out: Option<Dense>,
    pub(crate) down: Vec<Block>,
    pub(crate) middle: (Resnet, Transformer, Resnet),
}

impl Encoder {
    fn read(
        file: &mut ShardedSafeTensors,
        config: &UnetConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let down = (0..config.block_channels.len())
            .map(|level| {
                read_block(
                    file,
                    &format!("down_blocks.{level}"),
                    config,
                    level,
                    precision,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let deepest = config.heads.len() - 1;
        Ok(Self {
            conv_in: read_conv(file, "conv_in", 1, 1)?,
            time_in: dense(file, "time_embedding.linear_1", precision)?,
            time_out: dense(file, "time_embedding.linear_2", precision)?,
            add_in: dense(file, "add_embedding.linear_1", precision).ok(),
            add_out: dense(file, "add_embedding.linear_2", precision).ok(),
            middle: (
                read_resnet(file, "mid_block.resnets.0", config, precision)?,
                read_transformer(file, "mid_block.attentions.0", config, deepest, precision)?,
                read_resnet(file, "mid_block.resnets.1", config, precision)?,
            ),
            down,
        })
    }

    /// The vector a step index, and for XL the pooled prompt and the size,
    /// become.
    fn time(
        &self,
        config: &UnetConfig,
        step: f32,
        latent: &FeatureMap,
        pooled: &[f32],
    ) -> Result<Vec<f32>, NetworkError> {
        let embedded = sinusoid(step, config.freq_dim);
        let hidden = self.time_in.apply(&embedded)?;
        let activated: Vec<f32> = hidden
            .iter()
            .map(|value| crate::ffn::silu(*value))
            .collect();
        let mut time = self.time_out.apply(&activated)?;

        // Stable Diffusion XL adds the pooled prompt and the crop it is meant
        // to be producing to the same vector.
        if let (Some(input), Some(output)) = (&self.add_in, &self.add_out) {
            let mut added = pooled.to_vec();
            let upscale = 8;
            let (height, width) = (
                (latent.height * upscale) as f32,
                (latent.width * upscale) as f32,
            );
            for value in [height, width, 0.0, 0.0, height, width] {
                added.extend(sinusoid(value, config.addition_time_dim));
            }
            let hidden = input.apply(&added)?;
            let activated: Vec<f32> = hidden
                .iter()
                .map(|value| crate::ffn::silu(*value))
                .collect();
            for (time, added) in time.iter_mut().zip(output.apply(&activated)?) {
                *time += added;
            }
        }
        Ok(time)
    }

    /// The down blocks and the middle, from what `conv_in` produced. Every
    /// rung keeps what it produced for the matching rung on the way up.
    fn run(
        &self,
        eps: f32,
        mut sample: FeatureMap,
        time: &[f32],
        text: &Matrix,
    ) -> Result<(FeatureMap, Vec<FeatureMap>), NetworkError> {
        let mut skips = vec![sample.clone()];
        for block in &self.down {
            for (index, resnet) in block.resnets.iter().enumerate() {
                sample = resnet.forward(&sample, time)?;
                if let Some(transformer) = block.attentions.get(index) {
                    sample = transformer.forward(&sample, text, eps)?;
                }
                skips.push(sample.clone());
            }
            if let Some(resampler) = &block.resampler {
                sample = resampler.forward(&sample)?;
                skips.push(sample.clone());
            }
        }
        sample = self.middle.0.forward(&sample, time)?;
        sample = self.middle.1.forward(&sample, text, eps)?;
        sample = self.middle.2.forward(&sample, time)?;
        Ok((sample, skips))
    }
}

/// A ControlNet: a trained copy of a UNet's way down that reads a second
/// picture (edges, depth, a pose) and hands the UNet one correction for every
/// rung, which is how the picture steers the layout.
pub struct ControlNet {
    config: UnetConfig,
    encoder: Encoder,
    /// The small convolution stack that brings the picture down to the
    /// latent's size, from `controlnet_cond_embedding`.
    hint: Vec<Conv2d>,
    /// The one-by-one convolutions every rung's correction leaves through,
    /// which start at zero in training.
    zero: Vec<Conv2d>,
    zero_middle: Conv2d,
}

impl ControlNet {
    /// Reads a ControlNet out of a diffusers checkpoint; its `config.json`
    /// reads with [`UnetConfig::from_file`].
    pub fn load(
        path: impl AsRef<std::path::Path>,
        config: UnetConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let file = &mut ShardedSafeTensors::open(path)?;
        let base = "controlnet_cond_embedding";
        let mut hint = vec![read_conv(file, &format!("{base}.conv_in"), 1, 1)?];
        // The stack alternates a convolution that keeps the size with one
        // that halves it.
        while let Ok(conv) = read_conv(
            file,
            &format!("{base}.blocks.{}", hint.len() - 1),
            1 + (hint.len() - 1) % 2,
            1,
        ) {
            hint.push(conv);
        }
        hint.push(read_conv(file, &format!("{base}.conv_out"), 1, 1)?);
        let mut zero = Vec::new();
        while let Ok(conv) = read_conv(
            file,
            &format!("controlnet_down_blocks.{}", zero.len()),
            1,
            0,
        ) {
            zero.push(conv);
        }
        Ok(Self {
            encoder: Encoder::read(file, &config, precision)?,
            zero_middle: read_conv(file, "controlnet_mid_block", 1, 0)?,
            config,
            hint,
            zero,
        })
    }

    /// The corrections for every skip and for the middle, scaled.
    fn residuals(
        &self,
        latent: &FeatureMap,
        step: f32,
        control: &Control,
        conditioning: &Conditioning,
    ) -> Result<(Vec<FeatureMap>, FeatureMap), NetworkError> {
        let time = self
            .encoder
            .time(&self.config, step, latent, &conditioning.pooled)?;
        let mut hint = control.image.clone();
        for (index, conv) in self.hint.iter().enumerate() {
            hint = conv.forward(&hint)?;
            if index + 1 < self.hint.len() {
                silu(&mut hint);
            }
        }
        let mut sample = self.encoder.conv_in.forward(latent)?;
        if (hint.channels, hint.height, hint.width)
            != (sample.channels, sample.height, sample.width)
        {
            return Err(NetworkError::InvalidConfig(format!(
                "a ControlNet picture of {}x{} does not match a {}x{} latent; it is the size of \
                 the image being made",
                control.image.height, control.image.width, latent.height, latent.width
            )));
        }
        add_maps(&mut sample, &hint)?;
        let (middle, skips) =
            self.encoder
                .run(self.config.eps, sample, &time, &conditioning.text)?;
        if skips.len() != self.zero.len() {
            return Err(NetworkError::InvalidConfig(format!(
                "the ControlNet has {} rungs and {} corrections for them",
                skips.len(),
                self.zero.len()
            )));
        }
        let scale = |map: FeatureMap| {
            let mut map = map;
            map.data
                .iter_mut()
                .for_each(|value| *value *= control.scale);
            map
        };
        let skips = skips
            .iter()
            .zip(&self.zero)
            .map(|(skip, conv)| conv.forward(skip).map(scale))
            .collect::<Result<_, _>>()?;
        Ok((skips, scale(self.zero_middle.forward(&middle)?)))
    }
}

/// A ControlNet with the picture it is to follow and how hard. Through an
/// [`crate::ImagePipeline`], hand it to the UNet in `pipeline.denoiser` and
/// generate as usual.
#[derive(Clone)]
pub struct Control {
    pub net: std::sync::Arc<ControlNet>,
    /// RGB in `[0, 1]`, at the size of the image being made: eight times the
    /// latent's.
    pub image: FeatureMap,
    /// Diffusers' `controlnet_conditioning_scale`; 1.0 is what it was trained at.
    pub scale: f32,
}

/// A UNet denoiser.
pub struct Unet {
    config: UnetConfig,
    pub(crate) encoder: Encoder,
    pub(crate) up: Vec<Block>,
    pub(crate) norm_out: GroupNorm,
    pub(crate) conv_out: Conv2d,
    /// `alpha_bar` for the schedule this model was trained on, which is how a
    /// noise level becomes the step index the network counts in.
    alphas: Vec<f32>,
    conditioning: Option<Conditioning>,
    unconditional: Option<Conditioning>,
    /// What an inpainting checkpoint reads after the noisy latent.
    inpaint: Option<FeatureMap>,
    controls: Vec<Control>,
    /// The same weights on a CUDA device, when one was attached. The host copy
    /// stays where it is, which is what keeps the CPU path a fallback rather
    /// than a rewrite.
    #[cfg(feature = "cuda")]
    device: Option<crate::cuda_image::DeviceUnet>,
}

impl Unet {
    /// The configuration this model was built for.
    pub fn config(&self) -> &UnetConfig {
        &self.config
    }

    /// The width of the pooled prompt vector this model takes, if it takes one.
    pub fn pooled_dim(&self) -> Option<usize> {
        self.encoder.add_in.as_ref().and(self.config.pooled_dim)
    }

    /// Reads a UNet out of a checkpoint.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        config: UnetConfig,
    ) -> Result<Self, NetworkError> {
        Self::load_at(path, config, Precision::F32)
    }

    /// The same, holding the attention and feed-forward projections at a chosen
    /// precision. The convolutions stay in `f32`: they are most of the model's
    /// arithmetic and none of its parameters.
    pub fn load_at(
        path: impl AsRef<std::path::Path>,
        config: UnetConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        Self::read_at(&mut file, config, precision)
    }

    /// The same, from a checkpoint that is already open.
    pub fn read_at(
        file: &mut ShardedSafeTensors,
        config: UnetConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        // Climbing up reverses the ladder, so the head counts reverse with it.
        let up = (0..config.block_channels.len())
            .map(|level| {
                let mirrored = config.block_channels.len() - 1 - level;
                read_block(
                    file,
                    &format!("up_blocks.{level}"),
                    &config,
                    mirrored,
                    precision,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            encoder: Encoder::read(file, &config, precision)?,
            norm_out: read_group_norm(file, "conv_norm_out", config.norm_groups, config.eps)?,
            conv_out: read_conv(file, "conv_out", 1, 1)?,
            alphas: alphas_cumprod(0.00085, 0.012, 1000),
            conditioning: None,
            unconditional: None,
            inpaint: None,
            controls: Vec::new(),
            #[cfg(feature = "cuda")]
            device: None,
            config,
            up,
        })
    }

    /// Tells the model which schedule its checkpoint was trained on, which is
    /// what turns a noise level back into the step index it counts in.
    ///
    /// The default is the one every Stable Diffusion release shipped with, so
    /// this is only needed for a model whose scheduler says otherwise.
    pub fn set_schedule(&mut self, scheduler: Scheduler) {
        if let Scheduler::Ddim {
            beta_start,
            beta_end,
            train_steps,
        } = scheduler
        {
            self.alphas = alphas_cumprod(beta_start, beta_end, train_steps);
        }
    }

    /// Hands the model the prompt it is to follow.
    pub fn set_conditioning(&mut self, conditioning: Conditioning) {
        self.conditioning = Some(conditioning);
    }

    /// Hands it the empty prompt to push away from, which is what
    /// classifier-free guidance needs.
    pub fn set_unconditional(&mut self, conditioning: Conditioning) {
        self.unconditional = Some(conditioning);
    }

    /// Hands an inpainting checkpoint, one whose `in_channels` exceed its
    /// `out_channels`, the channels it reads after the noisy latent: the mask
    /// (1.0 where to repaint) and then the latent of the picture with that
    /// part blanked out, at the latent's size. `None` takes them away.
    pub fn set_inpaint(&mut self, extra: Option<FeatureMap>) {
        self.inpaint = extra;
    }

    /// Hands the model the ControlNets to follow, whose corrections add up.
    /// An empty list takes them away.
    pub fn set_controls(&mut self, controls: Vec<Control>) {
        self.controls = controls;
    }

    /// One pass: a noisy latent and a noise level in, the noise the model
    /// believes is in there out. An inpainting checkpoint's input carries its
    /// extra channels after the latent's, and only the latent's are scaled.
    pub fn forward(
        &self,
        latent: &FeatureMap,
        sigma: f32,
        conditioning: &Conditioning,
    ) -> Result<FeatureMap, NetworkError> {
        // The network was trained on unit-variance inputs at every level, and
        // a Karras-style latent grows with the level.
        let mut scaled = latent.clone();
        let scale = (sigma * sigma + 1.0).sqrt().recip();
        let noisy = self.config.out_channels.min(latent.channels) * latent.height * latent.width;
        scaled.data[..noisy]
            .iter_mut()
            .for_each(|value| *value *= scale);

        let step = self.timestep(sigma);
        let time = self
            .encoder
            .time(&self.config, step, latent, &conditioning.pooled)?;

        // NOTE: a ControlNet runs on the CPU, so a model with one falls
        // back there whole; give the device a ControlNet when that is slow.
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device
            && self.controls.is_empty()
        {
            return device.forward(&scaled, &time, &conditioning.text);
        }

        let sample = self.encoder.conv_in.forward(&scaled)?;
        let (mut sample, mut skips) =
            self.encoder
                .run(self.config.eps, sample, &time, &conditioning.text)?;
        if !self.controls.is_empty() {
            let noisy = FeatureMap {
                channels: self.config.out_channels.min(latent.channels),
                height: latent.height,
                width: latent.width,
                data: scaled.data[..noisy].to_vec(),
            };
            for control in &self.controls {
                let (corrections, middle) =
                    control.net.residuals(&noisy, step, control, conditioning)?;
                if corrections.len() != skips.len() {
                    return Err(NetworkError::InvalidConfig(
                        "the ControlNet does not have the UNet's rungs".into(),
                    ));
                }
                for (skip, correction) in skips.iter_mut().zip(&corrections) {
                    add_maps(skip, correction)?;
                }
                add_maps(&mut sample, &middle)?;
            }
        }

        for block in &self.up {
            for (index, resnet) in block.resnets.iter().enumerate() {
                let skip = skips.pop().ok_or_else(|| {
                    NetworkError::InvalidConfig(
                        "the way up has more rungs than the way down".into(),
                    )
                })?;
                sample = resnet.forward(&concatenate(&sample, &skip), &time)?;
                if let Some(transformer) = block.attentions.get(index) {
                    sample = transformer.forward(&sample, &conditioning.text, self.config.eps)?;
                }
            }
            if let Some(resampler) = &block.resampler {
                sample = resampler.forward(&upsample_nearest(&sample, 2))?;
            }
        }

        self.norm_out.forward(&mut sample)?;
        silu(&mut sample);
        self.conv_out.forward(&sample)
    }

    /// Uploads every weight to a CUDA device and runs there from now on.
    ///
    /// Fails closed: an allocation that does not fit, or a device that is not
    /// there, is an error and leaves the model exactly as it was, still on the
    /// CPU. Callers that want a silent fallback ignore the error.
    #[cfg(feature = "cuda")]
    pub fn attach_device(
        &mut self,
        gpu: std::sync::Arc<crate::cuda_image::ImageGpu>,
    ) -> Result<(), NetworkError> {
        self.device = Some(crate::cuda_image::DeviceUnet::upload(&gpu, self)?);
        Ok(())
    }

    /// The step index a noise level stands for, interpolated between the two
    /// the training schedule holds.
    fn timestep(&self, sigma: f32) -> f32 {
        // `sigma` rises with the index, so the first index that overtakes it
        // brackets the answer.
        let level = |index: usize| {
            let alpha = self.alphas[index];
            ((1.0 - alpha) / alpha).sqrt()
        };
        let last = self.alphas.len() - 1;
        let above = (0..=last)
            .find(|index| level(*index) >= sigma)
            .unwrap_or(last);
        if above == 0 {
            return 0.0;
        }
        let (low, high) = (level(above - 1), level(above));
        let fraction = match high > low {
            true => ((sigma - low) / (high - low)).clamp(0.0, 1.0),
            false => 0.0,
        };
        (above - 1) as f32 + fraction
    }

    /// The prediction for a latent held as `[pixels, channels]`, which is the
    /// shape the sampler works in.
    fn predict(
        &self,
        latents: &Matrix,
        sigma: f32,
        conditioning: &Conditioning,
    ) -> Result<Matrix, NetworkError> {
        let mut latent = FeatureMap::from_tokens(latents, conditioning.height, conditioning.width)?;
        if self.config.in_channels > latent.channels {
            let extra = self.inpaint.as_ref().ok_or_else(|| {
                NetworkError::InvalidConfig(
                    "an inpainting model needs the mask and the masked picture".into(),
                )
            })?;
            if (extra.channels + latent.channels, extra.height, extra.width)
                != (self.config.in_channels, latent.height, latent.width)
            {
                return Err(NetworkError::InvalidConfig(format!(
                    "the model reads {} channels and was handed {} latent and {} extra, at \
                     {}x{} against {}x{}",
                    self.config.in_channels,
                    latent.channels,
                    extra.channels,
                    extra.height,
                    extra.width,
                    latent.height,
                    latent.width
                )));
            }
            latent = concatenate(&latent, extra);
        }
        Ok(self.forward(&latent, sigma, conditioning)?.to_tokens())
    }
}

impl Denoiser for Unet {
    fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
        let conditioning = self.conditioning.as_ref().ok_or_else(|| {
            NetworkError::InvalidConfig("this denoiser has not been given a prompt".into())
        })?;
        self.predict(latents, sigma, conditioning)
    }

    fn denoise_unconditional(
        &mut self,
        latents: &Matrix,
        sigma: f32,
    ) -> Result<Option<Matrix>, NetworkError> {
        match &self.unconditional {
            Some(conditioning) => Ok(Some(self.predict(latents, sigma, conditioning)?)),
            None => Ok(None),
        }
    }
}

/// Two feature maps of the same size, stacked along their channels.
pub(crate) fn concatenate(first: &FeatureMap, second: &FeatureMap) -> FeatureMap {
    let mut output = FeatureMap::new(first.channels + second.channels, first.height, first.width);
    let split = first.data.len();
    output.data[..split].copy_from_slice(&first.data);
    output.data[split..].copy_from_slice(&second.data);
    output
}

fn add_maps(target: &mut FeatureMap, source: &FeatureMap) -> Result<(), NetworkError> {
    if target.data.len() != source.data.len() {
        return Err(NetworkError::InvalidConfig(format!(
            "a ControlNet correction of {}x{}x{} does not fit a {}x{}x{} rung",
            source.channels,
            source.height,
            source.width,
            target.channels,
            target.height,
            target.width
        )));
    }
    for (target, source) in target.data.iter_mut().zip(&source.data) {
        *target += source;
    }
    Ok(())
}

fn add(target: &mut Matrix, source: &Matrix) {
    target
        .data
        .par_iter_mut()
        .zip(source.data.par_iter())
        .for_each(|(target, source)| *target += source);
}

fn read_block(
    file: &mut ShardedSafeTensors,
    base: &str,
    config: &UnetConfig,
    level: usize,
    precision: Precision,
) -> Result<Block, NetworkError> {
    let mut resnets = Vec::new();
    // How many rungs this block holds is not worth reading from the
    // configuration when the checkpoint says it directly.
    while file
        .names()
        .any(|name| name == format!("{base}.resnets.{}.conv1.weight", resnets.len()))
    {
        let index = resnets.len();
        resnets.push(read_resnet(
            file,
            &format!("{base}.resnets.{index}"),
            config,
            precision,
        )?);
    }
    if resnets.is_empty() {
        return Err(NetworkError::InvalidConfig(format!(
            "the checkpoint holds no {base}, so it is not the shape its configuration says"
        )));
    }

    let mut attentions = Vec::new();
    while file
        .names()
        .any(|name| name.starts_with(&format!("{base}.attentions.{}.", attentions.len())))
    {
        let index = attentions.len();
        attentions.push(read_transformer(
            file,
            &format!("{base}.attentions.{index}"),
            config,
            level,
            precision,
        )?);
    }

    // A block resamples with a stride on the way down and with a plain
    // convolution after a nearest-neighbour doubling on the way up.
    let resampler = match base.starts_with("down") {
        true => read_conv(file, &format!("{base}.downsamplers.0.conv"), 2, 1).ok(),
        false => read_conv(file, &format!("{base}.upsamplers.0.conv"), 1, 1).ok(),
    };

    Ok(Block {
        resnets,
        attentions,
        resampler,
    })
}

fn read_resnet(
    file: &mut ShardedSafeTensors,
    base: &str,
    config: &UnetConfig,
    precision: Precision,
) -> Result<Resnet, NetworkError> {
    let conv1 = read_conv(file, &format!("{base}.conv1"), 1, 1)?;
    let conv2 = read_conv(file, &format!("{base}.conv2"), 1, 1)?;
    Ok(Resnet {
        norm1: read_group_norm(
            file,
            &format!("{base}.norm1"),
            config.norm_groups,
            config.eps,
        )?,
        time: dense(file, &format!("{base}.time_emb_proj"), precision)?,
        norm2: read_group_norm(
            file,
            &format!("{base}.norm2"),
            config.norm_groups,
            config.eps,
        )?,
        // The shortcut exists only where the channel count changes.
        shortcut: match conv1.in_channels == conv2.out_channels() {
            true => None,
            false => Some(read_conv(file, &format!("{base}.conv_shortcut"), 1, 0)?),
        },
        conv1,
        conv2,
    })
}

fn read_transformer(
    file: &mut ShardedSafeTensors,
    base: &str,
    config: &UnetConfig,
    level: usize,
    precision: Precision,
) -> Result<Transformer, NetworkError> {
    let mut blocks = Vec::new();
    while file.names().any(|name| {
        name == format!(
            "{base}.transformer_blocks.{}.attn1.to_q.weight",
            blocks.len()
        )
    }) {
        let block = format!("{base}.transformer_blocks.{}", blocks.len());
        blocks.push(TransformerBlock {
            norm1: norm(file, &format!("{block}.norm1"))?,
            attention: read_attention(file, &format!("{block}.attn1"), precision)?,
            norm2: norm(file, &format!("{block}.norm2"))?,
            cross: read_attention(file, &format!("{block}.attn2"), precision)?,
            norm3: norm(file, &format!("{block}.norm3"))?,
            gate: dense(file, &format!("{block}.ff.net.0.proj"), precision)?,
            output: dense(file, &format!("{block}.ff.net.2"), precision)?,
        });
    }
    if blocks.is_empty() {
        return Err(NetworkError::InvalidConfig(format!(
            "the checkpoint holds no transformer blocks under {base}"
        )));
    }

    Ok(Transformer {
        // A transformer's own group norm is tighter than the blocks around it,
        // which is what the reference implementation does.
        norm: read_group_norm(file, &format!("{base}.norm"), config.norm_groups, 1e-6)?,
        input: read_projection(file, &format!("{base}.proj_in"), precision)?,
        output: read_projection(file, &format!("{base}.proj_out"), precision)?,
        heads: config.heads.get(level).copied().unwrap_or(8),
        blocks,
    })
}

fn read_projection(
    file: &mut ShardedSafeTensors,
    name: &str,
    precision: Precision,
) -> Result<Projection, NetworkError> {
    let (values, shape) = file.tensor(&format!("{name}.weight"))?;
    let bias = file
        .tensor(&format!("{name}.bias"))
        .ok()
        .map(|(bias, _)| bias);
    // A rank-four weight is a one-by-one convolution and a rank-two one is a
    // linear layer, and they hold the same numbers.
    match shape.len() {
        4 => Ok(Projection::Convolution(Conv2d::new(
            Matrix::from_vec(shape[0], shape[1], values),
            bias,
            shape[1],
            1,
            1,
            0,
        )?)),
        2 => {
            let mut layer = Dense::new(Matrix::from_vec(shape[0], shape[1], values), bias)?;
            if precision == Precision::Q8 {
                layer.quantize();
            }
            Ok(Projection::Linear(layer))
        }
        _ => Err(NetworkError::InvalidConfig(format!(
            "{name}.weight is {shape:?}, which is neither a projection nor a convolution"
        ))),
    }
}

fn read_attention(
    file: &mut ShardedSafeTensors,
    base: &str,
    precision: Precision,
) -> Result<Attention, NetworkError> {
    Ok(Attention {
        query: dense(file, &format!("{base}.to_q"), precision)?,
        key: dense(file, &format!("{base}.to_k"), precision)?,
        value: dense(file, &format!("{base}.to_v"), precision)?,
        output: dense(file, &format!("{base}.to_out.0"), precision)?,
    })
}

fn read_group_norm(
    file: &mut ShardedSafeTensors,
    name: &str,
    groups: usize,
    eps: f32,
) -> Result<GroupNorm, NetworkError> {
    let (weight, _) = file.tensor(&format!("{name}.weight"))?;
    let (bias, _) = file.tensor(&format!("{name}.bias"))?;
    GroupNorm::new(groups, weight, bias, eps)
}

fn dense(
    file: &mut ShardedSafeTensors,
    name: &str,
    precision: Precision,
) -> Result<Dense, NetworkError> {
    let (values, shape) = file.tensor(&format!("{name}.weight"))?;
    if shape.len() != 2 {
        return Err(NetworkError::InvalidConfig(format!(
            "{name}.weight is {shape:?}, which is not a linear layer"
        )));
    }
    let bias = file
        .tensor(&format!("{name}.bias"))
        .ok()
        .map(|(bias, _)| bias);
    let mut layer = Dense::new(Matrix::from_vec(shape[0], shape[1], values), bias)?;
    if precision == Precision::Q8 {
        layer.quantize();
    }
    Ok(layer)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// A four-channel latent, two rungs of ladder, small enough to run in a
    /// test and shaped like the real thing.
    pub(crate) fn tiny() -> UnetConfig {
        UnetConfig {
            in_channels: 4,
            out_channels: 4,
            block_channels: vec![8, 16],
            heads: vec![2, 2],
            cross_dim: 12,
            norm_groups: 4,
            eps: 1e-5,
            time_dim: 32,
            freq_dim: 8,
            pooled_dim: None,
            addition_time_dim: 4,
        }
    }

    struct Writer {
        tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    }

    impl Writer {
        fn add(&mut self, name: String, shape: Vec<usize>) {
            let count = shape.iter().product::<usize>();
            let values = (0..count)
                .map(|index| ((index % 13) as f32 - 6.0) * 0.03)
                .collect();
            self.tensors.insert(name, (shape, values));
        }

        fn conv(&mut self, name: &str, out: usize, input: usize, kernel: usize) {
            self.add(format!("{name}.weight"), vec![out, input, kernel, kernel]);
            self.add(format!("{name}.bias"), vec![out]);
        }

        fn dense(&mut self, name: &str, out: usize, input: usize) {
            self.add(format!("{name}.weight"), vec![out, input]);
            self.add(format!("{name}.bias"), vec![out]);
        }

        fn norm(&mut self, name: &str, width: usize) {
            self.add(format!("{name}.weight"), vec![width]);
            self.add(format!("{name}.bias"), vec![width]);
        }

        fn resnet(&mut self, name: &str, input: usize, out: usize, time: usize) {
            self.norm(&format!("{name}.norm1"), input);
            self.conv(&format!("{name}.conv1"), out, input, 3);
            self.dense(&format!("{name}.time_emb_proj"), out, time);
            self.norm(&format!("{name}.norm2"), out);
            self.conv(&format!("{name}.conv2"), out, out, 3);
            if input != out {
                self.conv(&format!("{name}.conv_shortcut"), out, input, 1);
            }
        }

        /// `linear` picks which of the two ways a checkpoint can write the
        /// projections into and out of a transformer.
        fn transformer(&mut self, name: &str, width: usize, cross: usize, linear: bool) {
            self.norm(&format!("{name}.norm"), width);
            for projection in ["proj_in", "proj_out"] {
                match linear {
                    true => self.dense(&format!("{name}.{projection}"), width, width),
                    false => self.conv(&format!("{name}.{projection}"), width, width, 1),
                }
            }
            let block = format!("{name}.transformer_blocks.0");
            for index in 1..=3 {
                self.norm(&format!("{block}.norm{index}"), width);
            }
            for name in ["to_q", "to_k", "to_v"] {
                self.add(format!("{block}.attn1.{name}.weight"), vec![width, width]);
            }
            self.dense(&format!("{block}.attn1.to_out.0"), width, width);
            self.add(format!("{block}.attn2.to_q.weight"), vec![width, width]);
            for name in ["to_k", "to_v"] {
                self.add(format!("{block}.attn2.{name}.weight"), vec![width, cross]);
            }
            self.dense(&format!("{block}.attn2.to_out.0"), width, width);
            self.dense(&format!("{block}.ff.net.0.proj"), width * 8, width);
            self.dense(&format!("{block}.ff.net.2"), width, width * 4);
        }
    }

    /// Writes a checkpoint shaped the way diffusers writes one.
    pub(crate) fn checkpoint(config: &UnetConfig, path: &std::path::Path) {
        crate::safetensors::write_checkpoint(path, &unet_writer(config).tensors);
    }

    fn unet_writer(config: &UnetConfig) -> Writer {
        let mut writer = Writer {
            tensors: BTreeMap::new(),
        };
        let time = config.time_dim;
        writer.conv("conv_in", config.block_channels[0], config.in_channels, 3);
        writer.dense("time_embedding.linear_1", time, config.freq_dim);
        writer.dense("time_embedding.linear_2", time, time);
        if let Some(pooled) = config.pooled_dim {
            let width = pooled + 6 * config.addition_time_dim;
            writer.dense("add_embedding.linear_1", time, width);
            writer.dense("add_embedding.linear_2", time, time);
        }

        // The way down: two residual blocks a rung, halving everywhere but the
        // last rung.
        let levels = config.block_channels.len();
        let mut input = config.block_channels[0];
        let mut skips = vec![input];
        for (level, out) in config.block_channels.iter().copied().enumerate() {
            for index in 0..2 {
                let base = format!("down_blocks.{level}.resnets.{index}");
                writer.resnet(&base, input, out, time);
                input = out;
                skips.push(out);
                writer.transformer(
                    &format!("down_blocks.{level}.attentions.{index}"),
                    out,
                    config.cross_dim,
                    false,
                );
            }
            if level + 1 < levels {
                writer.conv(
                    &format!("down_blocks.{level}.downsamplers.0.conv"),
                    out,
                    out,
                    3,
                );
                skips.push(out);
            }
        }

        writer.resnet("mid_block.resnets.0", input, input, time);
        writer.transformer("mid_block.attentions.0", input, config.cross_dim, false);
        writer.resnet("mid_block.resnets.1", input, input, time);

        // And back up, three residual blocks a rung because each one is handed
        // a rung of the way down alongside what the last one produced.
        for (level, out) in config.block_channels.iter().copied().rev().enumerate() {
            for index in 0..3 {
                let skip = skips.pop().expect("a rung for every step up");
                let base = format!("up_blocks.{level}.resnets.{index}");
                writer.resnet(&base, input + skip, out, time);
                input = out;
                writer.transformer(
                    &format!("up_blocks.{level}.attentions.{index}"),
                    out,
                    config.cross_dim,
                    true,
                );
            }
            if level + 1 < levels {
                writer.conv(&format!("up_blocks.{level}.upsamplers.0.conv"), out, out, 3);
            }
        }

        writer.norm("conv_norm_out", config.block_channels[0]);
        writer.conv("conv_out", config.out_channels, config.block_channels[0], 3);
        writer
    }

    #[test]
    fn a_controlnet_steers_the_unet_only_as_hard_as_its_scale() {
        let config = tiny();
        let mut model = load(
            &config,
            "rusting_brain_controlnet_unet.safetensors",
            Precision::F32,
        );
        // The UNet's own weights with a ControlNet's extras on top, which
        // the loader reads the way down out of and ignores the way up of.
        let mut writer = unet_writer(&config);
        let base = "controlnet_cond_embedding";
        writer.conv(&format!("{base}.conv_in"), 4, 3, 3);
        for index in 0..6 {
            writer.conv(&format!("{base}.blocks.{index}"), 4, 4, 3);
        }
        writer.conv(&format!("{base}.conv_out"), 8, 4, 3);
        for (index, width) in [8, 8, 8, 8, 16, 16].into_iter().enumerate() {
            writer.conv(&format!("controlnet_down_blocks.{index}"), width, width, 1);
        }
        writer.conv("controlnet_mid_block", 16, 16, 1);
        let path = std::env::temp_dir().join("rusting_brain_controlnet.safetensors");
        crate::safetensors::write_checkpoint(&path, &writer.tensors);
        let net =
            std::sync::Arc::new(ControlNet::load(&path, config.clone(), Precision::F32).unwrap());
        std::fs::remove_file(&path).ok();

        let mut picture = FeatureMap::new(3, 64, 64);
        for (index, value) in picture.data.iter_mut().enumerate() {
            *value = (index as f32 * 0.37).sin().abs();
        }
        let conditioning = conditioning(&config, 8, 1.0);
        let run = |model: &mut Unet, scale: f32, image: &FeatureMap| {
            model.set_controls(vec![Control {
                net: net.clone(),
                image: image.clone(),
                scale,
            }]);
            model.forward(&latent(&config, 8), 1.0, &conditioning)
        };
        let plain = model
            .forward(&latent(&config, 8), 1.0, &conditioning)
            .unwrap();
        assert_eq!(run(&mut model, 0.0, &picture).unwrap().data, plain.data);
        let steered = run(&mut model, 1.0, &picture).unwrap();
        assert!(steered.data.iter().all(|value| value.is_finite()));
        assert_ne!(steered.data, plain.data);
        assert!(run(&mut model, 1.0, &FeatureMap::new(3, 32, 32)).is_err());
    }

    fn conditioning(config: &UnetConfig, size: usize, seed: f32) -> Conditioning {
        let mut text = Matrix::new(5, config.cross_dim);
        for (index, value) in text.data.iter_mut().enumerate() {
            *value = ((index % 7) as f32 - 3.0) * 0.1 * seed;
        }
        Conditioning {
            text,
            pooled: vec![0.1 * seed; config.pooled_dim.unwrap_or(0)],
            height: size,
            width: size,
            guidance: 0.0,
        }
    }

    fn latent(config: &UnetConfig, size: usize) -> FeatureMap {
        let mut latent = FeatureMap::new(config.in_channels, size, size);
        for (index, value) in latent.data.iter_mut().enumerate() {
            *value = ((index % 17) as f32 - 8.0) * 0.1;
        }
        latent
    }

    fn load(config: &UnetConfig, name: &str, precision: Precision) -> Unet {
        let path = std::env::temp_dir().join(name);
        checkpoint(config, &path);
        let model = Unet::load_at(&path, config.clone(), precision).expect("the model to load");
        std::fs::remove_file(path).ok();
        model
    }

    #[test]
    fn an_inpainting_model_reads_the_mask_after_the_latent() {
        let config = UnetConfig {
            in_channels: 9,
            ..tiny()
        };
        let mut model = load(
            &config,
            "rusting_brain_unet_inpaint.safetensors",
            Precision::F32,
        );
        model.set_conditioning(conditioning(&config, 8, 1.0));
        let noisy = FeatureMap::new(4, 8, 8).to_tokens();
        assert!(model.denoise(&noisy, 1.0).is_err());
        model.set_inpaint(Some(FeatureMap::new(4, 8, 8)));
        assert!(model.denoise(&noisy, 1.0).is_err());

        let mut predictions = Vec::new();
        for repaint in [0.0, 1.0] {
            let mut extra = FeatureMap::new(5, 8, 8);
            extra.data[..64].fill(repaint);
            model.set_inpaint(Some(extra));
            predictions.push(model.denoise(&noisy, 1.0).unwrap());
        }
        assert_eq!(predictions[0].cols, 4);
        assert_ne!(predictions[0].data, predictions[1].data);
    }

    /// Parity with diffusers for an inpainting checkpoint, against the files
    /// `scripts/unet_inpaint_reference.py` writes: run it, then this test with
    /// `RB_UNET_REF` set to its output directory. Worst error on 2026-10-08,
    /// 5.5e-7 against a mean output of 0.075.
    #[test]
    #[ignore = "needs RB_UNET_REF from scripts/unet_inpaint_reference.py"]
    fn an_inpainting_model_matches_diffusers() {
        let dir = std::path::PathBuf::from(std::env::var("RB_UNET_REF").unwrap());
        let config = UnetConfig::from_file(dir.join("unet/config.json")).unwrap();
        let mut model =
            Unet::load(dir.join("unet/diffusion_pytorch_model.safetensors"), config).unwrap();
        let r: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("ref.json")).unwrap()).unwrap();
        let v = |k: &str| -> Vec<f32> {
            r[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let latent = FeatureMap::from_vec(4, 8, 8, v("latent")).unwrap();
        let extra = FeatureMap::from_vec(5, 8, 8, v("extra")).unwrap();
        let conditioning = Conditioning {
            text: Matrix::from_vec(5, 16, v("text")),
            pooled: vec![],
            height: 8,
            width: 8,
            guidance: 0.0,
        };
        model.set_conditioning(conditioning);
        assert!(
            model
                .denoise(&latent.to_tokens(), r["sigma"].as_f64().unwrap() as f32)
                .is_err()
        );
        model.set_inpaint(Some(extra));
        let out = model
            .denoise(&latent.to_tokens(), r["sigma"].as_f64().unwrap() as f32)
            .unwrap();
        let out = FeatureMap::from_tokens(&out, 8, 8).unwrap();
        let reference = v("out");
        let worst = out
            .data
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean = reference.iter().map(|x| x.abs()).sum::<f32>() / reference.len() as f32;
        assert!(worst < 1e-3, "worst {worst} against a mean of {mean}");
    }

    /// Needs `RB_LORA_REF` from `scripts/unet_lora_reference.py`.
    #[test]
    #[ignore]
    fn a_lora_file_merges_as_peft_does() {
        let dir = std::path::PathBuf::from(std::env::var("RB_LORA_REF").unwrap());
        let r: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("ref.json")).unwrap()).unwrap();
        let v = |k: &str| -> Vec<f32> {
            r[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let scale = r["scale"].as_f64().unwrap() as f32;
        let config = UnetConfig::from_file(dir.join("unet/config.json")).unwrap();
        let open =
            || ShardedSafeTensors::open(dir.join("unet/diffusion_pytorch_model.safetensors"));
        for (lora, prefixes) in [
            ("diffusers_lora.safetensors", ["unet", ""]),
            ("kohya_lora.safetensors", ["lora_unet", ""]),
        ] {
            let mut file = open().unwrap();
            // Twelve of each attention projection and resnet, six of the rest.
            assert_eq!(file.add_lora(dir.join(lora), &prefixes, scale).unwrap(), 72);
            let mut model = Unet::read_at(&mut file, config.clone(), Precision::F32).unwrap();
            model.set_conditioning(Conditioning {
                text: Matrix::from_vec(5, 16, v("text")),
                pooled: vec![],
                height: 8,
                width: 8,
                guidance: 0.0,
            });
            let latent = FeatureMap::from_vec(4, 8, 8, v("latent")).unwrap();
            let out = model
                .denoise(&latent.to_tokens(), r["sigma"].as_f64().unwrap() as f32)
                .unwrap();
            let out = FeatureMap::from_tokens(&out, 8, 8).unwrap();
            let reference = v("out");
            let worst = out
                .data
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mean = reference.iter().map(|x| x.abs()).sum::<f32>() / reference.len() as f32;
            assert!(
                worst < 1e-3,
                "{lora}: worst {worst} against a mean of {mean}"
            );
        }
        // The kohya file's text encoder adapter is an error once it is asked for.
        let mut file = open().unwrap();
        assert!(
            file.add_lora(
                dir.join("kohya_lora.safetensors"),
                &["lora_te", "lora_unet"],
                1.0
            )
            .is_err()
        );
    }

    /// Needs `RB_CONTROLNET_REF` from `scripts/controlnet_reference.py`.
    #[test]
    #[ignore]
    fn a_controlnet_matches_diffusers() {
        let dir = std::path::PathBuf::from(std::env::var("RB_CONTROLNET_REF").unwrap());
        let read = |name: &str| {
            let config = UnetConfig::from_file(dir.join(name).join("config.json")).unwrap();
            (
                config,
                dir.join(name).join("diffusion_pytorch_model.safetensors"),
            )
        };
        let (config, path) = read("unet");
        let mut model = Unet::load(path, config).unwrap();
        let (config, path) = read("controlnet");
        let net = std::sync::Arc::new(ControlNet::load(path, config, Precision::F32).unwrap());
        let r: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("ref.json")).unwrap()).unwrap();
        let v = |k: &str| -> Vec<f32> {
            r[k].as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let latent = FeatureMap::from_vec(4, 8, 8, v("latent")).unwrap();
        let sigma = r["sigma"].as_f64().unwrap() as f32;
        model.set_conditioning(Conditioning {
            text: Matrix::from_vec(5, 16, v("text")),
            pooled: vec![],
            height: 8,
            width: 8,
            guidance: 0.0,
        });
        let control = |image: FeatureMap| Control {
            net: net.clone(),
            image,
            scale: r["scale"].as_f64().unwrap() as f32,
        };
        model.set_controls(vec![control(FeatureMap::new(3, 32, 32))]);
        assert!(model.denoise(&latent.to_tokens(), sigma).is_err());
        model.set_controls(vec![control(
            FeatureMap::from_vec(3, 64, 64, v("image")).unwrap(),
        )]);
        let out = model.denoise(&latent.to_tokens(), sigma).unwrap();
        let out = FeatureMap::from_tokens(&out, 8, 8).unwrap();
        let reference = v("out");
        let worst = out
            .data
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean = reference.iter().map(|x| x.abs()).sum::<f32>() / reference.len() as f32;
        assert!(worst < 1e-3, "worst {worst} against a mean of {mean}");
    }

    #[test]
    fn the_ladder_comes_back_to_the_size_it_left() {
        let config = tiny();
        let model = load(
            &config,
            "rusting_brain_unet_shape.safetensors",
            Precision::F32,
        );
        let output = model
            .forward(&latent(&config, 8), 1.0, &conditioning(&config, 8, 1.0))
            .expect("a prediction");
        assert_eq!(output.channels, config.out_channels);
        assert_eq!((output.height, output.width), (8, 8));
        assert!(output.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn the_prompt_changes_what_comes_back() {
        let config = tiny();
        let model = load(
            &config,
            "rusting_brain_unet_prompt.safetensors",
            Precision::F32,
        );
        let latent = latent(&config, 8);
        let first = model
            .forward(&latent, 1.0, &conditioning(&config, 8, 1.0))
            .expect("a prediction");
        let second = model
            .forward(&latent, 1.0, &conditioning(&config, 8, -2.0))
            .expect("a prediction");
        let moved = first
            .data
            .iter()
            .zip(&second.data)
            .any(|(first, second)| (first - second).abs() > 1e-4);
        assert!(moved, "cross-attention is not reading the prompt");
    }

    #[test]
    fn the_noise_level_becomes_the_step_index_it_was_trained_on() {
        let config = tiny();
        let model = load(
            &config,
            "rusting_brain_unet_time.safetensors",
            Precision::F32,
        );
        for step in [1usize, 250, 500, 999] {
            let alpha = model.alphas[step];
            let sigma = ((1.0 - alpha) / alpha).sqrt();
            let recovered = model.timestep(sigma);
            assert!(
                (recovered - step as f32).abs() < 0.5,
                "{sigma} came back as {recovered} rather than {step}"
            );
        }
        // Below the first level there is nowhere further to go.
        assert_eq!(model.timestep(0.0), 0.0);
        assert!(model.timestep(1e9) >= 998.0);
    }

    #[test]
    fn a_model_with_a_pooled_conditioning_reads_it() {
        let mut config = tiny();
        config.pooled_dim = Some(6);
        let model = load(
            &config,
            "rusting_brain_unet_pooled.safetensors",
            Precision::F32,
        );
        assert_eq!(model.pooled_dim(), Some(6));

        let latent = latent(&config, 8);
        let mut first = conditioning(&config, 8, 1.0);
        let second = Conditioning {
            pooled: vec![0.5; 6],
            ..first.clone()
        };
        first.pooled = vec![-0.5; 6];
        let quiet = model.forward(&latent, 1.0, &first).expect("a prediction");
        let loud = model.forward(&latent, 1.0, &second).expect("a prediction");
        let moved = quiet
            .data
            .iter()
            .zip(&loud.data)
            .any(|(quiet, loud)| (quiet - loud).abs() > 1e-4);
        assert!(moved, "the pooled vector is not reaching the blocks");
    }

    #[test]
    fn a_quantized_model_says_what_the_float_one_does() {
        let config = tiny();
        let float = load(
            &config,
            "rusting_brain_unet_float.safetensors",
            Precision::F32,
        );
        let quantized = load(
            &config,
            "rusting_brain_unet_eight.safetensors",
            Precision::Q8,
        );
        let latent = latent(&config, 8);
        let conditioning = conditioning(&config, 8, 1.0);
        let float = float.forward(&latent, 1.0, &conditioning).expect("float");
        let eight = quantized
            .forward(&latent, 1.0, &conditioning)
            .expect("quantized");
        let error = float
            .data
            .iter()
            .zip(&eight.data)
            .map(|(float, eight)| (float - eight).abs())
            .fold(0.0f32, f32::max);
        assert!(error < 0.2, "int8 drifted by {error}");
    }

    #[test]
    fn a_denoiser_without_a_prompt_says_so() {
        let config = tiny();
        let mut model = load(
            &config,
            "rusting_brain_unet_denoise.safetensors",
            Precision::F32,
        );
        let tokens = latent(&config, 8).to_tokens();
        assert!(model.denoise(&tokens, 1.0).is_err());

        model.set_conditioning(conditioning(&config, 8, 1.0));
        let predicted = model.denoise(&tokens, 1.0).expect("a prediction");
        assert_eq!((predicted.rows, predicted.cols), (64, config.out_channels));
        assert!(
            model
                .denoise_unconditional(&tokens, 1.0)
                .expect("no error")
                .is_none()
        );
    }

    #[test]
    fn a_configuration_is_read_the_way_diffusers_writes_one() {
        let path = std::env::temp_dir().join("rusting_brain_unet_config.json");
        std::fs::write(
            &path,
            r#"{
                "in_channels": 4,
                "out_channels": 4,
                "block_out_channels": [320, 640, 1280],
                "attention_head_dim": [5, 10, 20],
                "cross_attention_dim": 2048,
                "norm_num_groups": 32,
                "norm_eps": 1e-05,
                "addition_time_embed_dim": 256,
                "projection_class_embeddings_input_dim": 2816
            }"#,
        )
        .expect("the config to write");
        let config = UnetConfig::from_file(&path).expect("the config to read");
        assert_eq!(config.heads, vec![5, 10, 20]);
        assert_eq!(config.freq_dim, 320);
        assert_eq!(config.time_dim, 1280);
        // 2816 is the pooled prompt plus six size numbers at 256 apiece.
        assert_eq!(config.pooled_dim, Some(1280));

        // One head count covers every level, and a model without the pooled
        // projection takes no pooled vector.
        std::fs::write(
            &path,
            r#"{"block_out_channels": [320, 640], "attention_head_dim": 8, "cross_attention_dim": [768, 768]}"#,
        )
        .expect("the config to write");
        let config = UnetConfig::from_file(&path).expect("the config to read");
        assert_eq!(config.heads, vec![8, 8]);
        assert_eq!(config.cross_dim, 768);
        assert_eq!(config.pooled_dim, None);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_checkpoint_that_is_not_a_unet_is_refused() {
        let path = std::env::temp_dir().join("rusting_brain_unet_stub.safetensors");
        let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        tensors.insert("conv_in.weight".into(), (vec![8, 4, 3, 3], vec![0.0; 288]));
        crate::safetensors::write_checkpoint(&path, &tensors);
        assert!(Unet::load(&path, tiny()).is_err());
        std::fs::remove_file(path).ok();
    }
}
