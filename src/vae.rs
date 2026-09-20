//! The decoder that turns a diffusion model's latent into pixels.
//!
//! Every latent diffusion model — Stable Diffusion 1.x, SDXL, SD3, FLUX,
//! FLUX.2 — ends the same way: the sampler produces a small, many-channel
//! latent, and a convolutional decoder expands it to an image. The decoder is
//! the half of a variational autoencoder that runs at inference; the encoder
//! only matters for image-to-image work.
//!
//! The architecture below is the one `diffusers` calls `AutoencoderKL`, which
//! is what every checkpoint in that line ships: a stem convolution, a middle
//! pair of residual blocks with one self-attention between them, then a ladder
//! of residual blocks that doubles the resolution at each rung, and a final
//! normalize-activate-convolve to three channels.
//!
//! [`VaeDecoder::load`] reads such a checkpoint straight from
//! `.safetensors` under the `diffusers` key names, so hosting a published
//! decoder needs no conversion step.
//!
//! Inference only, like [`crate::conv`]: no gradients, no CUDA.

use crate::conv::{Conv2d, Dense, FeatureMap, GroupNorm, silu, upsample_nearest};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;

/// The shape of a decoder, which a checkpoint's `config.json` names.
#[derive(Clone, Debug, PartialEq)]
pub struct VaeConfig {
    /// Channels in the latent the sampler produces. Four for Stable Diffusion
    /// 1.x and XL, sixteen for SD3 and FLUX, thirty-two for FLUX.2.
    pub latent_channels: usize,
    /// Channels in the image. Three, unless something unusual is going on.
    pub out_channels: usize,
    /// Channels at each resolution, from the image's resolution inward, which
    /// is the order the checkpoint's own configuration lists them in.
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub norm_groups: usize,
    /// The latent's scale and offset, from the same configuration. A latent is
    /// stored as `(value - shift) * scaling`, so decoding undoes that first.
    pub scaling_factor: f32,
    pub shift_factor: f32,
}

impl Default for VaeConfig {
    /// The Stable Diffusion 1.x and XL decoder, which is the most common one.
    fn default() -> Self {
        Self {
            latent_channels: 4,
            out_channels: 3,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_groups: 32,
            scaling_factor: 0.18215,
            shift_factor: 0.0,
        }
    }
}

impl VaeConfig {
    /// The FLUX and FLUX.2 decoder: sixteen latent channels and the scale and
    /// shift those models were trained with.
    pub fn flux() -> Self {
        Self {
            latent_channels: 16,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
            ..Self::default()
        }
    }

    /// Reads a diffusers `vae/config.json`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let text = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| NetworkError::InvalidDataset(format!("vae config.json: {error}")))?;

        let default = Self::default();
        let number = |name: &str, fallback: usize| -> usize {
            json.get(name)
                .and_then(|value| value.as_u64())
                .map_or(fallback, |value| value as usize)
        };
        let float = |name: &str, fallback: f32| -> f32 {
            json.get(name)
                .and_then(|value| value.as_f64())
                .map_or(fallback, |value| value as f32)
        };

        Ok(Self {
            latent_channels: number("latent_channels", default.latent_channels),
            out_channels: number("out_channels", default.out_channels),
            block_out_channels: match json
                .get("block_out_channels")
                .and_then(|value| value.as_array())
            {
                Some(channels) => channels
                    .iter()
                    .map(|count| count.as_u64().map(|count| count as usize))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| {
                        NetworkError::InvalidDataset(
                            "vae config.json: block_out_channels is not a list of sizes".into(),
                        )
                    })?,
                None => default.block_out_channels,
            },
            layers_per_block: number("layers_per_block", default.layers_per_block),
            norm_groups: number("norm_num_groups", default.norm_groups),
            scaling_factor: float("scaling_factor", default.scaling_factor),
            // Only the models that need an offset record one.
            shift_factor: float("shift_factor", 0.0),
        })
    }

    /// How much larger the image is than the latent: one doubling per rung of
    /// the ladder above the first.
    pub fn upscale(&self) -> usize {
        1 << (self.block_out_channels.len() - 1)
    }
}

/// Two convolutions with a skip connection, the unit the whole decoder is
/// built from.
#[derive(Clone, Debug)]
pub(crate) struct ResnetBlock {
    pub(crate) norm1: GroupNorm,
    pub(crate) conv1: Conv2d,
    pub(crate) norm2: GroupNorm,
    pub(crate) conv2: Conv2d,
    /// A one-by-one convolution on the skip path, present only where the block
    /// changes the channel count.
    pub(crate) shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn forward(&self, input: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        let mut hidden = input.clone();
        self.norm1.forward(&mut hidden)?;
        silu(&mut hidden);
        let mut hidden = self.conv1.forward(&hidden)?;
        self.norm2.forward(&mut hidden)?;
        silu(&mut hidden);
        let mut hidden = self.conv2.forward(&hidden)?;

        let skip = match &self.shortcut {
            Some(shortcut) => shortcut.forward(input)?,
            None => input.clone(),
        };
        if skip.data.len() != hidden.data.len() {
            return Err(NetworkError::InvalidTarget {
                expected: hidden.data.len(),
                actual: skip.data.len(),
            });
        }
        for (value, skip) in hidden.data.iter_mut().zip(&skip.data) {
            *value += skip;
        }
        Ok(hidden)
    }
}

/// Self-attention over pixels, which the decoder runs once at its lowest
/// resolution, where the pixel count is small enough for the quadratic cost.
#[derive(Clone, Debug)]
pub(crate) struct AttentionBlock {
    pub(crate) norm: GroupNorm,
    pub(crate) query: Dense,
    pub(crate) key: Dense,
    pub(crate) value: Dense,
    pub(crate) output: Dense,
}

impl AttentionBlock {
    fn forward(&self, input: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        let mut normed = input.clone();
        self.norm.forward(&mut normed)?;
        let tokens = normed.to_tokens();

        let queries = self.query.forward(&tokens)?;
        let keys = self.key.forward(&tokens)?;
        let values = self.value.forward(&tokens)?;

        let scale = (queries.cols as f32).sqrt().recip();
        let mut attended = Matrix::new(tokens.rows, values.cols);
        let mut weights = vec![0.0; tokens.rows];
        for pixel in 0..tokens.rows {
            let query = queries.row(pixel);
            let mut largest = f32::NEG_INFINITY;
            for (weight, other) in weights.iter_mut().zip(0..tokens.rows) {
                let score = query
                    .iter()
                    .zip(keys.row(other))
                    .map(|(query, key)| query * key)
                    .sum::<f32>()
                    * scale;
                *weight = score;
                largest = largest.max(score);
            }
            let mut total = 0.0;
            for weight in &mut weights {
                *weight = (*weight - largest).exp();
                total += *weight;
            }
            let row = attended.row_mut(pixel);
            for (other, weight) in weights.iter().enumerate() {
                let weight = weight / total;
                for (value, source) in row.iter_mut().zip(values.row(other)) {
                    *value += weight * source;
                }
            }
        }

        let projected = self.output.forward(&attended)?;
        let mut output = FeatureMap::from_tokens(&projected, input.height, input.width)?;
        for (value, skip) in output.data.iter_mut().zip(&input.data) {
            *value += skip;
        }
        Ok(output)
    }
}

/// One rung of the ladder: several residual blocks, then a doubling.
#[derive(Clone, Debug)]
pub(crate) struct UpBlock {
    pub(crate) resnets: Vec<ResnetBlock>,
    /// The convolution that follows the nearest-neighbour upsample. Absent on
    /// the last rung, which is already at the image's resolution.
    pub(crate) upsampler: Option<Conv2d>,
}

/// The decoder half of a latent diffusion model's autoencoder.
#[derive(Clone, Debug)]
pub struct VaeDecoder {
    config: VaeConfig,
    /// The one-by-one convolution the Stable Diffusion line puts between the
    /// latent and the decoder, mirroring the encoder's `quant_conv`. It lives
    /// beside `decoder.` rather than inside it, and the FLUX line has none.
    pub(crate) post_quant: Option<Conv2d>,
    pub(crate) conv_in: Conv2d,
    pub(crate) mid_first: ResnetBlock,
    pub(crate) mid_attention: AttentionBlock,
    pub(crate) mid_last: ResnetBlock,
    pub(crate) up_blocks: Vec<UpBlock>,
    pub(crate) norm_out: GroupNorm,
    pub(crate) conv_out: Conv2d,
    /// The same weights on a CUDA device, when one was attached.
    #[cfg(feature = "cuda")]
    device: Option<std::sync::Arc<crate::cuda_image::DeviceVae>>,
}

impl VaeDecoder {
    /// The configuration this decoder was built for.
    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    /// Reads a decoder out of a checkpoint under the `diffusers` key names.
    ///
    /// `path` is either a `.safetensors` file or the
    /// `model.safetensors.index.json` of a sharded one. `prefix` is what the
    /// decoder's keys start with: `"decoder."` in a standalone VAE file, and
    /// `"vae.decoder."` in a checkpoint that holds the whole pipeline.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        prefix: &str,
        config: VaeConfig,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        Self::read(&mut file, prefix, config)
    }

    /// The same, from a checkpoint that is already open.
    pub fn read(
        file: &mut ShardedSafeTensors,
        prefix: &str,
        config: VaeConfig,
    ) -> Result<Self, NetworkError> {
        if config.block_out_channels.is_empty() {
            return Err(NetworkError::InvalidConfig(
                "a decoder needs at least one resolution".into(),
            ));
        }
        // The checkpoint lists channels from the image inward and runs them
        // outward, so the ladder is the reverse of the configuration.
        let ladder: Vec<usize> = config.block_out_channels.iter().rev().copied().collect();
        let deepest = ladder[0];

        // `post_quant_conv` sits one level up from the decoder's own keys.
        let root = prefix.strip_suffix("decoder.").unwrap_or("");
        let post_quant = read_conv(file, &format!("{root}post_quant_conv"), 1, 0).ok();
        let conv_in = read_conv(file, &format!("{prefix}conv_in"), 1, 1)?;
        let mid = format!("{prefix}mid_block");
        let mid_first = read_resnet(file, &format!("{mid}.resnets.0"), config.norm_groups)?;
        let mid_attention =
            read_attention(file, &format!("{mid}.attentions.0"), config.norm_groups)?;
        let mid_last = read_resnet(file, &format!("{mid}.resnets.1"), config.norm_groups)?;

        let mut up_blocks = Vec::with_capacity(ladder.len());
        for (rung, _) in ladder.iter().enumerate() {
            let base = format!("{prefix}up_blocks.{rung}");
            // One more residual block per rung than the configuration's count,
            // which is the autoencoder's own asymmetry.
            let resnets = (0..config.layers_per_block + 1)
                .map(|index| {
                    read_resnet(file, &format!("{base}.resnets.{index}"), config.norm_groups)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let upsampler = if rung + 1 < ladder.len() {
                Some(read_conv(file, &format!("{base}.upsamplers.0.conv"), 1, 1)?)
            } else {
                None
            };
            up_blocks.push(UpBlock { resnets, upsampler });
        }

        let norm_out =
            read_group_norm(file, &format!("{prefix}conv_norm_out"), config.norm_groups)?;
        let conv_out = read_conv(file, &format!("{prefix}conv_out"), 1, 1)?;

        // With a `post_quant_conv` in front, the stem reads what that
        // produces and the latent is what the convolution reads.
        let takes = post_quant
            .as_ref()
            .map_or(conv_in.in_channels, |quant| quant.in_channels);
        if takes != config.latent_channels || conv_in.out_channels() != deepest {
            return Err(NetworkError::InvalidConfig(format!(
                "the checkpoint's stem maps {} channels to {}, and the configuration says {} to \
                 {deepest}",
                takes,
                conv_in.out_channels(),
                config.latent_channels
            )));
        }

        Ok(Self {
            config,
            post_quant,
            conv_in,
            mid_first,
            mid_attention,
            mid_last,
            up_blocks,
            norm_out,
            conv_out,
            #[cfg(feature = "cuda")]
            device: None,
        })
    }

    /// Uploads every weight to a CUDA device and decodes there from now on.
    ///
    /// Fails closed, leaving the decoder on the CPU if the upload does not fit.
    #[cfg(feature = "cuda")]
    pub fn attach_device(
        &mut self,
        gpu: std::sync::Arc<crate::cuda_image::ImageGpu>,
    ) -> Result<(), NetworkError> {
        self.device = Some(std::sync::Arc::new(crate::cuda_image::DeviceVae::upload(
            &gpu, self,
        )?));
        Ok(())
    }

    /// Decodes a latent into an image whose values sit in roughly `[-1, 1]`.
    ///
    /// `latent` is what the sampler produced, in channel-height-width order and
    /// still carrying the model's scale and shift; both are undone here.
    pub fn decode(&self, latent: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        if latent.channels != self.config.latent_channels {
            return Err(NetworkError::InvalidTarget {
                expected: self.config.latent_channels,
                actual: latent.channels,
            });
        }

        let mut scaled = latent.clone();
        for value in &mut scaled.data {
            *value = *value / self.config.scaling_factor + self.config.shift_factor;
        }

        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            // The device mirror starts at `post_quant`, so the scale and shift
            // above are the only arithmetic either path does on the host.
            return device.decode(&scaled);
        }

        let scaled = match &self.post_quant {
            Some(quant) => quant.forward(&scaled)?,
            None => scaled,
        };

        let mut hidden = self.conv_in.forward(&scaled)?;
        hidden = self.mid_first.forward(&hidden)?;
        hidden = self.mid_attention.forward(&hidden)?;
        hidden = self.mid_last.forward(&hidden)?;

        for block in &self.up_blocks {
            for resnet in &block.resnets {
                hidden = resnet.forward(&hidden)?;
            }
            if let Some(upsampler) = &block.upsampler {
                hidden = upsampler.forward(&upsample_nearest(&hidden, 2))?;
            }
        }

        self.norm_out.forward(&mut hidden)?;
        silu(&mut hidden);
        self.conv_out.forward(&hidden)
    }
}

/// One rung of the encoder's ladder: several residual blocks, then a halving.
#[derive(Clone, Debug)]
struct DownBlock {
    resnets: Vec<ResnetBlock>,
    /// The stride-two convolution that ends the rung. Absent on the last one,
    /// which is already at the latent's resolution.
    downsampler: Option<Conv2d>,
}

/// The encoder half of a latent diffusion model's autoencoder.
///
/// Only image-to-image work needs one: text-to-image starts from noise, and
/// the encoder is what turns a starting picture into the latent the sampler
/// then partially renoises. The architecture mirrors [`VaeDecoder`] — a stem,
/// a ladder that halves the resolution at each rung, and a middle pair of
/// residual blocks around one self-attention — and its keys sit under
/// `encoder.` in the same file.
#[derive(Clone, Debug)]
pub struct VaeEncoder {
    config: VaeConfig,
    conv_in: Conv2d,
    down_blocks: Vec<DownBlock>,
    mid_first: ResnetBlock,
    mid_attention: AttentionBlock,
    mid_last: ResnetBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    /// The one-by-one convolution some checkpoints put between the encoder and
    /// the latent. It lives beside `encoder.` rather than inside it.
    quant: Option<Conv2d>,
}

impl VaeEncoder {
    /// The configuration this encoder was built for.
    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    /// Reads an encoder out of a checkpoint under the `diffusers` key names.
    ///
    /// `path` and `prefix` mean what they do for [`VaeDecoder::load`], with
    /// `"encoder."` in place of `"decoder."`.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        prefix: &str,
        config: VaeConfig,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        Self::read(&mut file, prefix, config)
    }

    /// The same, from a checkpoint that is already open.
    pub fn read(
        file: &mut ShardedSafeTensors,
        prefix: &str,
        config: VaeConfig,
    ) -> Result<Self, NetworkError> {
        if config.block_out_channels.is_empty() {
            return Err(NetworkError::InvalidConfig(
                "an encoder needs at least one resolution".into(),
            ));
        }
        // The encoder runs the configuration's channels in the order they are
        // written: from the image's resolution inward.
        let ladder = &config.block_out_channels;

        let conv_in = read_conv(file, &format!("{prefix}conv_in"), 1, 1)?;

        let mut down_blocks = Vec::with_capacity(ladder.len());
        for rung in 0..ladder.len() {
            let base = format!("{prefix}down_blocks.{rung}");
            let resnets = (0..config.layers_per_block)
                .map(|index| {
                    read_resnet(file, &format!("{base}.resnets.{index}"), config.norm_groups)
                })
                .collect::<Result<Vec<_>, _>>()?;
            // The halving convolution pads on the right and bottom only, which
            // `encode` does for it, so the convolution itself pads nothing.
            let downsampler = if rung + 1 < ladder.len() {
                Some(read_conv(
                    file,
                    &format!("{base}.downsamplers.0.conv"),
                    2,
                    0,
                )?)
            } else {
                None
            };
            down_blocks.push(DownBlock {
                resnets,
                downsampler,
            });
        }

        let mid = format!("{prefix}mid_block");
        let mid_first = read_resnet(file, &format!("{mid}.resnets.0"), config.norm_groups)?;
        let mid_attention =
            read_attention(file, &format!("{mid}.attentions.0"), config.norm_groups)?;
        let mid_last = read_resnet(file, &format!("{mid}.resnets.1"), config.norm_groups)?;

        let norm_out =
            read_group_norm(file, &format!("{prefix}conv_norm_out"), config.norm_groups)?;
        let conv_out = read_conv(file, &format!("{prefix}conv_out"), 1, 1)?;
        // `quant_conv` sits one level up from the encoder's own keys.
        let root = prefix.strip_suffix("encoder.").unwrap_or("");
        let quant = read_conv(file, &format!("{root}quant_conv"), 1, 0).ok();

        if conv_in.in_channels != config.out_channels || conv_in.out_channels() != ladder[0] {
            return Err(NetworkError::InvalidConfig(format!(
                "the checkpoint's stem maps {} channels to {}, and the configuration says {} to {}",
                conv_in.in_channels,
                conv_in.out_channels(),
                config.out_channels,
                ladder[0]
            )));
        }
        // Most checkpoints predict a mean and a log-variance, so the last
        // convolution is twice as wide as the latent; a few predict the mean
        // alone. Either is fine, and anything else is the wrong file.
        let produced = quant
            .as_ref()
            .map_or(conv_out.out_channels(), |quant| quant.out_channels());
        if produced != config.latent_channels && produced != 2 * config.latent_channels {
            return Err(NetworkError::InvalidConfig(format!(
                "the encoder ends in {produced} channels, which is neither {} nor twice it",
                config.latent_channels
            )));
        }

        Ok(Self {
            config,
            conv_in,
            down_blocks,
            mid_first,
            mid_attention,
            mid_last,
            norm_out,
            conv_out,
            quant,
        })
    }

    /// Encodes an image whose values sit in roughly `[-1, 1]` into the latent
    /// the sampler works on, scale and shift already applied.
    ///
    /// The result is the distribution's mean rather than a draw from it, which
    /// is what image-to-image wants: the noise it needs is added afterwards, at
    /// the level the run starts from.
    pub fn encode(&self, image: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        if image.channels != self.config.out_channels {
            return Err(NetworkError::InvalidTarget {
                expected: self.config.out_channels,
                actual: image.channels,
            });
        }

        let mut hidden = self.conv_in.forward(image)?;
        for block in &self.down_blocks {
            for resnet in &block.resnets {
                hidden = resnet.forward(&hidden)?;
            }
            if let Some(downsampler) = &block.downsampler {
                hidden = downsampler.forward(&pad_right_bottom(&hidden))?;
            }
        }

        hidden = self.mid_first.forward(&hidden)?;
        hidden = self.mid_attention.forward(&hidden)?;
        hidden = self.mid_last.forward(&hidden)?;

        self.norm_out.forward(&mut hidden)?;
        silu(&mut hidden);
        let mut moments = self.conv_out.forward(&hidden)?;
        if let Some(quant) = &self.quant {
            moments = quant.forward(&moments)?;
        }

        // The second half of the channels is the log-variance, which only
        // sampling from the distribution would need.
        let channels = self.config.latent_channels;
        let pixels = moments.pixels();
        let mut latent = FeatureMap::from_vec(
            channels,
            moments.height,
            moments.width,
            moments.data[..channels * pixels].to_vec(),
        )?;
        for value in &mut latent.data {
            *value = (*value - self.config.shift_factor) * self.config.scaling_factor;
        }
        Ok(latent)
    }
}

/// Adds one row at the bottom and one column at the right, which is the
/// padding `diffusers` applies before an encoder's halving convolution.
fn pad_right_bottom(map: &FeatureMap) -> FeatureMap {
    let (height, width) = (map.height + 1, map.width + 1);
    let mut padded = FeatureMap::new(map.channels, height, width);
    for channel in 0..map.channels {
        for row in 0..map.height {
            let from = (channel * map.height + row) * map.width;
            let to = (channel * height + row) * width;
            padded.data[to..to + map.width].copy_from_slice(&map.data[from..from + map.width]);
        }
    }
    padded
}

/// An image in `[-1, 1]` as the eight-bit interleaved rows a PNG encoder wants.
pub fn to_rgb8(image: &FeatureMap) -> Vec<u8> {
    let pixels = image.pixels();
    let mut bytes = Vec::with_capacity(pixels * image.channels);
    for pixel in 0..pixels {
        for channel in 0..image.channels {
            let value = image.data[channel * pixels + pixel];
            bytes.push(((value * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    bytes
}

/// Writes a decoded image to a PNG.
#[cfg(feature = "images")]
pub fn save_png(image: &FeatureMap, path: impl AsRef<std::path::Path>) -> Result<(), NetworkError> {
    let bytes = to_rgb8(image);
    let (width, height) = (image.width as u32, image.height as u32);
    let path = path.as_ref();
    let write = |error: image::ImageError| {
        NetworkError::InvalidDataset(format!("{}: {error}", path.display()))
    };
    match image.channels {
        3 => image::RgbImage::from_raw(width, height, bytes)
            .ok_or_else(|| {
                NetworkError::InvalidConfig("the image's shape and its bytes disagree".into())
            })?
            .save(path)
            .map_err(write),
        1 => image::GrayImage::from_raw(width, height, bytes)
            .ok_or_else(|| {
                NetworkError::InvalidConfig("the image's shape and its bytes disagree".into())
            })?
            .save(path)
            .map_err(write),
        channels => Err(NetworkError::InvalidConfig(format!(
            "a {channels}-channel image is neither grey nor RGB"
        ))),
    }
}

pub(crate) fn read_conv(
    file: &mut ShardedSafeTensors,
    name: &str,
    stride: usize,
    padding: usize,
) -> Result<Conv2d, NetworkError> {
    let (values, shape) = file.tensor(&format!("{name}.weight"))?;
    if shape.len() != 4 || shape[2] != shape[3] {
        return Err(NetworkError::InvalidConfig(format!(
            "{name}.weight is {shape:?}, which is not a square convolution kernel"
        )));
    }
    let (out_channels, in_channels, kernel) = (shape[0], shape[1], shape[2]);
    let bias = file
        .tensor(&format!("{name}.bias"))
        .ok()
        .map(|(bias, _)| bias);
    // A one-by-one convolution needs no padding whatever the caller asked for.
    let padding = if kernel == 1 { 0 } else { padding };
    Conv2d::new(
        Matrix::from_vec(out_channels, in_channels * kernel * kernel, values),
        bias,
        in_channels,
        kernel,
        stride,
        padding,
    )
}

fn read_group_norm(
    file: &mut ShardedSafeTensors,
    name: &str,
    groups: usize,
) -> Result<GroupNorm, NetworkError> {
    let (weight, _) = file.tensor(&format!("{name}.weight"))?;
    let (bias, _) = file.tensor(&format!("{name}.bias"))?;
    GroupNorm::new(groups, weight, bias, 1e-6)
}

fn read_resnet(
    file: &mut ShardedSafeTensors,
    name: &str,
    groups: usize,
) -> Result<ResnetBlock, NetworkError> {
    let conv1 = read_conv(file, &format!("{name}.conv1"), 1, 1)?;
    let conv2 = read_conv(file, &format!("{name}.conv2"), 1, 1)?;
    // The shortcut exists only where the channel count changes, so its absence
    // from the checkpoint is information rather than an error.
    let shortcut = match read_conv(file, &format!("{name}.conv_shortcut"), 1, 0) {
        Ok(conv) => Some(conv),
        Err(_) if conv1.in_channels == conv2.out_channels() => None,
        Err(error) => return Err(error),
    };
    Ok(ResnetBlock {
        norm1: read_group_norm(file, &format!("{name}.norm1"), groups)?,
        conv1,
        norm2: read_group_norm(file, &format!("{name}.norm2"), groups)?,
        conv2,
        shortcut,
    })
}

fn read_attention(
    file: &mut ShardedSafeTensors,
    name: &str,
    groups: usize,
) -> Result<AttentionBlock, NetworkError> {
    let projection = |file: &mut ShardedSafeTensors, suffix: &str| -> Result<Dense, NetworkError> {
        let (values, shape) = file.tensor(&format!("{name}.{suffix}.weight"))?;
        // Older checkpoints store these as `[c, c, 1, 1]` convolutions and
        // newer ones as `[c, c]` linear weights; the values are the same.
        let channels = shape[0];
        let (bias, _) = file.tensor(&format!("{name}.{suffix}.bias"))?;
        Dense::new(
            Matrix::from_vec(channels, values.len() / channels, values),
            Some(bias),
        )
    };
    Ok(AttentionBlock {
        norm: read_group_norm(file, &format!("{name}.group_norm"), groups)?,
        query: projection(file, "to_q")?,
        key: projection(file, "to_k")?,
        value: projection(file, "to_v")?,
        output: projection(file, "to_out.0")?,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// A checkpoint with every key the loader reads, at a size that runs fast.
    pub(crate) fn checkpoint(config: &VaeConfig, path: &std::path::Path) {
        checkpoint_at(config, path, true)
    }

    /// The same, with the choice of carrying the convolution the Stable
    /// Diffusion line puts in front of the decoder. The FLUX line has none.
    pub(crate) fn checkpoint_at(config: &VaeConfig, path: &std::path::Path, post_quant: bool) {
        let ladder: Vec<usize> = config.block_out_channels.iter().rev().copied().collect();
        let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let count = shape.iter().product::<usize>();
            // Small deterministic values: a real decoder's weights are small,
            // and the test cares about shapes, not about the image.
            let values = (0..count)
                .map(|index| ((index % 7) as f32 - 3.0) * 0.02)
                .collect();
            tensors.insert(name, (shape, values));
        };

        let deepest = ladder[0];
        add(
            "decoder.conv_in.weight".into(),
            vec![deepest, config.latent_channels, 3, 3],
        );
        add("decoder.conv_in.bias".into(), vec![deepest]);

        let resnet =
            |name: &str, input: usize, output: usize, add: &mut dyn FnMut(String, Vec<usize>)| {
                add(format!("{name}.norm1.weight"), vec![input]);
                add(format!("{name}.norm1.bias"), vec![input]);
                add(format!("{name}.conv1.weight"), vec![output, input, 3, 3]);
                add(format!("{name}.conv1.bias"), vec![output]);
                add(format!("{name}.norm2.weight"), vec![output]);
                add(format!("{name}.norm2.bias"), vec![output]);
                add(format!("{name}.conv2.weight"), vec![output, output, 3, 3]);
                add(format!("{name}.conv2.bias"), vec![output]);
                if input != output {
                    add(
                        format!("{name}.conv_shortcut.weight"),
                        vec![output, input, 1, 1],
                    );
                    add(format!("{name}.conv_shortcut.bias"), vec![output]);
                }
            };

        resnet("decoder.mid_block.resnets.0", deepest, deepest, &mut add);
        resnet("decoder.mid_block.resnets.1", deepest, deepest, &mut add);
        for suffix in ["group_norm", "to_q", "to_k", "to_v", "to_out.0"] {
            let shape = if suffix == "group_norm" {
                vec![deepest]
            } else {
                vec![deepest, deepest]
            };
            add(
                format!("decoder.mid_block.attentions.0.{suffix}.weight"),
                shape,
            );
            add(
                format!("decoder.mid_block.attentions.0.{suffix}.bias"),
                vec![deepest],
            );
        }

        let mut input = deepest;
        for (rung, channels) in ladder.iter().enumerate() {
            for index in 0..config.layers_per_block + 1 {
                resnet(
                    &format!("decoder.up_blocks.{rung}.resnets.{index}"),
                    input,
                    *channels,
                    &mut add,
                );
                input = *channels;
            }
            if rung + 1 < ladder.len() {
                add(
                    format!("decoder.up_blocks.{rung}.upsamplers.0.conv.weight"),
                    vec![*channels, *channels, 3, 3],
                );
                add(
                    format!("decoder.up_blocks.{rung}.upsamplers.0.conv.bias"),
                    vec![*channels],
                );
            }
        }

        add("decoder.conv_norm_out.weight".into(), vec![input]);
        add("decoder.conv_norm_out.bias".into(), vec![input]);
        add(
            "decoder.conv_out.weight".into(),
            vec![config.out_channels, input, 3, 3],
        );
        add("decoder.conv_out.bias".into(), vec![config.out_channels]);

        // The encoder half, which image-to-image reads.
        let forward = &config.block_out_channels;
        add(
            "encoder.conv_in.weight".into(),
            vec![forward[0], config.out_channels, 3, 3],
        );
        add("encoder.conv_in.bias".into(), vec![forward[0]]);

        let mut input = forward[0];
        for (rung, channels) in forward.iter().enumerate() {
            for index in 0..config.layers_per_block {
                resnet(
                    &format!("encoder.down_blocks.{rung}.resnets.{index}"),
                    input,
                    *channels,
                    &mut add,
                );
                input = *channels;
            }
            if rung + 1 < forward.len() {
                add(
                    format!("encoder.down_blocks.{rung}.downsamplers.0.conv.weight"),
                    vec![*channels, *channels, 3, 3],
                );
                add(
                    format!("encoder.down_blocks.{rung}.downsamplers.0.conv.bias"),
                    vec![*channels],
                );
            }
        }

        resnet("encoder.mid_block.resnets.0", input, input, &mut add);
        resnet("encoder.mid_block.resnets.1", input, input, &mut add);
        for suffix in ["group_norm", "to_q", "to_k", "to_v", "to_out.0"] {
            let shape = if suffix == "group_norm" {
                vec![input]
            } else {
                vec![input, input]
            };
            add(
                format!("encoder.mid_block.attentions.0.{suffix}.weight"),
                shape,
            );
            add(
                format!("encoder.mid_block.attentions.0.{suffix}.bias"),
                vec![input],
            );
        }

        add("encoder.conv_norm_out.weight".into(), vec![input]);
        add("encoder.conv_norm_out.bias".into(), vec![input]);
        let moments = 2 * config.latent_channels;
        add("encoder.conv_out.weight".into(), vec![moments, input, 3, 3]);
        add("encoder.conv_out.bias".into(), vec![moments]);
        add("quant_conv.weight".into(), vec![moments, moments, 1, 1]);
        add("quant_conv.bias".into(), vec![moments]);
        if post_quant {
            let latent = config.latent_channels;
            add("post_quant_conv.weight".into(), vec![latent, latent, 1, 1]);
            add("post_quant_conv.bias".into(), vec![latent]);
        }

        crate::safetensors::write_checkpoint(path, &tensors);
    }

    pub(crate) fn tiny() -> VaeConfig {
        VaeConfig {
            latent_channels: 4,
            out_channels: 3,
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            norm_groups: 4,
            scaling_factor: 0.5,
            shift_factor: 0.25,
        }
    }

    #[test]
    fn a_decoder_runs_the_convolution_that_sits_in_front_of_its_stem() {
        // Reading `post_quant_conv` and not applying it changes no shape and
        // no edge: the picture comes back with the same composition and the
        // wrong colours, which is what every other test here would pass.
        let config = tiny();
        let latent = FeatureMap::from_vec(
            4,
            3,
            3,
            (0..36).map(|value| (value as f32 * 0.1).sin()).collect(),
        )
        .unwrap();

        let mut images = Vec::new();
        for (carries, name) in [(true, "with"), (false, "without")] {
            let path = std::env::temp_dir()
                .join(format!("rusting_brain_vae_post_quant_{name}.safetensors"));
            checkpoint_at(&config, &path, carries);
            let decoder = VaeDecoder::load(&path, "decoder.", config.clone()).unwrap();
            assert_eq!(decoder.post_quant.is_some(), carries);
            images.push(decoder.decode(&latent).unwrap());
        }

        let (with, without) = (&images[0], &images[1]);
        let apart = with
            .data
            .iter()
            .zip(&without.data)
            .map(|(with, without)| (with - without).abs())
            .fold(0.0f32, f32::max);
        assert!(
            apart > 1e-4,
            "the convolution in front of the stem changed nothing, so it was not applied"
        );
    }

    #[test]
    fn a_decoder_reads_a_checkpoint_and_turns_a_latent_into_an_image() {
        let config = tiny();
        let path = std::env::temp_dir().join("rusting_brain_vae_decode.safetensors");
        checkpoint(&config, &path);

        let decoder = VaeDecoder::load(&path, "decoder.", config.clone()).unwrap();
        assert_eq!(decoder.config().upscale(), 2);

        let latent = FeatureMap::from_vec(
            4,
            3,
            3,
            (0..36).map(|value| (value as f32 * 0.1).sin()).collect(),
        )
        .unwrap();
        let image = decoder.decode(&latent).unwrap();

        assert_eq!(image.channels, 3);
        assert_eq!((image.height, image.width), (6, 6));
        assert!(image.data.iter().all(|value| value.is_finite()));

        // A latent of the wrong depth is refused rather than reinterpreted.
        assert!(decoder.decode(&FeatureMap::new(3, 3, 3)).is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn an_encoder_turns_an_image_into_a_latent_the_decoder_accepts() {
        let config = tiny();
        let path = std::env::temp_dir().join("rusting_brain_vae_encode.safetensors");
        checkpoint(&config, &path);

        let encoder = VaeEncoder::load(&path, "encoder.", config.clone()).unwrap();
        let decoder = VaeDecoder::load(&path, "decoder.", config.clone()).unwrap();

        let image = FeatureMap::from_vec(
            3,
            6,
            6,
            (0..108).map(|value| (value as f32 * 0.07).cos()).collect(),
        )
        .unwrap();
        let latent = encoder.encode(&image).unwrap();

        // The ladder halves once, so the latent is half the image's size and
        // as deep as the configuration says.
        assert_eq!(latent.channels, config.latent_channels);
        assert_eq!((latent.height, latent.width), (3, 3));
        assert!(latent.data.iter().all(|value| value.is_finite()));

        // What comes out is what the decoder takes, at the size it started.
        let round_trip = decoder.decode(&latent).unwrap();
        assert_eq!((round_trip.height, round_trip.width), (6, 6));

        // An image of the wrong depth is refused rather than reinterpreted.
        assert!(encoder.encode(&FeatureMap::new(1, 6, 6)).is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_missing_layer_is_an_error_rather_than_a_silent_hole() {
        let config = tiny();
        let path = std::env::temp_dir().join("rusting_brain_vae_missing.safetensors");
        checkpoint(&config, &path);

        assert!(VaeDecoder::load(&path, "vae.decoder.", config.clone()).is_err());
        // The same file read with one rung too many looks for keys that are
        // not there.
        let mut deeper = config;
        deeper.block_out_channels.push(32);
        assert!(VaeDecoder::load(&path, "decoder.", deeper).is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn eight_bit_output_clamps_the_range_the_decoder_produces() {
        let image = FeatureMap::from_vec(3, 1, 2, vec![-1.0, 0.0, 1.0, 2.0, -5.0, 0.5]).unwrap();
        // Interleaved, so the first pixel takes one value from each channel.
        assert_eq!(to_rgb8(&image), vec![0, 255, 0, 128, 255, 191]);
    }

    #[test]
    fn the_flux_configuration_carries_its_own_latent_shape() {
        let flux = VaeConfig::flux();
        assert_eq!(flux.latent_channels, 16);
        assert_eq!(flux.upscale(), 8);
        assert!(flux.shift_factor > 0.0);
    }
}
