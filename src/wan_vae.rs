//! Wan 2.2's video autoencoder, `AutoencoderKLWan` in `diffusers`.
//!
//! The TI2V-5B checkpoint compresses a clip sixteen times on each side and four
//! times in time into 48 latent channels. Every layer is causal in time: a
//! three-frame convolution only sees the current frame and the two before it,
//! and `diffusers` runs the clip one chunk at a time with a cache of the frames
//! each convolution saw last. The first latent frame decodes to one video
//! frame, every later one to four, so `T` latent frames make `1 + 4 (T - 1)`.
//!
//! Frames are kept as one [`FeatureMap`] each. A three-frame convolution is one
//! two-dimensional convolution over the three frames stacked along the
//! channels, with a missing frame (before the start of the clip) as zeros.
//! Streaming the chunks through each layer with its own history is then the
//! same as running the layer over the whole sequence at once, which is what
//! the chunked loop of `diffusers` computes too.
//!
//! [`WanVaeDecoder`] and [`WanVaeEncoder`] read the `diffusers` checkpoint
//! (`vae/diffusion_pytorch_model.safetensors`) as stored, in F32.

use crate::conv::{Conv2d, Dense, FeatureMap, upsample_nearest};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::vae::{attend, pad_right_bottom, read_conv};

/// Frames of input a three-frame convolution keeps from one chunk to the next.
const CACHE_FRAMES: usize = 2;

/// The shape of a Wan autoencoder, from its `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct WanVaeConfig {
    /// Latent channels, 48 for TI2V-5B.
    pub z_dim: usize,
    /// The encoder's base width.
    pub base_dim: usize,
    /// The decoder's base width.
    pub decoder_base_dim: usize,
    pub dim_mult: Vec<usize>,
    pub num_res_blocks: usize,
    /// Which of the encoder's halvings also halve time. The decoder's
    /// doublings are the reverse.
    pub temperal_downsample: Vec<bool>,
    /// Pixels on each side folded into channels before the encoder and
    /// unfolded after the decoder.
    pub patch_size: usize,
    /// Channels after that folding: three colours times `patch_size^2`.
    pub in_channels: usize,
    pub out_channels: usize,
    /// Per-channel statistics a latent is normalised with.
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

impl WanVaeConfig {
    /// Reads `config.json`. Only the residual variant without attention at
    /// any scale (Wan 2.2's) is supported.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let text = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| NetworkError::InvalidDataset(format!("wan vae config: {error}")))?;
        let missing = |name: &str| NetworkError::InvalidConfig(format!("wan vae config: {name}"));
        let number = |name: &str| -> Result<usize, NetworkError> {
            json.get(name)
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
                .ok_or_else(|| missing(name))
        };
        let list = |name: &str| -> Result<&Vec<serde_json::Value>, NetworkError> {
            json.get(name)
                .and_then(|value| value.as_array())
                .ok_or_else(|| missing(name))
        };
        let floats = |name: &str| -> Result<Vec<f32>, NetworkError> {
            list(name)?
                .iter()
                .map(|value| value.as_f64().map(|value| value as f32))
                .collect::<Option<Vec<f32>>>()
                .ok_or_else(|| missing(name))
        };
        if json.get("is_residual").and_then(|value| value.as_bool()) != Some(true) {
            return Err(NetworkError::InvalidConfig(
                "only the residual Wan autoencoder (Wan 2.2) is supported".into(),
            ));
        }
        if list("attn_scales").is_ok_and(|scales| !scales.is_empty()) {
            return Err(NetworkError::InvalidConfig(
                "a Wan autoencoder with attention at some scale is not supported".into(),
            ));
        }
        let config = Self {
            z_dim: number("z_dim")?,
            base_dim: number("base_dim")?,
            decoder_base_dim: number("decoder_base_dim")?,
            dim_mult: list("dim_mult")?
                .iter()
                .map(|value| value.as_u64().map(|value| value as usize))
                .collect::<Option<Vec<usize>>>()
                .ok_or_else(|| missing("dim_mult"))?,
            num_res_blocks: number("num_res_blocks")?,
            temperal_downsample: list("temperal_downsample")?
                .iter()
                .map(|value| value.as_bool())
                .collect::<Option<Vec<bool>>>()
                .ok_or_else(|| missing("temperal_downsample"))?,
            patch_size: number("patch_size")?,
            in_channels: number("in_channels")?,
            out_channels: number("out_channels")?,
            latents_mean: floats("latents_mean")?,
            latents_std: floats("latents_std")?,
        };
        if config.dim_mult.is_empty()
            || config.temperal_downsample.len() + 1 != config.dim_mult.len()
            || config.latents_mean.len() != config.z_dim
            || config.latents_std.len() != config.z_dim
        {
            return Err(NetworkError::InvalidConfig(
                "wan vae config: inconsistent dim_mult, temperal_downsample or latent statistics"
                    .into(),
            ));
        }
        Ok(config)
    }

    /// `(latent - mean) / std` per channel, what the encoder's mean becomes
    /// before the denoiser sees it.
    pub fn normalize(&self, latent: &mut [FeatureMap]) {
        self.affine(latent, |value, mean, std| (value - mean) / std);
    }

    /// `latent * std + mean`, the inverse, before decoding.
    pub fn denormalize(&self, latent: &mut [FeatureMap]) {
        self.affine(latent, |value, mean, std| value * std + mean);
    }

    fn affine(&self, latent: &mut [FeatureMap], map: impl Fn(f32, f32, f32) -> f32) {
        for frame in latent {
            let pixels = frame.pixels();
            for (channel, plane) in frame.data.chunks_mut(pixels).enumerate() {
                let (mean, std) = (self.latents_mean[channel], self.latents_std[channel]);
                for value in plane {
                    *value = map(*value, mean, std);
                }
            }
        }
    }
}

/// Folds each `patch x patch` square of pixels into channels: channel
/// `c * p^2 + dx * p + dy` at `(y, x)` holds pixel `(y * p + dy, x * p + dx)`
/// of channel `c` (the column offset first, as `diffusers` orders it).
pub fn patchify(frame: &FeatureMap, patch: usize) -> Result<FeatureMap, NetworkError> {
    if !frame.height.is_multiple_of(patch) || !frame.width.is_multiple_of(patch) {
        return Err(NetworkError::InvalidConfig(format!(
            "a {}x{} frame does not split into {patch}x{patch} patches",
            frame.height, frame.width
        )));
    }
    let (height, width) = (frame.height / patch, frame.width / patch);
    let mut output = FeatureMap::new(frame.channels * patch * patch, height, width);
    for channel in 0..frame.channels {
        for dx in 0..patch {
            for dy in 0..patch {
                let to = (channel * patch * patch + dx * patch + dy) * height * width;
                for y in 0..height {
                    for x in 0..width {
                        output.data[to + y * width + x] =
                            frame.data[(channel * frame.height + y * patch + dy) * frame.width
                                + x * patch
                                + dx];
                    }
                }
            }
        }
    }
    Ok(output)
}

/// The inverse of [`patchify`].
pub fn unpatchify(frame: &FeatureMap, patch: usize) -> FeatureMap {
    let channels = frame.channels / (patch * patch);
    let (height, width) = (frame.height * patch, frame.width * patch);
    let mut output = FeatureMap::new(channels, height, width);
    let plane = frame.pixels();
    for channel in 0..channels {
        for dx in 0..patch {
            for dy in 0..patch {
                let from = (channel * patch * patch + dx * patch + dy) * plane;
                for y in 0..frame.height {
                    for x in 0..frame.width {
                        output.data[(channel * height + y * patch + dy) * width + x * patch + dx] =
                            frame.data[from + y * frame.width + x];
                    }
                }
            }
        }
    }
    output
}

/// One history per causal layer, taken in the order the layers run, the way
/// `diffusers` indexes its `feat_cache`. `None` marks a layer that has not
/// seen a chunk yet.
pub(crate) struct Streams<T> {
    slots: Vec<Option<Vec<T>>>,
    cursor: usize,
}

impl<T> Streams<T> {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            cursor: 0,
        }
    }

    /// Starts the next chunk at the first layer.
    pub(crate) fn rewind(&mut self) {
        self.cursor = 0;
    }

    /// The history of the next layer.
    pub(crate) fn next(&mut self) -> &mut Option<Vec<T>> {
        if self.cursor == self.slots.len() {
            self.slots.push(None);
        }
        self.cursor += 1;
        &mut self.slots[self.cursor - 1]
    }
}

/// A convolution over `taps` consecutive frames, causal in time.
///
/// The weight is the checkpoint's `[out, in, kt, kh, kw]` reordered to
/// `[out, kt, in, kh, kw]`, so the frames stacked oldest first along the
/// channels make one ordinary convolution.
#[derive(Clone, Debug)]
pub(crate) struct CausalConv3d {
    pub(crate) conv: Conv2d,
    pub(crate) taps: usize,
}

impl CausalConv3d {
    /// The output for every frame of `chunk`, given the frames this layer saw
    /// before it in `history`, which is then replaced by the last two frames
    /// of the stream.
    fn forward(
        &self,
        history: &mut Option<Vec<FeatureMap>>,
        chunk: Vec<FeatureMap>,
    ) -> Result<Vec<FeatureMap>, NetworkError> {
        let mut stream = history.take().unwrap_or_default();
        let past = stream.len();
        stream.extend(chunk);
        let outputs = (past..stream.len())
            .map(|end| self.conv.forward(&self.stack(&stream, end)))
            .collect::<Result<Vec<_>, _>>()?;
        let keep = stream.len().saturating_sub(CACHE_FRAMES);
        stream.drain(..keep);
        *history = Some(stream);
        Ok(outputs)
    }

    /// The `taps` frames ending at `stream[last]`, oldest first, along the
    /// channels; frames before the stream are zeros.
    fn stack(&self, stream: &[FeatureMap], last: usize) -> FeatureMap {
        let frame = &stream[last];
        let size = frame.data.len();
        let mut stacked = FeatureMap::new(self.conv.in_channels, frame.height, frame.width);
        for tap in 0..self.taps {
            let back = self.taps - 1 - tap;
            if back <= last {
                stacked.data[tap * size..(tap + 1) * size]
                    .copy_from_slice(&stream[last - back].data);
            }
        }
        stacked
    }
}

/// The RMS norm over channels at each pixel, `x / max(|x|, 1e-12) * sqrt(C) *
/// gamma`, with `sqrt(C)` already folded into `scale`. Optionally followed by
/// SiLU.
pub(crate) fn rms_norm(map: &mut FeatureMap, scale: &[f32], silu: bool) {
    let pixels = map.pixels();
    let mut norms = vec![0.0f32; pixels];
    for plane in map.data.chunks(pixels) {
        for (norm, value) in norms.iter_mut().zip(plane) {
            *norm += value * value;
        }
    }
    for norm in &mut norms {
        *norm = norm.sqrt().max(1e-12).recip();
    }
    for (plane, gamma) in map.data.chunks_mut(pixels).zip(scale) {
        for (value, norm) in plane.iter_mut().zip(&norms) {
            *value *= norm * gamma;
            if silu {
                *value = crate::ffn::silu(*value);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct WanResnet {
    pub(crate) norm1: Vec<f32>,
    pub(crate) conv1: CausalConv3d,
    pub(crate) norm2: Vec<f32>,
    pub(crate) conv2: CausalConv3d,
    /// A one-by-one convolution when the block changes the width.
    pub(crate) shortcut: Option<Conv2d>,
}

impl WanResnet {
    fn forward(
        &self,
        streams: &mut Streams<FeatureMap>,
        input: Vec<FeatureMap>,
    ) -> Result<Vec<FeatureMap>, NetworkError> {
        let skip = match &self.shortcut {
            Some(conv) => input
                .iter()
                .map(|frame| conv.forward(frame))
                .collect::<Result<Vec<_>, _>>()?,
            None => input.clone(),
        };
        let mut hidden = input;
        for frame in &mut hidden {
            rms_norm(frame, &self.norm1, true);
        }
        hidden = self.conv1.forward(streams.next(), hidden)?;
        for frame in &mut hidden {
            rms_norm(frame, &self.norm2, true);
        }
        hidden = self.conv2.forward(streams.next(), hidden)?;
        for (frame, skip) in hidden.iter_mut().zip(&skip) {
            for (value, skip) in frame.data.iter_mut().zip(&skip.data) {
                *value += skip;
            }
        }
        Ok(hidden)
    }
}

/// Single-head self-attention within each frame.
#[derive(Clone, Debug)]
pub(crate) struct WanAttention {
    pub(crate) norm: Vec<f32>,
    pub(crate) query: Dense,
    pub(crate) key: Dense,
    pub(crate) value: Dense,
    pub(crate) proj: Dense,
}

impl WanAttention {
    fn forward(&self, frame: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        let mut normed = frame.clone();
        rms_norm(&mut normed, &self.norm, false);
        let tokens = normed.to_tokens();
        let attended = attend(
            &self.query.forward(&tokens)?,
            &self.key.forward(&tokens)?,
            &self.value.forward(&tokens)?,
        );
        let mut output =
            FeatureMap::from_tokens(&self.proj.forward(&attended)?, frame.height, frame.width)?;
        for (value, skip) in output.data.iter_mut().zip(&frame.data) {
            *value += skip;
        }
        Ok(output)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct WanMid {
    pub(crate) first: WanResnet,
    pub(crate) attention: WanAttention,
    pub(crate) last: WanResnet,
}

impl WanMid {
    fn forward(
        &self,
        streams: &mut Streams<FeatureMap>,
        input: Vec<FeatureMap>,
    ) -> Result<Vec<FeatureMap>, NetworkError> {
        let hidden = self.first.forward(streams, input)?;
        let hidden = hidden
            .iter()
            .map(|frame| self.attention.forward(frame))
            .collect::<Result<Vec<_>, _>>()?;
        self.last.forward(streams, hidden)
    }
}

/// The channel-to-space shortcut of an up block (`DupUp3D`): each input
/// channel repeated, then the channels unfolded into `time x side x side`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DupUp {
    pub(crate) out_channels: usize,
    pub(crate) time: usize,
    pub(crate) side: usize,
}

impl DupUp {
    /// `first` drops the leading `time - 1` frames, as the first chunk does.
    fn forward(&self, input: &[FeatureMap], first: bool) -> Vec<FeatureMap> {
        let (fs, factor) = (self.side, self.time * self.side * self.side);
        let mut frames = Vec::with_capacity(input.len() * self.time);
        for frame in input {
            let repeats = self.out_channels * factor / frame.channels;
            let (height, width) = (frame.height * fs, frame.width * fs);
            for dt in 0..self.time {
                let mut output = FeatureMap::new(self.out_channels, height, width);
                for out in 0..self.out_channels {
                    for dy in 0..fs {
                        for dx in 0..fs {
                            let source = (out * factor + dt * fs * fs + dy * fs + dx) / repeats;
                            let plane = frame.plane(source);
                            for y in 0..frame.height {
                                for x in 0..frame.width {
                                    output.data
                                        [(out * height + y * fs + dy) * width + x * fs + dx] =
                                        plane[y * frame.width + x];
                                }
                            }
                        }
                    }
                }
                frames.push(output);
            }
        }
        if first {
            frames.drain(..self.time - 1);
        }
        frames
    }
}

/// The space-to-channel shortcut of a down block (`AvgDown3D`): `time x side
/// x side` folded into channels, then neighbouring channels averaged. The
/// clip is padded at the front with zero frames to a multiple of `time`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AvgDown {
    pub(crate) out_channels: usize,
    pub(crate) time: usize,
    pub(crate) side: usize,
}

impl AvgDown {
    fn forward(&self, input: &[FeatureMap]) -> Vec<FeatureMap> {
        let (fs, factor) = (self.side, self.time * self.side * self.side);
        let pad = (self.time - input.len() % self.time) % self.time;
        let first = &input[0];
        let (channels, height, width) = (first.channels, first.height / fs, first.width / fs);
        let group = channels * factor / self.out_channels;
        let groups = (input.len() + pad) / self.time;
        (0..groups)
            .map(|index| {
                let mut output = FeatureMap::new(self.out_channels, height, width);
                for j in 0..channels * factor {
                    let (channel, rest) = (j / factor, j % factor);
                    let (dt, dy, dx) = (rest / (fs * fs), rest / fs % fs, rest % fs);
                    // Frames before the clip are the zeros of the padding.
                    let Some(source) = (index * self.time + dt).checked_sub(pad) else {
                        continue;
                    };
                    let source = &input[source];
                    let to = j / group * height * width;
                    for y in 0..height {
                        for x in 0..width {
                            output.data[to + y * width + x] +=
                                source.data[(channel * source.height + y * fs + dy) * source.width
                                    + x * fs
                                    + dx];
                        }
                    }
                }
                let share = (group as f32).recip();
                for value in &mut output.data {
                    *value *= share;
                }
                output
            })
            .collect()
    }
}

/// A decoder doubling: optionally twice the frames first (the
/// `upsample3d` mode), then twice the side and a 3x3 convolution.
#[derive(Clone, Debug)]
pub(crate) struct WanUpsample {
    /// A three-frame convolution to twice the channels, the two halves being
    /// the two frames each input frame becomes.
    pub(crate) time: Option<CausalConv3d>,
    pub(crate) conv: Conv2d,
}

impl WanUpsample {
    fn forward(
        &self,
        streams: &mut Streams<FeatureMap>,
        mut input: Vec<FeatureMap>,
    ) -> Result<Vec<FeatureMap>, NetworkError> {
        if let Some(time) = &self.time {
            let history = streams.next();
            if history.is_none() {
                // The first chunk keeps its frame count (`diffusers` marks the
                // cache "Rep"); the next starts the time convolution from zeros.
                *history = Some(Vec::new());
            } else {
                input = time
                    .forward(history, input)?
                    .into_iter()
                    .flat_map(|frame| {
                        let half = frame.data.len() / 2;
                        let (channels, height, width) =
                            (frame.channels / 2, frame.height, frame.width);
                        let (early, late) = frame.data.split_at(half);
                        [early.to_vec(), late.to_vec()].map(|data| FeatureMap {
                            channels,
                            height,
                            width,
                            data,
                        })
                    })
                    .collect();
            }
        }
        input
            .iter()
            .map(|frame| self.conv.forward(&upsample_nearest(frame, 2)))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct WanUpBlock {
    pub(crate) resnets: Vec<WanResnet>,
    pub(crate) upsample: Option<WanUpsample>,
    pub(crate) shortcut: Option<DupUp>,
}

/// An encoder halving: a 3x3 stride-2 convolution after padding the right
/// and bottom by one, then optionally a stride-2 three-frame convolution
/// over the last frame of the previous chunk and this chunk.
#[derive(Clone, Debug)]
pub(crate) struct WanDownsample {
    pub(crate) conv: Conv2d,
    pub(crate) time: Option<CausalConv3d>,
}

impl WanDownsample {
    fn forward(
        &self,
        streams: &mut Streams<FeatureMap>,
        input: Vec<FeatureMap>,
    ) -> Result<Vec<FeatureMap>, NetworkError> {
        let mut frames = input
            .iter()
            .map(|frame| self.conv.forward(&pad_right_bottom(frame)))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(time) = &self.time {
            let history = streams.next();
            let last = frames.last().cloned().into_iter().collect();
            if let Some(previous) = history.replace(last) {
                // Not the first chunk, which passes through unchanged.
                let mut stream: Vec<FeatureMap> = previous.into_iter().collect();
                stream.extend(frames);
                frames = (time.taps - 1..stream.len())
                    .step_by(2)
                    .map(|end| time.conv.forward(&time.stack(&stream, end)))
                    .collect::<Result<Vec<_>, _>>()?;
            }
        }
        Ok(frames)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct WanDownBlock {
    pub(crate) resnets: Vec<WanResnet>,
    pub(crate) downsample: Option<WanDownsample>,
    pub(crate) shortcut: AvgDown,
}

/// The decoder of a Wan 2.2 autoencoder, `post_quant_conv` included.
#[derive(Clone, Debug)]
pub struct WanVaeDecoder {
    pub config: WanVaeConfig,
    pub(crate) post_quant: Conv2d,
    pub(crate) conv_in: CausalConv3d,
    pub(crate) mid: WanMid,
    pub(crate) up_blocks: Vec<WanUpBlock>,
    pub(crate) norm_out: Vec<f32>,
    pub(crate) conv_out: CausalConv3d,
}

impl WanVaeDecoder {
    /// Reads the decoder from `vae/diffusion_pytorch_model.safetensors`.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        config: WanVaeConfig,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        let base = config.decoder_base_dim;
        let deepest = base * config.dim_mult[config.dim_mult.len() - 1];
        let mut dims = vec![deepest];
        dims.extend(config.dim_mult.iter().rev().map(|mult| base * mult));
        let upsample_time: Vec<bool> = config.temperal_downsample.iter().rev().copied().collect();

        let post_quant = read_conv3d(&mut file, "post_quant_conv", 1, 0)?.conv;
        let conv_in = read_conv3d(&mut file, "decoder.conv_in", 1, 1)?;
        let mid = read_mid(&mut file, "decoder.mid_block")?;
        let mut up_blocks = Vec::new();
        for (index, pair) in dims.windows(2).enumerate() {
            let (in_dim, out_dim) = (pair[0], pair[1]);
            let prefix = format!("decoder.up_blocks.{index}");
            let resnets = (0..=config.num_res_blocks)
                .map(|block| read_resnet(&mut file, &format!("{prefix}.resnets.{block}")))
                .collect::<Result<Vec<_>, _>>()?;
            let up = index + 1 < config.dim_mult.len();
            let (upsample, shortcut) = if up {
                let time = upsample_time[index];
                let upsample = WanUpsample {
                    time: time
                        .then(|| {
                            read_conv3d(&mut file, &format!("{prefix}.upsampler.time_conv"), 1, 0)
                        })
                        .transpose()?,
                    conv: read_conv(&mut file, &format!("{prefix}.upsampler.resample.1"), 1, 1)?,
                };
                let shortcut = DupUp {
                    out_channels: out_dim,
                    time: if time { 2 } else { 1 },
                    side: 2,
                };
                if !(out_dim * shortcut.time * 4).is_multiple_of(in_dim) {
                    return Err(NetworkError::InvalidConfig(format!(
                        "{prefix}: {in_dim} channels do not repeat into {out_dim}"
                    )));
                }
                (Some(upsample), Some(shortcut))
            } else {
                (None, None)
            };
            up_blocks.push(WanUpBlock {
                resnets,
                upsample,
                shortcut,
            });
        }
        Ok(Self {
            post_quant,
            conv_in,
            mid,
            up_blocks,
            norm_out: read_rms(&mut file, "decoder.norm_out")?,
            conv_out: read_conv3d(&mut file, "decoder.conv_out", 1, 1)?,
            config,
        })
    }

    /// Decodes a latent (already denormalised, see
    /// [`WanVaeConfig::denormalize`]) given as one `[z_dim, h, w]` map per
    /// latent frame, into `1 + 4 (T - 1)` RGB frames in `[-1, 1]`.
    pub fn decode(&self, latent: &[FeatureMap]) -> Result<Vec<FeatureMap>, NetworkError> {
        if latent.is_empty()
            || latent
                .iter()
                .any(|frame| frame.channels != self.config.z_dim)
        {
            return Err(NetworkError::InvalidConfig(format!(
                "a Wan latent is a non-empty list of {}-channel frames",
                self.config.z_dim
            )));
        }
        let mut streams = Streams::new();
        let mut video = Vec::new();
        for (index, frame) in latent.iter().enumerate() {
            streams.rewind();
            let chunk = vec![self.post_quant.forward(frame)?];
            for frame in self.decode_chunk(&mut streams, chunk, index == 0)? {
                let mut frame = unpatchify(&frame, self.config.patch_size);
                for value in &mut frame.data {
                    *value = value.clamp(-1.0, 1.0);
                }
                video.push(frame);
            }
        }
        Ok(video)
    }

    fn decode_chunk(
        &self,
        streams: &mut Streams<FeatureMap>,
        chunk: Vec<FeatureMap>,
        first: bool,
    ) -> Result<Vec<FeatureMap>, NetworkError> {
        let mut hidden = self.conv_in.forward(streams.next(), chunk)?;
        hidden = self.mid.forward(streams, hidden)?;
        for block in &self.up_blocks {
            let skip = block
                .shortcut
                .map(|shortcut| shortcut.forward(&hidden, first));
            for resnet in &block.resnets {
                hidden = resnet.forward(streams, hidden)?;
            }
            if let Some(upsample) = &block.upsample {
                hidden = upsample.forward(streams, hidden)?;
            }
            if let Some(skip) = skip {
                for (frame, skip) in hidden.iter_mut().zip(&skip) {
                    for (value, skip) in frame.data.iter_mut().zip(&skip.data) {
                        *value += skip;
                    }
                }
            }
        }
        for frame in &mut hidden {
            rms_norm(frame, &self.norm_out, true);
        }
        self.conv_out.forward(streams.next(), hidden)
    }
}

/// The encoder of a Wan 2.2 autoencoder, `quant_conv` included.
#[derive(Clone, Debug)]
pub struct WanVaeEncoder {
    pub config: WanVaeConfig,
    pub(crate) conv_in: CausalConv3d,
    pub(crate) down_blocks: Vec<WanDownBlock>,
    pub(crate) mid: WanMid,
    pub(crate) norm_out: Vec<f32>,
    pub(crate) conv_out: CausalConv3d,
    pub(crate) quant: Conv2d,
}

impl WanVaeEncoder {
    /// Reads the encoder from `vae/diffusion_pytorch_model.safetensors`.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        config: WanVaeConfig,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        let base = config.base_dim;
        let mut dims = vec![base];
        dims.extend(config.dim_mult.iter().map(|mult| base * mult));
        let mut down_blocks = Vec::new();
        for (index, pair) in dims.windows(2).enumerate() {
            let (in_dim, out_dim) = (pair[0], pair[1]);
            let prefix = format!("encoder.down_blocks.{index}");
            let resnets = (0..config.num_res_blocks)
                .map(|block| read_resnet(&mut file, &format!("{prefix}.resnets.{block}")))
                .collect::<Result<Vec<_>, _>>()?;
            let down = index + 1 < config.dim_mult.len();
            let time = down && config.temperal_downsample[index];
            let downsample = down
                .then(|| -> Result<WanDownsample, NetworkError> {
                    Ok(WanDownsample {
                        conv: read_conv(
                            &mut file,
                            &format!("{prefix}.downsampler.resample.1"),
                            2,
                            0,
                        )?,
                        time: time
                            .then(|| {
                                read_conv3d(
                                    &mut file,
                                    &format!("{prefix}.downsampler.time_conv"),
                                    1,
                                    0,
                                )
                            })
                            .transpose()?,
                    })
                })
                .transpose()?;
            let shortcut = AvgDown {
                out_channels: out_dim,
                time: if time { 2 } else { 1 },
                side: if down { 2 } else { 1 },
            };
            if !(in_dim * shortcut.time * shortcut.side * shortcut.side).is_multiple_of(out_dim) {
                return Err(NetworkError::InvalidConfig(format!(
                    "{prefix}: {in_dim} channels do not average into {out_dim}"
                )));
            }
            down_blocks.push(WanDownBlock {
                resnets,
                downsample,
                shortcut,
            });
        }
        Ok(Self {
            conv_in: read_conv3d(&mut file, "encoder.conv_in", 1, 1)?,
            down_blocks,
            mid: read_mid(&mut file, "encoder.mid_block")?,
            norm_out: read_rms(&mut file, "encoder.norm_out")?,
            conv_out: read_conv3d(&mut file, "encoder.conv_out", 1, 1)?,
            quant: read_conv3d(&mut file, "quant_conv", 1, 0)?.conv,
            config,
        })
    }

    /// The latent distribution's mean (`latent_dist.mode()`), not normalised,
    /// of `1 + 4k` RGB frames in `[-1, 1]` whose sides are multiples of 16:
    /// one `[z_dim, h / 16, w / 16]` map per latent frame.
    pub fn encode_mean(&self, clip: &[FeatureMap]) -> Result<Vec<FeatureMap>, NetworkError> {
        if clip.is_empty() || !(clip.len() - 1).is_multiple_of(4) {
            return Err(NetworkError::InvalidConfig(format!(
                "a Wan clip has 1 + 4k frames, not {}",
                clip.len()
            )));
        }
        let patched = clip
            .iter()
            .map(|frame| patchify(frame, self.config.patch_size))
            .collect::<Result<Vec<_>, _>>()?;
        if patched[0].channels != self.config.in_channels {
            return Err(NetworkError::InvalidTarget {
                expected: self.config.in_channels,
                actual: patched[0].channels,
            });
        }
        let mut streams = Streams::new();
        let mut latent = Vec::new();
        let chunks = std::iter::once(&patched[..1]).chain(patched[1..].chunks(4));
        for chunk in chunks {
            streams.rewind();
            for frame in self.encode_chunk(&mut streams, chunk.to_vec())? {
                let moments = self.quant.forward(&frame)?;
                let size = self.config.z_dim * moments.pixels();
                latent.push(FeatureMap::from_vec(
                    self.config.z_dim,
                    moments.height,
                    moments.width,
                    moments.data[..size].to_vec(),
                )?);
            }
        }
        Ok(latent)
    }

    /// The mean, normalised the way the denoiser expects.
    pub fn encode(&self, clip: &[FeatureMap]) -> Result<Vec<FeatureMap>, NetworkError> {
        let mut latent = self.encode_mean(clip)?;
        self.config.normalize(&mut latent);
        Ok(latent)
    }

    fn encode_chunk(
        &self,
        streams: &mut Streams<FeatureMap>,
        chunk: Vec<FeatureMap>,
    ) -> Result<Vec<FeatureMap>, NetworkError> {
        let mut hidden = self.conv_in.forward(streams.next(), chunk)?;
        for block in &self.down_blocks {
            let skip = block.shortcut.forward(&hidden);
            for resnet in &block.resnets {
                hidden = resnet.forward(streams, hidden)?;
            }
            if let Some(downsample) = &block.downsample {
                hidden = downsample.forward(streams, hidden)?;
            }
            for (frame, skip) in hidden.iter_mut().zip(&skip) {
                for (value, skip) in frame.data.iter_mut().zip(&skip.data) {
                    *value += skip;
                }
            }
        }
        hidden = self.mid.forward(streams, hidden)?;
        for frame in &mut hidden {
            rms_norm(frame, &self.norm_out, true);
        }
        self.conv_out.forward(streams.next(), hidden)
    }
}

/// A `[out, in, kt, kh, kw]` convolution, reordered for [`CausalConv3d`].
/// `padding` is the spatial padding; time is always padded causally.
fn read_conv3d(
    file: &mut ShardedSafeTensors,
    name: &str,
    stride: usize,
    padding: usize,
) -> Result<CausalConv3d, NetworkError> {
    let (values, shape) = file.tensor(&format!("{name}.weight"))?;
    if shape.len() != 5 || shape[3] != shape[4] {
        return Err(NetworkError::InvalidConfig(format!(
            "{name}.weight is {shape:?}, not a 3D convolution with a square kernel"
        )));
    }
    let (out, input, taps, kernel) = (shape[0], shape[1], shape[2], shape[3]);
    let area = kernel * kernel;
    let mut weight = vec![0.0; values.len()];
    for o in 0..out {
        for i in 0..input {
            for t in 0..taps {
                let from = ((o * input + i) * taps + t) * area;
                let to = ((o * taps + t) * input + i) * area;
                weight[to..to + area].copy_from_slice(&values[from..from + area]);
            }
        }
    }
    let bias = file
        .tensor(&format!("{name}.bias"))
        .ok()
        .map(|(bias, _)| bias);
    Ok(CausalConv3d {
        conv: Conv2d::new(
            Matrix::from_vec(out, taps * input * area, weight),
            bias,
            taps * input,
            kernel,
            stride,
            if kernel == 1 { 0 } else { padding },
        )?,
        taps,
    })
}

/// An RMS norm's gamma times `sqrt(C)`.
fn read_rms(file: &mut ShardedSafeTensors, name: &str) -> Result<Vec<f32>, NetworkError> {
    let (gamma, _) = file.tensor(&format!("{name}.gamma"))?;
    let root = (gamma.len() as f32).sqrt();
    Ok(gamma.into_iter().map(|value| value * root).collect())
}

fn read_resnet(file: &mut ShardedSafeTensors, name: &str) -> Result<WanResnet, NetworkError> {
    let key = format!("{name}.conv_shortcut.weight");
    let shortcut = if file.names().any(|name| name == key) {
        Some(read_conv3d(file, &format!("{name}.conv_shortcut"), 1, 0)?.conv)
    } else {
        None
    };
    Ok(WanResnet {
        norm1: read_rms(file, &format!("{name}.norm1"))?,
        conv1: read_conv3d(file, &format!("{name}.conv1"), 1, 1)?,
        norm2: read_rms(file, &format!("{name}.norm2"))?,
        conv2: read_conv3d(file, &format!("{name}.conv2"), 1, 1)?,
        shortcut,
    })
}

fn read_mid(file: &mut ShardedSafeTensors, name: &str) -> Result<WanMid, NetworkError> {
    let first = read_resnet(file, &format!("{name}.resnets.0"))?;
    let prefix = format!("{name}.attentions.0");
    let (qkv, shape) = file.tensor(&format!("{prefix}.to_qkv.weight"))?;
    let (qkv_bias, _) = file.tensor(&format!("{prefix}.to_qkv.bias"))?;
    let width = shape[1];
    let part = |index: usize| {
        Dense::new(
            Matrix::from_vec(
                width,
                width,
                qkv[index * width * width..(index + 1) * width * width].to_vec(),
            ),
            Some(qkv_bias[index * width..(index + 1) * width].to_vec()),
        )
    };
    let (proj, _) = file.tensor(&format!("{prefix}.proj.weight"))?;
    let (proj_bias, _) = file.tensor(&format!("{prefix}.proj.bias"))?;
    let attention = WanAttention {
        norm: read_rms(file, &format!("{prefix}.norm"))?,
        query: part(0)?,
        key: part(1)?,
        value: part(2)?,
        proj: Dense::new(Matrix::from_vec(width, width, proj), Some(proj_bias))?,
    };
    Ok(WanMid {
        first,
        attention,
        last: read_resnet(file, &format!("{name}.resnets.1"))?,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn map(channels: usize, height: usize, width: usize, seed: u32) -> FeatureMap {
        let data = (0..channels * height * width)
            .map(|index| {
                ((index as u32).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32 / 500.0
                    - 1.0
            })
            .collect();
        FeatureMap::from_vec(channels, height, width, data).unwrap()
    }

    fn causal(input: usize, out: usize, taps: usize, seed: u32) -> CausalConv3d {
        let weights = map(out, input * taps, 9, seed).data;
        CausalConv3d {
            conv: Conv2d::new(
                Matrix::from_vec(out, taps * input * 9, weights),
                Some(map(out, 1, 1, seed + 1).data),
                taps * input,
                3,
                1,
                1,
            )
            .unwrap(),
            taps,
        }
    }

    /// Streaming a clip in chunks of one, then four, gives the outputs of one
    /// pass over the whole clip.
    #[test]
    fn chunked_causal_convolution_matches_one_pass() {
        let conv = causal(3, 2, 3, 7);
        let clip: Vec<FeatureMap> = (0..9).map(|frame| map(3, 4, 5, frame)).collect();
        let whole = conv.forward(&mut None, clip.clone()).unwrap();
        let mut history = None;
        let mut chunked = conv.forward(&mut history, clip[..1].to_vec()).unwrap();
        chunked.extend(conv.forward(&mut history, clip[1..5].to_vec()).unwrap());
        chunked.extend(conv.forward(&mut history, clip[5..].to_vec()).unwrap());
        assert_eq!(whole.len(), 9);
        for (a, b) in whole.iter().zip(&chunked) {
            assert_eq!(a.data, b.data);
        }
        // Frame 0 sees only itself: the last tap alone, the others are zeros.
        let last_tap = Conv2d::new(
            Matrix::from_vec(
                2,
                27,
                (0..2)
                    .flat_map(|out| conv.conv.weight.row(out)[54..].to_vec())
                    .collect(),
            ),
            conv.conv.bias.clone(),
            3,
            3,
            1,
            1,
        )
        .unwrap();
        assert_eq!(whole[0].data, last_tap.forward(&clip[0]).unwrap().data);
    }

    #[test]
    fn patchify_round_trips_with_the_column_offset_first() {
        let frame = map(3, 4, 6, 3);
        let patched = patchify(&frame, 2).unwrap();
        assert_eq!(
            (patched.channels, patched.height, patched.width),
            (12, 2, 3)
        );
        // Channel c * 4 + dx * 2 + dy at (y, x) is pixel (2y + dy, 2x + dx).
        let (c, dx, dy, y, x) = (2, 1, 0, 1, 2);
        assert_eq!(
            patched.data[((c * 4 + dx * 2 + dy) * 2 + y) * 3 + x],
            frame.data[(c * 4 + 2 * y + dy) * 6 + 2 * x + dx]
        );
        assert_eq!(unpatchify(&patched, 2).data, frame.data);
    }

    /// `DupUp3D` repeats channels then unfolds them; `AvgDown3D` folds then
    /// averages. With one input channel per group the second undoes the
    /// first, and the first chunk's leading frame is dropped.
    #[test]
    fn dup_up_and_avg_down_follow_the_diffusers_layout() {
        let frames = vec![map(4, 2, 3, 1), map(4, 2, 3, 2)];
        let up = DupUp {
            out_channels: 4,
            time: 2,
            side: 2,
        };
        let expanded = up.forward(&frames, false);
        assert_eq!(expanded.len(), 4);
        // out[o, 2t + dt, 2y + dy, 2x + dx] = in[(o * 8 + dt * 4 + dy * 2 + dx) / 8].
        let (o, dt, dy, dx, y, x) = (3, 1, 1, 0, 1, 2);
        assert_eq!(
            expanded[2 + dt].data[(o * 4 + 2 * y + dy) * 6 + 2 * x + dx],
            frames[1].data[(o * 2 + y) * 3 + x]
        );
        assert_eq!(up.forward(&frames, true).len(), 3);
        let down = AvgDown {
            out_channels: 4,
            time: 2,
            side: 2,
        };
        let back = down.forward(&expanded);
        assert_eq!(back.len(), 2);
        for (a, b) in back.iter().zip(&frames) {
            for (a, b) in a.data.iter().zip(&b.data) {
                assert!((a - b).abs() < 1e-6);
            }
        }
        // A single frame is padded at the front by a zero frame, so the
        // channels its dt = 0 half feeds average in zeros.
        let one = down.forward(&expanded[..1]);
        assert_eq!(one.len(), 1);
        for (a, b) in one[0].data.iter().zip(&frames[0].data) {
            assert!((a - b / 2.0).abs() < 1e-6);
        }
    }

    /// Relative L2 distance of `got` from `want`.
    pub(crate) fn relative(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let (gap, norm) = got
            .iter()
            .zip(want)
            .fold((0.0f64, 0.0f64), |(g, n), (a, b)| {
                (g + ((a - b) as f64).powi(2), n + (*b as f64).powi(2))
            });
        (gap / norm).sqrt() as f32
    }

    /// A `[C, T, H, W]` reference tensor as one map per frame.
    pub(crate) fn frames(data: &[f32], shape: &[usize]) -> Vec<FeatureMap> {
        let (channels, time, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        let plane = height * width;
        (0..time)
            .map(|t| {
                let mut frame = FeatureMap::new(channels, height, width);
                for c in 0..channels {
                    let from = (c * time + t) * plane;
                    frame.data[c * plane..(c + 1) * plane]
                        .copy_from_slice(&data[from..from + plane]);
                }
                frame
            })
            .collect()
    }

    /// One map per frame back to `[C, T, H, W]`.
    pub(crate) fn stack(frames: &[FeatureMap]) -> Vec<f32> {
        let (channels, plane) = (frames[0].channels, frames[0].pixels());
        let mut data = vec![0.0; channels * frames.len() * plane];
        for (t, frame) in frames.iter().enumerate() {
            for c in 0..channels {
                let to = (c * frames.len() + t) * plane;
                data[to..to + plane].copy_from_slice(frame.plane(c));
            }
        }
        data
    }

    /// Wan 2.2's autoencoder against diffusers' `AutoencoderKLWan` in F32 on
    /// the output of RustingFrame's `tools/wan_refs.py w2`: a two-frame latent
    /// decoded to five 128x128 frames, a five-frame 64x64 clip and its first
    /// frame encoded. `RB_WAN_REF` is the directory holding `w2.safetensors`
    /// and `reference.json`, `RB_WAN_MODEL` the model (default the local
    /// copy), and `RB_WAN_CUDA=1` checks the device decoder as well. About
    /// 3 GB of host memory.
    #[test]
    #[ignore = "needs Wan2.2-TI2V-5B and the output of RustingFrame's tools/wan_refs.py w2"]
    fn wan22_vae_matches_diffusers() {
        let env = |name: &str| std::env::var(name).ok();
        let dir = std::path::PathBuf::from(env("RB_WAN_REF").expect("RB_WAN_REF"));
        let model = std::path::PathBuf::from(
            env("RB_WAN_MODEL")
                .unwrap_or_else(|| "/home/vasylt/Code/ai/ml/models/Wan2.2-TI2V-5B".into()),
        )
        .join("vae");
        let mut reference =
            crate::safetensors::SafeTensors::open(dir.join("w2.safetensors")).unwrap();
        let mut read = |name: &str| reference.tensor(name).unwrap();
        let mut report = Vec::new();
        let mut gate = |stage: &str, error: f32, limit: f32| {
            eprintln!("wan vae {stage}: relative L2 {error:.3e} (gate {limit:.0e})");
            report.push((stage.to_string(), error, limit));
        };

        let config = WanVaeConfig::from_file(model.join("config.json")).unwrap();
        let weights = model.join("diffusion_pytorch_model.safetensors");

        let (latent, shape) = read("dec.latent");
        let mut latent = frames(&latent, &shape);
        config.denormalize(&mut latent);
        let (denorm, _) = read("dec.latent_denorm");
        gate(
            "latent denormalise",
            relative(&stack(&latent), &denorm),
            1e-6,
        );
        let (video, _) = read("dec.video");

        {
            let started = std::time::Instant::now();
            let decoder = WanVaeDecoder::load(&weights, config.clone()).unwrap();
            eprintln!("wan vae decoder load: {:?}", started.elapsed());
            let started = std::time::Instant::now();
            let decoded = decoder.decode(&latent).unwrap();
            eprintln!("wan vae host decode: {:?}", started.elapsed());
            assert_eq!(decoded.len(), 5);
            gate("host decode", relative(&stack(&decoded), &video), 1e-5);
        }

        let encoder = WanVaeEncoder::load(&weights, config.clone()).unwrap();
        for name in ["enc", "enc1"] {
            let (clip, shape) = read(&format!("{name}.clip"));
            let clip = frames(&clip, &shape);
            let started = std::time::Instant::now();
            let mean = encoder.encode_mean(&clip).unwrap();
            eprintln!("wan vae host {name}: {:?}", started.elapsed());
            gate(
                &format!("host {name} mean"),
                relative(&stack(&mean), &read(&format!("{name}.mean")).0),
                1e-5,
            );
            let mut normed = mean;
            config.normalize(&mut normed);
            gate(
                &format!("host {name} mean_norm"),
                relative(&stack(&normed), &read(&format!("{name}.mean_norm")).0),
                1e-5,
            );
        }

        let failures: Vec<_> = report
            .iter()
            .filter(|(_, error, limit)| error.is_nan() || error > limit)
            .collect();
        assert!(failures.is_empty(), "over the gate: {failures:?}");
    }
}
