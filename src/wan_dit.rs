//! Wan 2.2's video transformer, `WanTransformer3DModel` in `diffusers`.
//!
//! The TI2V-5B checkpoint: 30 blocks of 3072 wide, 24 heads of 128, over
//! patches of one latent frame by two by two. Every block is self-attention
//! with a three-axis rotary position (frame, row, column), cross-attention to
//! the umT5 prompt states, and a tanh-GELU feed-forward; the self-attention
//! and the feed-forward sit behind adaptive layer norms whose shift, scale and
//! gate come from the timestep. Queries and keys are RMS-normed across all
//! heads at once, not per head as FLUX does.
//!
//! TI2V conditions on the first frame by giving its tokens timestep 0 and the
//! rest the step's timestep, so the timestep is one per token frame here
//! rather than one per call: `diffusers` takes one per token, but its mask is
//! always whole frames.
//!
//! [`WanDit`] reads the `diffusers` checkpoint (`transformer/`, five F32
//! shards) and checks every tensor's shape against the configuration before
//! keeping it. At F32 the model is about 20 GB, so [`WanDit::read_head`] and
//! [`WanBlock::read`] let a caller hold one block at a time.

use crate::conv::{Dense, FeatureMap};
use crate::ffn::{gelu, silu};
use crate::matrix::Matrix;
use crate::mmdit::{Positions, attention, layer_norm, sinusoid};
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::text_encoder::rms_norm;
use crate::transformer::Precision;
use rayon::prelude::*;

/// The shape of a Wan transformer, from its `config.json`.
#[derive(Clone, Debug, PartialEq)]
pub struct WanDitConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub ffn_dim: usize,
    pub freq_dim: usize,
    pub text_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    /// Frames, rows and columns per patch.
    pub patch: [usize; 3],
    pub eps: f32,
    /// The longest axis the rotary tables cover, in patches.
    pub rope_max_seq_len: usize,
}

impl Default for WanDitConfig {
    /// TI2V-5B.
    fn default() -> Self {
        Self {
            num_heads: 24,
            head_dim: 128,
            layers: 30,
            ffn_dim: 14336,
            freq_dim: 256,
            text_dim: 4096,
            in_channels: 48,
            out_channels: 48,
            patch: [1, 2, 2],
            eps: 1e-6,
            rope_max_seq_len: 1024,
        }
    }
}

impl WanDitConfig {
    /// Reads a `diffusers` `config.json`, refusing the variants this model
    /// does not implement (image conditioning, per-head or no query norm).
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let text = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| NetworkError::InvalidConfig(format!("wan config.json: {error}")))?;
        let refused =
            |what: String| NetworkError::InvalidConfig(format!("wan transformer: {what}"));
        let default = Self::default();
        let number = |name: &str, fallback: usize| {
            json.get(name)
                .and_then(|value| value.as_u64())
                .map_or(fallback, |value| value as usize)
        };
        if json.get("qk_norm").and_then(|value| value.as_str()) != Some("rms_norm_across_heads") {
            return Err(refused(format!("qk_norm {:?}", json.get("qk_norm"))));
        }
        if json
            .get("cross_attn_norm")
            .and_then(|value| value.as_bool())
            != Some(true)
        {
            return Err(refused("cross_attn_norm must be true".into()));
        }
        for name in ["image_dim", "added_kv_proj_dim", "pos_embed_seq_len"] {
            if json.get(name).is_some_and(|value| !value.is_null()) {
                return Err(refused(format!(
                    "{name} (image conditioning) is not supported"
                )));
            }
        }
        let patch: Vec<usize> = match json.get("patch_size").and_then(|value| value.as_array()) {
            Some(values) => values
                .iter()
                .map(|value| value.as_u64().map(|value| value as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| refused("patch_size".into()))?,
            None => default.patch.to_vec(),
        };
        let config = Self {
            num_heads: number("num_attention_heads", default.num_heads),
            head_dim: number("attention_head_dim", default.head_dim),
            layers: number("num_layers", default.layers),
            ffn_dim: number("ffn_dim", default.ffn_dim),
            freq_dim: number("freq_dim", default.freq_dim),
            text_dim: number("text_dim", default.text_dim),
            in_channels: number("in_channels", default.in_channels),
            out_channels: number("out_channels", default.out_channels),
            patch: patch
                .try_into()
                .map_err(|patch| refused(format!("patch_size {patch:?}")))?,
            eps: json
                .get("eps")
                .and_then(|value| value.as_f64())
                .unwrap_or(default.eps as f64) as f32,
            rope_max_seq_len: number("rope_max_seq_len", default.rope_max_seq_len),
        };
        config.validate()?;
        Ok(config)
    }

    /// Model width.
    pub fn dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Head dimensions per rotary axis, frame then row then column: the row
    /// and the column take a third each, rounded down to even, and the frame
    /// the rest.
    pub fn rope_dims(&self) -> [usize; 3] {
        let side = 2 * (self.head_dim / 6);
        [self.head_dim - 2 * side, side, side]
    }

    /// Values per patch, `channels * frames * rows * columns`.
    fn patch_values(&self, channels: usize) -> usize {
        channels * self.patch.iter().product::<usize>()
    }

    /// Refuses a shape no layer could run: a zero size, an odd head, or a
    /// rotary axis with no dimensions.
    pub fn validate(&self) -> Result<(), NetworkError> {
        let sizes = [
            self.num_heads,
            self.head_dim,
            self.ffn_dim,
            self.freq_dim,
            self.text_dim,
            self.in_channels,
            self.out_channels,
            self.rope_max_seq_len,
        ];
        if sizes.contains(&0) || self.patch.contains(&0) {
            return Err(NetworkError::InvalidConfig(format!(
                "wan transformer: a zero size in {self:?}"
            )));
        }
        if !self.head_dim.is_multiple_of(2) || !self.freq_dim.is_multiple_of(2) || self.head_dim < 6
        {
            return Err(NetworkError::InvalidConfig(format!(
                "wan transformer: head_dim {} and freq_dim {} must be even, the head at least 6",
                self.head_dim, self.freq_dim
            )));
        }
        Ok(())
    }
}

/// One block's weights.
pub struct WanBlock {
    /// `[6 * dim]`: shift, scale and gate of the self-attention, then of the
    /// feed-forward, added to the timestep's projection.
    pub(crate) table: Vec<f32>,
    pub(crate) query: Dense,
    pub(crate) key: Dense,
    pub(crate) value: Dense,
    pub(crate) output: Dense,
    pub(crate) query_norm: Vec<f32>,
    pub(crate) key_norm: Vec<f32>,
    pub(crate) cross_query: Dense,
    pub(crate) cross_key: Dense,
    pub(crate) cross_value: Dense,
    pub(crate) cross_output: Dense,
    pub(crate) cross_query_norm: Vec<f32>,
    pub(crate) cross_key_norm: Vec<f32>,
    /// The cross-attention's layer norm, which has its own affine.
    pub(crate) norm_weight: Vec<f32>,
    pub(crate) norm_bias: Vec<f32>,
    pub(crate) ffn_in: Dense,
    pub(crate) ffn_out: Dense,
}

/// The state of one denoiser call between its stages, which the parity test
/// reads after each.
pub(crate) struct Step {
    /// Patches per axis: frames, rows, columns.
    pub(crate) grid: [usize; 3],
    /// `[tokens, dim]`, frame-major, then row, then column.
    pub(crate) hidden: Matrix,
    /// `[frames, dim]`, one per token frame.
    pub(crate) temb: Matrix,
    /// `[frames, 6 * dim]`, the timestep's share of every block's modulation.
    pub(crate) projection: Matrix,
    /// `[prompt tokens, dim]`.
    pub(crate) context: Matrix,
    pub(crate) rope: Positions,
}

impl Step {
    fn per_frame(&self) -> usize {
        self.grid[1] * self.grid[2]
    }
}

/// A Wan video transformer.
pub struct WanDit {
    pub(crate) config: WanDitConfig,
    /// The patch convolution as a linear map over `[channels, pt, ph, pw]`.
    pub(crate) patch: Dense,
    pub(crate) time_in: Dense,
    pub(crate) time_out: Dense,
    pub(crate) time_projection: Dense,
    pub(crate) text_in: Dense,
    pub(crate) text_out: Dense,
    /// `[2 * dim]`: the output norm's shift and scale, added to the timestep
    /// embedding.
    pub(crate) out_table: Vec<f32>,
    pub(crate) proj_out: Dense,
    pub(crate) blocks: Vec<WanBlock>,
    #[cfg(feature = "cuda")]
    device: Option<crate::cuda_wan_dit::DeviceWanDit>,
}

impl WanDit {
    /// The configuration this model was built for.
    pub fn config(&self) -> &WanDitConfig {
        &self.config
    }

    /// Reads the model out of a checkpoint (a shard index or one file).
    pub fn load_at(
        path: impl AsRef<std::path::Path>,
        config: WanDitConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        let mut model = Self::read_head(&mut file, config, precision)?;
        model.blocks = (0..model.config.layers)
            .map(|index| WanBlock::read(&mut file, &model.config, index, precision))
            .collect::<Result<_, _>>()?;
        Ok(model)
    }

    /// Everything but the blocks, which are left empty for the caller to
    /// stream through [`WanBlock::read`].
    pub fn read_head(
        file: &mut ShardedSafeTensors,
        config: WanDitConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        config.validate()?;
        let dim = config.dim();
        let (values, shape) = file.tensor("patch_embedding.weight")?;
        let [pt, ph, pw] = config.patch;
        expect_shape(
            "patch_embedding.weight",
            &shape,
            &[dim, config.in_channels, pt, ph, pw],
        )?;
        let patch_in = config.patch_values(config.in_channels);
        let mut patch = Dense::new(
            Matrix::from_vec(dim, patch_in, values),
            Some(vector(file, "patch_embedding.bias", dim)?),
        )?;
        if precision == Precision::Q8 {
            patch.quantize();
        }
        let mut linear =
            |name: &str, out: usize, inner: usize| linear(file, name, out, inner, precision);
        let embedder = "condition_embedder";
        let model = Self {
            patch,
            time_in: linear(
                &format!("{embedder}.time_embedder.linear_1"),
                dim,
                config.freq_dim,
            )?,
            time_out: linear(&format!("{embedder}.time_embedder.linear_2"), dim, dim)?,
            time_projection: linear(&format!("{embedder}.time_proj"), 6 * dim, dim)?,
            text_in: linear(
                &format!("{embedder}.text_embedder.linear_1"),
                dim,
                config.text_dim,
            )?,
            text_out: linear(&format!("{embedder}.text_embedder.linear_2"), dim, dim)?,
            proj_out: linear("proj_out", config.patch_values(config.out_channels), dim)?,
            out_table: table(file, "scale_shift_table", 2, dim)?,
            blocks: Vec::new(),
            config,
            #[cfg(feature = "cuda")]
            device: None,
        };
        Ok(model)
    }

    /// Moves the blocks to the device; [`WanDit::forward`] runs there from
    /// then on. The host keeps the embedders, the source of the timestep
    /// embedding, and drops its copy of the blocks.
    #[cfg(feature = "cuda")]
    pub fn attach_device(
        &mut self,
        context: &std::sync::Arc<crate::gpu_transformer::GpuContext>,
    ) -> Result<(), NetworkError> {
        self.attach_device_keeping_host(context)?;
        self.blocks = Vec::new();
        Ok(())
    }

    /// [`WanDit::attach_device`], but the host keeps its blocks (4.6 GB at q8)
    /// so [`WanDit::detach_device`] and a second attach is only an upload.
    #[cfg(feature = "cuda")]
    pub fn attach_device_keeping_host(
        &mut self,
        context: &std::sync::Arc<crate::gpu_transformer::GpuContext>,
    ) -> Result<(), NetworkError> {
        self.device = Some(crate::cuda_wan_dit::DeviceWanDit::upload(context, self)?);
        Ok(())
    }

    /// Frees the device copy. Only for a model attached with
    /// [`WanDit::attach_device_keeping_host`]: [`WanDit::forward`] then runs
    /// on the kept host blocks until the next attach.
    #[cfg(feature = "cuda")]
    pub fn detach_device(&mut self) {
        self.device = None;
    }

    /// Whether [`WanDit::forward`] runs on the device.
    pub fn on_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        return self.device.is_some();
        #[cfg(not(feature = "cuda"))]
        false
    }

    /// The velocity for `latent`, one map of `[in_channels, height, width]`
    /// per latent frame, at `timesteps` (one for every frame, or one per
    /// token frame), under the umT5 `context`, `[tokens, text_dim]`. Returns
    /// one map of `[out_channels, height, width]` per latent frame.
    pub fn forward(
        &self,
        latent: &[FeatureMap],
        timesteps: &[f32],
        context: &Matrix,
    ) -> Result<Vec<FeatureMap>, NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            return device.forward(self, latent, timesteps, context);
        }
        let mut step = self.prepare(latent, timesteps, context)?;
        for block in &self.blocks {
            block.forward(&self.config, &mut step)?;
        }
        self.finish(&step)
    }

    /// Checks the inputs and returns the patch grid.
    pub(crate) fn grid(
        &self,
        latent: &[FeatureMap],
        timesteps: &[f32],
        context: &Matrix,
    ) -> Result<[usize; 3], NetworkError> {
        let config = &self.config;
        let refused =
            |what: String| Err(NetworkError::InvalidConfig(format!("wan forward: {what}")));
        let Some(first) = latent.first() else {
            return refused("no latent frames".into());
        };
        if latent.iter().any(|frame| {
            frame.channels != config.in_channels
                || frame.height != first.height
                || frame.width != first.width
                || frame.data.len() != frame.channels * frame.pixels()
        }) {
            return refused(format!(
                "every frame must be {} x {} x {}",
                config.in_channels, first.height, first.width
            ));
        }
        let sizes = [latent.len(), first.height, first.width];
        let mut grid = [0; 3];
        for axis in 0..3 {
            if sizes[axis] == 0 || !sizes[axis].is_multiple_of(config.patch[axis]) {
                return refused(format!(
                    "latent {sizes:?} does not split into patches of {:?}",
                    config.patch
                ));
            }
            grid[axis] = sizes[axis] / config.patch[axis];
            if grid[axis] > config.rope_max_seq_len {
                return refused(format!(
                    "{} patches on one axis, past the rotary tables' {}",
                    grid[axis], config.rope_max_seq_len
                ));
            }
        }
        if timesteps.len() != 1 && timesteps.len() != grid[0] {
            return refused(format!(
                "{} timesteps for {} token frames",
                timesteps.len(),
                grid[0]
            ));
        }
        if context.rows == 0 || context.cols != config.text_dim {
            return refused(format!(
                "context is {} x {}, wanted rows of {}",
                context.rows, context.cols, config.text_dim
            ));
        }
        Ok(grid)
    }

    /// The timestep embedding and its projection, one row per token frame.
    pub(crate) fn time(
        &self,
        timesteps: &[f32],
        frames: usize,
    ) -> Result<(Matrix, Matrix), NetworkError> {
        let dim = self.config.dim();
        let mut temb = Matrix::new(frames, dim);
        let mut projection = Matrix::new(frames, 6 * dim);
        for frame in 0..frames {
            let time = timesteps[frame.min(timesteps.len() - 1)];
            let hidden = self.time_in.apply(&sinusoid(time, self.config.freq_dim))?;
            let embedded = self
                .time_out
                .apply(&hidden.into_iter().map(silu).collect::<Vec<_>>())?;
            let projected = self.time_projection.apply(
                &embedded
                    .iter()
                    .map(|value| silu(*value))
                    .collect::<Vec<_>>(),
            )?;
            temb.row_mut(frame).copy_from_slice(&embedded);
            projection.row_mut(frame).copy_from_slice(&projected);
        }
        Ok((temb, projection))
    }

    /// The patches of `latent`, `[tokens, channels * pt * ph * pw]` in the
    /// convolution weight's order.
    pub(crate) fn patches(&self, latent: &[FeatureMap], grid: [usize; 3]) -> Matrix {
        let [pt, ph, pw] = self.config.patch;
        let channels = self.config.in_channels;
        let width = latent[0].width;
        let mut patches = Matrix::new(grid.iter().product(), self.config.patch_values(channels));
        patches
            .data
            .par_chunks_mut(patches.cols)
            .enumerate()
            .for_each(|(token, row)| {
                let (f, y, x) = (
                    token / (grid[1] * grid[2]),
                    token / grid[2] % grid[1],
                    token % grid[2],
                );
                let mut at = 0;
                for c in 0..channels {
                    for kt in 0..pt {
                        let plane = latent[f * pt + kt].plane(c);
                        for kh in 0..ph {
                            for kw in 0..pw {
                                row[at] = plane[(y * ph + kh) * width + x * pw + kw];
                                at += 1;
                            }
                        }
                    }
                }
            });
        patches
    }

    /// The prompt states projected to the model's width.
    pub(crate) fn embed_context(&self, context: &Matrix) -> Result<Matrix, NetworkError> {
        let mut hidden = self.text_in.forward(context)?;
        hidden
            .data
            .par_iter_mut()
            .for_each(|value| *value = gelu(*value));
        self.text_out.forward(&hidden)
    }

    /// Everything before the first block.
    pub(crate) fn prepare(
        &self,
        latent: &[FeatureMap],
        timesteps: &[f32],
        context: &Matrix,
    ) -> Result<Step, NetworkError> {
        let grid = self.grid(latent, timesteps, context)?;
        let (temb, projection) = self.time(timesteps, grid[0])?;
        Ok(Step {
            hidden: self.patch.forward(&self.patches(latent, grid))?,
            temb,
            projection,
            context: self.embed_context(context)?,
            rope: rope(&self.config, grid),
            grid,
        })
    }

    /// The output norm, the projection and the inverse of the patching.
    pub(crate) fn finish(&self, step: &Step) -> Result<Vec<FeatureMap>, NetworkError> {
        let dim = self.config.dim();
        let mut modulation = Matrix::new(step.grid[0], 2 * dim);
        for frame in 0..step.grid[0] {
            for (index, value) in modulation.row_mut(frame).iter_mut().enumerate() {
                *value = self.out_table[index] + step.temb.row(frame)[index % dim];
            }
        }
        let mut normed = layer_norm(&step.hidden, self.config.eps);
        modulate_frames(&mut normed, &modulation, step.per_frame(), 0);
        let output = self.proj_out.forward(&normed)?;
        Ok(self.unpatchify(&output, step.grid, self.config.out_channels))
    }

    /// `[tokens, pt * ph * pw * channels]` (channels fastest) back to one map
    /// per latent frame.
    pub(crate) fn unpatchify(
        &self,
        tokens: &Matrix,
        grid: [usize; 3],
        channels: usize,
    ) -> Vec<FeatureMap> {
        let [pt, ph, pw] = self.config.patch;
        let (height, width) = (grid[1] * ph, grid[2] * pw);
        let mut frames = vec![FeatureMap::new(channels, height, width); grid[0] * pt];
        for (token, row) in tokens.data.chunks(tokens.cols).enumerate() {
            let (f, y, x) = (
                token / (grid[1] * grid[2]),
                token / grid[2] % grid[1],
                token % grid[2],
            );
            for kt in 0..pt {
                let frame = &mut frames[f * pt + kt];
                for kh in 0..ph {
                    for kw in 0..pw {
                        let pixel = (y * ph + kh) * width + x * pw + kw;
                        let base = ((kt * ph + kh) * pw + kw) * channels;
                        for c in 0..channels {
                            frame.data[c * height * width + pixel] = row[base + c];
                        }
                    }
                }
            }
        }
        frames
    }
}

impl WanBlock {
    /// Block `index`'s weights, each checked against `config`.
    pub fn read(
        file: &mut ShardedSafeTensors,
        config: &WanDitConfig,
        index: usize,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let (dim, ffn) = (config.dim(), config.ffn_dim);
        let name = |part: &str| format!("blocks.{index}.{part}");
        let mut linear =
            |part: &str, out: usize, inner: usize| linear(file, &name(part), out, inner, precision);
        let query = linear("attn1.to_q", dim, dim)?;
        let key = linear("attn1.to_k", dim, dim)?;
        let value = linear("attn1.to_v", dim, dim)?;
        let output = linear("attn1.to_out.0", dim, dim)?;
        let cross_query = linear("attn2.to_q", dim, dim)?;
        let cross_key = linear("attn2.to_k", dim, dim)?;
        let cross_value = linear("attn2.to_v", dim, dim)?;
        let cross_output = linear("attn2.to_out.0", dim, dim)?;
        let ffn_in = linear("ffn.net.0.proj", ffn, dim)?;
        let ffn_out = linear("ffn.net.2", dim, ffn)?;
        Ok(Self {
            table: table(file, &name("scale_shift_table"), 6, dim)?,
            query,
            key,
            value,
            output,
            query_norm: vector(file, &name("attn1.norm_q.weight"), dim)?,
            key_norm: vector(file, &name("attn1.norm_k.weight"), dim)?,
            cross_query,
            cross_key,
            cross_value,
            cross_output,
            cross_query_norm: vector(file, &name("attn2.norm_q.weight"), dim)?,
            cross_key_norm: vector(file, &name("attn2.norm_k.weight"), dim)?,
            norm_weight: vector(file, &name("norm2.weight"), dim)?,
            norm_bias: vector(file, &name("norm2.bias"), dim)?,
            ffn_in,
            ffn_out,
        })
    }

    /// This block's modulation for every token frame, `[frames, 6 * dim]`.
    pub(crate) fn modulation(&self, step: &Step) -> Matrix {
        let mut modulation = step.projection.clone();
        modulation
            .data
            .par_chunks_mut(self.table.len())
            .for_each(|row| row.iter_mut().zip(&self.table).for_each(|(v, t)| *v += t));
        modulation
    }

    /// Runs the block on `step.hidden` in place.
    pub(crate) fn forward(
        &self,
        config: &WanDitConfig,
        step: &mut Step,
    ) -> Result<(), NetworkError> {
        let (heads, eps) = (config.num_heads, config.eps);
        let per_frame = step.per_frame();
        let modulation = self.modulation(step);

        let mut normed = layer_norm(&step.hidden, eps);
        modulate_frames(&mut normed, &modulation, per_frame, 0);
        let mut query = rms_norm(&self.query.forward(&normed)?, &self.query_norm, eps);
        let mut key = rms_norm(&self.key.forward(&normed)?, &self.key_norm, eps);
        let value = self.value.forward(&normed)?;
        step.rope.apply(&mut query, heads);
        step.rope.apply(&mut key, heads);
        let attended = self
            .output
            .forward(&attention(&query, &key, &value, heads, heads, false))?;
        gate_frames(&mut step.hidden, &attended, &modulation, per_frame, 2);

        let mut normed = layer_norm(&step.hidden, eps);
        normed.data.par_chunks_mut(normed.cols).for_each(|row| {
            for ((value, weight), bias) in
                row.iter_mut().zip(&self.norm_weight).zip(&self.norm_bias)
            {
                *value = *value * weight + bias;
            }
        });
        let query = rms_norm(
            &self.cross_query.forward(&normed)?,
            &self.cross_query_norm,
            eps,
        );
        let key = rms_norm(
            &self.cross_key.forward(&step.context)?,
            &self.cross_key_norm,
            eps,
        );
        let value = self.cross_value.forward(&step.context)?;
        let attended = self
            .cross_output
            .forward(&attention(&query, &key, &value, heads, heads, false))?;
        step.hidden
            .data
            .par_iter_mut()
            .zip(&attended.data)
            .for_each(|(value, add)| *value += add);

        let mut normed = layer_norm(&step.hidden, eps);
        modulate_frames(&mut normed, &modulation, per_frame, 3);
        let mut inner = self.ffn_in.forward(&normed)?;
        inner
            .data
            .par_iter_mut()
            .for_each(|value| *value = gelu(*value));
        let output = self.ffn_out.forward(&inner)?;
        gate_frames(&mut step.hidden, &output, &modulation, per_frame, 5);
        Ok(())
    }
}

/// The rotary table for a patch grid: per token (frame-major), the frame's
/// angles, then the row's, then the column's, each axis with its own
/// frequencies over its share of the head. Angles in double precision, as
/// `diffusers` builds them.
pub(crate) fn rope(config: &WanDitConfig, grid: [usize; 3]) -> Positions {
    let pairs = config.head_dim / 2;
    let tokens: usize = grid.iter().product();
    let (mut cos, mut sin) = (vec![0.0; tokens * pairs], vec![0.0; tokens * pairs]);
    for token in 0..tokens {
        let position = [
            token / (grid[1] * grid[2]),
            token / grid[2] % grid[1],
            token % grid[2],
        ];
        let mut pair = 0;
        for (axis, dim) in config.rope_dims().into_iter().enumerate() {
            for index in 0..dim / 2 {
                let omega = 10_000f64.powf(-2.0 * index as f64 / dim as f64);
                let angle = position[axis] as f64 * omega;
                cos[token * pairs + pair] = angle.cos() as f32;
                sin[token * pairs + pair] = angle.sin() as f32;
                pair += 1;
            }
        }
    }
    Positions { cos, sin, pairs }
}

/// `x * (1 + scale) + shift` on every row, with the shift at chunk `chunk`
/// and the scale at `chunk + 1` of the row's frame's modulation.
pub(crate) fn modulate_frames(
    tokens: &mut Matrix,
    modulation: &Matrix,
    per_frame: usize,
    chunk: usize,
) {
    let dim = tokens.cols;
    tokens
        .data
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(token, row)| {
            let frame = &modulation.row(token / per_frame)[chunk * dim..(chunk + 2) * dim];
            let (shift, scale) = frame.split_at(dim);
            for ((value, shift), scale) in row.iter_mut().zip(shift).zip(scale) {
                *value = *value * (1.0 + scale) + shift;
            }
        });
}

/// `x += branch * gate`, the gate at chunk `chunk` of the row's frame's
/// modulation.
pub(crate) fn gate_frames(
    target: &mut Matrix,
    branch: &Matrix,
    modulation: &Matrix,
    per_frame: usize,
    chunk: usize,
) {
    let dim = target.cols;
    target
        .data
        .par_chunks_mut(dim)
        .zip(branch.data.par_chunks(dim))
        .enumerate()
        .for_each(|(token, (row, branch))| {
            let gate = &modulation.row(token / per_frame)[chunk * dim..(chunk + 1) * dim];
            for ((value, branch), gate) in row.iter_mut().zip(branch).zip(gate) {
                *value += branch * gate;
            }
        });
}

fn expect_shape(name: &str, got: &[usize], want: &[usize]) -> Result<(), NetworkError> {
    match got == want {
        true => Ok(()),
        false => Err(NetworkError::InvalidConfig(format!(
            "{name} is {got:?}, the configuration says {want:?}"
        ))),
    }
}

fn vector(file: &mut ShardedSafeTensors, name: &str, len: usize) -> Result<Vec<f32>, NetworkError> {
    let (values, shape) = file.tensor(name)?;
    expect_shape(name, &shape, &[len])?;
    Ok(values)
}

/// A `[1, chunks, dim]` modulation table, flattened.
fn table(
    file: &mut ShardedSafeTensors,
    name: &str,
    chunks: usize,
    dim: usize,
) -> Result<Vec<f32>, NetworkError> {
    let (values, shape) = file.tensor(name)?;
    expect_shape(name, &shape, &[1, chunks, dim])?;
    Ok(values)
}

/// A linear layer with its bias, both checked against `[out, inner]`.
fn linear(
    file: &mut ShardedSafeTensors,
    name: &str,
    out: usize,
    inner: usize,
    precision: Precision,
) -> Result<Dense, NetworkError> {
    let (values, shape) = file.tensor(&format!("{name}.weight"))?;
    expect_shape(&format!("{name}.weight"), &shape, &[out, inner])?;
    let bias = vector(file, &format!("{name}.bias"), out)?;
    let mut layer = Dense::new(Matrix::from_vec(out, inner, values), Some(bias))?;
    if precision == Precision::Q8 {
        layer.quantize();
    }
    Ok(layer)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Heads of 64, which the fused attention takes, so the device test can
    /// share it.
    pub(crate) fn tiny() -> WanDitConfig {
        WanDitConfig {
            num_heads: 2,
            head_dim: 64,
            layers: 2,
            ffn_dim: 256,
            freq_dim: 32,
            text_dim: 48,
            in_channels: 8,
            out_channels: 8,
            patch: [1, 2, 2],
            eps: 1e-6,
            rope_max_seq_len: 16,
        }
    }

    /// Small values that do not repeat along a row.
    pub(crate) fn noise(count: usize, seed: u32, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let x = ((index as f32 + 1.0) * 12.9898 + seed as f32 * 78.233).sin() * 43_758.547;
                (x - x.floor() - 0.5) * scale
            })
            .collect()
    }

    pub(crate) fn tensors(config: &WanDitConfig) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let mut tensors = BTreeMap::new();
        let mut seed = 0;
        let mut add = |name: String, shape: Vec<usize>| {
            seed += 1;
            let inner = *shape.last().unwrap();
            // A linear layer's scale keeps the activations near one.
            let scale = match shape.len() {
                2 => 2.0 / (inner as f32).sqrt(),
                _ => 0.4,
            };
            let values = noise(shape.iter().product(), seed, scale);
            tensors.insert(name, (shape, values));
        };
        let dim = config.dim();
        let [pt, ph, pw] = config.patch;
        let linear = |add: &mut dyn FnMut(String, Vec<usize>), name: &str, out, inner| {
            add(format!("{name}.weight"), vec![out, inner]);
            add(format!("{name}.bias"), vec![out]);
        };
        add(
            "patch_embedding.weight".into(),
            vec![dim, config.in_channels, pt, ph, pw],
        );
        add("patch_embedding.bias".into(), vec![dim]);
        let embedder = "condition_embedder";
        let head = [
            (
                format!("{embedder}.time_embedder.linear_1"),
                dim,
                config.freq_dim,
            ),
            (format!("{embedder}.time_embedder.linear_2"), dim, dim),
            (format!("{embedder}.time_proj"), 6 * dim, dim),
            (
                format!("{embedder}.text_embedder.linear_1"),
                dim,
                config.text_dim,
            ),
            (format!("{embedder}.text_embedder.linear_2"), dim, dim),
            (
                "proj_out".into(),
                config.patch_values(config.out_channels),
                dim,
            ),
        ];
        for (name, out, inner) in head {
            linear(&mut add, &name, out, inner);
        }
        add("scale_shift_table".into(), vec![1, 2, dim]);
        for index in 0..config.layers {
            let name = |part: &str| format!("blocks.{index}.{part}");
            for attention in ["attn1", "attn2"] {
                for part in ["to_q", "to_k", "to_v", "to_out.0"] {
                    linear(&mut add, &name(&format!("{attention}.{part}")), dim, dim);
                }
                add(name(&format!("{attention}.norm_q.weight")), vec![dim]);
                add(name(&format!("{attention}.norm_k.weight")), vec![dim]);
            }
            add(name("norm2.weight"), vec![dim]);
            add(name("norm2.bias"), vec![dim]);
            linear(&mut add, &name("ffn.net.0.proj"), config.ffn_dim, dim);
            linear(&mut add, &name("ffn.net.2"), dim, config.ffn_dim);
            add(name("scale_shift_table"), vec![1, 6, dim]);
        }
        tensors
    }

    pub(crate) fn checkpoint(config: &WanDitConfig, name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(name);
        crate::safetensors::write_checkpoint(&path, &tensors(config));
        path
    }

    /// A latent of `frames` maps of `[in_channels, height, width]`.
    pub(crate) fn latent(
        config: &WanDitConfig,
        frames: usize,
        height: usize,
        width: usize,
    ) -> Vec<FeatureMap> {
        (0..frames)
            .map(|frame| {
                let count = config.in_channels * height * width;
                FeatureMap::from_vec(
                    config.in_channels,
                    height,
                    width,
                    noise(count, 100 + frame as u32, 2.0),
                )
                .unwrap()
            })
            .collect()
    }

    pub(crate) fn context(config: &WanDitConfig, tokens: usize) -> Matrix {
        Matrix::from_vec(
            tokens,
            config.text_dim,
            noise(tokens * config.text_dim, 200, 0.4),
        )
    }

    #[test]
    fn the_rotary_table_gives_each_axis_its_share_of_the_head() {
        assert_eq!(WanDitConfig::default().rope_dims(), [44, 42, 42]);
        let config = WanDitConfig {
            head_dim: 12,
            ..tiny()
        };
        assert_eq!(config.rope_dims(), [4, 4, 4]);
        let grid = [2, 3, 4];
        let table = rope(&config, grid);
        assert_eq!(table.pairs, 6);
        // Frame 1, row 2, column 3; the second pair of each axis turns at a
        // hundredth of the first.
        let token = 4 * 3 + 2 * 4 + 3;
        let angles: Vec<f32> = (0..6)
            .map(|pair| table.sin[token * 6 + pair].atan2(table.cos[token * 6 + pair]))
            .collect();
        let want = [1.0, 0.01, 2.0, 0.02, 3.0, 0.03];
        for (got, want) in angles.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{angles:?}");
        }
        // The first token sits at the origin.
        assert!(table.cos[..6].iter().all(|cos| *cos == 1.0));
        assert!(table.sin[..6].iter().all(|sin| *sin == 0.0));
    }

    #[test]
    fn the_timestep_embedding_is_cosine_first_and_one_per_frame() {
        // `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)`: the
        // frequencies are exp(-ln(10000) i / half), with no scaling of t.
        let embedding = sinusoid(1.0, 4);
        let want = [1f32.cos(), 0.01f32.cos(), 1f32.sin(), 0.01f32.sin()];
        for (got, want) in embedding.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{embedding:?}");
        }

        let config = tiny();
        let path = checkpoint(&config, "rusting_brain_wan_dit_time.safetensors");
        let model = WanDit::load_at(&path, config.clone(), Precision::F32).unwrap();
        let (temb, projection) = model.time(&[0.0, 952.0, 952.0], 3).unwrap();
        assert_eq!((temb.rows, temb.cols), (3, config.dim()));
        assert_eq!((projection.rows, projection.cols), (3, 6 * config.dim()));
        assert_eq!(temb.row(1), temb.row(2));
        assert_ne!(temb.row(0), temb.row(1));
        // One timestep stands for every frame.
        let (single, _) = model.time(&[952.0], 2).unwrap();
        assert_eq!(single.row(0), temb.row(1));
        assert_eq!(single.row(1), temb.row(1));
        // The projection reads the SiLU of the embedding.
        let silu_temb: Vec<f32> = temb.row(0).iter().map(|value| silu(*value)).collect();
        assert_eq!(
            model.time_projection.apply(&silu_temb).unwrap(),
            projection.row(0)
        );
    }

    #[test]
    fn modulation_and_gates_follow_each_rows_frame() {
        let dim = 2;
        // Two frames of six chunks each, the chunk index in the tens.
        let modulation = Matrix::from_vec(
            2,
            6 * dim,
            (0..2 * 6 * dim)
                .map(|index| (index / (6 * dim)) as f32 + (index % (6 * dim) / dim) as f32 * 10.0)
                .collect(),
        );
        let mut tokens = Matrix::from_vec(4, dim, vec![1.0; 4 * dim]);
        modulate_frames(&mut tokens, &modulation, 2, 3);
        // Frame 0: shift 30, scale 40; frame 1: shift 31, scale 41.
        assert_eq!(
            tokens.data,
            [71.0, 71.0, 71.0, 71.0, 73.0, 73.0, 73.0, 73.0]
        );

        let branch = Matrix::from_vec(4, dim, vec![1.0; 4 * dim]);
        let mut target = Matrix::new(4, dim);
        gate_frames(&mut target, &branch, &modulation, 2, 5);
        assert_eq!(
            target.data,
            [50.0, 50.0, 50.0, 50.0, 51.0, 51.0, 51.0, 51.0]
        );
    }

    #[test]
    fn patches_unpatchify_back_with_the_channel_fastest() {
        let config = tiny();
        let path = checkpoint(&config, "rusting_brain_wan_dit_patch.safetensors");
        let model = WanDit::read_head(
            &mut ShardedSafeTensors::open(&path).unwrap(),
            config.clone(),
            Precision::F32,
        )
        .unwrap();
        let latent = latent(&config, 2, 4, 6);
        let grid = model.grid(&latent, &[1.0], &context(&config, 3)).unwrap();
        assert_eq!(grid, [2, 2, 3]);
        let patches = model.patches(&latent, grid);
        // The convolution's order is `[channel, kt, kh, kw]`; the output's is
        // `[kt, kh, kw, channel]`.
        let spatial = 4;
        let mut output = Matrix::new(patches.rows, patches.cols);
        for (row, patch) in output
            .data
            .chunks_mut(patches.cols)
            .zip(patches.data.chunks(patches.cols))
        {
            for c in 0..config.in_channels {
                for k in 0..spatial {
                    row[k * config.in_channels + c] = patch[c * spatial + k];
                }
            }
        }
        assert_eq!(model.unpatchify(&output, grid, config.in_channels), latent);
    }

    #[test]
    fn malformed_shapes_are_refused() {
        fn refused<T>(result: Result<T, NetworkError>) {
            assert!(
                matches!(result, Err(NetworkError::InvalidConfig(_))),
                "{:?}",
                result.err()
            )
        }
        let config = tiny();
        refused(
            WanDitConfig {
                head_dim: 63,
                ..tiny()
            }
            .validate(),
        );
        refused(
            WanDitConfig {
                patch: [1, 0, 2],
                ..tiny()
            }
            .validate(),
        );

        // A block tensor of the wrong shape.
        let mut tensors = tensors(&config);
        tensors.insert(
            "blocks.1.attn2.to_k.weight".into(),
            (vec![128, 64], vec![0.0; 128 * 64]),
        );
        let path = std::env::temp_dir().join("rusting_brain_wan_dit_bad.safetensors");
        crate::safetensors::write_checkpoint(&path, &tensors);
        refused(WanDit::load_at(&path, config.clone(), Precision::F32).map(drop));
        let mut tensors = super::tests::tensors(&config);
        tensors.insert(
            "blocks.0.scale_shift_table".into(),
            (vec![1, 5, 128], vec![0.0; 5 * 128]),
        );
        crate::safetensors::write_checkpoint(&path, &tensors);
        refused(WanDit::load_at(&path, config.clone(), Precision::F32).map(drop));

        let path = checkpoint(&config, "rusting_brain_wan_dit_inputs.safetensors");
        let model = WanDit::load_at(&path, config.clone(), Precision::F32).unwrap();
        let context = context(&config, 3);
        refused(model.forward(&[], &[1.0], &context));
        refused(model.forward(&latent(&config, 2, 5, 6), &[1.0], &context));
        refused(model.forward(&latent(&config, 2, 4, 6), &[1.0, 2.0, 3.0], &context));
        refused(model.forward(&latent(&config, 2, 4, 6), &[1.0], &Matrix::new(3, 47)));
        refused(model.forward(&latent(&config, 1, 4, 34), &[1.0], &context));
        let mut uneven = latent(&config, 2, 4, 6);
        uneven[1] = FeatureMap::new(config.in_channels, 4, 8);
        refused(model.forward(&uneven, &[1.0], &context));
        let mut short = latent(&config, 2, 4, 6);
        short[0].data.pop();
        refused(model.forward(&short, &[1.0], &context));
    }

    #[test]
    fn a_quantized_model_says_what_the_float_one_does() {
        let config = tiny();
        let path = checkpoint(&config, "rusting_brain_wan_dit_q8.safetensors");
        let float = WanDit::load_at(&path, config.clone(), Precision::F32).unwrap();
        let quantized = WanDit::load_at(&path, config.clone(), Precision::Q8).unwrap();
        let (latent, context) = (latent(&config, 2, 4, 6), context(&config, 5));
        let want = float.forward(&latent, &[0.0, 500.0], &context).unwrap();
        let got = quantized.forward(&latent, &[0.0, 500.0], &context).unwrap();
        assert_eq!(want.len(), 2);
        assert_eq!((want[0].channels, want[0].height, want[0].width), (8, 4, 6));
        let flat = |frames: &[FeatureMap]| {
            frames
                .iter()
                .flat_map(|f| f.data.clone())
                .collect::<Vec<_>>()
        };
        let error = crate::wan_vae::tests::relative(&flat(&got), &flat(&want));
        assert!(error < 2e-2 && error > 0.0, "{error}");
    }

    /// The repeat-interleaved `[tokens, head_dim]` layout diffusers keeps
    /// its rotary table in.
    fn interleaved(half: &[f32], pairs: usize) -> Vec<f32> {
        half.chunks(pairs)
            .flat_map(|row| row.iter().flat_map(|value| [*value, *value]))
            .collect()
    }

    /// Per-frame rows repeated for each of the frame's `per_frame` tokens.
    fn tokens_of(frames: &Matrix, per_frame: usize) -> Vec<f32> {
        frames
            .data
            .chunks(frames.cols)
            .flat_map(|row| std::iter::repeat_n(row, per_frame).flatten().copied())
            .collect()
    }

    /// One F32 step of Wan 2.2's transformer against diffusers'
    /// `WanTransformer3DModel` on the output of RustingFrame's
    /// `tools/wan_refs.py w3`: a two-frame 16x16 latent at t = 952 under the
    /// short prompt of w1, every frame but the first at the step's timestep.
    /// `RB_WAN_REF` is the directory holding `w3.safetensors` and
    /// `reference.json`, `RB_WAN_MODEL` the model (default the local copy),
    /// and `RB_WAN_CUDA=1` checks the int8 device model as well, with
    /// `RB_WAN_BENCH` timing it at a larger latent. The host runs read the
    /// blocks one at a time, about 1.2 GB of host memory; the device model
    /// holds its int8 copy on the host until upload, about 4.6 GB.
    #[test]
    #[ignore = "needs Wan2.2-TI2V-5B and the output of RustingFrame's tools/wan_refs.py w3"]
    fn wan22_transformer_matches_diffusers() {
        use crate::wan_vae::tests::{frames, relative, stack};
        let env = |name: &str| std::env::var(name).ok();
        let dir = std::path::PathBuf::from(env("RB_WAN_REF").expect("RB_WAN_REF"));
        let model = std::path::PathBuf::from(
            env("RB_WAN_MODEL")
                .unwrap_or_else(|| "/mnt/ollama/RustingFrameModels/Wan2.2-TI2V-5B".into()),
        )
        .join("transformer");
        let mut reference =
            crate::safetensors::SafeTensors::open(dir.join("w3.safetensors")).unwrap();
        let mut read = |name: &str| reference.tensor(name).unwrap();
        let mut report = Vec::new();
        let mut gate = |stage: &str, error: f32, limit: f32| {
            eprintln!("wan dit {stage}: relative L2 {error:.3e} (gate {limit:.0e})");
            report.push((stage.to_string(), error, limit));
        };

        let config = WanDitConfig::from_file(model.join("config.json")).unwrap();
        let weights = model.join("diffusion_pytorch_model.safetensors.index.json");
        let (hidden, shape) = read("hidden_states");
        let latent = frames(&hidden, &shape);
        let (context, shape) = read("context");
        let context = Matrix::from_vec(shape[0], shape[1], context);
        // The reference keeps one timestep per token; they are even over
        // each frame.
        let (per_token, _) = read("timestep");
        let per_frame = per_token.len() / latent.len();
        let timesteps: Vec<f32> = per_token.iter().step_by(per_frame).copied().collect();
        for (frame, t) in per_token.chunks(per_frame).zip(&timesteps) {
            assert!(frame.iter().all(|value| value == t), "{per_token:?}");
        }
        assert_eq!(timesteps, [0.0, 952.0]);

        let started = std::time::Instant::now();
        let mut file = ShardedSafeTensors::open(&weights).unwrap();
        let head = WanDit::read_head(&mut file, config.clone(), Precision::F32).unwrap();
        eprintln!("wan dit head load: {:?}", started.elapsed());
        let started = std::time::Instant::now();
        let mut step = head.prepare(&latent, &timesteps, &context).unwrap();
        let pairs = step.rope.pairs;
        gate(
            "rope cos",
            relative(&interleaved(&step.rope.cos, pairs), &read("rope.cos").0),
            1e-6,
        );
        gate(
            "rope sin",
            relative(&interleaved(&step.rope.sin, pairs), &read("rope.sin").0),
            1e-6,
        );
        gate(
            "patch_embed",
            relative(&step.hidden.data, &read("patch_embed").0),
            1e-5,
        );
        let tokens = step.per_frame();
        gate(
            "temb",
            relative(&tokens_of(&step.temb, tokens), &read("temb").0),
            1e-5,
        );
        gate(
            "timestep_proj",
            relative(
                &tokens_of(&step.projection, tokens),
                &read("timestep_proj").0,
            ),
            1e-5,
        );
        gate(
            "context_embed",
            relative(&step.context.data, &read("context_embed").0),
            1e-5,
        );
        let mut reading = std::time::Duration::ZERO;
        for index in 0..config.layers {
            let loaded = std::time::Instant::now();
            let block = WanBlock::read(&mut file, &config, index, Precision::F32).unwrap();
            reading += loaded.elapsed();
            block.forward(&config, &mut step).unwrap();
            if index == 0 {
                gate(
                    "block0",
                    relative(&step.hidden.data, &read("block0").0),
                    1e-5,
                );
            }
        }
        let output = head.finish(&step).unwrap();
        eprintln!(
            "wan dit host step: {:?} ({reading:?} of it reading blocks)",
            started.elapsed()
        );
        // The F32 rounding of thirty blocks, each about 1e-6 (block0), adds up;
        // see docs/baseline.md.
        gate("output", relative(&stack(&output), &read("output").0), 5e-5);
        drop((head, step));

        // The same step with int8 weights on the host, which is what the
        // device's error is to be read against.
        let head = WanDit::read_head(&mut file, config.clone(), Precision::Q8).unwrap();
        let mut step = head.prepare(&latent, &timesteps, &context).unwrap();
        for index in 0..config.layers {
            let block = WanBlock::read(&mut file, &config, index, Precision::Q8).unwrap();
            block.forward(&config, &mut step).unwrap();
        }
        let host_q8 = stack(&head.finish(&step).unwrap());
        gate(
            "host Q8 output",
            relative(&host_q8, &read("output").0),
            5e-2,
        );
        drop((head, step, file));

        #[cfg(feature = "cuda")]
        if env("RB_WAN_CUDA").as_deref() == Some("1") {
            let want = read("output").0;
            let started = std::time::Instant::now();
            let mut model = WanDit::load_at(&weights, config.clone(), Precision::Q8).unwrap();
            eprintln!("wan dit Q8 host load: {:?}", started.elapsed());
            let gpu = crate::gpu_transformer::GpuContext::new(0).unwrap();
            let started = std::time::Instant::now();
            model.attach_device(&gpu).unwrap();
            eprintln!("wan dit upload: {:?}", started.elapsed());
            for run in 0..3 {
                let started = std::time::Instant::now();
                let output = model.forward(&latent, &timesteps, &context).unwrap();
                eprintln!("wan dit device step {run}: {:?}", started.elapsed());
                gate("device Q8 output", relative(&stack(&output), &want), 5e-2);
                gate(
                    "device Q8 against host Q8",
                    relative(&stack(&output), &host_q8),
                    5e-2,
                );
            }
            // `RB_WAN_BENCH=FxHxW`, a latent size, times a random latent of
            // it: 5x44x80 is 704x1280 at 17 frames.
            if let Some(size) = env("RB_WAN_BENCH") {
                let [frames, height, width] = size
                    .split('x')
                    .map(|part| part.parse().unwrap())
                    .collect::<Vec<usize>>()[..]
                else {
                    panic!("RB_WAN_BENCH is FxHxW")
                };
                let latent = self::latent(&config, frames, height, width);
                let mut timesteps = vec![952.0; frames];
                timesteps[0] = 0.0;
                for run in 0..3 {
                    let started = std::time::Instant::now();
                    model.forward(&latent, &timesteps, &context).unwrap();
                    eprintln!(
                        "wan dit device step at {size} {run}: {:?}",
                        started.elapsed()
                    );
                }
            }
        }

        let failures: Vec<_> = report
            .iter()
            .filter(|(_, error, limit)| error.is_nan() || error > limit)
            .collect();
        assert!(failures.is_empty(), "over the gate: {failures:?}");
    }
}
