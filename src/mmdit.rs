//! The transformer that denoises a latent: MMDiT, as FLUX and Stable
//! Diffusion 3 build it.
//!
//! The shape is not the language model's. A denoiser reads two sequences —
//! the image's patches and the prompt's tokens — and it has to condition every
//! layer on the noise level. MMDiT answers both with the same trick: the two
//! sequences keep separate weights but share one attention, and a vector built
//! from the timestep drives a per-layer scale, shift and gate on top of a
//! plain layer norm. That is *adaptive layer norm*, and it is where the
//! timestep enters the network.
//!
//! A block comes in two flavours, and a model stacks both:
//!
//! - **Double stream**: image and text each have their own norms, projections
//!   and feed-forward, and meet only inside the attention, where the two
//!   token sequences are concatenated. FLUX runs nineteen of these,
//!   FLUX.2-klein eight.
//! - **Single stream**: the two sequences are already concatenated, so one set
//!   of weights covers both, and the attention and the feed-forward share one
//!   fused projection. FLUX runs thirty-eight, FLUX.2-klein forty-eight.
//!
//! Positions are rotary, in two dimensions: each attention head's dimensions
//! are split between the patch's row and its column, so the model knows where
//! in the image a patch sits. Text tokens sit at the origin, where the
//! rotation is the identity.
//!
//! [`Dit::load`] reads a checkpoint under the key names Black Forest Labs
//! publishes FLUX with. Inference only, like the rest of the hosting path.

use crate::conv::Dense;
use crate::diffusion::Denoiser;
use crate::ffn::{gelu, silu};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::transformer::Precision;
use rayon::prelude::*;

/// The shape of a denoiser, which a checkpoint's own configuration names.
#[derive(Clone, Debug, PartialEq)]
pub struct DitConfig {
    /// Channels per patch of the latent the sampler works on. FLUX packs a
    /// 16-channel latent into 2x2 patches, so 64.
    pub in_channels: usize,
    pub out_channels: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub double_blocks: usize,
    pub single_blocks: usize,
    /// Feed-forward width as a multiple of `d_model`.
    pub mlp_ratio: f32,
    /// Width of the text encoder's output.
    pub text_dim: usize,
    /// Width of the pooled prompt vector that joins the timestep.
    pub pooled_dim: usize,
    /// How each head's dimensions split across the position axes. The values
    /// must be even and sum to `d_model / num_heads`. Three axes is what FLUX
    /// ships — one of them always zero, so it contributes no rotation — and
    /// two is the same model written down without it.
    pub axes_dim: Vec<usize>,
    pub theta: f32,
    /// Whether the model takes a distilled guidance scale. FLUX dev does,
    /// FLUX schnell and the klein models do not.
    pub guidance: bool,
    pub eps: f32,
}

impl Default for DitConfig {
    /// FLUX.1, which is the published model this layout was written from.
    fn default() -> Self {
        Self {
            in_channels: 64,
            out_channels: 64,
            d_model: 3072,
            num_heads: 24,
            double_blocks: 19,
            single_blocks: 38,
            mlp_ratio: 4.0,
            text_dim: 4096,
            pooled_dim: 768,
            axes_dim: vec![16, 56, 56],
            theta: 10_000.0,
            guidance: true,
            eps: 1e-6,
        }
    }
}

impl DitConfig {
    /// Reads a diffusers `transformer/config.json`, which is how a published
    /// model states its own shape.
    ///
    /// The width is `attention_head_dim * num_attention_heads`, because that is
    /// the pair those files record rather than the product.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let text = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
            NetworkError::InvalidDataset(format!("transformer config.json: {error}"))
        })?;

        let number = |name: &str| -> Option<usize> {
            json.get(name)
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
        };
        let missing = |name: &str| {
            NetworkError::InvalidDataset(format!("transformer config.json has no {name}"))
        };

        let num_heads =
            number("num_attention_heads").ok_or_else(|| missing("num_attention_heads"))?;
        let head_dim = number("attention_head_dim").ok_or_else(|| missing("attention_head_dim"))?;
        let in_channels = number("in_channels").ok_or_else(|| missing("in_channels"))?;
        let axes_dim = match json
            .get("axes_dims_rope")
            .and_then(|value| value.as_array())
        {
            Some(axes) => axes
                .iter()
                .map(|axis| axis.as_u64().map(|axis| axis as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| missing("a numeric axes_dims_rope"))?,
            // Two axes for the image and nothing else is the arrangement a
            // model without the entry is using.
            None => vec![head_dim / 2, head_dim - head_dim / 2],
        };

        Ok(Self {
            in_channels,
            out_channels: number("out_channels").unwrap_or(in_channels),
            d_model: head_dim * num_heads,
            num_heads,
            double_blocks: number("num_layers").ok_or_else(|| missing("num_layers"))?,
            single_blocks: number("num_single_layers").unwrap_or(0),
            mlp_ratio: json
                .get("mlp_ratio")
                .and_then(|value| value.as_f64())
                .unwrap_or(4.0) as f32,
            text_dim: number("joint_attention_dim")
                .ok_or_else(|| missing("joint_attention_dim"))?,
            pooled_dim: number("pooled_projection_dim").unwrap_or(0),
            axes_dim,
            theta: json
                .get("rope_theta")
                .and_then(|value| value.as_f64())
                .unwrap_or(10_000.0) as f32,
            guidance: json
                .get("guidance_embeds")
                .and_then(|value| value.as_bool())
                .unwrap_or(false),
            eps: 1e-6,
        })
    }

    /// Dimensions per attention head.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads
    }

    /// Feed-forward width.
    pub fn mlp_dim(&self) -> usize {
        (self.d_model as f32 * self.mlp_ratio) as usize
    }

    fn check(&self) -> Result<(), NetworkError> {
        if self.num_heads == 0 || self.d_model % self.num_heads != 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "{} heads do not divide {} model dimensions",
                self.num_heads, self.d_model
            )));
        }
        if self.axes_dim.iter().any(|dim| dim % 2 != 0)
            || self.axes_dim.iter().sum::<usize>() != self.head_dim()
        {
            return Err(NetworkError::InvalidConfig(format!(
                "the position axes {:?} must be even and sum to the head's {} dimensions",
                self.axes_dim,
                self.head_dim()
            )));
        }
        Ok(())
    }
}

/// What the prompt contributes: the text encoder's tokens, the pooled vector
/// that rides with the timestep, and the shape of the image being made.
#[derive(Clone, Debug)]
pub struct Conditioning {
    /// `[tokens, text_dim]` from the text encoder.
    pub text: Matrix,
    /// The pooled prompt vector, `[pooled_dim]`.
    pub pooled: Vec<f32>,
    /// The latent's patch grid, which is what the rotary positions count over.
    pub height: usize,
    pub width: usize,
    /// The distilled guidance scale, for a model that takes one.
    pub guidance: f32,
}

/// The rotation table for one sequence: a cosine and a sine per head dimension
/// pair per token.
struct Positions {
    cos: Vec<f32>,
    sin: Vec<f32>,
    pairs: usize,
}

impl Positions {
    /// Text tokens at the origin, then one entry per patch in row-major order.
    fn new(config: &DitConfig, text_len: usize, height: usize, width: usize) -> Self {
        let pairs = config.head_dim() / 2;
        let tokens = text_len + height * width;
        let mut cos = vec![0.0; tokens * pairs];
        let mut sin = vec![0.0; tokens * pairs];

        for token in 0..tokens {
            // Axes beyond the last two are the ones FLUX keeps at zero, so
            // only the final two carry the patch's row and column.
            let (row, column) = if token < text_len {
                (0.0, 0.0)
            } else {
                let patch = token - text_len;
                ((patch / width) as f32, (patch % width) as f32)
            };

            let mut pair = 0;
            for (axis, dim) in config.axes_dim.iter().enumerate() {
                let position = match config.axes_dim.len() - axis {
                    2 => row,
                    1 => column,
                    _ => 0.0,
                };
                for index in 0..dim / 2 {
                    let omega = config.theta.powf(-2.0 * index as f32 / *dim as f32);
                    let angle = position * omega;
                    cos[token * pairs + pair] = angle.cos();
                    sin[token * pairs + pair] = angle.sin();
                    pair += 1;
                }
            }
        }

        Self { cos, sin, pairs }
    }

    /// Rotates every head of a `[tokens, heads * head_dim]` matrix in place.
    fn apply(&self, values: &mut Matrix, num_heads: usize) {
        let head_dim = self.pairs * 2;
        values
            .data
            .par_chunks_mut(num_heads * head_dim)
            .enumerate()
            .for_each(|(token, row)| {
                let (cos, sin) = (
                    &self.cos[token * self.pairs..(token + 1) * self.pairs],
                    &self.sin[token * self.pairs..(token + 1) * self.pairs],
                );
                for head in 0..num_heads {
                    let head = &mut row[head * head_dim..(head + 1) * head_dim];
                    for pair in 0..self.pairs {
                        let (first, second) = (head[pair * 2], head[pair * 2 + 1]);
                        head[pair * 2] = first * cos[pair] - second * sin[pair];
                        head[pair * 2 + 1] = first * sin[pair] + second * cos[pair];
                    }
                }
            });
    }
}

/// The per-head RMS normalization every modern denoiser puts on its queries
/// and keys, which is what keeps attention logits from drifting at scale.
#[derive(Clone, Debug)]
struct HeadNorm {
    scale: Vec<f32>,
}

impl HeadNorm {
    fn apply(&self, values: &mut Matrix, num_heads: usize) {
        let head_dim = self.scale.len();
        values
            .data
            .par_chunks_mut(num_heads * head_dim)
            .for_each(|row| {
                for head in 0..num_heads {
                    let head = &mut row[head * head_dim..(head + 1) * head_dim];
                    let mean_square =
                        head.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
                    let inverse = (mean_square + 1e-6).sqrt().recip();
                    for (value, scale) in head.iter_mut().zip(&self.scale) {
                        *value *= inverse * scale;
                    }
                }
            });
    }
}

/// Attention over one sequence.
///
/// `kv_heads` below `num_heads` is grouped-query attention, where several
/// query heads share one key and value head. `causal` masks every key after
/// the query, which a text decoder needs and an image does not: an image has
/// no reading order.
pub(crate) fn attention(
    queries: &Matrix,
    keys: &Matrix,
    values: &Matrix,
    num_heads: usize,
    kv_heads: usize,
    causal: bool,
) -> Matrix {
    let tokens = queries.rows;
    // Cross-attention reads a different sequence from the one it writes, so the
    // keys are counted on their own.
    let context = keys.rows;
    let head_dim = queries.cols / num_heads;
    let group = num_heads / kv_heads.max(1);
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = Matrix::new(tokens, queries.cols);

    let heads: Vec<Vec<f32>> = (0..num_heads)
        .into_par_iter()
        .map(|head| {
            let offset = head * head_dim;
            let kv_offset = head / group * head_dim;
            let mut attended = vec![0.0; tokens * head_dim];
            let mut weights = vec![0.0; context];
            for query_token in 0..tokens {
                let query = &queries.row(query_token)[offset..offset + head_dim];
                let visible = if causal { query_token + 1 } else { context };
                let mut largest = f32::NEG_INFINITY;
                for (key_token, weight) in weights[..visible].iter_mut().enumerate() {
                    let key = &keys.row(key_token)[kv_offset..kv_offset + head_dim];
                    let score = query
                        .iter()
                        .zip(key)
                        .map(|(query, key)| query * key)
                        .sum::<f32>()
                        * scale;
                    *weight = score;
                    largest = largest.max(score);
                }
                let mut total = 0.0;
                for weight in weights[..visible].iter_mut() {
                    *weight = (*weight - largest).exp();
                    total += *weight;
                }
                let target = &mut attended[query_token * head_dim..(query_token + 1) * head_dim];
                for (key_token, weight) in weights[..visible].iter().enumerate() {
                    let weight = weight / total;
                    let value = &values.row(key_token)[kv_offset..kv_offset + head_dim];
                    for (target, value) in target.iter_mut().zip(value) {
                        *target += weight * value;
                    }
                }
            }
            attended
        })
        .collect();

    for (head, attended) in heads.iter().enumerate() {
        for token in 0..tokens {
            let target = &mut output.row_mut(token)[head * head_dim..(head + 1) * head_dim];
            target.copy_from_slice(&attended[token * head_dim..(token + 1) * head_dim]);
        }
    }
    output
}

/// Layer norm with no learned scale: the scale arrives from the timestep
/// instead, which is the whole point of adaptive layer norm.
pub(crate) fn layer_norm(tokens: &Matrix, eps: f32) -> Matrix {
    let mut output = tokens.clone();
    output.data.par_chunks_mut(tokens.cols).for_each(|row| {
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let variance =
            row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / row.len() as f32;
        let inverse = (variance + eps).sqrt().recip();
        for value in row.iter_mut() {
            *value = (*value - mean) * inverse;
        }
    });
    output
}

/// `x * (1 + scale) + shift`, the modulation an adaptive norm applies.
fn modulate(tokens: &Matrix, shift: &[f32], scale: &[f32]) -> Matrix {
    let mut output = tokens.clone();
    output.data.par_chunks_mut(tokens.cols).for_each(|row| {
        for ((value, shift), scale) in row.iter_mut().zip(shift).zip(scale) {
            *value = *value * (1.0 + scale) + shift;
        }
    });
    output
}

/// `target += gate * source`, the gated residual add every block ends with.
fn gated_add(target: &mut Matrix, source: &Matrix, gate: &[f32]) {
    let width = target.cols;
    target
        .data
        .par_chunks_mut(width)
        .zip(source.data.par_chunks(width))
        .for_each(|(target, source)| {
            for ((target, source), gate) in target.iter_mut().zip(source).zip(gate) {
                *target += gate * source;
            }
        });
}

/// The sinusoidal encoding of a noise level, which the timestep embedder
/// reads.
fn timestep_embedding(timestep: f32, dim: usize) -> Vec<f32> {
    // The published models scale the flow-matching time by a thousand before
    // embedding it, so the same table covers the 1000-step discrete schedules.
    sinusoid(timestep * 1000.0, dim)
}

/// The sinusoidal table itself, cosines first, over a time that is already on
/// the scale the model counts in.
pub(crate) fn sinusoid(time: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut embedding = vec![0.0; dim];
    for index in 0..half {
        let frequency = (-(10_000.0f32).ln() * index as f32 / half as f32).exp();
        embedding[index] = (time * frequency).cos();
        embedding[half + index] = (time * frequency).sin();
    }
    embedding
}

/// The two-layer embedder that turns a vector into a conditioning signal.
#[derive(Clone, Debug)]
struct Embedder {
    input: Dense,
    output: Dense,
}

impl Embedder {
    fn forward(&self, values: &[f32]) -> Result<Vec<f32>, NetworkError> {
        let hidden = self.input.apply(values)?;
        let activated: Vec<f32> = hidden.iter().map(|value| silu(*value)).collect();
        self.output.apply(&activated)
    }
}

/// One stream's modulation: a projection from the conditioning vector into as
/// many `(shift, scale, gate)` triples as the block needs.
#[derive(Clone, Debug)]
struct Modulation {
    projection: Dense,
    chunks: usize,
}

impl Modulation {
    /// Triples of `(shift, scale, gate)`, in the checkpoint's order.
    fn forward(&self, vector: &[f32]) -> Result<Vec<[Vec<f32>; 3]>, NetworkError> {
        let activated: Vec<f32> = vector.iter().map(|value| silu(*value)).collect();
        let values = self.projection.apply(&activated)?;
        let width = values.len() / (self.chunks * 3);
        Ok((0..self.chunks)
            .map(|chunk| {
                let base = chunk * 3 * width;
                [
                    values[base..base + width].to_vec(),
                    values[base + width..base + 2 * width].to_vec(),
                    values[base + 2 * width..base + 3 * width].to_vec(),
                ]
            })
            .collect())
    }
}

/// Image and text, separate weights, one attention.
#[derive(Clone, Debug)]
struct DoubleBlock {
    image: Stream,
    text: Stream,
}

/// One side of a double-stream block.
#[derive(Clone, Debug)]
struct Stream {
    modulation: Modulation,
    qkv: Dense,
    query_norm: HeadNorm,
    key_norm: HeadNorm,
    projection: Dense,
    mlp_in: Dense,
    mlp_out: Dense,
}

impl Stream {
    /// The modulated queries, keys and values this stream contributes.
    fn project(
        &self,
        tokens: &Matrix,
        shift: &[f32],
        scale: &[f32],
        num_heads: usize,
        eps: f32,
    ) -> Result<(Matrix, Matrix, Matrix), NetworkError> {
        let normed = modulate(&layer_norm(tokens, eps), shift, scale);
        let qkv = self.qkv.forward(&normed)?;
        let width = qkv.cols / 3;
        let mut parts = (0..3)
            .map(|part| {
                let mut slice = Matrix::new(qkv.rows, width);
                for token in 0..qkv.rows {
                    slice
                        .row_mut(token)
                        .copy_from_slice(&qkv.row(token)[part * width..(part + 1) * width]);
                }
                slice
            })
            .collect::<Vec<_>>();
        let values = parts.pop().expect("three parts");
        let mut keys = parts.pop().expect("three parts");
        let mut queries = parts.pop().expect("three parts");
        self.query_norm.apply(&mut queries, num_heads);
        self.key_norm.apply(&mut keys, num_heads);
        Ok((queries, keys, values))
    }

    /// The feed-forward half, which runs after the attention.
    fn feed_forward(
        &self,
        tokens: &Matrix,
        shift: &[f32],
        scale: &[f32],
        eps: f32,
    ) -> Result<Matrix, NetworkError> {
        let normed = modulate(&layer_norm(tokens, eps), shift, scale);
        let mut hidden = self.mlp_in.forward(&normed)?;
        hidden
            .data
            .par_iter_mut()
            .for_each(|value| *value = gelu(*value));
        self.mlp_out.forward(&hidden)
    }
}

/// One set of weights over the concatenated sequence, with the attention and
/// the feed-forward sharing a projection.
#[derive(Clone, Debug)]
struct SingleBlock {
    modulation: Modulation,
    fused_in: Dense,
    fused_out: Dense,
    query_norm: HeadNorm,
    key_norm: HeadNorm,
}

/// The denoising transformer.
pub struct Dit {
    config: DitConfig,
    image_in: Dense,
    text_in: Dense,
    time_in: Embedder,
    /// The pooled-prompt embedder, which the models that take no pooled
    /// vector do not have.
    vector_in: Option<Embedder>,
    guidance_in: Option<Embedder>,
    double: Vec<DoubleBlock>,
    single: Vec<SingleBlock>,
    final_modulation: Dense,
    final_out: Dense,
    conditioning: Option<Conditioning>,
    unconditional: Option<Conditioning>,
}

impl Dit {
    /// The configuration this model was built for.
    pub fn config(&self) -> &DitConfig {
        &self.config
    }

    /// The width of the pooled prompt vector this checkpoint takes, if it
    /// takes one at all. Read from the weights rather than the configuration,
    /// because it is the weights that have to match.
    pub fn pooled_dim(&self) -> Option<usize> {
        self.vector_in
            .as_ref()
            .map(|embedder| embedder.input.in_dim())
    }

    /// Hands the model the prompt it should denoise towards.
    ///
    /// [`Denoiser::denoise`] reads it, so this has to be called before a
    /// sampling run and again for every new prompt.
    pub fn set_conditioning(&mut self, conditioning: Conditioning) {
        self.conditioning = Some(conditioning);
    }

    /// Hands it the prompt to push away from, which is the empty one for
    /// classifier-free guidance.
    ///
    /// A distilled model needs none: it was trained with the guidance already
    /// in its weights, and a second pass per step would halve the speed for
    /// nothing.
    pub fn set_unconditional(&mut self, conditioning: Conditioning) {
        self.unconditional = Some(conditioning);
    }

    /// Reads a denoiser out of a checkpoint under the key names FLUX ships
    /// with.
    ///
    /// `path` is a `.safetensors` file or the index of a sharded one, and
    /// `prefix` is what the keys start with — empty for a standalone
    /// transformer file, `"model.diffusion_model."` inside a full pipeline.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        prefix: &str,
        config: DitConfig,
    ) -> Result<Self, NetworkError> {
        Self::load_at(path, prefix, config, Precision::F32)
    }

    /// The same, holding every projection at whatever precision is asked for.
    ///
    /// [`Precision::Q8`] is a quarter of the memory and about a fifth slower
    /// per step on this machine, because the byte path multiplies one token at
    /// a time where the float path multiplies a block. It is what makes a
    /// model that does not fit run at all.
    pub fn load_at(
        path: impl AsRef<std::path::Path>,
        prefix: &str,
        config: DitConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        Self::read_at(&mut file, prefix, config, precision)
    }

    /// The same, from a checkpoint that is already open.
    ///
    /// Both published spellings are read: the one Black Forest Labs ships,
    /// and the one diffusers repacks the same weights under. Which is in the
    /// file is a question the file answers, so it is not asked of the caller.
    pub fn read(
        file: &mut ShardedSafeTensors,
        prefix: &str,
        config: DitConfig,
    ) -> Result<Self, NetworkError> {
        Self::read_at(file, prefix, config, Precision::F32)
    }

    /// The same, at a chosen precision.
    pub fn read_at(
        file: &mut ShardedSafeTensors,
        prefix: &str,
        config: DitConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        config.check()?;
        let layout = Layout::detect(file, prefix);
        let embedder =
            |file: &mut ShardedSafeTensors, name: &str| -> Result<Embedder, NetworkError> {
                let (input, output) = layout.embedder_layers();
                Ok(Embedder {
                    input: dense(file, &format!("{name}.{input}"), precision)?,
                    output: dense(file, &format!("{name}.{output}"), precision)?,
                })
            };

        let double = (0..config.double_blocks)
            .map(|index| {
                let base = format!("{prefix}{}.{index}", layout.double_blocks());
                Ok(DoubleBlock {
                    image: read_stream(file, &base, layout, true, precision)?,
                    text: read_stream(file, &base, layout, false, precision)?,
                })
            })
            .collect::<Result<Vec<_>, NetworkError>>()?;

        let single = (0..config.single_blocks)
            .map(|index| {
                let base = format!("{prefix}{}.{index}", layout.single_blocks());
                read_single(file, &base, layout, precision)
            })
            .collect::<Result<Vec<_>, NetworkError>>()?;

        let image_in = dense(file, &format!("{prefix}{}", layout.image_in()), precision)?;
        if image_in.in_dim() != config.in_channels || image_in.out_dim() != config.d_model {
            return Err(NetworkError::InvalidConfig(format!(
                "the checkpoint's patch embedding maps {} channels to {}, and the configuration \
                 says {} to {}",
                image_in.in_dim(),
                image_in.out_dim(),
                config.in_channels,
                config.d_model
            )));
        }

        let mut final_modulation = dense(
            file,
            &format!("{prefix}{}", layout.final_modulation()),
            precision,
        )?;
        if layout.final_scale_first() {
            swap_halves(&mut final_modulation);
        }

        Ok(Self {
            image_in,
            text_in: dense(file, &format!("{prefix}{}", layout.text_in()), precision)?,
            time_in: embedder(file, &format!("{prefix}{}", layout.time_in()))?,
            // A model with no pooled prompt has no embedder for one, and the
            // absence is read from the file rather than configured.
            vector_in: embedder(file, &format!("{prefix}{}", layout.vector_in())).ok(),
            guidance_in: match config.guidance {
                true => Some(embedder(
                    file,
                    &format!("{prefix}{}", layout.guidance_in()),
                )?),
                false => None,
            },
            double,
            single,
            final_modulation,
            final_out: dense(file, &format!("{prefix}{}", layout.final_out()), precision)?,
            config,
            conditioning: None,
            unconditional: None,
        })
    }

    /// Stores every projection at one byte per value, a quarter of the
    /// memory.
    ///
    /// Inference only and one-way: the `f32` weights are dropped. The
    /// normalization scales and the biases stay as they are, because they are
    /// a thousandth of the model and the rounding shows in them.
    pub fn quantize(&mut self) {
        let mut layers: Vec<&mut Dense> = vec![
            &mut self.image_in,
            &mut self.text_in,
            &mut self.time_in.input,
            &mut self.time_in.output,
            &mut self.final_modulation,
            &mut self.final_out,
        ];
        for embedder in [self.vector_in.as_mut(), self.guidance_in.as_mut()]
            .into_iter()
            .flatten()
        {
            layers.push(&mut embedder.input);
            layers.push(&mut embedder.output);
        }
        for block in &mut self.double {
            for stream in [&mut block.image, &mut block.text] {
                layers.extend([
                    &mut stream.modulation.projection,
                    &mut stream.qkv,
                    &mut stream.projection,
                    &mut stream.mlp_in,
                    &mut stream.mlp_out,
                ]);
            }
        }
        for block in &mut self.single {
            layers.extend([
                &mut block.modulation.projection,
                &mut block.fused_in,
                &mut block.fused_out,
            ]);
        }
        layers.into_par_iter().for_each(Dense::quantize);
    }

    /// One forward pass: patches in, predicted velocity out.
    ///
    /// `patches` is `[height * width, in_channels]`, the latent packed into the
    /// patch grid the conditioning names.
    pub fn forward(&self, patches: &Matrix, timestep: f32) -> Result<Matrix, NetworkError> {
        let conditioning = self.conditioning.as_ref().ok_or_else(|| {
            NetworkError::InvalidConfig(
                "this model has no prompt: call `set_conditioning` before sampling".into(),
            )
        })?;
        self.forward_with(patches, timestep, conditioning)
    }

    /// The same pass against a conditioning handed in rather than the one the
    /// model is holding, which is how the unconditional half of a guided step
    /// runs.
    pub fn forward_with(
        &self,
        patches: &Matrix,
        timestep: f32,
        conditioning: &Conditioning,
    ) -> Result<Matrix, NetworkError> {
        if patches.rows != conditioning.height * conditioning.width {
            return Err(NetworkError::InvalidTarget {
                expected: conditioning.height * conditioning.width,
                actual: patches.rows,
            });
        }

        // The conditioning vector: noise level first, then the pooled prompt,
        // then the distilled guidance scale if the model takes one.
        let mut vector = self
            .time_in
            .forward(&timestep_embedding(timestep, self.time_in.input.in_dim()))?;
        if let Some(vector_in) = &self.vector_in {
            for (value, pooled) in vector
                .iter_mut()
                .zip(vector_in.forward(&conditioning.pooled)?)
            {
                *value += pooled;
            }
        }
        if let Some(guidance) = &self.guidance_in {
            // The distilled scale goes through the same sinusoidal table as
            // the noise level, with the same thousandfold scaling, so it is
            // handed over as it is.
            let embedded = guidance.forward(&timestep_embedding(
                conditioning.guidance,
                guidance.input.in_dim(),
            ))?;
            for (value, guidance) in vector.iter_mut().zip(embedded) {
                *value += guidance;
            }
        }

        let mut image = self.image_in.forward(patches)?;
        let mut text = self.text_in.forward(&conditioning.text)?;
        let positions = Positions::new(
            &self.config,
            text.rows,
            conditioning.height,
            conditioning.width,
        );
        let heads = self.config.num_heads;
        let eps = self.config.eps;

        for block in &self.double {
            let image_mod = block.image.modulation.forward(&vector)?;
            let text_mod = block.text.modulation.forward(&vector)?;
            let (image_queries, image_keys, image_values) =
                block
                    .image
                    .project(&image, &image_mod[0][0], &image_mod[0][1], heads, eps)?;
            let (text_queries, text_keys, text_values) =
                block
                    .text
                    .project(&text, &text_mod[0][0], &text_mod[0][1], heads, eps)?;

            // Text first, then image: the order the rotary table was built in.
            let mut queries = concatenate(&text_queries, &image_queries);
            let mut keys = concatenate(&text_keys, &image_keys);
            let values = concatenate(&text_values, &image_values);
            positions.apply(&mut queries, heads);
            positions.apply(&mut keys, heads);
            let attended = attention(&queries, &keys, &values, heads, heads, false);
            let (text_attended, image_attended) = split(&attended, text.rows);

            gated_add(
                &mut image,
                &block.image.projection.forward(&image_attended)?,
                &image_mod[0][2],
            );
            gated_add(
                &mut text,
                &block.text.projection.forward(&text_attended)?,
                &text_mod[0][2],
            );
            let image_mlp =
                block
                    .image
                    .feed_forward(&image, &image_mod[1][0], &image_mod[1][1], eps)?;
            gated_add(&mut image, &image_mlp, &image_mod[1][2]);
            let text_mlp = block
                .text
                .feed_forward(&text, &text_mod[1][0], &text_mod[1][1], eps)?;
            gated_add(&mut text, &text_mlp, &text_mod[1][2]);
        }

        let mut hidden = concatenate(&text, &image);
        let mlp_dim = self.config.mlp_dim();
        for block in &self.single {
            let modulation = block.modulation.forward(&vector)?;
            let (shift, scale, gate) = (&modulation[0][0], &modulation[0][1], &modulation[0][2]);
            let normed = modulate(&layer_norm(&hidden, eps), shift, scale);
            let fused = block.fused_in.forward(&normed)?;

            let width = self.config.d_model;
            let mut queries = Matrix::new(fused.rows, width);
            let mut keys = Matrix::new(fused.rows, width);
            let mut values = Matrix::new(fused.rows, width);
            let mut mlp = Matrix::new(fused.rows, mlp_dim);
            for token in 0..fused.rows {
                let row = fused.row(token);
                queries.row_mut(token).copy_from_slice(&row[..width]);
                keys.row_mut(token).copy_from_slice(&row[width..2 * width]);
                values
                    .row_mut(token)
                    .copy_from_slice(&row[2 * width..3 * width]);
                mlp.row_mut(token)
                    .copy_from_slice(&row[3 * width..3 * width + mlp_dim]);
            }
            block.query_norm.apply(&mut queries, heads);
            block.key_norm.apply(&mut keys, heads);
            positions.apply(&mut queries, heads);
            positions.apply(&mut keys, heads);
            let attended = attention(&queries, &keys, &values, heads, heads, false);

            mlp.data
                .par_iter_mut()
                .for_each(|value| *value = gelu(*value));
            let output = block
                .fused_out
                .forward(&concatenate_columns(&attended, &mlp))?;
            gated_add(&mut hidden, &output, gate);
        }

        let (_, image) = split(&hidden, text.rows);
        let modulation = self
            .final_modulation
            .apply(&vector.iter().map(|value| silu(*value)).collect::<Vec<_>>())?;
        // This one is shift then scale, with no gate.
        let width = modulation.len() / 2;
        let normed = modulate(
            &layer_norm(&image, eps),
            &modulation[..width],
            &modulation[width..],
        );
        self.final_out.forward(&normed)
    }
}

impl Denoiser for Dit {
    fn denoise(&mut self, latents: &Matrix, sigma: f32) -> Result<Matrix, NetworkError> {
        // A rectified-flow model's timestep is the noise level itself.
        self.forward(latents, sigma)
    }

    fn denoise_unconditional(
        &mut self,
        latents: &Matrix,
        sigma: f32,
    ) -> Result<Option<Matrix>, NetworkError> {
        match &self.unconditional {
            Some(conditioning) => Ok(Some(self.forward_with(latents, sigma, conditioning)?)),
            None => Ok(None),
        }
    }
}

/// Two sequences, one after the other.
fn concatenate(first: &Matrix, second: &Matrix) -> Matrix {
    let mut data = Vec::with_capacity(first.data.len() + second.data.len());
    data.extend_from_slice(&first.data);
    data.extend_from_slice(&second.data);
    Matrix::from_vec(first.rows + second.rows, first.cols, data)
}

/// Two matrices of the same height, side by side.
fn concatenate_columns(left: &Matrix, right: &Matrix) -> Matrix {
    let mut output = Matrix::new(left.rows, left.cols + right.cols);
    for row in 0..left.rows {
        let target = output.row_mut(row);
        target[..left.cols].copy_from_slice(left.row(row));
        target[left.cols..].copy_from_slice(right.row(row));
    }
    output
}

/// The inverse of [`concatenate`].
fn split(tokens: &Matrix, first: usize) -> (Matrix, Matrix) {
    let cut = first * tokens.cols;
    (
        Matrix::from_vec(first, tokens.cols, tokens.data[..cut].to_vec()),
        Matrix::from_vec(
            tokens.rows - first,
            tokens.cols,
            tokens.data[cut..].to_vec(),
        ),
    )
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
        // Quantizing as each tensor arrives is what keeps the peak at one
        // `f32` tensor rather than the whole model.
        layer.quantize();
    }
    Ok(layer)
}

fn head_norm(file: &mut ShardedSafeTensors, name: &str) -> Result<HeadNorm, NetworkError> {
    let (scale, _) = file.tensor(name)?;
    Ok(HeadNorm { scale })
}

/// Which spelling of the same weights a checkpoint uses.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Layout {
    /// The reference implementation's names, which the original checkpoints
    /// carry.
    Reference,
    /// The names diffusers repacks them under.
    Diffusers,
}

impl Layout {
    /// Reads the layout off the file, by the name its blocks go by.
    fn detect(file: &ShardedSafeTensors, prefix: &str) -> Self {
        let blocks = format!("{prefix}transformer_blocks.");
        match file.names().any(|name| name.starts_with(&blocks)) {
            true => Self::Diffusers,
            false => Self::Reference,
        }
    }

    fn double_blocks(self) -> &'static str {
        match self {
            Self::Reference => "double_blocks",
            Self::Diffusers => "transformer_blocks",
        }
    }

    fn single_blocks(self) -> &'static str {
        match self {
            Self::Reference => "single_blocks",
            Self::Diffusers => "single_transformer_blocks",
        }
    }

    fn embedder_layers(self) -> (&'static str, &'static str) {
        match self {
            Self::Reference => ("in_layer", "out_layer"),
            Self::Diffusers => ("linear_1", "linear_2"),
        }
    }

    fn image_in(self) -> &'static str {
        match self {
            Self::Reference => "img_in",
            Self::Diffusers => "x_embedder",
        }
    }

    fn text_in(self) -> &'static str {
        match self {
            Self::Reference => "txt_in",
            Self::Diffusers => "context_embedder",
        }
    }

    fn time_in(self) -> &'static str {
        match self {
            Self::Reference => "time_in",
            Self::Diffusers => "time_text_embed.timestep_embedder",
        }
    }

    fn vector_in(self) -> &'static str {
        match self {
            Self::Reference => "vector_in",
            Self::Diffusers => "time_text_embed.text_embedder",
        }
    }

    fn guidance_in(self) -> &'static str {
        match self {
            Self::Reference => "guidance_in",
            Self::Diffusers => "time_text_embed.guidance_embedder",
        }
    }

    fn final_modulation(self) -> &'static str {
        match self {
            Self::Reference => "final_layer.adaLN_modulation.1",
            Self::Diffusers => "norm_out.linear",
        }
    }

    fn final_out(self) -> &'static str {
        match self {
            Self::Reference => "final_layer.linear",
            Self::Diffusers => "proj_out",
        }
    }

    /// Whether the final modulation emits the scale before the shift, which
    /// diffusers does and the reference implementation does not.
    fn final_scale_first(self) -> bool {
        self == Self::Diffusers
    }
}

/// Puts the second half of a projection's outputs first, which is how one
/// spelling's (scale, shift) becomes the other's (shift, scale).
fn swap_halves(layer: &mut Dense) {
    let half = layer.weight.rows / 2;
    let width = layer.weight.cols;
    layer.weight.data.rotate_left(half * width);
    if let Some(bias) = &mut layer.bias {
        bias.rotate_left(half);
    }
}

fn read_stream(
    file: &mut ShardedSafeTensors,
    base: &str,
    layout: Layout,
    image: bool,
    precision: Precision,
) -> Result<Stream, NetworkError> {
    let names: [&str; 8] = match (layout, image) {
        (Layout::Reference, true) => [
            "img_mod.lin",
            "img_attn.qkv",
            "",
            "",
            "img_attn.norm.query_norm.scale",
            "img_attn.norm.key_norm.scale",
            "img_attn.proj",
            "img_mlp",
        ],
        (Layout::Reference, false) => [
            "txt_mod.lin",
            "txt_attn.qkv",
            "",
            "",
            "txt_attn.norm.query_norm.scale",
            "txt_attn.norm.key_norm.scale",
            "txt_attn.proj",
            "txt_mlp",
        ],
        (Layout::Diffusers, true) => [
            "norm1.linear",
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.norm_q.weight",
            "attn.norm_k.weight",
            "attn.to_out.0",
            "ff.net",
        ],
        (Layout::Diffusers, false) => [
            "norm1_context.linear",
            "attn.add_q_proj",
            "attn.add_k_proj",
            "attn.add_v_proj",
            "attn.norm_added_q.weight",
            "attn.norm_added_k.weight",
            "attn.to_add_out",
            "ff_context.net",
        ],
    };
    let [
        modulation,
        queries,
        keys,
        values,
        query_norm,
        key_norm,
        projection,
        mlp,
    ] = names;
    let (mlp_in, mlp_out) = match layout {
        Layout::Reference => (format!("{mlp}.0"), format!("{mlp}.2")),
        Layout::Diffusers => (format!("{mlp}.0.proj"), format!("{mlp}.2")),
    };

    Ok(Stream {
        modulation: Modulation {
            projection: dense(file, &format!("{base}.{modulation}"), precision)?,
            chunks: 2,
        },
        qkv: match layout {
            Layout::Reference => dense(file, &format!("{base}.{queries}"), precision)?,
            // Three projections written separately are the one fused
            // projection this model runs, stacked in query, key, value order.
            Layout::Diffusers => stacked(file, base, &[queries, keys, values], precision)?,
        },
        query_norm: head_norm(file, &format!("{base}.{query_norm}"))?,
        key_norm: head_norm(file, &format!("{base}.{key_norm}"))?,
        projection: dense(file, &format!("{base}.{projection}"), precision)?,
        mlp_in: dense(file, &format!("{base}.{mlp_in}"), precision)?,
        mlp_out: dense(file, &format!("{base}.{mlp_out}"), precision)?,
    })
}

fn read_single(
    file: &mut ShardedSafeTensors,
    base: &str,
    layout: Layout,
    precision: Precision,
) -> Result<SingleBlock, NetworkError> {
    Ok(match layout {
        Layout::Reference => SingleBlock {
            modulation: Modulation {
                projection: dense(file, &format!("{base}.modulation.lin"), precision)?,
                chunks: 1,
            },
            fused_in: dense(file, &format!("{base}.linear1"), precision)?,
            fused_out: dense(file, &format!("{base}.linear2"), precision)?,
            query_norm: head_norm(file, &format!("{base}.norm.query_norm.scale"))?,
            key_norm: head_norm(file, &format!("{base}.norm.key_norm.scale"))?,
        },
        Layout::Diffusers => SingleBlock {
            modulation: Modulation {
                projection: dense(file, &format!("{base}.norm.linear"), precision)?,
                chunks: 1,
            },
            // Attention and feed-forward share one projection here too, and
            // the feed-forward's half is written after the values.
            fused_in: stacked(
                file,
                base,
                &["attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp"],
                precision,
            )?,
            fused_out: dense(file, &format!("{base}.proj_out"), precision)?,
            query_norm: head_norm(file, &format!("{base}.attn.norm_q.weight"))?,
            key_norm: head_norm(file, &format!("{base}.attn.norm_k.weight"))?,
        },
    })
}

/// Several projections of the same input, read as the one wider projection
/// they add up to.
fn stacked(
    file: &mut ShardedSafeTensors,
    base: &str,
    names: &[&str],
    precision: Precision,
) -> Result<Dense, NetworkError> {
    let parts = names
        .iter()
        .map(|name| dense(file, &format!("{base}.{name}"), Precision::F32))
        .collect::<Result<Vec<_>, NetworkError>>()?;
    let width = parts[0].in_dim();
    if parts.iter().any(|part| part.in_dim() != width) {
        return Err(NetworkError::InvalidConfig(format!(
            "{base}: the projections {names:?} read the same input but disagree about its width"
        )));
    }

    let rows = parts.iter().map(Dense::out_dim).sum();
    let mut weight = Matrix::new(rows, width);
    let mut bias = Vec::with_capacity(rows);
    let mut cursor = 0;
    for part in &parts {
        let end = cursor + part.out_dim() * width;
        weight.data[cursor..end].copy_from_slice(&part.weight.data);
        cursor = end;
        match &part.bias {
            Some(values) => bias.extend_from_slice(values),
            // One projection without a bias among others that have one is a
            // zero bias, not a missing one.
            None => bias.resize(bias.len() + part.out_dim(), 0.0),
        }
    }
    let bias = parts.iter().any(|part| part.bias.is_some()).then_some(bias);
    let mut layer = Dense::new(weight, bias)?;
    if precision == Precision::Q8 {
        layer.quantize();
    }
    Ok(layer)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::diffusion::{SamplingConfig, Scheduler, sample};
    use std::collections::BTreeMap;

    fn tiny() -> DitConfig {
        DitConfig {
            in_channels: 4,
            out_channels: 4,
            d_model: 16,
            num_heads: 2,
            double_blocks: 2,
            single_blocks: 2,
            mlp_ratio: 2.0,
            text_dim: 8,
            pooled_dim: 8,
            axes_dim: vec![4, 4],
            theta: 10_000.0,
            guidance: false,
            eps: 1e-6,
        }
    }

    /// A checkpoint holding every key the loader reads, at a size that runs in
    /// milliseconds.
    pub(crate) fn checkpoint(config: &DitConfig, path: &std::path::Path) {
        crate::safetensors::write_checkpoint(path, &tensors(config));
    }

    /// The same weights, under the reference implementation's names.
    fn tensors(config: &DitConfig) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let count = shape.iter().product::<usize>();
            let values = (0..count)
                .map(|index| ((index % 11) as f32 - 5.0) * 0.03)
                .collect();
            tensors.insert(name, (shape, values));
        };
        let linear =
            |name: &str, input: usize, output: usize, add: &mut dyn FnMut(String, Vec<usize>)| {
                add(format!("{name}.weight"), vec![output, input]);
                add(format!("{name}.bias"), vec![output]);
            };

        let (dim, mlp, head) = (config.d_model, config.mlp_dim(), config.head_dim());
        linear("img_in", config.in_channels, dim, &mut add);
        linear("txt_in", config.text_dim, dim, &mut add);
        linear("time_in.in_layer", 256, dim, &mut add);
        linear("time_in.out_layer", dim, dim, &mut add);
        linear("vector_in.in_layer", config.pooled_dim, dim, &mut add);
        linear("vector_in.out_layer", dim, dim, &mut add);
        if config.guidance {
            linear("guidance_in.in_layer", 256, dim, &mut add);
            linear("guidance_in.out_layer", dim, dim, &mut add);
        }

        for index in 0..config.double_blocks {
            let base = format!("double_blocks.{index}");
            for side in ["img", "txt"] {
                linear(&format!("{base}.{side}_mod.lin"), dim, 6 * dim, &mut add);
                linear(&format!("{base}.{side}_attn.qkv"), dim, 3 * dim, &mut add);
                add(
                    format!("{base}.{side}_attn.norm.query_norm.scale"),
                    vec![head],
                );
                add(
                    format!("{base}.{side}_attn.norm.key_norm.scale"),
                    vec![head],
                );
                linear(&format!("{base}.{side}_attn.proj"), dim, dim, &mut add);
                linear(&format!("{base}.{side}_mlp.0"), dim, mlp, &mut add);
                linear(&format!("{base}.{side}_mlp.2"), mlp, dim, &mut add);
            }
        }
        for index in 0..config.single_blocks {
            let base = format!("single_blocks.{index}");
            linear(&format!("{base}.modulation.lin"), dim, 3 * dim, &mut add);
            linear(&format!("{base}.linear1"), dim, 3 * dim + mlp, &mut add);
            linear(&format!("{base}.linear2"), dim + mlp, dim, &mut add);
            add(format!("{base}.norm.query_norm.scale"), vec![head]);
            add(format!("{base}.norm.key_norm.scale"), vec![head]);
        }
        linear("final_layer.adaLN_modulation.1", dim, 2 * dim, &mut add);
        linear("final_layer.linear", dim, config.out_channels, &mut add);

        tensors
    }

    /// The same weights again, under the names diffusers repacks them with:
    /// the fused projections split apart, and the final modulation written
    /// scale before shift.
    fn diffusers_tensors(config: &DitConfig) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let source = tensors(config);
        let mut out: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let take = |source: &BTreeMap<String, (Vec<usize>, Vec<f32>)>, name: &str| {
            source
                .get(name)
                .unwrap_or_else(|| panic!("{name} is in the reference checkpoint"))
                .clone()
        };
        let copy = |out: &mut BTreeMap<_, _>, from: &str, to: &str| {
            for suffix in ["weight", "bias"] {
                out.insert(
                    format!("{to}.{suffix}"),
                    take(&source, &format!("{from}.{suffix}")),
                );
            }
        };
        // A projection of several stacked outputs, cut back into its parts.
        let split = |out: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
                     from: &str,
                     parts: &[(&str, usize)]| {
            let (shape, values) = take(&source, &format!("{from}.weight"));
            let (_, bias) = take(&source, &format!("{from}.bias"));
            let width = shape[1];
            let mut cursor = 0;
            for (name, rows) in parts {
                out.insert(
                    format!("{name}.weight"),
                    (
                        vec![*rows, width],
                        values[cursor * width..(cursor + rows) * width].to_vec(),
                    ),
                );
                out.insert(
                    format!("{name}.bias"),
                    (vec![*rows], bias[cursor..cursor + rows].to_vec()),
                );
                cursor += rows;
            }
        };

        let (dim, mlp, head) = (config.d_model, config.mlp_dim(), config.head_dim());
        copy(&mut out, "img_in", "x_embedder");
        copy(&mut out, "txt_in", "context_embedder");
        copy(
            &mut out,
            "time_in.in_layer",
            "time_text_embed.timestep_embedder.linear_1",
        );
        copy(
            &mut out,
            "time_in.out_layer",
            "time_text_embed.timestep_embedder.linear_2",
        );
        copy(
            &mut out,
            "vector_in.in_layer",
            "time_text_embed.text_embedder.linear_1",
        );
        copy(
            &mut out,
            "vector_in.out_layer",
            "time_text_embed.text_embedder.linear_2",
        );

        for index in 0..config.double_blocks {
            let from = format!("double_blocks.{index}");
            let to = format!("transformer_blocks.{index}");
            for side in ["img", "txt"] {
                let image = side == "img";
                copy(
                    &mut out,
                    &format!("{from}.{side}_mod.lin"),
                    &match image {
                        true => format!("{to}.norm1.linear"),
                        false => format!("{to}.norm1_context.linear"),
                    },
                );
                let names: [String; 3] = match image {
                    true => ["to_q", "to_k", "to_v"].map(|part| format!("{to}.attn.{part}")),
                    false => ["add_q_proj", "add_k_proj", "add_v_proj"]
                        .map(|part| format!("{to}.attn.{part}")),
                };
                split(
                    &mut out,
                    &format!("{from}.{side}_attn.qkv"),
                    &names
                        .iter()
                        .map(|name| (name.as_str(), dim))
                        .collect::<Vec<_>>(),
                );
                for (part, diffusers) in [("query_norm", "norm_q"), ("key_norm", "norm_k")] {
                    let diffusers = match image {
                        true => format!("{to}.attn.{diffusers}.weight"),
                        false => format!("{to}.attn.{}.weight", diffusers.replace('_', "_added_")),
                    };
                    out.insert(
                        diffusers,
                        take(&source, &format!("{from}.{side}_attn.norm.{part}.scale")),
                    );
                }
                copy(
                    &mut out,
                    &format!("{from}.{side}_attn.proj"),
                    &match image {
                        true => format!("{to}.attn.to_out.0"),
                        false => format!("{to}.attn.to_add_out"),
                    },
                );
                let feed_forward = match image {
                    true => format!("{to}.ff.net"),
                    false => format!("{to}.ff_context.net"),
                };
                copy(
                    &mut out,
                    &format!("{from}.{side}_mlp.0"),
                    &format!("{feed_forward}.0.proj"),
                );
                copy(
                    &mut out,
                    &format!("{from}.{side}_mlp.2"),
                    &format!("{feed_forward}.2"),
                );
            }
        }

        for index in 0..config.single_blocks {
            let from = format!("single_blocks.{index}");
            let to = format!("single_transformer_blocks.{index}");
            copy(
                &mut out,
                &format!("{from}.modulation.lin"),
                &format!("{to}.norm.linear"),
            );
            split(
                &mut out,
                &format!("{from}.linear1"),
                &[
                    (&format!("{to}.attn.to_q"), dim),
                    (&format!("{to}.attn.to_k"), dim),
                    (&format!("{to}.attn.to_v"), dim),
                    (&format!("{to}.proj_mlp"), mlp),
                ],
            );
            copy(
                &mut out,
                &format!("{from}.linear2"),
                &format!("{to}.proj_out"),
            );
            for (part, diffusers) in [("query_norm", "norm_q"), ("key_norm", "norm_k")] {
                out.insert(
                    format!("{to}.attn.{diffusers}.weight"),
                    take(&source, &format!("{from}.norm.{part}.scale")),
                );
            }
        }
        assert_eq!(head, config.head_dim());

        // Scale before shift here, the other way round in the reference.
        let (shape, values) = take(&source, "final_layer.adaLN_modulation.1.weight");
        let (_, bias) = take(&source, "final_layer.adaLN_modulation.1.bias");
        let half = shape[0] / 2 * shape[1];
        out.insert(
            "norm_out.linear.weight".into(),
            (
                shape.clone(),
                values[half..]
                    .iter()
                    .chain(&values[..half])
                    .copied()
                    .collect(),
            ),
        );
        out.insert(
            "norm_out.linear.bias".into(),
            (
                vec![shape[0]],
                bias[shape[0] / 2..]
                    .iter()
                    .chain(&bias[..shape[0] / 2])
                    .copied()
                    .collect(),
            ),
        );
        copy(&mut out, "final_layer.linear", "proj_out");
        out
    }

    fn conditioning(config: &DitConfig, height: usize, width: usize) -> Conditioning {
        Conditioning {
            text: Matrix::from_vec(
                3,
                config.text_dim,
                (0..3 * config.text_dim)
                    .map(|index| (index as f32 * 0.37).sin())
                    .collect(),
            ),
            pooled: (0..config.pooled_dim)
                .map(|index| (index as f32 * 0.11).cos())
                .collect(),
            height,
            width,
            guidance: 3.5,
        }
    }

    fn model(config: &DitConfig, name: &str) -> Dit {
        let path = std::env::temp_dir().join(name);
        checkpoint(config, &path);
        let model = Dit::load(&path, "", config.clone()).unwrap();
        std::fs::remove_file(path).ok();
        model
    }

    #[test]
    fn the_two_published_spellings_of_a_checkpoint_load_to_the_same_model() {
        let config = tiny();
        let path = std::env::temp_dir().join("rusting_brain_dit_diffusers.safetensors");
        crate::safetensors::write_checkpoint(&path, &diffusers_tensors(&config));
        let mut diffusers = Dit::load(&path, "", config.clone()).unwrap();
        std::fs::remove_file(&path).ok();
        let mut reference = model(&config, "rusting_brain_dit_reference.safetensors");

        diffusers.set_conditioning(conditioning(&config, 2, 3));
        reference.set_conditioning(conditioning(&config, 2, 3));
        let patches = Matrix::from_vec(
            6,
            config.in_channels,
            (0..6 * config.in_channels)
                .map(|index| (index as f32 * 0.2).sin())
                .collect(),
        );

        let expected = reference.denoise(&patches, 0.7).unwrap();
        let actual = diffusers.denoise(&patches, 0.7).unwrap();
        assert_eq!((actual.rows, actual.cols), (expected.rows, expected.cols));
        for (actual, expected) in actual.data.iter().zip(&expected.data) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "{actual} is not {expected}"
            );
        }
    }

    #[test]
    fn a_distilled_guidance_scale_reaches_the_prediction() {
        let config = DitConfig {
            guidance: true,
            ..tiny()
        };
        let mut model = model(&config, "rusting_brain_dit_guidance.safetensors");
        let patches = Matrix::from_vec(
            6,
            config.in_channels,
            (0..6 * config.in_channels)
                .map(|index| (index as f32 * 0.2).sin())
                .collect(),
        );

        model.set_conditioning(Conditioning {
            guidance: 3.5,
            ..conditioning(&config, 2, 3)
        });
        let first = model.forward(&patches, 0.7).unwrap();
        model.set_conditioning(Conditioning {
            guidance: 7.0,
            ..conditioning(&config, 2, 3)
        });
        let second = model.forward(&patches, 0.7).unwrap();

        assert!(
            first
                .data
                .iter()
                .zip(&second.data)
                .any(|(first, second)| (first - second).abs() > 1e-5)
        );

        // The scale is embedded on the same sinusoidal table as the noise
        // level, without a second thousandfold scaling of its own: a scale of
        // 3.5 is the table's 3500, not its 3.5.
        let embedded = timestep_embedding(3.5, 8);
        assert!((embedded[0] - (3500.0f32).cos()).abs() < 1e-3);
    }

    #[test]
    fn a_quantized_denoiser_predicts_what_the_float_one_does() {
        let config = tiny();
        let path = std::env::temp_dir().join("rusting_brain_dit_quantized.safetensors");
        checkpoint(&config, &path);
        let mut float = Dit::load(&path, "", config.clone()).unwrap();
        let mut bytes = Dit::load_at(&path, "", config.clone(), Precision::Q8).unwrap();
        std::fs::remove_file(&path).ok();

        float.set_conditioning(conditioning(&config, 2, 3));
        bytes.set_conditioning(conditioning(&config, 2, 3));
        let patches = Matrix::from_vec(
            6,
            config.in_channels,
            (0..6 * config.in_channels)
                .map(|index| (index as f32 * 0.2).sin())
                .collect(),
        );

        let expected = float.denoise(&patches, 0.7).unwrap();
        let actual = bytes.denoise(&patches, 0.7).unwrap();
        let error = actual
            .data
            .iter()
            .zip(&expected.data)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        let scale = expected
            .data
            .iter()
            .fold(0.0f32, |most, value| most.max(value.abs()));
        assert!(error < 0.05 * scale.max(1e-3), "{error} against {scale}");
    }

    #[test]
    fn a_denoiser_reads_a_checkpoint_and_predicts_a_velocity_per_patch() {
        let config = tiny();
        let mut model = model(&config, "rusting_brain_dit_forward.safetensors");
        model.set_conditioning(conditioning(&config, 2, 3));

        let patches = Matrix::from_vec(
            6,
            config.in_channels,
            (0..24).map(|index| (index as f32 * 0.2).sin()).collect(),
        );
        let velocity = model.denoise(&patches, 0.8).unwrap();

        assert_eq!((velocity.rows, velocity.cols), (6, config.out_channels));
        assert!(velocity.data.iter().all(|value| value.is_finite()));

        // The noise level reaches the output, which is what the modulation is
        // there for.
        let other = model.denoise(&patches, 0.2).unwrap();
        assert!(
            velocity
                .data
                .iter()
                .zip(&other.data)
                .any(|(first, second)| (first - second).abs() > 1e-5)
        );
    }

    #[test]
    fn the_prompt_and_the_patch_grid_both_change_the_prediction() {
        let config = tiny();
        let mut model = model(&config, "rusting_brain_dit_prompt.safetensors");
        let patches = Matrix::from_vec(6, config.in_channels, vec![0.1; 24]);

        model.set_conditioning(conditioning(&config, 2, 3));
        let first = model.denoise(&patches, 0.5).unwrap();

        let mut other = conditioning(&config, 2, 3);
        other.pooled[0] += 1.0;
        model.set_conditioning(other);
        let second = model.denoise(&patches, 0.5).unwrap();
        assert!(
            first
                .data
                .iter()
                .zip(&second.data)
                .any(|(first, second)| (first - second).abs() > 1e-5)
        );

        // Positions are two-dimensional, so the same patches at a different
        // aspect ratio denoise differently.
        model.set_conditioning(conditioning(&config, 3, 2));
        let third = model.denoise(&patches, 0.5).unwrap();
        assert!(
            first
                .data
                .iter()
                .zip(&third.data)
                .any(|(first, third)| (first - third).abs() > 1e-5)
        );
    }

    #[test]
    fn a_model_without_a_prompt_or_with_the_wrong_grid_refuses() {
        let config = tiny();
        let mut model = model(&config, "rusting_brain_dit_refuse.safetensors");
        let patches = Matrix::from_vec(6, config.in_channels, vec![0.0; 24]);
        assert!(model.denoise(&patches, 0.5).is_err());

        model.set_conditioning(conditioning(&config, 2, 2));
        assert!(model.denoise(&patches, 0.5).is_err());

        let mut broken = config.clone();
        broken.axes_dim = vec![4, 6];
        let path = std::env::temp_dir().join("rusting_brain_dit_broken.safetensors");
        checkpoint(&config, &path);
        assert!(Dit::load(&path, "", broken).is_err());
        // A prefix that is not in the file is an error, not an empty model.
        assert!(Dit::load(&path, "model.diffusion_model.", config).is_err());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn the_sampler_drives_the_denoiser_end_to_end() {
        let config = tiny();
        let mut model = model(&config, "rusting_brain_dit_sample.safetensors");
        model.set_conditioning(conditioning(&config, 2, 2));

        let start = crate::diffusion::noise(4, config.in_channels, Some(7));
        let image = sample(
            &mut model,
            Scheduler::flow_match(3.0),
            start.clone(),
            &SamplingConfig {
                steps: 4,
                ..SamplingConfig::default()
            },
            |_, _| true,
        )
        .unwrap();

        assert_eq!((image.rows, image.cols), (4, config.in_channels));
        assert!(image.data.iter().all(|value| value.is_finite()));
        assert_ne!(image.data, start.data);
    }

    #[test]
    fn the_rotary_table_leaves_text_alone_and_turns_patches() {
        let config = tiny();
        let positions = Positions::new(&config, 2, 2, 2);
        let mut values = Matrix::from_vec(
            6,
            config.d_model,
            (0..6 * config.d_model).map(|index| index as f32).collect(),
        );
        let before = values.clone();
        positions.apply(&mut values, config.num_heads);

        // Text tokens sit at the origin, where the rotation is the identity.
        assert_eq!(values.row(0), before.row(0));
        assert_eq!(values.row(1), before.row(1));
        // The patch at (0, 0) is also unrotated; the rest are not.
        assert_eq!(values.row(2), before.row(2));
        assert_ne!(values.row(3), before.row(3));
        assert_ne!(values.row(4), before.row(4));

        // A rotation preserves each pair's length.
        let norm = |row: &[f32]| row.iter().map(|value| value * value).sum::<f32>();
        assert!((norm(values.row(5)) - norm(before.row(5))).abs() < 1e-2);
    }
}
