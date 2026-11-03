//! SANA-Sprint 0.6B: a one-to-four step text-to-image model.
//!
//! Three published pieces, each ported from what diffusers 0.42.1's
//! `SanaSprintPipeline` runs:
//!
//! - [`GemmaEncoder`]: Gemma-2-2B-IT, whose final hidden states over a long
//!   fixed instruction plus the prompt are the condition.
//! - [`SanaDit`]: `SanaTransformer2DModel`, linear self-attention, masked
//!   cross-attention and a convolutional feed-forward over a 32x-downsampled
//!   latent grid.
//! - [`crate::dcae::DcaeDecoder`]: the DC-AE that turns 32 latent channels
//!   back into pixels.
//!
//! [`SanaSprint`] chains them with the SCM/TrigFlow sampler.

use crate::conv::{Dense, FeatureMap};
use crate::dcae::DcaeDecoder;
use crate::ffn::{gelu, silu};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::tokenizer::Bpe;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// The instruction diffusers puts in front of every prompt, joined by
/// newlines (`SanaSprintPipeline.__call__`'s `complex_human_instruction`).
pub const INSTRUCTION: [&str; 8] = [
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
];

/// Tokens of the prompt the transformer sees: the first one plus the last
/// `PROMPT_TOKENS - 1`.
pub const PROMPT_TOKENS: usize = 300;
/// What the latent is scaled by around the sampler (`sigma_data`).
pub const SIGMA_DATA: f32 = 0.5;
/// The guidance the distilled model was trained around, times
/// `guidance_embeds_scale`.
pub const GUIDANCE: f32 = 4.5 * 0.1;
/// TrigFlow's start time, `pi / 2` rounded the way diffusers writes it
/// (`max_timesteps=1.57080`), kept literal so the schedule matches its own.
#[allow(clippy::approx_constant)]
pub const MAX_TIME: f32 = 1.5708;
/// DC-AE `scaling_factor`.
pub const LATENT_SCALE: f32 = 0.41407;
/// Latent channels, and how many pixels one latent covers on a side.
pub const LATENT_CHANNELS: usize = 32;
pub const DOWNSCALE: usize = 32;

pub(crate) const GEMMA_HIDDEN: usize = 2304;
pub(crate) const GEMMA_LAYERS: usize = 26;
pub(crate) const GEMMA_HEADS: usize = 8;
pub(crate) const GEMMA_KV_HEADS: usize = 4;
pub(crate) const GEMMA_HEAD_DIM: usize = 256;
pub(crate) const GEMMA_EPS: f32 = 1e-6;
pub(crate) const GEMMA_SOFTCAP: f32 = 50.0;
pub(crate) const GEMMA_THETA: f32 = 10_000.0;
const BOS: u32 = 2;
const PAD: u32 = 0;

/// Gemma-2-2B as the prompt encoder.
///
/// The host path reads each layer from disk while it runs it, so encoding
/// needs one layer's weights in memory rather than ten gigabytes of `f32`.
pub struct GemmaEncoder {
    pub(crate) weights: PathBuf,
    tokenizer: Bpe,
}

impl GemmaEncoder {
    /// Opens `text_encoder/` and `tokenizer/` under a diffusers model folder.
    pub fn open(model: impl AsRef<Path>) -> Result<Self, NetworkError> {
        let model = model.as_ref();
        Ok(Self {
            weights: model.join("text_encoder/model.safetensors.index.json"),
            tokenizer: Bpe::from_file(model.join("tokenizer/tokenizer.json"))?,
        })
    }

    /// The padded token ids and their mask, exactly as diffusers builds them:
    /// BOS, the instruction, the lowercased prompt, truncated and padded to
    /// the instruction's length plus 298.
    pub fn prompt_ids(&self, prompt: &str) -> Result<(Vec<u32>, Vec<bool>), NetworkError> {
        let instruction = INSTRUCTION.join("\n");
        let length = 1 + self.tokenizer.encode(&instruction)?.len() + PROMPT_TOKENS - 2;
        let mut ids = vec![BOS];
        ids.extend(
            self.tokenizer
                .encode(&format!("{instruction}{}", prompt.to_lowercase().trim()))?,
        );
        ids.truncate(length);
        let mask = (0..length).map(|at| at < ids.len()).collect();
        ids.resize(length, PAD);
        Ok((ids, mask))
    }

    /// Gemma's final (normed) hidden states over `ids`, `[ids, 2304]`.
    pub fn hidden(&self, ids: &[u32], mask: &[bool]) -> Result<Matrix, NetworkError> {
        let mut file = ShardedSafeTensors::open(&self.weights)?;
        let mut hidden = embed(&mut file, ids)?;
        let rotary = crate::text_encoder::Rotary::new(ids.len(), GEMMA_HEAD_DIM, GEMMA_THETA);
        for layer in 0..GEMMA_LAYERS {
            let weights = GemmaLayer::read(&mut file, layer)?;
            weights.forward(&mut hidden, &rotary, mask)?;
        }
        Ok(crate::text_encoder::rms_norm(
            &hidden,
            &gemma_norm(&mut file, "norm.weight")?,
            GEMMA_EPS,
        ))
    }

    /// The condition for `prompt`: `[300, 2304]` and which of those rows are
    /// real tokens.
    pub fn encode(&self, prompt: &str) -> Result<(Matrix, Vec<bool>), NetworkError> {
        let (ids, mask) = self.prompt_ids(prompt)?;
        let hidden = self.hidden(&ids, &mask)?;
        Ok(select_prompt(&hidden, &mask))
    }
}

/// diffusers keeps row 0 and the last 299: `[0] + range(-299, 0)`.
pub(crate) fn select_prompt(hidden: &Matrix, mask: &[bool]) -> (Matrix, Vec<bool>) {
    let rows: Vec<usize> = std::iter::once(0)
        .chain(hidden.rows + 1 - PROMPT_TOKENS..hidden.rows)
        .collect();
    let mut out = Matrix::new(rows.len(), hidden.cols);
    for (target, &row) in rows.iter().enumerate() {
        out.row_mut(target).copy_from_slice(hidden.row(row));
    }
    (out, rows.iter().map(|&row| mask[row]).collect())
}

/// Embedding rows times `sqrt(hidden)`, as Gemma scales them.
fn embed(file: &mut ShardedSafeTensors, ids: &[u32]) -> Result<Matrix, NetworkError> {
    let (table, _) = file.tensor("embed_tokens.weight")?;
    let scale = (GEMMA_HIDDEN as f32).sqrt();
    let mut out = Matrix::new(ids.len(), GEMMA_HIDDEN);
    for (row, &id) in ids.iter().enumerate() {
        let source = &table[id as usize * GEMMA_HIDDEN..(id as usize + 1) * GEMMA_HIDDEN];
        for (value, weight) in out.row_mut(row).iter_mut().zip(source) {
            *value = weight * scale;
        }
    }
    Ok(out)
}

/// A Gemma norm gain with its `1 +` folded in.
pub(crate) fn gemma_norm(file: &mut ShardedSafeTensors, name: &str) -> Result<Vec<f32>, NetworkError> {
    Ok(file.tensor(name)?.0.into_iter().map(|w| 1.0 + w).collect())
}

pub(crate) fn linear(file: &mut ShardedSafeTensors, name: &str, bias: bool) -> Result<Dense, NetworkError> {
    let (values, shape) = file.tensor(&format!("{name}.weight"))?;
    let rows = shape[0];
    let cols = values.len() / rows;
    let bias = match bias {
        true => Some(file.tensor(&format!("{name}.bias"))?.0),
        false => None,
    };
    Dense::new(Matrix::from_vec(rows, cols, values), bias)
}

struct GemmaLayer {
    input_norm: Vec<f32>,
    post_attention_norm: Vec<f32>,
    pre_feedforward_norm: Vec<f32>,
    post_feedforward_norm: Vec<f32>,
    query: Dense,
    key: Dense,
    value: Dense,
    output: Dense,
    gate: Dense,
    up: Dense,
    down: Dense,
}

impl GemmaLayer {
    fn read(file: &mut ShardedSafeTensors, layer: usize) -> Result<Self, NetworkError> {
        let name = |part: &str| format!("layers.{layer}.{part}");
        Ok(Self {
            input_norm: gemma_norm(file, &name("input_layernorm.weight"))?,
            post_attention_norm: gemma_norm(file, &name("post_attention_layernorm.weight"))?,
            pre_feedforward_norm: gemma_norm(file, &name("pre_feedforward_layernorm.weight"))?,
            post_feedforward_norm: gemma_norm(file, &name("post_feedforward_layernorm.weight"))?,
            query: linear(file, &name("self_attn.q_proj"), false)?,
            key: linear(file, &name("self_attn.k_proj"), false)?,
            value: linear(file, &name("self_attn.v_proj"), false)?,
            output: linear(file, &name("self_attn.o_proj"), false)?,
            gate: linear(file, &name("mlp.gate_proj"), false)?,
            up: linear(file, &name("mlp.up_proj"), false)?,
            down: linear(file, &name("mlp.down_proj"), false)?,
        })
    }

    fn forward(&self, hidden: &mut Matrix, rotary: &crate::text_encoder::Rotary, mask: &[bool]) -> Result<(), NetworkError> {
        use crate::text_encoder::rms_norm;
        let x = rms_norm(hidden, &self.input_norm, GEMMA_EPS);
        let mut query = self.query.forward(&x)?;
        let mut key = self.key.forward(&x)?;
        let value = self.value.forward(&x)?;
        rotary.apply(&mut query, GEMMA_HEADS);
        rotary.apply(&mut key, GEMMA_KV_HEADS);
        let tokens = hidden.rows;
        let mut attended = Matrix::new(tokens, GEMMA_HEADS * GEMMA_HEAD_DIM);
        let group = GEMMA_HEADS / GEMMA_KV_HEADS;
        for head in 0..GEMMA_HEADS {
            let kv = head / group;
            let out = attend(
                &columns(&query, head * GEMMA_HEAD_DIM, GEMMA_HEAD_DIM),
                &columns(&key, kv * GEMMA_HEAD_DIM, GEMMA_HEAD_DIM),
                &columns(&value, kv * GEMMA_HEAD_DIM, GEMMA_HEAD_DIM),
                (GEMMA_HEAD_DIM as f32).powf(-0.5),
                Some(GEMMA_SOFTCAP),
                |row, key| key <= row && mask[key],
            );
            set_columns(&mut attended, head * GEMMA_HEAD_DIM, &out);
        }
        let out = rms_norm(&self.output.forward(&attended)?, &self.post_attention_norm, GEMMA_EPS);
        add(hidden, &out);

        let x = rms_norm(hidden, &self.pre_feedforward_norm, GEMMA_EPS);
        let mut gate = self.gate.forward(&x)?;
        let up = self.up.forward(&x)?;
        gate.data
            .par_iter_mut()
            .zip(&up.data)
            .for_each(|(g, u)| *g = gelu(*g) * u);
        let out = rms_norm(&self.down.forward(&gate)?, &self.post_feedforward_norm, GEMMA_EPS);
        add(hidden, &out);
        Ok(())
    }
}

/// `width` columns of `matrix` from `start`, as their own matrix.
fn columns(matrix: &Matrix, start: usize, width: usize) -> Matrix {
    let mut out = Matrix::new(matrix.rows, width);
    for row in 0..matrix.rows {
        out.row_mut(row)
            .copy_from_slice(&matrix.row(row)[start..start + width]);
    }
    out
}

fn set_columns(matrix: &mut Matrix, start: usize, part: &Matrix) {
    for row in 0..part.rows {
        matrix.row_mut(row)[start..start + part.cols].copy_from_slice(part.row(row));
    }
}

fn add(target: &mut Matrix, other: &Matrix) {
    target
        .data
        .par_iter_mut()
        .zip(&other.data)
        .for_each(|(a, b)| *a += b);
}

/// Softmax attention of one head in `f32`, with an optional `tanh` cap on the
/// scores and `allowed(row, key)` as the mask.
fn attend(
    query: &Matrix,
    key: &Matrix,
    value: &Matrix,
    scale: f32,
    softcap: Option<f32>,
    allowed: impl Fn(usize, usize) -> bool + Sync,
) -> Matrix {
    let mut scores = Matrix::new(query.rows, key.rows);
    query.dot_rhs_transposed(key, &mut scores);
    scores
        .data
        .par_chunks_mut(key.rows)
        .enumerate()
        .for_each(|(row, scores)| {
            let mut max = f32::NEG_INFINITY;
            for (at, score) in scores.iter_mut().enumerate() {
                let mut s = *score * scale;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                *score = match allowed(row, at) {
                    true => s,
                    false => f32::NEG_INFINITY,
                };
                max = max.max(*score);
            }
            let mut sum = 0.0;
            for score in scores.iter_mut() {
                *score = (*score - max).exp();
                sum += *score;
            }
            for score in scores.iter_mut() {
                *score /= sum;
            }
        });
    let mut out = Matrix::new(query.rows, value.cols);
    scores.dot(value, &mut out);
    out
}

/// Layer norm over each row without a gain, then `* (1 + scale) + shift`.
fn modulated_norm(x: &Matrix, shift: &[f32], scale: &[f32], eps: f32) -> Matrix {
    let mut out = x.clone();
    out.data.par_chunks_mut(x.cols).for_each(|row| {
        let n = row.len() as f32;
        let mean = row.iter().sum::<f32>() / n;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let inverse = (var + eps).sqrt().recip();
        for ((v, shift), scale) in row.iter_mut().zip(shift).zip(scale) {
            *v = (*v - mean) * inverse * (1.0 + scale) + shift;
        }
    });
    out
}

/// diffusers' `get_timestep_embedding(t, 256, flip_sin_to_cos=True)`.
pub(crate) fn timestep_features(t: f32) -> Vec<f32> {
    let half = 128;
    let mut out = vec![0.0; 2 * half];
    for i in 0..half {
        let frequency = (-(10_000f32.ln()) * i as f32 / half as f32).exp();
        let angle = t * frequency;
        out[i] = angle.cos();
        out[half + i] = angle.sin();
    }
    out
}

pub(crate) const DIT_DIM: usize = 1152;
pub(crate) const DIT_HEADS: usize = 36;
pub(crate) const DIT_HEAD_DIM: usize = 32;
pub(crate) const CROSS_HEADS: usize = 16;
pub(crate) const CROSS_HEAD_DIM: usize = 72;
pub(crate) const FF_HIDDEN: usize = 2880;
pub(crate) const DIT_LAYERS: usize = 28;
pub(crate) const QK_EPS: f32 = 1e-5;
pub(crate) const NORM_EPS: f32 = 1e-6;

pub(crate) struct SanaBlock {
    pub(crate) table: Vec<f32>,
    pub(crate) query: Dense,
    pub(crate) key: Dense,
    pub(crate) value: Dense,
    pub(crate) out: Dense,
    pub(crate) query_norm: Vec<f32>,
    pub(crate) key_norm: Vec<f32>,
    pub(crate) cross_query: Dense,
    pub(crate) cross_key: Dense,
    pub(crate) cross_value: Dense,
    pub(crate) cross_out: Dense,
    pub(crate) cross_query_norm: Vec<f32>,
    pub(crate) cross_key_norm: Vec<f32>,
    pub(crate) inverted: Dense,
    pub(crate) depth_weight: Vec<f32>,
    pub(crate) depth_bias: Vec<f32>,
    pub(crate) point: Dense,
}

/// `SanaTransformer2DModel`, patch size 1.
pub struct SanaDit {
    pub(crate) patch: Dense,
    pub(crate) time: [Dense; 2],
    pub(crate) guidance: [Dense; 2],
    pub(crate) time_linear: Dense,
    pub(crate) caption: [Dense; 2],
    pub(crate) caption_norm: Vec<f32>,
    pub(crate) table: Vec<f32>,
    pub(crate) proj_out: Dense,
    pub(crate) blocks: Vec<SanaBlock>,
}

impl SanaDit {
    /// Reads `transformer/diffusion_pytorch_model.safetensors`.
    pub fn load(model: impl AsRef<Path>) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(
            model.as_ref().join("transformer/diffusion_pytorch_model.safetensors"),
        )?;
        let f = &mut file;
        let mut blocks = Vec::with_capacity(DIT_LAYERS);
        for layer in 0..DIT_LAYERS {
            let name = |part: &str| format!("transformer_blocks.{layer}.{part}");
            blocks.push(SanaBlock {
                table: f.tensor(&name("scale_shift_table"))?.0,
                query: linear(f, &name("attn1.to_q"), false)?,
                key: linear(f, &name("attn1.to_k"), false)?,
                value: linear(f, &name("attn1.to_v"), false)?,
                out: linear(f, &name("attn1.to_out.0"), true)?,
                query_norm: f.tensor(&name("attn1.norm_q.weight"))?.0,
                key_norm: f.tensor(&name("attn1.norm_k.weight"))?.0,
                cross_query: linear(f, &name("attn2.to_q"), true)?,
                cross_key: linear(f, &name("attn2.to_k"), true)?,
                cross_value: linear(f, &name("attn2.to_v"), true)?,
                cross_out: linear(f, &name("attn2.to_out.0"), true)?,
                cross_query_norm: f.tensor(&name("attn2.norm_q.weight"))?.0,
                cross_key_norm: f.tensor(&name("attn2.norm_k.weight"))?.0,
                inverted: linear(f, &name("ff.conv_inverted"), true)?,
                depth_weight: f.tensor(&name("ff.conv_depth.weight"))?.0,
                depth_bias: f.tensor(&name("ff.conv_depth.bias"))?.0,
                point: linear(f, &name("ff.conv_point"), false)?,
            });
        }
        Ok(Self {
            patch: linear(f, "patch_embed.proj", true)?,
            time: [
                linear(f, "time_embed.timestep_embedder.linear_1", true)?,
                linear(f, "time_embed.timestep_embedder.linear_2", true)?,
            ],
            guidance: [
                linear(f, "time_embed.guidance_embedder.linear_1", true)?,
                linear(f, "time_embed.guidance_embedder.linear_2", true)?,
            ],
            time_linear: linear(f, "time_embed.linear", true)?,
            caption: [
                linear(f, "caption_projection.linear_1", true)?,
                linear(f, "caption_projection.linear_2", true)?,
            ],
            caption_norm: f.tensor("caption_norm.weight")?.0,
            table: f.tensor("scale_shift_table")?.0,
            proj_out: linear(f, "proj_out", true)?,
            blocks,
        })
    }

    /// The projected, normed prompt, keeping only its real tokens: the
    /// masked ones get `-10000` before a softmax, which is zero in `f32`.
    pub fn context(&self, embeds: &Matrix, mask: &[bool]) -> Result<Matrix, NetworkError> {
        let rows: Vec<usize> = (0..embeds.rows).filter(|&row| mask[row]).collect();
        let mut real = Matrix::new(rows.len(), embeds.cols);
        for (target, &row) in rows.iter().enumerate() {
            real.row_mut(target).copy_from_slice(embeds.row(row));
        }
        let mut x = self.caption[0].forward(&real)?;
        x.data.par_iter_mut().for_each(|v| *v = gelu(*v));
        let x = self.caption[1].forward(&x)?;
        Ok(crate::text_encoder::rms_norm(&x, &self.caption_norm, 1e-5))
    }

    /// `(temb [6 * 1152], embedded [1152])` for a time and a guidance.
    pub(crate) fn conditioning(&self, t: f32, guidance: f32) -> Result<(Vec<f32>, Vec<f32>), NetworkError> {
        let embed = |layers: &[Dense; 2], value: f32| -> Result<Vec<f32>, NetworkError> {
            let mut x = layers[0].apply(&timestep_features(value))?;
            x.iter_mut().for_each(|v| *v = silu(*v));
            layers[1].apply(&x)
        };
        let time = embed(&self.time, t)?;
        let guide = embed(&self.guidance, guidance)?;
        let embedded: Vec<f32> = time.iter().zip(&guide).map(|(a, b)| a + b).collect();
        let activated: Vec<f32> = embedded.iter().map(|v| silu(*v)).collect();
        Ok((self.time_linear.apply(&activated)?, embedded))
    }

    /// One denoiser call: `latent` is `[32, height * width]`, channel-major,
    /// and so is the prediction.
    pub fn forward(
        &self,
        latent: &Matrix,
        height: usize,
        width: usize,
        context: &Matrix,
        t: f32,
        guidance: f32,
    ) -> Result<Matrix, NetworkError> {
        let tokens = transpose(latent);
        let mut hidden = self.patch.forward(&tokens)?;
        let (temb, embedded) = self.conditioning(t, guidance)?;
        for block in &self.blocks {
            block.forward(&mut hidden, height, width, context, &temb)?;
        }
        let shift: Vec<f32> = (0..DIT_DIM).map(|i| self.table[i] + embedded[i]).collect();
        let scale: Vec<f32> = (0..DIT_DIM)
            .map(|i| self.table[DIT_DIM + i] + embedded[i])
            .collect();
        let x = modulated_norm(&hidden, &shift, &scale, NORM_EPS);
        Ok(transpose(&self.proj_out.forward(&x)?))
    }
}

pub(crate) fn transpose(m: &Matrix) -> Matrix {
    let mut out = Matrix::new(m.cols, m.rows);
    for r in 0..m.rows {
        for c in 0..m.cols {
            out.data[c * m.rows + r] = m.data[r * m.cols + c];
        }
    }
    out
}

impl SanaBlock {
    pub(crate) fn forward(
        &self,
        hidden: &mut Matrix,
        height: usize,
        width: usize,
        context: &Matrix,
        temb: &[f32],
    ) -> Result<(), NetworkError> {
        let modulation: Vec<f32> = self.table.iter().zip(temb).map(|(a, b)| a + b).collect();
        let part = |i: usize| &modulation[i * DIT_DIM..(i + 1) * DIT_DIM];

        // Linear self-attention.
        let x = modulated_norm(hidden, part(0), part(1), NORM_EPS);
        let attended = self.linear_attention(&x)?;
        gated_add(hidden, &self.out.forward(&attended)?, part(2));

        // Cross-attention over the prompt, no gate.
        let query = crate::text_encoder::rms_norm(&self.cross_query.forward(hidden)?, &self.cross_query_norm, QK_EPS);
        let key = crate::text_encoder::rms_norm(&self.cross_key.forward(context)?, &self.cross_key_norm, QK_EPS);
        let value = self.cross_value.forward(context)?;
        let mut attended = Matrix::new(hidden.rows, DIT_DIM);
        for head in 0..CROSS_HEADS {
            let at = head * CROSS_HEAD_DIM;
            let out = attend(
                &columns(&query, at, CROSS_HEAD_DIM),
                &columns(&key, at, CROSS_HEAD_DIM),
                &columns(&value, at, CROSS_HEAD_DIM),
                (CROSS_HEAD_DIM as f32).powf(-0.5),
                None,
                |_, _| true,
            );
            set_columns(&mut attended, at, &out);
        }
        add(hidden, &self.cross_out.forward(&attended)?);

        // GLUMBConv feed-forward on the latent grid.
        let x = modulated_norm(hidden, part(3), part(4), NORM_EPS);
        let mut inverted = self.inverted.forward(&x)?;
        inverted.data.par_iter_mut().for_each(|v| *v = silu(*v));
        let depth = crate::dcae::depthwise(&inverted, height, width, &self.depth_weight, Some(&self.depth_bias), 3);
        let mut glu = Matrix::new(depth.rows, FF_HIDDEN);
        glu.data
            .par_chunks_mut(FF_HIDDEN)
            .zip(depth.data.par_chunks(2 * FF_HIDDEN))
            .for_each(|(out, row)| {
                for i in 0..FF_HIDDEN {
                    out[i] = row[i] * silu(row[FF_HIDDEN + i]);
                }
            });
        gated_add(hidden, &self.point.forward(&glu)?, part(5));
        Ok(())
    }

    /// `SanaLinearAttnProcessor2_0`: RMS-normed, ReLU'd queries and keys,
    /// `out = (V K^T) q / (sum(K) q + 1e-15)` per head.
    fn linear_attention(&self, x: &Matrix) -> Result<Matrix, NetworkError> {
        let mut query = crate::text_encoder::rms_norm(&self.query.forward(x)?, &self.query_norm, QK_EPS);
        let mut key = crate::text_encoder::rms_norm(&self.key.forward(x)?, &self.key_norm, QK_EPS);
        let value = self.value.forward(x)?;
        query.data.par_iter_mut().for_each(|v| *v = v.max(0.0));
        key.data.par_iter_mut().for_each(|v| *v = v.max(0.0));
        Ok(linear_attention(&query, &key, &value, DIT_HEADS, DIT_HEAD_DIM))
    }
}

/// Linear attention over token-major `[n, heads * d]` inputs, with the
/// one-padded value row as the normalizer.
pub(crate) fn linear_attention(query: &Matrix, key: &Matrix, value: &Matrix, heads: usize, d: usize) -> Matrix {
    let n = query.rows;
    let width = query.cols;
    let mut out = Matrix::new(n, heads * d);
    let per_head: Vec<Vec<f32>> = (0..heads)
        .into_par_iter()
        .map(|head| {
            // kv[i][j] = sum_n v[n, i] k[n, j], row d is sum_n k[n, j].
            let mut kv = vec![0f32; (d + 1) * d];
            for token in 0..n {
                let k = &key.data[token * width + head * d..][..d];
                let v = &value.data[token * width + head * d..][..d];
                for i in 0..d {
                    let row = &mut kv[i * d..(i + 1) * d];
                    for j in 0..d {
                        row[j] += v[i] * k[j];
                    }
                }
                let row = &mut kv[d * d..];
                for j in 0..d {
                    row[j] += k[j];
                }
            }
            let mut result = vec![0f32; n * d];
            for token in 0..n {
                let q = &query.data[token * width + head * d..][..d];
                let dot = |row: &[f32]| row.iter().zip(q).map(|(a, b)| a * b).sum::<f32>();
                let normalizer = dot(&kv[d * d..]) + 1e-15;
                for i in 0..d {
                    result[token * d + i] = dot(&kv[i * d..(i + 1) * d]) / normalizer;
                }
            }
            result
        })
        .collect();
    for (head, values) in per_head.iter().enumerate() {
        for token in 0..n {
            out.data[token * heads * d + head * d..][..d].copy_from_slice(&values[token * d..(token + 1) * d]);
        }
    }
    out
}

fn gated_add(hidden: &mut Matrix, update: &Matrix, gate: &[f32]) {
    hidden
        .data
        .par_chunks_mut(hidden.cols)
        .zip(update.data.par_chunks(update.cols))
        .for_each(|(row, update)| {
            for ((h, u), g) in row.iter_mut().zip(update).zip(gate) {
                *h += g * u;
            }
        });
}

/// The TrigFlow times `SCMScheduler` walks for `steps` steps.
pub fn timesteps(steps: usize) -> Vec<f32> {
    match steps {
        2 => vec![MAX_TIME, 1.3, 0.0],
        _ => (0..=steps)
            .map(|i| MAX_TIME * (1.0 - i as f32 / steps as f32))
            .collect(),
    }
}

/// `SanaSprintPipeline`'s loop: `noise(i)` is the standard normal draw for
/// step `i`'s re-noising, `start` the initial one, both `[32, h * w]`.
/// Returns the final latent, already divided by `sigma_data`.
pub fn sample(
    start: &Matrix,
    steps: usize,
    mut denoise: impl FnMut(&Matrix, f32) -> Result<Matrix, NetworkError>,
    mut noise: impl FnMut(usize) -> Matrix,
) -> Result<Matrix, NetworkError> {
    let times = timesteps(steps);
    let mut latents = start.clone();
    latents.data.iter_mut().for_each(|v| *v *= SIGMA_DATA);
    let mut denoised = latents.clone();
    for i in 0..steps {
        let (s, next) = (times[i], times[i + 1]);
        let scm = s.sin() / (s.cos() + s.sin());
        let norm = (scm * scm + (1.0 - scm) * (1.0 - scm)).sqrt();
        let mut input = latents.clone();
        input.data.iter_mut().for_each(|v| *v = *v / SIGMA_DATA * norm);
        let raw = denoise(&input, scm)?;
        let draw = noise(i);
        for at in 0..latents.data.len() {
            let pred = ((1.0 - 2.0 * scm) * input.data[at]
                + (1.0 - 2.0 * scm + 2.0 * scm * scm) * raw.data[at])
                / norm
                * SIGMA_DATA;
            denoised.data[at] = s.cos() * latents.data[at] - s.sin() * pred;
            latents.data[at] =
                next.cos() * denoised.data[at] + next.sin() * draw.data[at] * SIGMA_DATA;
        }
    }
    denoised.data.iter_mut().for_each(|v| *v /= SIGMA_DATA);
    Ok(denoised)
}

/// The whole host pipeline.
pub struct SanaSprint {
    pub encoder: GemmaEncoder,
    pub dit: SanaDit,
    pub decoder: DcaeDecoder,
}

impl SanaSprint {
    /// Loads a diffusers `Sana_Sprint_0.6B_1024px_diffusers` folder.
    pub fn load(model: impl AsRef<Path>) -> Result<Self, NetworkError> {
        let model = model.as_ref();
        Ok(Self {
            encoder: GemmaEncoder::open(model)?,
            dit: SanaDit::load(model)?,
            decoder: DcaeDecoder::load(model)?,
        })
    }

    /// One image in `[-1, 1]`, `[3, height, width]`. Sizes must be multiples
    /// of 32.
    pub fn generate(
        &self,
        prompt: &str,
        width: usize,
        height: usize,
        steps: usize,
        seed: Option<u64>,
    ) -> Result<FeatureMap, NetworkError> {
        check_size(width, height, steps)?;
        let (embeds, mask) = self.encoder.encode(prompt)?;
        let context = self.dit.context(&embeds, &mask)?;
        let (h, w) = (height / DOWNSCALE, width / DOWNSCALE);
        let start = crate::diffusion::noise(LATENT_CHANNELS, h * w, seed);
        let latent = sample(
            &start,
            steps,
            |x, t| self.dit.forward(x, h, w, &context, t, GUIDANCE),
            |i| crate::diffusion::noise(LATENT_CHANNELS, h * w, seed.map(|s| s.wrapping_add(1 + i as u64))),
        )?;
        self.decoder.decode(&latent_map(&latent, h, w))
    }
}

pub(crate) fn check_size(width: usize, height: usize, steps: usize) -> Result<(), NetworkError> {
    if width == 0 || height == 0 || !width.is_multiple_of(DOWNSCALE) || !height.is_multiple_of(DOWNSCALE) || steps == 0 {
        return Err(NetworkError::InvalidConfig(format!(
            "SANA-Sprint needs sizes that are positive multiples of 32 and at least one step, \
             not {width}x{height} in {steps}"
        )));
    }
    Ok(())
}

/// The final latent divided by the DC-AE scale, as the decoder's input.
pub(crate) fn latent_map(latent: &Matrix, h: usize, w: usize) -> FeatureMap {
    FeatureMap {
        channels: LATENT_CHANNELS,
        height: h,
        width: w,
        data: latent.data.iter().map(|v| v / LATENT_SCALE).collect(),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::safetensors::SafeTensors;

    pub(crate) fn relative(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let (gap, norm) = got.iter().zip(want).fold((0.0f64, 0.0f64), |(g, n), (a, b)| {
            (g + ((a - b) as f64).powi(2), n + (*b as f64).powi(2))
        });
        (gap / norm).sqrt() as f32
    }

    pub(crate) fn gate(name: &str, got: &[f32], want: &[f32], limit: f32) -> f32 {
        let error = relative(got, want);
        println!("{name}: relative L2 {error:.3e} (gate {limit:.0e})");
        assert!(error <= limit, "{name} misses its gate: {error:.3e} > {limit:.0e}");
        error
    }

    /// The reference folder from `RustingFrame/tools/sana_refs.py` and the
    /// model it was made from.
    pub(crate) fn reference() -> Option<(SafeTensors, PathBuf)> {
        let dir = PathBuf::from(std::env::var("RB_SANA_REF").ok()?);
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).ok()?).ok()?;
        let model = PathBuf::from(json["model"].as_str()?);
        Some((SafeTensors::open(dir.join("sana.safetensors")).ok()?, model))
    }

    pub(crate) fn tensor(file: &mut SafeTensors, name: &str) -> Vec<f32> {
        file.tensor(name).unwrap().0
    }

    #[test]
    fn the_timesteps_are_the_schedulers() {
        assert_eq!(timesteps(2), vec![MAX_TIME, 1.3, 0.0]);
        let four = timesteps(4);
        assert_eq!(four.len(), 5);
        assert!((four[2] - MAX_TIME / 2.0).abs() < 1e-6 && four[4] == 0.0);
    }

    #[test]
    fn a_size_off_the_latent_grid_is_refused() {
        assert!(check_size(1000, 1024, 2).is_err());
        assert!(check_size(1024, 1024, 0).is_err());
        assert!(check_size(512, 1024, 1).is_ok());
    }

    #[test]
    fn linear_attention_normalizes_by_the_key_sum() {
        // One head of width 1: out = sum(v k) q / (sum(k) q).
        let q = Matrix::from_vec(2, 1, vec![1.0, 2.0]);
        let k = Matrix::from_vec(2, 1, vec![1.0, 3.0]);
        let v = Matrix::from_vec(2, 1, vec![2.0, 4.0]);
        let out = linear_attention(&q, &k, &v, 1, 1);
        assert!((out.data[0] - 14.0 / 4.0).abs() < 1e-6);
        assert!((out.data[1] - 14.0 / 4.0).abs() < 1e-6);
    }

    #[test]
    #[ignore = "needs RB_SANA_REF and the SANA-Sprint weights"]
    fn sana_gemma_matches_diffusers() {
        let Some((mut reference, model)) = reference() else {
            panic!("set RB_SANA_REF to the sana_refs.py output folder");
        };
        let encoder = GemmaEncoder::open(&model).unwrap();
        let prompt = "a lighthouse on a rocky coast in fog";
        let (ids, mask) = encoder.prompt_ids(prompt).unwrap();
        let want: Vec<u32> = tensor(&mut reference, "input_ids").iter().map(|v| *v as u32).collect();
        let want_mask: Vec<bool> = tensor(&mut reference, "attention_mask").iter().map(|v| *v != 0.0).collect();
        assert_eq!(ids, want, "token ids");
        assert_eq!(mask, want_mask, "attention mask");
        let start = std::time::Instant::now();
        let hidden = encoder.hidden(&ids, &mask).unwrap();
        println!("host gemma: {:.1} s", start.elapsed().as_secs_f32());
        gate("gemma hidden", &hidden.data, &tensor(&mut reference, "gemma_hidden"), 1e-2);
        let (embeds, selected) = select_prompt(&hidden, &mask);
        let want_selected: Vec<bool> = tensor(&mut reference, "prompt_mask").iter().map(|v| *v != 0.0).collect();
        assert_eq!(selected, want_selected);
        gate("prompt embeds", &embeds.data, &tensor(&mut reference, "prompt_embeds"), 1e-2);
    }

    #[test]
    #[ignore = "needs RB_SANA_REF and the SANA-Sprint weights"]
    fn sana_dit_matches_diffusers() {
        let Some((mut reference, model)) = reference() else {
            panic!("set RB_SANA_REF to the sana_refs.py output folder");
        };
        let dit = SanaDit::load(&model).unwrap();
        let embeds = Matrix::from_vec(PROMPT_TOKENS, GEMMA_HIDDEN, tensor(&mut reference, "prompt_embeds"));
        let mask: Vec<bool> = tensor(&mut reference, "prompt_mask").iter().map(|v| *v != 0.0).collect();
        let context = dit.context(&embeds, &mask).unwrap();
        let input = Matrix::from_vec(LATENT_CHANNELS, 1024, tensor(&mut reference, "step0_input"));
        let times = timesteps(2);
        let scm = times[0].sin() / (times[0].cos() + times[0].sin());
        let start = std::time::Instant::now();
        let raw = dit.forward(&input, 32, 32, &context, scm, GUIDANCE).unwrap();
        println!("host dit step: {:.1} s", start.elapsed().as_secs_f32());
        gate("dit step 0", &raw.data, &tensor(&mut reference, "step0_raw"), 1e-2);

        let noise = |name: &str, reference: &mut SafeTensors| {
            Matrix::from_vec(LATENT_CHANNELS, 1024, tensor(reference, name))
        };
        let start_noise = noise("noise", &mut reference);
        let draws = [noise("step0_noise", &mut reference), noise("step1_noise", &mut reference)];
        let latent = sample(
            &start_noise,
            2,
            |x, t| dit.forward(x, 32, 32, &context, t, GUIDANCE),
            |i| draws[i].clone(),
        )
        .unwrap();
        gate("final latent", &latent.data, &tensor(&mut reference, "final_latents"), 3e-2);
    }
}
