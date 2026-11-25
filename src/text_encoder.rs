//! The text encoder a diffusion model conditions on.
//!
//! An image model does not read a prompt itself. A language model reads it, and
//! the image model conditions on that model's hidden states. FLUX.2 uses
//! Qwen3-4B for this, Stable Diffusion 3 uses two CLIPs and a T5, and the shape
//! that covers the largest share of them — and all of the recent ones — is the
//! LLaMA-family decoder: RMSNorm, rotary positions, grouped-query attention,
//! SwiGLU, and no biases.
//!
//! This is that shape, read straight from a Hugging Face checkpoint's
//! `.safetensors` and `config.json`, and run forward only. It covers Qwen2,
//! Qwen3, LLaMA, Mistral and the models that copy them, including the per-head
//! query and key normalization Qwen3 added, which the loader picks up from the
//! checkpoint when it is there.
//!
//! What it is not: T5, whose encoder uses relative position biases, and CLIP,
//! whose text tower uses LayerNorm and learned positions. Both are separate
//! shapes rather than options on this one.
//!
//! [`crate::transformer::TransformerLm`] is the trainable transformer. This is
//! a second, leaner path on purpose: a checkpoint read from someone else's file
//! needs no gradients, no optimizer state and no KV cache, and it needs details
//! — query normalization, attention biases — that the training model does not
//! have.

use crate::conv::Dense;
use crate::matrix::Matrix;
use crate::mmdit::attention_over;
use crate::network::NetworkError;
use crate::quantized::Quantized;
use crate::safetensors::ShardedSafeTensors;
use crate::tokenizer::Bpe;
use crate::transformer::Precision;
use rayon::prelude::*;

/// The shape of an encoder, as a checkpoint's `config.json` names it.
#[derive(Clone, Debug, PartialEq)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub layers: usize,
    pub num_heads: usize,
    /// Fewer than `num_heads` is grouped-query attention.
    pub kv_heads: usize,
    pub head_dim: usize,
    pub mlp_dim: usize,
    pub rope_theta: f32,
    pub eps: f32,
    /// Whether the model reads left to right. True for every decoder language
    /// model, which is what these encoders are.
    pub causal: bool,
}

impl TextEncoderConfig {
    /// Reads a Hugging Face `config.json`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let text = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| NetworkError::InvalidDataset(format!("config.json: {error}")))?;
        // A pipeline's config nests the encoder's own configuration.
        let json = json.get("text_config").unwrap_or(&json).clone();

        let number = |name: &str| -> Option<usize> {
            json.get(name)
                .and_then(|value| value.as_u64())
                .map(|value| value as usize)
        };
        let missing =
            |name: &str| NetworkError::InvalidDataset(format!("config.json has no {name}"));

        let d_model = number("hidden_size").ok_or_else(|| missing("hidden_size"))?;
        let num_heads =
            number("num_attention_heads").ok_or_else(|| missing("num_attention_heads"))?;
        Ok(Self {
            vocab_size: number("vocab_size").ok_or_else(|| missing("vocab_size"))?,
            d_model,
            layers: number("num_hidden_layers").ok_or_else(|| missing("num_hidden_layers"))?,
            num_heads,
            kv_heads: number("num_key_value_heads").unwrap_or(num_heads),
            head_dim: number("head_dim").unwrap_or(d_model / num_heads.max(1)),
            mlp_dim: number("intermediate_size").ok_or_else(|| missing("intermediate_size"))?,
            rope_theta: json
                .get("rope_theta")
                .and_then(|value| value.as_f64())
                .unwrap_or(10_000.0) as f32,
            eps: json
                .get("rms_norm_eps")
                .and_then(|value| value.as_f64())
                .unwrap_or(1e-6) as f32,
            causal: true,
        })
    }
}

/// One transformer layer of the encoder.
pub(crate) struct Layer {
    pub(crate) input_norm: Vec<f32>,
    pub(crate) query: Dense,
    pub(crate) key: Dense,
    pub(crate) value: Dense,
    pub(crate) output: Dense,
    /// Qwen3's per-head query and key normalization. Absent in Qwen2 and
    /// LLaMA, and its absence is read from the checkpoint rather than assumed.
    pub(crate) query_norm: Option<Vec<f32>>,
    pub(crate) key_norm: Option<Vec<f32>>,
    pub(crate) post_norm: Vec<f32>,
    pub(crate) gate: Dense,
    pub(crate) up: Dense,
    pub(crate) down: Dense,
}

/// A decoder language model run as a prompt encoder.
pub struct TextEncoder {
    config: TextEncoderConfig,
    embeddings: Matrix,
    pub(crate) layers: Vec<Layer>,
    final_norm: Vec<f32>,
    /// Where the padding rows of a prompt sit for RoPE, if not at their own
    /// index. FLUX 3 trained with them all at position 1.
    pub pad_position: Option<usize>,
    #[cfg(feature = "cuda")]
    device: Option<crate::cuda_text_encoder::DeviceTextEncoder>,
}

impl TextEncoder {
    /// The configuration this encoder was built for.
    pub fn config(&self) -> &TextEncoderConfig {
        &self.config
    }

    /// The RoPE tables for `tokens` rows of which the first `real` are not
    /// padding.
    pub(crate) fn rotary(&self, tokens: usize, real: usize) -> Rotary {
        let mut rotary = Rotary::new(tokens, self.config.head_dim, self.config.rope_theta);
        if let Some(position) = self.pad_position.filter(|position| *position < real) {
            let half = self.config.head_dim / 2;
            for row in real..tokens {
                rotary.cos.copy_within(position * half..(position + 1) * half, row * half);
                rotary.sin.copy_within(position * half..(position + 1) * half, row * half);
            }
        }
        rotary
    }

    /// Reads an encoder out of a checkpoint.
    ///
    /// `path` is a `.safetensors` file or the index of a sharded one, and
    /// `prefix` is what the keys start with — `"model."` in a standalone
    /// language model, `"text_encoder.model."` inside a pipeline.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        prefix: &str,
        config: TextEncoderConfig,
    ) -> Result<Self, NetworkError> {
        Self::load_at(path, prefix, config, Precision::F32)
    }

    /// The same, holding the projections at whatever precision is asked for.
    ///
    /// A prompt encoder is often the largest part of an image model, and
    /// [`Precision::Q8`] is a quarter of the memory for it. The embedding table
    /// stays as it is: it is read by lookup, not multiplied.
    pub fn load_at(
        path: impl AsRef<std::path::Path>,
        prefix: &str,
        config: TextEncoderConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        Self::read_at(&mut file, prefix, config, precision)
    }

    /// The same, from a checkpoint that is already open.
    pub fn read(
        file: &mut ShardedSafeTensors,
        prefix: &str,
        config: TextEncoderConfig,
    ) -> Result<Self, NetworkError> {
        Self::read_at(file, prefix, config, Precision::F32)
    }

    /// The same, at a chosen precision.
    pub fn read_at(
        file: &mut ShardedSafeTensors,
        prefix: &str,
        config: TextEncoderConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let embeddings = file.matrix(&format!("{prefix}embed_tokens.weight"))?;
        if embeddings.cols != config.d_model {
            return Err(NetworkError::InvalidConfig(format!(
                "the checkpoint's embeddings are {} wide and the configuration says {}",
                embeddings.cols, config.d_model
            )));
        }

        let layers = (0..config.layers)
            .map(|index| {
                let base = format!("{prefix}layers.{index}");
                Ok(Layer {
                    input_norm: vector(file, &format!("{base}.input_layernorm.weight"))?,
                    query: dense(file, &format!("{base}.self_attn.q_proj"), precision)?,
                    key: dense(file, &format!("{base}.self_attn.k_proj"), precision)?,
                    value: dense(file, &format!("{base}.self_attn.v_proj"), precision)?,
                    output: dense(file, &format!("{base}.self_attn.o_proj"), precision)?,
                    query_norm: vector(file, &format!("{base}.self_attn.q_norm.weight")).ok(),
                    key_norm: vector(file, &format!("{base}.self_attn.k_norm.weight")).ok(),
                    post_norm: vector(file, &format!("{base}.post_attention_layernorm.weight"))?,
                    gate: dense(file, &format!("{base}.mlp.gate_proj"), precision)?,
                    up: dense(file, &format!("{base}.mlp.up_proj"), precision)?,
                    down: dense(file, &format!("{base}.mlp.down_proj"), precision)?,
                })
            })
            .collect::<Result<Vec<_>, NetworkError>>()?;

        Ok(Self {
            final_norm: vector(file, &format!("{prefix}norm.weight"))?,
            config,
            embeddings,
            layers,
            pad_position: None,
            #[cfg(feature = "cuda")]
            device: None,
        })
    }

    /// Uploads every layer to a CUDA device and runs there from now on. The
    /// embedding table stays on the host: it is read by lookup.
    ///
    /// Fails closed: an error leaves the encoder on the CPU, as it was.
    #[cfg(feature = "cuda")]
    pub fn attach_device(
        &mut self,
        context: &std::sync::Arc<crate::gpu_transformer::GpuContext>,
    ) -> Result<(), NetworkError> {
        self.device = Some(crate::cuda_text_encoder::DeviceTextEncoder::upload(
            context, self,
        )?);
        Ok(())
    }

    /// The bytes [`TextEncoder::attach_device`] puts on the device: one a
    /// weight where a layer is int8, two (BF16) where it is not.
    #[cfg(feature = "cuda")]
    pub(crate) fn device_bytes(&self) -> usize {
        let dense = |layer: &Layer| {
            [
                &layer.query,
                &layer.key,
                &layer.value,
                &layer.output,
                &layer.gate,
                &layer.up,
                &layer.down,
            ]
            .map(|dense| dense.out_dim() * dense.in_dim() * (2 - dense.is_quantized() as usize))
        };
        self.layers.iter().flat_map(dense).sum()
    }

    /// Whether [`TextEncoder::attach_device`] moved this encoder.
    #[cfg(feature = "cuda")]
    pub fn on_device(&self) -> bool {
        self.device.is_some()
    }

    /// Drops the device copy, freeing its memory; the encoder runs on the
    /// host again.
    #[cfg(feature = "cuda")]
    pub fn detach_device(&mut self) {
        self.device = None;
    }

    /// The hidden states for a sequence of token ids, `[tokens, d_model]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Matrix, NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            let hidden = device.forward_layers(self, ids, ids.len(), &[self.layers.len()])?;
            return Ok(rms_norm(&hidden, &self.final_norm, self.config.eps));
        }
        let hidden = self.run(ids, ids.len(), self.layers.len(), None, |_, _| {})?;
        Ok(rms_norm(&hidden, &self.final_norm, self.config.eps))
    }

    /// The hidden states after each of `layers` layers, laid side by side per
    /// token, `[tokens, layers.len() * d_model]`: the output FLUX.2 reads,
    /// which takes three depths of the model rather than its last one, and
    /// takes them without the final norm.
    ///
    /// Only the first `real` of `ids` are the prompt; the rest are padding.
    /// Every position, padding included, attends causally and to none of the
    /// padding's keys, which is what the attention mask a padded batch is
    /// encoded with does. The layers past the deepest one asked for are not
    /// run.
    pub fn forward_layers(
        &self,
        ids: &[u32],
        real: usize,
        layers: &[usize],
    ) -> Result<Matrix, NetworkError> {
        let deepest = layers.iter().copied().max().unwrap_or(0);
        if layers.is_empty() || layers.contains(&0) || deepest > self.layers.len() {
            return Err(NetworkError::InvalidConfig(format!(
                "this encoder has {} layers and the output asks for those after {layers:?}",
                self.layers.len()
            )));
        }
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            return device.forward_layers(self, ids, real, layers);
        }
        let width = self.config.d_model;
        let mut output = Matrix::new(ids.len(), layers.len() * width);
        self.run(ids, real, deepest, None, |done, hidden| {
            for (slot, _) in layers
                .iter()
                .enumerate()
                .filter(|(_, layer)| **layer == done)
            {
                for token in 0..hidden.rows {
                    output.row_mut(token)[slot * width..(slot + 1) * width]
                        .copy_from_slice(hidden.row(token));
                }
            }
        })?;
        Ok(output)
    }

    /// The states after the last layer, with no final norm, of a prompt that
    /// holds a picture: `ids` with `visual` standing in for its image tokens.
    ///
    /// ponytail: host only. Qwen-Image 2.1 keeps its encoder off the card,
    /// which the denoiser fills; teach `cuda_text_encoder` the three-axis
    /// positions and the deepstack sums if an edit model's encoder ever fits.
    pub fn forward_visual(&self, ids: &[u32], visual: &Visual) -> Result<Matrix, NetworkError> {
        #[cfg(feature = "cuda")]
        if self.device.is_some() {
            return Err(NetworkError::InvalidConfig(
                "a prompt with a picture in it is only read on the host".into(),
            ));
        }
        let rows = visual.rows.len();
        if visual.embeddings.rows != rows
            || visual.embeddings.cols != self.config.d_model
            || visual.rows.iter().any(|row| *row >= ids.len())
            || visual.deepstack.iter().any(|deep| deep.rows != rows)
            || visual.positions.len() > ids.len()
        {
            return Err(NetworkError::InvalidConfig(
                "the picture's rows do not fit the prompt they are spliced into".into(),
            ));
        }
        self.run(ids, ids.len(), self.layers.len(), Some(visual), |_, _| {})
    }

    /// The first `depth` layers over `ids`, of which the first `real` are
    /// attended to. `after` sees the hidden states after every layer, counted
    /// from one.
    fn run(
        &self,
        ids: &[u32],
        real: usize,
        depth: usize,
        visual: Option<&Visual>,
        mut after: impl FnMut(usize, &Matrix),
    ) -> Result<Matrix, NetworkError> {
        let mut hidden = self.embed(ids)?;
        let scatter = |hidden: &mut Matrix, rows: &[usize], source: &Matrix, add: bool| {
            for (from, row) in rows.iter().enumerate() {
                for (value, source) in hidden.row_mut(*row).iter_mut().zip(source.row(from)) {
                    *value = source + if add { *value } else { 0.0 };
                }
            }
        };
        if let Some(visual) = visual {
            scatter(&mut hidden, &visual.rows, &visual.embeddings, false);
        }

        let (heads, kv_heads) = (self.config.num_heads, self.config.kv_heads);
        let rotary = match visual {
            Some(visual) => visual.rotary(ids.len(), self.config.head_dim, self.config.rope_theta),
            None => self.rotary(ids.len(), real),
        };

        for (index, layer) in self.layers[..depth].iter().enumerate() {
            let normed = rms_norm(&hidden, &layer.input_norm, self.config.eps);
            let mut queries = layer.query.forward(&normed)?;
            let mut keys = layer.key.forward(&normed)?;
            let values = layer.value.forward(&normed)?;

            if let Some(scale) = &layer.query_norm {
                normalize_heads(&mut queries, scale, self.config.eps);
            }
            if let Some(scale) = &layer.key_norm {
                normalize_heads(&mut keys, scale, self.config.eps);
            }
            rotary.apply(&mut queries, heads);
            rotary.apply(&mut keys, kv_heads);

            let attended = attention_over(
                &queries,
                &keys,
                &values,
                heads,
                kv_heads,
                self.config.causal,
                real,
            );
            let projected = layer.output.forward(&attended)?;
            for (value, delta) in hidden.data.iter_mut().zip(&projected.data) {
                *value += delta;
            }

            let normed = rms_norm(&hidden, &layer.post_norm, self.config.eps);
            let mut gate = layer.gate.forward(&normed)?;
            let up = layer.up.forward(&normed)?;
            gate.data
                .par_iter_mut()
                .zip(up.data.par_iter())
                .for_each(|(gate, up)| *gate = crate::ffn::silu(*gate) * up);
            let projected = layer.down.forward(&gate)?;
            for (value, delta) in hidden.data.iter_mut().zip(&projected.data) {
                *value += delta;
            }
            if let Some(deep) = visual.and_then(|visual| Some((visual, visual.deepstack.get(index)?))) {
                scatter(&mut hidden, &deep.0.rows, deep.1, true);
            }
            after(index + 1, &hidden);
        }
        Ok(hidden)
    }

    /// The embedding rows of `ids`, `[tokens, d_model]`.
    pub(crate) fn embed(&self, ids: &[u32]) -> Result<Matrix, NetworkError> {
        if ids.is_empty() {
            return Err(NetworkError::InvalidConfig(
                "an empty prompt has no hidden states".into(),
            ));
        }
        let mut hidden = Matrix::new(ids.len(), self.config.d_model);
        for (position, id) in ids.iter().enumerate() {
            if *id as usize >= self.embeddings.rows {
                return Err(NetworkError::InvalidTarget {
                    expected: self.embeddings.rows,
                    actual: *id as usize,
                });
            }
            hidden
                .row_mut(position)
                .copy_from_slice(self.embeddings.row(*id as usize));
        }
        Ok(hidden)
    }

    /// Tokenizes a prompt and returns its hidden states.
    pub fn encode(&self, tokenizer: &Bpe, prompt: &str) -> Result<Matrix, NetworkError> {
        self.forward(&tokenizer.encode(prompt)?)
    }

    /// The single vector a diffusion model takes alongside the per-token
    /// states: the last position's hidden state, which in a causal model is the
    /// one that has read the whole prompt.
    pub fn pool(&self, hidden: &Matrix) -> Vec<f32> {
        hidden.row(hidden.rows - 1).to_vec()
    }
}

/// The rotary table, in the half-split arrangement Hugging Face checkpoints
/// are trained with: dimension `i` pairs with `i + head_dim / 2`, rather than
/// with its neighbour.
pub(crate) struct Rotary {
    pub(crate) cos: Vec<f32>,
    pub(crate) sin: Vec<f32>,
    pub(crate) half: usize,
}

/// A picture spliced into a prompt, the way Qwen3-VL reads one: rows from
/// [`crate::qwen_vision::QwenVision`] in place of the embeddings of the
/// prompt's image tokens.
pub struct Visual {
    /// The rows of the prompt that hold an image token, ascending.
    pub rows: Vec<usize>,
    /// What those rows hold in place of an embedding, `[rows.len(), d_model]`.
    pub embeddings: Matrix,
    /// Added to those rows after the first layers, one matrix a layer.
    pub deepstack: Vec<Matrix>,
    /// Where each leading token sits in time, height and width. Text sits at
    /// one position on all three; a picture's tokens share a time and spread
    /// over the other two. The tokens past the end of this follow one apart
    /// from `next`.
    pub positions: Vec<[usize; 3]>,
    pub next: usize,
}

impl Visual {
    /// A prompt with no picture in it yet, for a model `d_model` wide.
    pub fn new(d_model: usize) -> Self {
        Self {
            rows: Vec::new(),
            embeddings: Matrix::new(0, d_model),
            deepstack: Vec::new(),
            positions: Vec::new(),
            next: 0,
        }
    }

    /// `count` tokens of text come next.
    pub fn text(&mut self, count: usize) {
        self.positions.extend((self.next..self.next + count).map(|at| [at; 3]));
        self.next += count;
    }

    /// A picture comes next: `rows * columns` image tokens in reading order,
    /// with what [`crate::qwen_vision::QwenVision::forward`] made of it.
    pub fn picture(&mut self, rows: usize, columns: usize, seen: (Matrix, Vec<Matrix>)) {
        for row in 0..rows {
            for column in 0..columns {
                self.rows.push(self.positions.len());
                self.positions.push([self.next, self.next + row, self.next + column]);
            }
        }
        // A picture is as long in time as its longer side.
        self.next += rows.max(columns);
        let (embeddings, deepstack) = seen;
        let append = |all: &mut Matrix, more: Matrix| {
            all.data.extend(more.data);
            all.rows += more.rows;
        };
        append(&mut self.embeddings, embeddings);
        for (index, deep) in deepstack.into_iter().enumerate() {
            match self.deepstack.get_mut(index) {
                Some(all) => append(all, deep),
                None => self.deepstack.push(deep),
            }
        }
    }

    /// The rotary table of a prompt of `tokens` rows.
    ///
    /// ponytail: Qwen3-VL's interleaving for `mrope_section` `[24, 20, 20]`,
    /// the only layout this has met: of the first 60 pairs every third turns
    /// with the height and the one after with the width, and the rest turn
    /// with the time. Read `rope_scaling.mrope_section` if another turns up.
    fn rotary(&self, tokens: usize, head_dim: usize, theta: f32) -> Rotary {
        let mut rotary = Rotary::new(tokens, head_dim, theta);
        let half = rotary.half;
        for token in 0..tokens {
            let position = match self.positions.get(token) {
                Some(position) => *position,
                None => [self.next + token - self.positions.len(); 3],
            };
            for index in 0..half {
                let axis = match index % 3 {
                    1 | 2 if index < 60 => index % 3,
                    _ => 0,
                };
                let frequency = theta.powf(-2.0 * index as f32 / head_dim as f32);
                let angle = position[axis] as f32 * frequency;
                rotary.cos[token * half + index] = angle.cos();
                rotary.sin[token * half + index] = angle.sin();
            }
        }
        rotary
    }
}

impl Rotary {
    pub(crate) fn new(tokens: usize, head_dim: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let mut cos = vec![0.0; tokens * half];
        let mut sin = vec![0.0; tokens * half];
        for position in 0..tokens {
            for index in 0..half {
                let frequency = theta.powf(-2.0 * index as f32 / head_dim as f32);
                let angle = position as f32 * frequency;
                cos[position * half + index] = angle.cos();
                sin[position * half + index] = angle.sin();
            }
        }
        Self { cos, sin, half }
    }

    pub(crate) fn apply(&self, values: &mut Matrix, heads: usize) {
        let head_dim = self.half * 2;
        let half = self.half;
        values
            .data
            .par_chunks_mut(heads * head_dim)
            .enumerate()
            .for_each(|(position, row)| {
                let cos = &self.cos[position * half..(position + 1) * half];
                let sin = &self.sin[position * half..(position + 1) * half];
                for head in 0..heads {
                    let head = &mut row[head * head_dim..(head + 1) * head_dim];
                    for index in 0..half {
                        let (first, second) = (head[index], head[index + half]);
                        head[index] = first * cos[index] - second * sin[index];
                        head[index + half] = second * cos[index] + first * sin[index];
                    }
                }
            });
    }
}

/// RMS normalization with a per-channel scale, which is what these models use
/// in place of a layer norm.
pub(crate) fn rms_norm(tokens: &Matrix, weight: &[f32], eps: f32) -> Matrix {
    let mut output = tokens.clone();
    output.data.par_chunks_mut(tokens.cols).for_each(|row| {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
        let inverse = (mean_square + eps).sqrt().recip();
        for (value, weight) in row.iter_mut().zip(weight) {
            *value *= inverse * weight;
        }
    });
    output
}

/// The same, over each attention head rather than each token.
fn normalize_heads(values: &mut Matrix, weight: &[f32], eps: f32) {
    let head_dim = weight.len();
    values.data.par_chunks_mut(values.cols).for_each(|row| {
        for head in row.chunks_mut(head_dim) {
            let mean_square = head.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
            let inverse = (mean_square + eps).sqrt().recip();
            for (value, weight) in head.iter_mut().zip(weight) {
                *value *= inverse * weight;
            }
        }
    });
}

fn dense(
    file: &mut ShardedSafeTensors,
    name: &str,
    precision: Precision,
) -> Result<Dense, NetworkError> {
    let weight = |file: &mut ShardedSafeTensors| -> Result<Matrix, NetworkError> {
        let (values, shape) = file.tensor(&format!("{name}.weight"))?;
        if shape.len() != 2 {
            return Err(NetworkError::InvalidConfig(format!(
                "{name}.weight is {shape:?}, which is not a linear layer"
            )));
        }
        Ok(Matrix::from_vec(shape[0], shape[1], values))
    };
    // Qwen2 carries attention biases and Qwen3 does not, so the bias is read
    // if it is there rather than required.
    let bias = file
        .tensor(&format!("{name}.bias"))
        .ok()
        .map(|(bias, _)| bias);
    if precision != Precision::Q8 {
        return Dense::new(weight(file)?, bias);
    }
    let quantized = file.quantized(&format!("{name}.weight"), |file| {
        Ok(Quantized::from_matrix(&weight(file)?))
    })?;
    Dense::from_quantized(quantized, bias)
}

fn vector(file: &mut ShardedSafeTensors, name: &str) -> Result<Vec<f32>, NetworkError> {
    Ok(file.tensor(name)?.0)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn tiny() -> TextEncoderConfig {
        TextEncoderConfig {
            vocab_size: 16,
            d_model: 16,
            layers: 2,
            num_heads: 4,
            kv_heads: 2,
            head_dim: 4,
            mlp_dim: 32,
            rope_theta: 10_000.0,
            eps: 1e-6,
            causal: true,
        }
    }

    pub(crate) fn checkpoint(config: &TextEncoderConfig, path: &std::path::Path, query_norm: bool) {
        let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let count = shape.iter().product::<usize>();
            let values = (0..count)
                .map(|index| ((index % 13) as f32 - 6.0) * 0.05)
                .collect();
            tensors.insert(name, (shape, values));
        };

        add(
            "model.embed_tokens.weight".into(),
            vec![config.vocab_size, config.d_model],
        );
        for index in 0..config.layers {
            let base = format!("model.layers.{index}");
            add(
                format!("{base}.input_layernorm.weight"),
                vec![config.d_model],
            );
            add(
                format!("{base}.post_attention_layernorm.weight"),
                vec![config.d_model],
            );
            let queries = config.num_heads * config.head_dim;
            let keys = config.kv_heads * config.head_dim;
            add(
                format!("{base}.self_attn.q_proj.weight"),
                vec![queries, config.d_model],
            );
            add(
                format!("{base}.self_attn.k_proj.weight"),
                vec![keys, config.d_model],
            );
            add(
                format!("{base}.self_attn.v_proj.weight"),
                vec![keys, config.d_model],
            );
            add(
                format!("{base}.self_attn.o_proj.weight"),
                vec![config.d_model, queries],
            );
            if query_norm {
                add(
                    format!("{base}.self_attn.q_norm.weight"),
                    vec![config.head_dim],
                );
                add(
                    format!("{base}.self_attn.k_norm.weight"),
                    vec![config.head_dim],
                );
            }
            add(
                format!("{base}.mlp.gate_proj.weight"),
                vec![config.mlp_dim, config.d_model],
            );
            add(
                format!("{base}.mlp.up_proj.weight"),
                vec![config.mlp_dim, config.d_model],
            );
            add(
                format!("{base}.mlp.down_proj.weight"),
                vec![config.d_model, config.mlp_dim],
            );
        }
        add("model.norm.weight".into(), vec![config.d_model]);

        crate::safetensors::write_checkpoint(path, &tensors);
    }

    fn encoder(config: &TextEncoderConfig, name: &str, query_norm: bool) -> TextEncoder {
        let path = std::env::temp_dir().join(name);
        checkpoint(config, &path, query_norm);
        let encoder = TextEncoder::load(&path, "model.", config.clone()).unwrap();
        std::fs::remove_file(path).ok();
        encoder
    }

    #[test]
    fn a_prompt_becomes_one_hidden_state_per_token() {
        let config = tiny();
        let encoder = encoder(&config, "rusting_brain_text_forward.safetensors", false);
        let hidden = encoder.forward(&[1, 5, 9, 2]).unwrap();

        assert_eq!((hidden.rows, hidden.cols), (4, config.d_model));
        assert!(hidden.data.iter().all(|value| value.is_finite()));
        assert_eq!(encoder.pool(&hidden), hidden.row(3).to_vec());
    }

    #[test]
    fn a_causal_encoder_leaves_earlier_states_alone_when_the_prompt_grows() {
        let encoder = encoder(&tiny(), "rusting_brain_text_causal.safetensors", false);
        let short = encoder.forward(&[3, 7]).unwrap();
        let long = encoder.forward(&[3, 7, 1, 4]).unwrap();

        for row in 0..2 {
            for (first, second) in short.row(row).iter().zip(long.row(row)) {
                assert!((first - second).abs() < 1e-4, "{first} vs {second}");
            }
        }
        // The later tokens are new, so the pooled vector moves.
        assert_ne!(encoder.pool(&short), encoder.pool(&long));
    }

    #[test]
    fn stacked_layers_are_the_states_after_each_depth_and_padding_is_not_read() {
        let config = TextEncoderConfig {
            layers: 3,
            ..tiny()
        };
        let encoder = encoder(&config, "rusting_brain_text_layers.safetensors", true);
        let width = config.d_model;

        // The last depth, normed, is the ordinary output.
        let ids = [3, 7, 1, 4];
        let stacked = encoder.forward_layers(&ids, ids.len(), &[1, 3]).unwrap();
        assert_eq!((stacked.rows, stacked.cols), (4, 2 * width));
        let mut last = Matrix::new(4, width);
        for token in 0..4 {
            last.row_mut(token)
                .copy_from_slice(&stacked.row(token)[width..]);
        }
        let normed = rms_norm(&last, &encoder.final_norm, config.eps);
        let plain = encoder.forward(&ids).unwrap();
        for (first, second) in normed.data.iter().zip(&plain.data) {
            assert!((first - second).abs() < 1e-5, "{first} vs {second}");
        }

        // Two real tokens, then padding. No state reads a padded key, so the
        // padding's ids change only the padded positions' own states, and a
        // padded position does not see the padding before it.
        let first = encoder.forward_layers(&[3, 7, 9, 9], 2, &[2, 3]).unwrap();
        let second = encoder.forward_layers(&[3, 7, 5, 9], 2, &[2, 3]).unwrap();
        for token in [0, 1, 3] {
            for (a, b) in first.row(token).iter().zip(second.row(token)) {
                assert!((a - b).abs() < 1e-5, "token {token}: {a} vs {b}");
            }
        }
        assert_ne!(first.row(2), second.row(2));
        assert!(encoder.forward_layers(&ids, 4, &[4]).is_err());
    }

    #[test]
    fn query_normalization_is_read_from_the_checkpoint_and_changes_the_output() {
        let config = tiny();
        let plain = encoder(&config, "rusting_brain_text_plain.safetensors", false);
        let normalized = encoder(&config, "rusting_brain_text_qknorm.safetensors", true);

        let ids = [2, 6, 4];
        let first = plain.forward(&ids).unwrap();
        let second = normalized.forward(&ids).unwrap();
        assert!(
            first
                .data
                .iter()
                .zip(&second.data)
                .any(|(first, second)| (first - second).abs() > 1e-5)
        );
    }

    #[test]
    fn an_impossible_prompt_is_refused() {
        let encoder = encoder(&tiny(), "rusting_brain_text_refuse.safetensors", false);
        assert!(encoder.forward(&[]).is_err());
        assert!(encoder.forward(&[999]).is_err());
    }

    #[test]
    fn a_hugging_face_configuration_reads_back() {
        let path = std::env::temp_dir().join("rusting_brain_text_config.json");
        std::fs::write(
            &path,
            r#"{"hidden_size": 2560, "num_hidden_layers": 36, "num_attention_heads": 32,
                "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 9728,
                "vocab_size": 151936, "rope_theta": 1000000.0, "rms_norm_eps": 1e-6}"#,
        )
        .unwrap();

        let config = TextEncoderConfig::from_file(&path).unwrap();
        assert_eq!(config.d_model, 2560);
        assert_eq!(config.kv_heads, 8);
        assert_eq!(config.head_dim, 128);
        assert_eq!(config.rope_theta, 1e6);
        assert!(config.causal);
        std::fs::remove_file(path).ok();

        assert!(TextEncoderConfig::from_file("/nonexistent/config.json").is_err());
    }
}
