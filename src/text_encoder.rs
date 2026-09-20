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
use crate::mmdit::attention;
use crate::network::NetworkError;
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
struct Layer {
    input_norm: Vec<f32>,
    query: Dense,
    key: Dense,
    value: Dense,
    output: Dense,
    /// Qwen3's per-head query and key normalization. Absent in Qwen2 and
    /// LLaMA, and its absence is read from the checkpoint rather than assumed.
    query_norm: Option<Vec<f32>>,
    key_norm: Option<Vec<f32>>,
    post_norm: Vec<f32>,
    gate: Dense,
    up: Dense,
    down: Dense,
}

/// A decoder language model run as a prompt encoder.
pub struct TextEncoder {
    config: TextEncoderConfig,
    embeddings: Matrix,
    layers: Vec<Layer>,
    final_norm: Vec<f32>,
}

impl TextEncoder {
    /// The configuration this encoder was built for.
    pub fn config(&self) -> &TextEncoderConfig {
        &self.config
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
        })
    }

    /// The hidden states for a sequence of token ids, `[tokens, d_model]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Matrix, NetworkError> {
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

        let (heads, kv_heads, head_dim) = (
            self.config.num_heads,
            self.config.kv_heads,
            self.config.head_dim,
        );
        let rotary = Rotary::new(ids.len(), head_dim, self.config.rope_theta);

        for layer in &self.layers {
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

            let attended = attention(
                &queries,
                &keys,
                &values,
                heads,
                kv_heads,
                self.config.causal,
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
        }

        Ok(rms_norm(&hidden, &self.final_norm, self.config.eps))
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
struct Rotary {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
}

impl Rotary {
    fn new(tokens: usize, head_dim: usize, theta: f32) -> Self {
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

    fn apply(&self, values: &mut Matrix, heads: usize) {
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
    let (values, shape) = file.tensor(&format!("{name}.weight"))?;
    if shape.len() != 2 {
        return Err(NetworkError::InvalidConfig(format!(
            "{name}.weight is {shape:?}, which is not a linear layer"
        )));
    }
    // Qwen2 carries attention biases and Qwen3 does not, so the bias is read
    // if it is there rather than required.
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
