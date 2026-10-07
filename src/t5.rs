//! The T5 encoder, which is the prompt encoder FLUX.1 and Stable Diffusion 3
//! read their per-token states from.
//!
//! It differs from the decoder in [`crate::text_encoder`] in three ways that
//! matter to a reader: attention is bidirectional, positions are not rotary but
//! a learned bias added to every attention score, and the queries are not
//! scaled — T5 folds that scale into its initialization instead, so dividing by
//! the square root of the head width here would quietly change every logit.
//!
//! The position bias is shared: the first block owns the table, and every block
//! after it adds the same numbers.
//!
//! NOTE: inference only, encoder only, no attention mask. The published
//! image pipelines pad a prompt and let the padding be attended, so a mask
//! would produce different numbers from the reference rather than better ones.

use crate::conv::Dense;
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::text_encoder::rms_norm;
use crate::tokenizer::Unigram;
use crate::transformer::Precision;
use rayon::prelude::*;

/// The shape of a T5 encoder, as its `config.json` states it.
#[derive(Clone, Debug, PartialEq)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    /// Width of one attention head, which T5 sets independently of `d_model`.
    pub head_dim: usize,
    pub num_heads: usize,
    pub layers: usize,
    pub mlp_dim: usize,
    pub buckets: usize,
    pub max_distance: usize,
    pub eps: f32,
    /// Whether the feed-forward is the gated pair the 1.1 models use, rather
    /// than the single ReLU projection of the original.
    pub gated: bool,
    /// The id the vocabulary ends a prompt with, and the one it pads with.
    pub eos_token: u32,
    pub pad_token: u32,
}

impl Default for T5Config {
    /// T5 v1.1 XXL, which is the encoder FLUX.1 and Stable Diffusion 3 ship.
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 4096,
            head_dim: 64,
            num_heads: 64,
            layers: 24,
            mlp_dim: 10240,
            buckets: 32,
            max_distance: 128,
            eps: 1e-6,
            gated: true,
            eos_token: 1,
            pad_token: 0,
        }
    }
}

impl T5Config {
    /// Reads a Hugging Face `config.json`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let text = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|error| NetworkError::InvalidDataset(format!("T5 config.json: {error}")))?;

        let default = Self::default();
        let number = |name: &str, fallback: usize| -> usize {
            json.get(name)
                .and_then(|value| value.as_u64())
                .map_or(fallback, |value| value as usize)
        };
        // The original models write `feed_forward_proj: "relu"`; the 1.1 line
        // writes `"gated-gelu"`, and the repacked ones write a flag instead.
        let gated = json
            .get("is_gated_act")
            .and_then(|value| value.as_bool())
            .unwrap_or_else(|| {
                json.get("feed_forward_proj")
                    .and_then(|value| value.as_str())
                    .is_none_or(|name| name.contains("gated"))
            });

        Ok(Self {
            vocab_size: number("vocab_size", default.vocab_size),
            d_model: number("d_model", default.d_model),
            head_dim: number("d_kv", default.head_dim),
            num_heads: number("num_heads", default.num_heads),
            layers: number("num_layers", default.layers),
            mlp_dim: number("d_ff", default.mlp_dim),
            buckets: number("relative_attention_num_buckets", default.buckets),
            max_distance: number("relative_attention_max_distance", default.max_distance),
            eps: json
                .get("layer_norm_epsilon")
                .and_then(|value| value.as_f64())
                .unwrap_or(default.eps as f64) as f32,
            gated,
            eos_token: number("eos_token_id", default.eos_token as usize) as u32,
            pad_token: number("pad_token_id", default.pad_token as usize) as u32,
        })
    }
}

/// One encoder block: bidirectional attention, then a feed-forward, each behind
/// its own root-mean-square norm.
pub(crate) struct Block {
    pub(crate) attention_norm: Vec<f32>,
    pub(crate) query: Dense,
    pub(crate) key: Dense,
    pub(crate) value: Dense,
    pub(crate) output: Dense,
    pub(crate) mlp_norm: Vec<f32>,
    /// The gate of a gated feed-forward, absent in the original models.
    pub(crate) gate: Option<Dense>,
    pub(crate) mlp_in: Dense,
    pub(crate) mlp_out: Dense,
}

/// A T5 encoder.
pub struct T5Encoder {
    pub(crate) config: T5Config,
    pub(crate) tokens: Matrix,
    /// The shared position bias, `[buckets, heads]`.
    pub(crate) position_bias: Matrix,
    pub(crate) blocks: Vec<Block>,
    pub(crate) final_norm: Vec<f32>,
    #[cfg(feature = "cuda")]
    device: Option<crate::cuda_t5::DeviceT5>,
}

impl T5Encoder {
    /// The configuration this encoder was built for.
    pub fn config(&self) -> &T5Config {
        &self.config
    }

    /// Reads an encoder out of a checkpoint.
    pub fn load(path: impl AsRef<std::path::Path>, config: T5Config) -> Result<Self, NetworkError> {
        Self::load_at(path, config, Precision::F32)
    }

    /// The same, holding the projections at a chosen precision.
    pub fn load_at(
        path: impl AsRef<std::path::Path>,
        config: T5Config,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        Self::read_at(&mut file, config, precision)
    }

    /// The same, from a checkpoint that is already open.
    pub fn read_at(
        file: &mut ShardedSafeTensors,
        config: T5Config,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        // An encoder published on its own writes the table bare; one published
        // beside its decoder shares it.
        let tokens = file
            .matrix("shared.weight")
            .or_else(|_| file.matrix("encoder.embed_tokens.weight"))?;
        if tokens.cols != config.d_model {
            return Err(NetworkError::InvalidConfig(format!(
                "the checkpoint's embeddings are {} wide and the configuration says {}",
                tokens.cols, config.d_model
            )));
        }
        let position_bias =
            file.matrix("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")?;

        let blocks = (0..config.layers)
            .map(|index| {
                let attention = format!("encoder.block.{index}.layer.0");
                let mlp = format!("encoder.block.{index}.layer.1");
                Ok(Block {
                    attention_norm: file.tensor(&format!("{attention}.layer_norm.weight"))?.0,
                    query: dense(file, &format!("{attention}.SelfAttention.q"), precision)?,
                    key: dense(file, &format!("{attention}.SelfAttention.k"), precision)?,
                    value: dense(file, &format!("{attention}.SelfAttention.v"), precision)?,
                    output: dense(file, &format!("{attention}.SelfAttention.o"), precision)?,
                    mlp_norm: file.tensor(&format!("{mlp}.layer_norm.weight"))?.0,
                    gate: match config.gated {
                        true => Some(dense(
                            file,
                            &format!("{mlp}.DenseReluDense.wi_1"),
                            precision,
                        )?),
                        false => None,
                    },
                    mlp_in: dense(
                        file,
                        &match config.gated {
                            true => format!("{mlp}.DenseReluDense.wi_0"),
                            false => format!("{mlp}.DenseReluDense.wi"),
                        },
                        precision,
                    )?,
                    mlp_out: dense(file, &format!("{mlp}.DenseReluDense.wo"), precision)?,
                })
            })
            .collect::<Result<Vec<_>, NetworkError>>()?;

        Ok(Self {
            final_norm: file.tensor("encoder.final_layer_norm.weight")?.0,
            config,
            tokens,
            position_bias,
            blocks,
            #[cfg(feature = "cuda")]
            device: None,
        })
    }

    /// The token ids for a prompt: the prompt, the end token, and padding out
    /// to `length`.
    ///
    /// The published pipelines pad every prompt to a fixed length and hand the
    /// padded states to the denoiser, so the length is part of what a model was
    /// tuned for rather than a detail of this call.
    pub fn tokenize(
        &self,
        tokenizer: &Unigram,
        prompt: &str,
        length: usize,
    ) -> Result<Vec<u32>, NetworkError> {
        if length == 0 {
            return Err(NetworkError::InvalidConfig(
                "a prompt padded to nothing has no states".into(),
            ));
        }
        let mut ids = tokenizer.encode(prompt)?;
        ids.truncate(length - 1);
        ids.push(self.config.eos_token);
        ids.resize(length, self.config.pad_token);
        Ok(ids)
    }

    /// Uploads every weight to a CUDA device and runs there from now on.
    ///
    /// Fails closed: an error leaves the encoder on the CPU, as it was.
    #[cfg(feature = "cuda")]
    pub fn attach_device(
        &mut self,
        context: &std::sync::Arc<crate::gpu_transformer::GpuContext>,
    ) -> Result<(), NetworkError> {
        self.device = Some(crate::cuda_t5::DeviceT5::upload(context, self)?);
        Ok(())
    }

    /// The hidden states for a sequence of token ids, `[tokens, d_model]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Matrix, NetworkError> {
        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            return device.forward(self, ids);
        }
        let mut hidden = self.embed(ids)?;
        let bias = self.bias(ids.len());
        for block in &self.blocks {
            let normed = rms_norm(&hidden, &block.attention_norm, self.config.eps);
            let attended = attention(
                &block.query.forward(&normed)?,
                &block.key.forward(&normed)?,
                &block.value.forward(&normed)?,
                self.config.num_heads,
                &bias,
            );
            add(&mut hidden, &block.output.forward(&attended)?);

            let normed = rms_norm(&hidden, &block.mlp_norm, self.config.eps);
            let mut wide = block.mlp_in.forward(&normed)?;
            match &block.gate {
                Some(gate) => {
                    let gate = gate.forward(&normed)?;
                    wide.data
                        .par_iter_mut()
                        .zip(gate.data.par_iter())
                        .for_each(|(value, gate)| *value = crate::ffn::gelu(*value) * gate);
                }
                None => wide
                    .data
                    .par_iter_mut()
                    .for_each(|value| *value = value.max(0.0)),
            }
            add(&mut hidden, &block.mlp_out.forward(&wide)?);
        }

        Ok(rms_norm(&hidden, &self.final_norm, self.config.eps))
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
            if *id as usize >= self.tokens.rows {
                return Err(NetworkError::InvalidTarget {
                    expected: self.tokens.rows,
                    actual: *id as usize,
                });
            }
            hidden
                .row_mut(position)
                .copy_from_slice(self.tokens.row(*id as usize));
        }
        Ok(hidden)
    }

    /// Tokenizing and running in one call.
    pub fn encode(
        &self,
        tokenizer: &Unigram,
        prompt: &str,
        length: usize,
    ) -> Result<Matrix, NetworkError> {
        self.forward(&self.tokenize(tokenizer, prompt, length)?)
    }

    /// The learned position bias for a sequence of this length, one
    /// `[tokens, tokens]` block per head.
    pub(crate) fn bias(&self, tokens: usize) -> Vec<Vec<f32>> {
        (0..self.config.num_heads)
            .map(|head| {
                let mut head_bias = vec![0.0; tokens * tokens];
                for query in 0..tokens {
                    for (key, slot) in head_bias[query * tokens..(query + 1) * tokens]
                        .iter_mut()
                        .enumerate()
                    {
                        let bucket = bucket(
                            key as isize - query as isize,
                            self.config.buckets,
                            self.config.max_distance,
                        );
                        *slot = self
                            .position_bias
                            .row(bucket.min(self.position_bias.rows - 1))[head];
                    }
                }
                head_bias
            })
            .collect()
    }
}

/// Which bucket a relative position falls in: exact for the near half of the
/// buckets, then logarithmically spaced out to `max_distance`, with the two
/// directions taking half the table each.
fn bucket(relative: isize, buckets: usize, max_distance: usize) -> usize {
    let half = buckets / 2;
    let forward = match relative > 0 {
        true => half,
        false => 0,
    };
    let distance = relative.unsigned_abs();
    let exact = half / 2;
    if distance < exact {
        return forward + distance;
    }
    // The far half is spaced so that the last bucket ends at `max_distance`.
    let spread = (distance as f32 / exact as f32).ln() / (max_distance as f32 / exact as f32).ln();
    let large = exact + (spread * (half - exact) as f32) as usize;
    forward + large.min(half - 1)
}

/// Bidirectional attention with a per-head position bias and no query scaling,
/// which is what T5 trains.
fn attention(
    queries: &Matrix,
    keys: &Matrix,
    values: &Matrix,
    num_heads: usize,
    bias: &[Vec<f32>],
) -> Matrix {
    let tokens = queries.rows;
    let head_dim = queries.cols / num_heads;
    let mut output = Matrix::new(tokens, queries.cols);

    let heads: Vec<Vec<f32>> = (0..num_heads)
        .into_par_iter()
        .map(|head| {
            let offset = head * head_dim;
            let mut attended = vec![0.0; tokens * head_dim];
            let mut weights = vec![0.0; tokens];
            for query_token in 0..tokens {
                let query = &queries.row(query_token)[offset..offset + head_dim];
                let row = &bias[head][query_token * tokens..(query_token + 1) * tokens];
                let mut largest = f32::NEG_INFINITY;
                for (key_token, weight) in weights.iter_mut().enumerate() {
                    let key = &keys.row(key_token)[offset..offset + head_dim];
                    *weight = query
                        .iter()
                        .zip(key)
                        .map(|(query, key)| query * key)
                        .sum::<f32>()
                        + row[key_token];
                    largest = largest.max(*weight);
                }
                let mut total = 0.0;
                for weight in weights.iter_mut() {
                    *weight = (*weight - largest).exp();
                    total += *weight;
                }
                let target = &mut attended[query_token * head_dim..(query_token + 1) * head_dim];
                for (key_token, weight) in weights.iter().enumerate() {
                    let weight = weight / total;
                    let value = &values.row(key_token)[offset..offset + head_dim];
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

fn add(target: &mut Matrix, source: &Matrix) {
    target
        .data
        .par_iter_mut()
        .zip(source.data.par_iter())
        .for_each(|(target, source)| *target += source);
}

/// T5's projections carry no bias, which is why none is read.
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
    let mut layer = Dense::new(Matrix::from_vec(shape[0], shape[1], values), None)?;
    if precision == Precision::Q8 {
        layer.quantize();
    }
    Ok(layer)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::BTreeMap;

    pub(crate) fn tiny() -> T5Config {
        T5Config {
            vocab_size: 32,
            d_model: 16,
            head_dim: 4,
            num_heads: 4,
            layers: 2,
            mlp_dim: 32,
            buckets: 8,
            max_distance: 16,
            eps: 1e-6,
            gated: true,
            eos_token: 1,
            pad_token: 0,
        }
    }

    pub(crate) fn checkpoint(config: &T5Config, path: &std::path::Path) {
        let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let count = shape.iter().product::<usize>();
            let values = (0..count)
                .map(|index| ((index % 11) as f32 - 5.0) * 0.05)
                .collect();
            tensors.insert(name, (shape, values));
        };

        let inner = config.num_heads * config.head_dim;
        add(
            "shared.weight".into(),
            vec![config.vocab_size, config.d_model],
        );
        add(
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            vec![config.buckets, config.num_heads],
        );
        for index in 0..config.layers {
            let attention = format!("encoder.block.{index}.layer.0");
            let mlp = format!("encoder.block.{index}.layer.1");
            add(
                format!("{attention}.layer_norm.weight"),
                vec![config.d_model],
            );
            for name in ["q", "k", "v"] {
                add(
                    format!("{attention}.SelfAttention.{name}.weight"),
                    vec![inner, config.d_model],
                );
            }
            add(
                format!("{attention}.SelfAttention.o.weight"),
                vec![config.d_model, inner],
            );
            add(format!("{mlp}.layer_norm.weight"), vec![config.d_model]);
            match config.gated {
                true => {
                    for name in ["wi_0", "wi_1"] {
                        add(
                            format!("{mlp}.DenseReluDense.{name}.weight"),
                            vec![config.mlp_dim, config.d_model],
                        );
                    }
                }
                false => add(
                    format!("{mlp}.DenseReluDense.wi.weight"),
                    vec![config.mlp_dim, config.d_model],
                ),
            }
            add(
                format!("{mlp}.DenseReluDense.wo.weight"),
                vec![config.d_model, config.mlp_dim],
            );
        }
        add(
            "encoder.final_layer_norm.weight".into(),
            vec![config.d_model],
        );

        crate::safetensors::write_checkpoint(path, &tensors);
    }

    fn encoder(config: &T5Config, name: &str, precision: Precision) -> T5Encoder {
        let path = std::env::temp_dir().join(name);
        checkpoint(config, &path);
        let encoder = T5Encoder::load_at(&path, config.clone(), precision).unwrap();
        std::fs::remove_file(path).ok();
        encoder
    }

    #[test]
    fn a_prompt_becomes_one_hidden_state_per_token() {
        let config = tiny();
        let encoder = encoder(
            &config,
            "rusting_brain_t5_forward.safetensors",
            Precision::F32,
        );
        let hidden = encoder.forward(&[5, 9, 3, 1]).unwrap();

        assert_eq!((hidden.rows, hidden.cols), (4, config.d_model));
        assert!(hidden.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn attention_reads_in_both_directions() {
        let config = tiny();
        let encoder = encoder(&config, "rusting_brain_t5_both.safetensors", Precision::F32);
        let first = encoder.forward(&[5, 9, 3, 1]).unwrap();
        // A causal model's first state cannot see the change; this one must.
        let second = encoder.forward(&[5, 9, 7, 1]).unwrap();

        assert_ne!(first.row(0), second.row(0));
    }

    #[test]
    fn a_prompt_ends_with_the_end_token_and_is_padded_to_length() {
        let config = tiny();
        let encoder = encoder(
            &config,
            "rusting_brain_t5_tokens.safetensors",
            Precision::F32,
        );
        let tokenizer = Unigram::from_json(&crate::tokenizer::tests::unigram_json()).unwrap();

        let ids = encoder.tokenize(&tokenizer, "abc", 6).unwrap();
        assert_eq!(ids.len(), 6);
        assert_eq!(ids[1], config.eos_token);
        assert!(ids[2..].iter().all(|id| *id == config.pad_token));

        // A prompt too long for the length keeps room for the end token.
        let long = encoder.tokenize(&tokenizer, "abc abc abc", 3).unwrap();
        assert_eq!(long.len(), 3);
        assert_eq!(long[2], config.eos_token);
        assert!(encoder.tokenize(&tokenizer, "abc", 0).is_err());
    }

    #[test]
    fn the_position_bias_is_exact_nearby_and_logarithmic_far_away() {
        let (buckets, max_distance) = (32, 128);
        // A relative position is the key's place minus the query's, and the
        // two directions take half the table each: keys ahead of the query
        // start at the halfway mark.
        assert_eq!(bucket(0, buckets, max_distance), 0);
        assert_eq!(bucket(-3, buckets, max_distance), 3);
        assert_eq!(bucket(3, buckets, max_distance), buckets / 2 + 3);

        // Past the exact range the buckets widen, and nothing leaves its half.
        let mut last = 0;
        for distance in 1..1000 {
            let bucket = bucket(-distance, buckets, max_distance);
            assert!(bucket >= last && bucket < buckets / 2);
            last = bucket;
        }
        assert_eq!(bucket(-100_000, buckets, max_distance), buckets / 2 - 1);
        assert_eq!(bucket(100_000, buckets, max_distance), buckets - 1);
    }

    #[test]
    fn the_original_feed_forward_is_read_where_a_checkpoint_has_one() {
        let config = T5Config {
            gated: false,
            ..tiny()
        };
        let encoder = encoder(&config, "rusting_brain_t5_relu.safetensors", Precision::F32);
        let hidden = encoder.forward(&[5, 9, 1]).unwrap();

        assert!(hidden.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn a_quantized_encoder_says_what_the_float_one_does() {
        let config = tiny();
        let ids = [5, 9, 3, 7, 1];
        let float = encoder(&config, "rusting_brain_t5_f32.safetensors", Precision::F32)
            .forward(&ids)
            .unwrap();
        let quantized = encoder(&config, "rusting_brain_t5_q8.safetensors", Precision::Q8)
            .forward(&ids)
            .unwrap();

        for (float, quantized) in float.data.iter().zip(&quantized.data) {
            assert!(
                (float - quantized).abs() < 0.05,
                "{float} against {quantized}"
            );
        }
    }

    #[test]
    fn a_published_config_states_the_shape() {
        let path = std::env::temp_dir().join("rusting_brain_t5_config.json");
        std::fs::write(
            &path,
            r#"{"d_model": 4096, "d_ff": 10240, "d_kv": 64, "num_heads": 64, "num_layers": 24,
                "relative_attention_num_buckets": 32, "relative_attention_max_distance": 128,
                "layer_norm_epsilon": 1e-6, "vocab_size": 32128,
                "feed_forward_proj": "gated-gelu"}"#,
        )
        .unwrap();
        let config = T5Config::from_file(&path).unwrap();
        std::fs::remove_file(path).ok();

        assert_eq!(config, T5Config::default());

        // The original line says so in the same field.
        let path = std::env::temp_dir().join("rusting_brain_t5_config_relu.json");
        std::fs::write(&path, r#"{"feed_forward_proj": "relu"}"#).unwrap();
        let config = T5Config::from_file(&path).unwrap();
        std::fs::remove_file(path).ok();
        assert!(!config.gated);
    }
}
