//! The T5 encoder, which is the prompt encoder FLUX.1 and Stable Diffusion 3
//! read their per-token states from.
//!
//! It differs from the decoder in [`crate::text_encoder`] in three ways that
//! matter to a reader: attention is bidirectional, positions are not rotary but
//! a learned bias added to every attention score, and the queries are not
//! scaled — T5 folds that scale into its initialization instead, so dividing by
//! the square root of the head width here would quietly change every logit.
//!
//! In T5 the position bias is shared: the first block owns the table, and every
//! block after it adds the same numbers. umT5 (Wan's prompt encoder) is the
//! same network with a table in every block, and a checkpoint that carries
//! them is read that way.
//!
//! NOTE: inference only, encoder only, no attention mask. FLUX.1 and SD3
//! pad a prompt and let the padding be attended, so a mask would produce
//! different numbers from the reference rather than better ones. Wan masks the
//! padding, which [`T5Encoder::encode_real`] gets by not running it at all.

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
    /// The position bias tables, `[buckets, heads]` each: one shared by every
    /// block (T5), or one per block (umT5).
    pub(crate) position_bias: Vec<Matrix>,
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
        let table = |index: usize| {
            format!("encoder.block.{index}.layer.0.SelfAttention.relative_attention_bias.weight")
        };
        let per_block = config.layers > 1 && file.names().any(|name| name == table(1));
        let position_bias = (0..if per_block { config.layers } else { 1 })
            .map(|index| file.matrix(&table(index)))
            .collect::<Result<Vec<_>, NetworkError>>()?;

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
        let mut ids = self.real_ids(tokenizer, prompt, length)?;
        ids.resize(length, self.config.pad_token);
        Ok(ids)
    }

    /// The same without the padding: the prompt cut to leave room for the end
    /// token, and the end token.
    fn real_ids(
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
        let shared = (self.position_bias.len() == 1).then(|| self.bias(0, ids.len()));
        for index in 0..self.blocks.len() {
            match &shared {
                Some(bias) => self.layer(index, &mut hidden, bias)?,
                None => self.layer(index, &mut hidden, &self.bias(index, ids.len()))?,
            }
        }

        Ok(rms_norm(&hidden, &self.final_norm, self.config.eps))
    }

    /// Block `index` on the host, in place, with the bias it reads.
    pub(crate) fn layer(
        &self,
        index: usize,
        hidden: &mut Matrix,
        bias: &[Vec<f32>],
    ) -> Result<(), NetworkError> {
        let block = &self.blocks[index];
        let normed = rms_norm(hidden, &block.attention_norm, self.config.eps);
        let attended = attention(
            &block.query.forward(&normed)?,
            &block.key.forward(&normed)?,
            &block.value.forward(&normed)?,
            self.config.num_heads,
            bias,
        );
        add(hidden, &block.output.forward(&attended)?);

        let normed = rms_norm(hidden, &block.mlp_norm, self.config.eps);
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
        add(hidden, &block.mlp_out.forward(&wide)?);
        Ok(())
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

    /// The states Wan hands its denoiser: `[length, d_model]`, the prompt's
    /// real tokens (the end token among them) and zero rows after them.
    ///
    /// Wan runs umT5 over the padded ids with the padding masked out of
    /// attention, then zeroes the padding's states. Running only the real
    /// tokens gives the same rows: a masked key adds nothing, and the bias
    /// depends on the distance between two tokens, not on the length.
    pub fn encode_real(
        &self,
        tokenizer: &Unigram,
        prompt: &str,
        length: usize,
    ) -> Result<Matrix, NetworkError> {
        let mut states = self.forward(&self.real_ids(tokenizer, prompt, length)?)?;
        states.data.resize(length * self.config.d_model, 0.0);
        states.rows = length;
        Ok(states)
    }

    /// The learned position bias block `layer` adds for a sequence of this
    /// length, one `[tokens, tokens]` block per head.
    pub(crate) fn bias(&self, layer: usize, tokens: usize) -> Vec<Vec<f32>> {
        let table = &self.position_bias[layer.min(self.position_bias.len() - 1)];
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
                        *slot = table.row(bucket.min(table.rows - 1))[head];
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
        checkpoint_with(config, path, false);
    }

    /// The same, with a position table in every block, as umT5 writes it.
    /// Each block's table differs from the one before it.
    pub(crate) fn checkpoint_with(config: &T5Config, path: &std::path::Path, per_block: bool) {
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

        for index in (1..config.layers).filter(|_| per_block) {
            let count = config.buckets * config.num_heads;
            let values = (0..count)
                .map(|at| ((at * (index + 2) % 13) as f32 - 6.0) * 0.1)
                .collect();
            tensors.insert(
                format!(
                    "encoder.block.{index}.layer.0.SelfAttention.relative_attention_bias.weight"
                ),
                (vec![config.buckets, config.num_heads], values),
            );
        }
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
    fn every_umt5_block_reads_its_own_position_table() {
        let config = tiny();
        let path = std::env::temp_dir().join("rusting_brain_umt5_tables.safetensors");
        checkpoint_with(&config, &path, true);
        let umt5 = T5Encoder::load(&path, config.clone()).unwrap();
        std::fs::remove_file(path).ok();
        let t5 = encoder(
            &config,
            "rusting_brain_umt5_shared.safetensors",
            Precision::F32,
        );

        assert_eq!((umt5.position_bias.len(), t5.position_bias.len()), (2, 1));
        assert_eq!(umt5.bias(0, 5), t5.bias(0, 5));
        assert_ne!(umt5.bias(1, 5), umt5.bias(0, 5));
        // T5 adds block 0's table everywhere.
        assert_eq!(t5.bias(1, 5), t5.bias(0, 5));
        let ids = [5, 9, 3, 1];
        assert_ne!(umt5.forward(&ids).unwrap(), t5.forward(&ids).unwrap());

        // Block by block, the host steps are the forward.
        let mut hidden = umt5.embed(&ids).unwrap();
        for index in 0..config.layers {
            umt5.layer(index, &mut hidden, &umt5.bias(index, ids.len()))
                .unwrap();
        }
        let hidden = rms_norm(&hidden, &umt5.final_norm, config.eps);
        assert_eq!(hidden, umt5.forward(&ids).unwrap());
    }

    #[test]
    fn the_real_tokens_run_alone_and_the_padding_is_zero() {
        let config = tiny();
        let encoder = encoder(&config, "rusting_brain_t5_real.safetensors", Precision::F32);
        let tokenizer = Unigram::from_json(&crate::tokenizer::tests::unigram_json()).unwrap();

        let states = encoder.encode_real(&tokenizer, "abc", 6).unwrap();
        let ids = encoder.tokenize(&tokenizer, "abc", 6).unwrap();
        let real = ids.iter().position(|&id| id == config.eos_token).unwrap() + 1;
        let alone = encoder.forward(&ids[..real]).unwrap();

        assert_eq!((states.rows, states.cols), (6, config.d_model));
        assert_eq!(&states.data[..alone.data.len()], &alone.data[..]);
        assert!(
            states.data[alone.data.len()..]
                .iter()
                .all(|&value| value == 0.0)
        );
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

    /// Relative L2 distance of `got` from `want`.
    fn relative(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let (gap, norm) = got
            .iter()
            .zip(want)
            .fold((0.0f64, 0.0f64), |(g, n), (a, b)| {
                (g + ((a - b) as f64).powi(2), n + (*b as f64).powi(2))
            });
        (gap / norm).sqrt() as f32
    }

    /// Wan 2.2's umT5-XXL at Q8 against transformers, stage by stage, for both
    /// prompts of RustingFrame's `tools/wan_refs.py w1`. `RB_WAN_REF` is the
    /// directory holding `w1.safetensors` and `reference.json`, `RB_WAN_MODEL`
    /// the model (default the local copy), and `RB_WAN_CUDA=1` checks the
    /// device encoder as well. About 9 GB of host memory: the embedding table
    /// is F32 (4.2 GB), the projections int8.
    #[test]
    #[ignore = "needs Wan2.2-TI2V-5B and the output of RustingFrame's tools/wan_refs.py w1"]
    fn umt5_xxl_matches_transformers() {
        let env = |name: &str| std::env::var(name).ok();
        let dir = std::path::PathBuf::from(env("RB_WAN_REF").expect("RB_WAN_REF"));
        let model = std::path::PathBuf::from(
            env("RB_WAN_MODEL")
                .unwrap_or_else(|| "/mnt/ollama/RustingFrameModels/Wan2.2-TI2V-5B".into()),
        );
        let settings: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).unwrap())
                .unwrap();
        let mut reference =
            crate::safetensors::SafeTensors::open(dir.join("w1.safetensors")).unwrap();
        let mut read = |name: &str| reference.tensor(name).unwrap().0;
        let mut report = Vec::new();
        let mut gate = |stage: &str, error: f32, limit: f32| {
            eprintln!("umt5 {stage}: relative L2 {error:.3e} (gate {limit:.0e})");
            report.push((stage.to_string(), error, limit));
        };

        let tokenizer = Unigram::from_file(model.join("tokenizer/tokenizer.json")).unwrap();
        let config = T5Config::from_file(model.join("text_encoder/config.json")).unwrap();
        let started = std::time::Instant::now();
        let encoder = T5Encoder::load_at(
            model.join("text_encoder/model.safetensors.index.json"),
            config,
            Precision::Q8,
        )
        .unwrap();
        eprintln!("umt5 load Q8: {:?}", started.elapsed());
        assert_eq!(
            encoder.position_bias.len(),
            24,
            "umT5 has a table per block"
        );
        let length = settings["w1"]["max_sequence_length"].as_u64().unwrap() as usize;
        let d_model = encoder.config.d_model;

        // The table block 0 adds at the full length; the others differ only in
        // which table they read.
        let bias = encoder.bias(0, length).concat();
        gate(
            "layer 0 position bias",
            relative(&bias, &read("layer0.position_bias")),
            1e-6,
        );
        drop(bias);

        #[cfg(feature = "cuda")]
        let device = (env("RB_WAN_CUDA").as_deref() == Some("1")).then(|| {
            let context = crate::gpu_transformer::GpuContext::new(0).unwrap();
            let started = std::time::Instant::now();
            let device = crate::cuda_t5::DeviceT5::upload(&context, &encoder).unwrap();
            eprintln!("umt5 upload Q8: {:?}", started.elapsed());
            device
        });

        for name in ["short", "long"] {
            let ids = encoder
                .tokenize(
                    &tokenizer,
                    settings["w1"]["prompts"][name].as_str().unwrap(),
                    length,
                )
                .unwrap();
            let want: Vec<u32> = read(&format!("{name}.input_ids"))
                .iter()
                .map(|&id| id as u32)
                .collect();
            let real = read(&format!("{name}.attention_mask"))
                .iter()
                .filter(|&&m| m > 0.0)
                .count();
            assert_eq!(ids, want, "{name}: token ids");
            assert_eq!(
                real,
                settings["w1"][format!("{name}_tokens")].as_u64().unwrap() as usize
            );

            let embed = encoder.embed(&ids).unwrap();
            gate(
                &format!("{name} embed"),
                relative(&embed.data, &read(&format!("{name}.embed"))),
                1e-6,
            );

            // Past the real tokens transformers' rows are masked queries, which
            // Wan zeroes; only the real rows are compared.
            let mut hidden = encoder.embed(&ids[..real]).unwrap();
            encoder
                .layer(0, &mut hidden, &encoder.bias(0, real))
                .unwrap();
            let layer0 = read(&format!("{name}.layer0"));
            gate(
                &format!("{name} layer 0, Q8"),
                relative(&hidden.data, &layer0[..real * d_model]),
                5e-3,
            );

            let prompt = settings["w1"]["prompts"][name].as_str().unwrap();
            let started = std::time::Instant::now();
            let states = encoder.encode_real(&tokenizer, prompt, length).unwrap();
            eprintln!(
                "umt5 {name} host encode Q8 ({real} tokens): {:?}",
                started.elapsed()
            );
            let embeds = read(&format!("{name}.prompt_embeds"));
            gate(
                &format!("{name} prompt embeds, host Q8"),
                relative(&states.data, &embeds),
                3e-2,
            );

            #[cfg(feature = "cuda")]
            if let Some(device) = &device {
                device.forward(&encoder, &ids[..real]).unwrap();
                let mut times = Vec::new();
                let mut got = None;
                for _ in 0..3 {
                    let started = std::time::Instant::now();
                    got = Some(device.forward(&encoder, &ids[..real]).unwrap());
                    times.push(started.elapsed());
                }
                eprintln!("umt5 {name} device encode Q8 ({real} tokens): {times:?}");
                let mut got = got.unwrap();
                got.data.resize(length * d_model, 0.0);
                gate(
                    &format!("{name} prompt embeds, device Q8"),
                    relative(&got.data, &embeds),
                    3e-2,
                );
            }
        }

        let failed: Vec<_> = report
            .iter()
            .filter(|(_, error, limit)| error > limit)
            .collect();
        assert!(failed.is_empty(), "over the gate: {failed:?}");
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
