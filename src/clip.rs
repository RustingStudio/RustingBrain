//! The CLIP text tower, which is the prompt encoder every Stable Diffusion
//! model and FLUX.1 read their pooled vector from.
//!
//! It is a small causal transformer with learned positions and LayerNorm,
//! rather than the rotary RMSNorm arrangement [`crate::text_encoder`] runs. The
//! two exist side by side because published models pair them: FLUX.1 takes its
//! per-token states from T5 and its pooled vector from here, and Stable
//! Diffusion XL concatenates the states of two of these.
//!
//! The pooled vector is the hidden state at the end-of-text token, not the last
//! position: a prompt is padded, and the padding has read the whole prompt too.
//!
//! The image tower is [`crate::vit_encoder::VitEncoder`].
//! [`zero_shot`] compares the two towers' vectors, with [`logit_scale`] read
//! from the same checkpoint.
//!
//! NOTE: inference only, `f32` or one byte per weight.

use crate::conv::Dense;
use crate::matrix::Matrix;
use crate::mmdit::{attention, layer_norm};
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::tokenizer::Bpe;
use crate::transformer::Precision;
use rayon::prelude::*;

/// The shape of a CLIP text tower, as its `config.json` states it.
#[derive(Clone, Debug, PartialEq)]
pub struct ClipTextConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub layers: usize,
    pub num_heads: usize,
    pub mlp_dim: usize,
    /// How long the learned position table is, which is also how long a prompt
    /// is padded to.
    pub max_positions: usize,
    pub eps: f32,
    /// CLIP's own `x * sigmoid(1.702 * x)`, which the published towers use in
    /// place of the usual GELU.
    pub quick_gelu: bool,
    /// The width the pooled vector is projected to, when the checkpoint carries
    /// a projection at all.
    pub projection_dim: Option<usize>,
    pub bos_token: u32,
    pub eos_token: u32,
}

impl Default for ClipTextConfig {
    /// The CLIP ViT-L/14 text tower, which is the one Stable Diffusion 1.x and
    /// FLUX.1 carry.
    fn default() -> Self {
        Self {
            vocab_size: 49408,
            d_model: 768,
            layers: 12,
            num_heads: 12,
            mlp_dim: 3072,
            max_positions: 77,
            eps: 1e-5,
            quick_gelu: true,
            projection_dim: None,
            bos_token: 49406,
            eos_token: 49407,
        }
    }
}

impl ClipTextConfig {
    /// Reads a Hugging Face `text_encoder/config.json`.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, NetworkError> {
        let text = std::fs::read_to_string(path)?;
        let json: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
            NetworkError::InvalidDataset(format!("text encoder config.json: {error}"))
        })?;
        // A whole pipeline's configuration nests the tower's own.
        let json = json.get("text_config").unwrap_or(&json).clone();

        let default = Self::default();
        let number = |name: &str, fallback: usize| -> usize {
            json.get(name)
                .and_then(|value| value.as_u64())
                .map_or(fallback, |value| value as usize)
        };
        let token = |name: &str, fallback: u32| -> u32 {
            json.get(name)
                .and_then(|value| value.as_u64())
                .map_or(fallback, |value| value as u32)
        };

        Ok(Self {
            vocab_size: number("vocab_size", default.vocab_size),
            d_model: number("hidden_size", default.d_model),
            layers: number("num_hidden_layers", default.layers),
            num_heads: number("num_attention_heads", default.num_heads),
            mlp_dim: number("intermediate_size", default.mlp_dim),
            max_positions: number("max_position_embeddings", default.max_positions),
            eps: json
                .get("layer_norm_eps")
                .and_then(|value| value.as_f64())
                .unwrap_or(default.eps as f64) as f32,
            quick_gelu: json
                .get("hidden_act")
                .and_then(|value| value.as_str())
                .is_none_or(|name| name == "quick_gelu"),
            // A tower with no projection is not an error: only the ones used as
            // a pooled encoder on their own carry one.
            projection_dim: json
                .get("projection_dim")
                .and_then(|value| value.as_u64())
                .map(|value| value as usize),
            bos_token: token("bos_token_id", default.bos_token),
            eos_token: token("eos_token_id", default.eos_token),
        }
        .legacy_special_tokens())
    }

    /// Older configurations (OpenAI's CLIP, Stable Diffusion 1.x, FLUX.1's
    /// CLIP-L) say `bos 0, eos 2`, which are ordinary tokens. Transformers
    /// treats `eos_token_id == 2` as that legacy case and pools at the highest
    /// id, which in CLIP's vocabulary is the end token, the last entry; the
    /// start token sits just before it.
    fn legacy_special_tokens(mut self) -> Self {
        if self.eos_token == 2 && self.vocab_size >= 2 {
            self.bos_token = self.vocab_size as u32 - 2;
            self.eos_token = self.vocab_size as u32 - 1;
        }
        self
    }

    /// Dimensions per attention head.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads.max(1)
    }
}

/// One transformer layer: attention, then feed-forward, each behind its own
/// layer norm.
pub(crate) struct Layer {
    pub(crate) attention_norm: Norm,
    pub(crate) query: Dense,
    pub(crate) key: Dense,
    pub(crate) value: Dense,
    pub(crate) output: Dense,
    pub(crate) mlp_norm: Norm,
    pub(crate) mlp_in: Dense,
    pub(crate) mlp_out: Dense,
}

/// LayerNorm with the learned scale and offset CLIP trains, which is the same
/// layer the UNet's transformer blocks use.
pub(crate) struct Norm {
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

impl Norm {
    pub(crate) fn forward(&self, tokens: &Matrix, eps: f32) -> Matrix {
        let mut output = layer_norm(tokens, eps);
        output.data.par_chunks_mut(tokens.cols).for_each(|row| {
            for ((value, weight), bias) in row.iter_mut().zip(&self.weight).zip(&self.bias) {
                *value = *value * weight + bias;
            }
        });
        output
    }
}

/// A CLIP text tower.
pub struct ClipTextEncoder {
    config: ClipTextConfig,
    tokens: Matrix,
    positions: Matrix,
    pub(crate) layers: Vec<Layer>,
    pub(crate) final_norm: Norm,
    projection: Option<Dense>,
    /// The token a prompt is padded with, which is the end token unless the
    /// tokenizer names another.
    pad: u32,
    /// The same weights on a CUDA device, when one was attached.
    #[cfg(feature = "cuda")]
    device: Option<crate::cuda_image::DeviceClip>,
}

impl ClipTextEncoder {
    /// The configuration this tower was built for.
    pub fn config(&self) -> &ClipTextConfig {
        &self.config
    }

    /// Reads a tower out of a checkpoint.
    ///
    /// `prefix` is what the keys start with, which is empty for the files
    /// published as `text_encoder/model.safetensors`.
    pub fn load(
        path: impl AsRef<std::path::Path>,
        prefix: &str,
        config: ClipTextConfig,
    ) -> Result<Self, NetworkError> {
        Self::load_at(path, prefix, config, Precision::F32)
    }

    /// The same, holding the projections at a chosen precision.
    pub fn load_at(
        path: impl AsRef<std::path::Path>,
        prefix: &str,
        config: ClipTextConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let mut file = ShardedSafeTensors::open(path)?;
        Self::read_at(&mut file, prefix, config, precision)
    }

    /// The same, from a checkpoint that is already open.
    pub fn read_at(
        file: &mut ShardedSafeTensors,
        prefix: &str,
        config: ClipTextConfig,
        precision: Precision,
    ) -> Result<Self, NetworkError> {
        let base = format!("{prefix}text_model");
        let tokens = file.matrix(&format!("{base}.embeddings.token_embedding.weight"))?;
        let positions = file.matrix(&format!("{base}.embeddings.position_embedding.weight"))?;
        if tokens.cols != config.d_model {
            return Err(NetworkError::InvalidConfig(format!(
                "the checkpoint's embeddings are {} wide and the configuration says {}",
                tokens.cols, config.d_model
            )));
        }

        let eos = config.eos_token;
        let layers = (0..config.layers)
            .map(|index| {
                let layer = format!("{base}.encoder.layers.{index}");
                Ok(Layer {
                    attention_norm: norm(file, &format!("{layer}.layer_norm1"))?,
                    query: dense(file, &format!("{layer}.self_attn.q_proj"), precision)?,
                    key: dense(file, &format!("{layer}.self_attn.k_proj"), precision)?,
                    value: dense(file, &format!("{layer}.self_attn.v_proj"), precision)?,
                    output: dense(file, &format!("{layer}.self_attn.out_proj"), precision)?,
                    mlp_norm: norm(file, &format!("{layer}.layer_norm2"))?,
                    mlp_in: dense(file, &format!("{layer}.mlp.fc1"), precision)?,
                    mlp_out: dense(file, &format!("{layer}.mlp.fc2"), precision)?,
                })
            })
            .collect::<Result<Vec<_>, NetworkError>>()?;

        Ok(Self {
            final_norm: norm(file, &format!("{base}.final_layer_norm"))?,
            // The projection lives outside the tower and only some checkpoints
            // carry it, so its absence is read rather than required.
            projection: file
                .matrix(&format!("{prefix}text_projection.weight"))
                .ok()
                .map(|weight| Dense::new(weight, None))
                .transpose()?,
            config,
            tokens,
            pad: eos,
            positions,
            layers,
            #[cfg(feature = "cuda")]
            device: None,
        })
    }

    /// The token ids for a prompt: the start token, the prompt, the end token,
    /// and padding out to the length the position table covers.
    ///
    /// Padding is what the published pipelines feed, and the pooled vector is
    /// read from the end token rather than the last position because of it.
    /// The padding is end tokens unless [`ClipTextEncoder::set_pad_token`]
    /// said otherwise. Attention is causal, so the padding cannot change the
    /// pooled state, but a denoiser cross-attends to every position, padding
    /// included, so it has to be the token the model was trained with.
    pub fn tokenize(&self, tokenizer: &Bpe, prompt: &str) -> Result<Vec<u32>, NetworkError> {
        let mut ids = vec![self.config.bos_token];
        // Two of the positions are the start and end tokens.
        let room = self.config.max_positions.saturating_sub(2);
        ids.extend(tokenizer.encode(prompt)?.into_iter().take(room));
        ids.push(self.config.eos_token);
        ids.resize(self.config.max_positions, self.pad);
        Ok(ids)
    }

    /// Pads prompts with this token from now on. Stable Diffusion XL's second
    /// tokenizer pads with `!` rather than the end token, and its UNet reads
    /// the padded positions.
    pub fn set_pad_token(&mut self, id: u32) {
        self.pad = id;
    }

    /// The hidden states for a sequence of token ids, `[tokens, d_model]`.
    pub fn forward(&self, ids: &[u32]) -> Result<Matrix, NetworkError> {
        Ok(self.forward_skipping(ids, 0)?.1)
    }

    /// The same pass, also handing back the states as they were `skip` layers
    /// from the end.
    ///
    /// Stable Diffusion XL reads its prompt from the second-to-last layer and
    /// its pooled vector from the end of the tower, so both come back here and
    /// the tower runs once. `skip` of zero makes the two the same states.
    pub fn forward_skipping(
        &self,
        ids: &[u32],
        skip: usize,
    ) -> Result<(Matrix, Matrix), NetworkError> {
        if ids.is_empty() {
            return Err(NetworkError::InvalidConfig(
                "an empty prompt has no hidden states".into(),
            ));
        }
        if ids.len() > self.positions.rows {
            return Err(NetworkError::InvalidTarget {
                expected: self.positions.rows,
                actual: ids.len(),
            });
        }

        let mut hidden = Matrix::new(ids.len(), self.config.d_model);
        for (position, id) in ids.iter().enumerate() {
            if *id as usize >= self.tokens.rows {
                return Err(NetworkError::InvalidTarget {
                    expected: self.tokens.rows,
                    actual: *id as usize,
                });
            }
            let row = hidden.row_mut(position);
            row.copy_from_slice(self.tokens.row(*id as usize));
            for (value, learned) in row.iter_mut().zip(self.positions.row(position)) {
                *value += learned;
            }
        }

        #[cfg(feature = "cuda")]
        if let Some(device) = &self.device {
            // The embedding gather above is a table lookup per token, so it
            // stays here and the device gets the one buffer it produced.
            return device.forward(&hidden, skip);
        }

        let heads = self.config.num_heads;
        let stop = self.layers.len().saturating_sub(skip);
        let mut skipped = None;
        for (index, layer) in self.layers.iter().enumerate() {
            // The states as they were before this layer ran are what a
            // pipeline that skips the last layers reads.
            if index == stop {
                skipped = Some(hidden.clone());
            }
            run_layer(
                layer,
                &mut hidden,
                heads,
                self.config.eps,
                self.config.quick_gelu,
                true,
            )?;
        }

        let normalized = self.final_norm.forward(&hidden, self.config.eps);
        // The layers that were skipped were skipped before the final norm,
        // which is what the reference pipeline reads.
        Ok((skipped.unwrap_or_else(|| normalized.clone()), normalized))
    }

    /// Uploads every layer to a CUDA device and runs there from now on.
    ///
    /// Fails closed, leaving the tower on the CPU if the upload does not fit.
    #[cfg(feature = "cuda")]
    pub fn attach_device(
        &mut self,
        gpu: std::sync::Arc<crate::cuda_image::ImageGpu>,
    ) -> Result<(), NetworkError> {
        self.device = Some(crate::cuda_image::DeviceClip::upload(&gpu, self)?);
        Ok(())
    }

    /// The pooled vector: the hidden state at the end token, through the
    /// projection if the checkpoint has one.
    pub fn pool(&self, ids: &[u32], hidden: &Matrix) -> Result<Vec<f32>, NetworkError> {
        // The first end token, because everything after it is padding.
        let position = ids
            .iter()
            .position(|id| *id == self.config.eos_token)
            .unwrap_or(hidden.rows - 1)
            .min(hidden.rows - 1);
        let pooled = hidden.row(position).to_vec();
        match &self.projection {
            Some(projection) => projection.apply(&pooled),
            None => Ok(pooled),
        }
    }

    /// Tokenizing, running and pooling in one call.
    pub fn encode(
        &self,
        tokenizer: &Bpe,
        prompt: &str,
    ) -> Result<(Matrix, Vec<f32>), NetworkError> {
        let ids = self.tokenize(tokenizer, prompt)?;
        let hidden = self.forward(&ids)?;
        let pooled = self.pool(&ids, &hidden)?;
        Ok((hidden, pooled))
    }
}

/// Runs one layer in place: attention behind its norm, then the feed-forward
/// behind its own.
///
/// Shared with the image tower in [`crate::vit_encoder`], which is the same
/// layer with `causal` off. The two towers differ in what they put in front of
/// the stack, not in the stack.
pub(crate) fn run_layer(
    layer: &Layer,
    hidden: &mut Matrix,
    heads: usize,
    eps: f32,
    quick_gelu: bool,
    causal: bool,
) -> Result<(), NetworkError> {
    let normed = layer.attention_norm.forward(hidden, eps);
    let attended = attention(
        &layer.query.forward(&normed)?,
        &layer.key.forward(&normed)?,
        &layer.value.forward(&normed)?,
        heads,
        heads,
        causal,
    );
    add(hidden, &layer.output.forward(&attended)?);

    let normed = layer.mlp_norm.forward(hidden, eps);
    let mut wide = layer.mlp_in.forward(&normed)?;
    wide.data.par_iter_mut().for_each(|value| {
        *value = match quick_gelu {
            true => quick_gelu_value(*value),
            false => crate::ffn::gelu(*value),
        }
    });
    add(hidden, &layer.mlp_out.forward(&wide)?);
    Ok(())
}

/// `x * sigmoid(1.702 * x)`, the activation the published towers were trained
/// with.
fn quick_gelu_value(value: f32) -> f32 {
    value / (1.0 + (-1.702 * value).exp())
}

pub(crate) fn add(target: &mut Matrix, source: &Matrix) {
    target
        .data
        .par_iter_mut()
        .zip(source.data.par_iter())
        .for_each(|(target, source)| *target += source);
}

/// The temperature a full CLIP checkpoint learned, `exp(logit_scale)`, which
/// is the multiplier [`zero_shot`] takes. About 100 for the published models.
pub fn logit_scale(file: &mut ShardedSafeTensors, prefix: &str) -> Result<f32, NetworkError> {
    let (values, _) = file.tensor(&format!("{prefix}logit_scale"))?;
    match values[..] {
        [value] => Ok(value.exp()),
        _ => Err(NetworkError::InvalidTarget {
            expected: 1,
            actual: values.len(),
        }),
    }
}

/// Zero-shot classification: how likely each caption is to describe the
/// image, as CLIP's `logits_per_image` through a softmax.
///
/// `image` is [`VitEncoder::embed`](crate::vit_encoder::VitEncoder::embed) and
/// each of `captions` is [`ClipTextEncoder::pool`], both from the same
/// checkpoint; `scale` is [`logit_scale`]. For retrieval, rank by the same
/// cosine similarities without the softmax.
pub fn zero_shot(image: &[f32], captions: &[Vec<f32>], scale: f32) -> Vec<f32> {
    let unit = |vector: &[f32]| {
        let length = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        vector
            .iter()
            .map(|value| value / length)
            .collect::<Vec<_>>()
    };
    let image = unit(image);
    let logits: Vec<f32> = captions
        .iter()
        .map(|caption| {
            scale
                * unit(caption)
                    .iter()
                    .zip(&image)
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
        })
        .collect();
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = logits.iter().map(|logit| (logit - top).exp()).collect();
    let total: f32 = weights.iter().sum();
    weights.iter().map(|weight| weight / total).collect()
}

pub(crate) fn norm(file: &mut ShardedSafeTensors, name: &str) -> Result<Norm, NetworkError> {
    Ok(Norm {
        weight: file.tensor(&format!("{name}.weight"))?.0,
        bias: file.tensor(&format!("{name}.bias"))?.0,
    })
}

pub(crate) fn dense(
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

    pub(crate) fn tiny() -> ClipTextConfig {
        ClipTextConfig {
            vocab_size: 24,
            d_model: 16,
            layers: 2,
            num_heads: 4,
            mlp_dim: 32,
            max_positions: 8,
            eps: 1e-5,
            quick_gelu: true,
            projection_dim: None,
            bos_token: 20,
            eos_token: 21,
        }
    }

    pub(crate) fn checkpoint(config: &ClipTextConfig, path: &std::path::Path, projection: bool) {
        let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let count = shape.iter().product::<usize>();
            let values = (0..count)
                .map(|index| ((index % 11) as f32 - 5.0) * 0.05)
                .collect();
            tensors.insert(name, (shape, values));
        };

        add(
            "text_model.embeddings.token_embedding.weight".into(),
            vec![config.vocab_size, config.d_model],
        );
        add(
            "text_model.embeddings.position_embedding.weight".into(),
            vec![config.max_positions, config.d_model],
        );
        for index in 0..config.layers {
            let base = format!("text_model.encoder.layers.{index}");
            for norm in ["layer_norm1", "layer_norm2"] {
                add(format!("{base}.{norm}.weight"), vec![config.d_model]);
                add(format!("{base}.{norm}.bias"), vec![config.d_model]);
            }
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                add(
                    format!("{base}.self_attn.{name}.weight"),
                    vec![config.d_model, config.d_model],
                );
                add(
                    format!("{base}.self_attn.{name}.bias"),
                    vec![config.d_model],
                );
            }
            add(
                format!("{base}.mlp.fc1.weight"),
                vec![config.mlp_dim, config.d_model],
            );
            add(format!("{base}.mlp.fc1.bias"), vec![config.mlp_dim]);
            add(
                format!("{base}.mlp.fc2.weight"),
                vec![config.d_model, config.mlp_dim],
            );
            add(format!("{base}.mlp.fc2.bias"), vec![config.d_model]);
        }
        add(
            "text_model.final_layer_norm.weight".into(),
            vec![config.d_model],
        );
        add(
            "text_model.final_layer_norm.bias".into(),
            vec![config.d_model],
        );
        if let Some(width) = config.projection_dim.filter(|_| projection) {
            add("text_projection.weight".into(), vec![width, config.d_model]);
        }

        crate::safetensors::write_checkpoint(path, &tensors);
    }

    fn encoder(config: &ClipTextConfig, name: &str, precision: Precision) -> ClipTextEncoder {
        let path = std::env::temp_dir().join(name);
        checkpoint(config, &path, true);
        let encoder = ClipTextEncoder::load_at(&path, "", config.clone(), precision).unwrap();
        std::fs::remove_file(path).ok();
        encoder
    }

    #[test]
    fn a_prompt_becomes_one_hidden_state_per_position() {
        let config = tiny();
        let encoder = encoder(
            &config,
            "rusting_brain_clip_forward.safetensors",
            Precision::F32,
        );
        let hidden = encoder.forward(&[20, 3, 7, 21]).unwrap();

        assert_eq!((hidden.rows, hidden.cols), (4, config.d_model));
        assert!(hidden.data.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn the_pooled_vector_is_read_at_the_end_token_and_not_the_last_position() {
        let config = tiny();
        let encoder = encoder(
            &config,
            "rusting_brain_clip_pool.safetensors",
            Precision::F32,
        );
        // Two real tokens, an end token, and then padding.
        let ids = [20, 3, 7, 21, 21, 21, 21, 21];
        let hidden = encoder.forward(&ids).unwrap();
        let pooled = encoder.pool(&ids, &hidden).unwrap();

        assert_eq!(pooled, hidden.row(3).to_vec());
        assert_ne!(pooled, hidden.row(hidden.rows - 1).to_vec());
    }

    #[test]
    fn a_prompt_is_wrapped_and_padded_to_the_position_table() {
        let config = tiny();
        let encoder = encoder(
            &config,
            "rusting_brain_clip_tokens.safetensors",
            Precision::F32,
        );
        let tokenizer = crate::tokenizer::tests::tokenizer(&["h i"], &["hi"], None);
        let ids = encoder.tokenize(&tokenizer, "hi").unwrap();

        assert_eq!(ids.len(), config.max_positions);
        assert_eq!(ids[0], config.bos_token);
        assert!(ids[1..].contains(&config.eos_token));
        // A prompt longer than the table still fits, with room kept for both
        // of the markers.
        let long = encoder.tokenize(&tokenizer, "hihihihihihi").unwrap();
        assert_eq!(long.len(), config.max_positions);
        assert_eq!(long[config.max_positions - 1], config.eos_token);
    }

    #[test]
    fn a_named_pad_token_fills_after_the_end_token() {
        let config = tiny();
        let mut encoder = encoder(
            &config,
            "rusting_brain_clip_pad.safetensors",
            Precision::F32,
        );
        let tokenizer = crate::tokenizer::tests::tokenizer(&["h i"], &["hi"], None);
        encoder.set_pad_token(0);
        let ids = encoder.tokenize(&tokenizer, "hi").unwrap();

        let end = ids.iter().position(|id| *id == config.eos_token).unwrap();
        assert!(ids[end + 1..].iter().all(|id| *id == 0));
    }

    #[test]
    fn zero_shot_is_a_softmax_over_scaled_cosines() {
        let image = [3.0, 0.0];
        // Lengths differ on purpose: only the direction may count.
        let captions = [vec![2.0, 0.0], vec![0.0, 5.0], vec![-0.5, 0.0]];

        let probabilities = zero_shot(&image, &captions, 1.0);
        let total: f32 = [1.0f32, 0.0, -1.0].iter().map(|x| x.exp()).sum();
        for (got, logit) in probabilities.iter().zip([1.0f32, 0.0, -1.0]) {
            assert!((got - logit.exp() / total).abs() < 1e-6, "{got}");
        }
        assert!(zero_shot(&image, &captions, 100.0)[0] > 0.999);
    }

    #[test]
    fn a_projection_widens_the_pooled_vector() {
        let mut config = tiny();
        config.projection_dim = Some(6);
        let encoder = encoder(
            &config,
            "rusting_brain_clip_projection.safetensors",
            Precision::F32,
        );
        let ids = [20, 3, 21];
        let hidden = encoder.forward(&ids).unwrap();

        assert_eq!(encoder.pool(&ids, &hidden).unwrap().len(), 6);
    }

    #[test]
    fn a_quantized_tower_says_what_the_float_one_does() {
        let config = tiny();
        let ids = [20, 3, 7, 12, 21];
        let float = encoder(
            &config,
            "rusting_brain_clip_f32.safetensors",
            Precision::F32,
        )
        .forward(&ids)
        .unwrap();
        let quantized = encoder(&config, "rusting_brain_clip_q8.safetensors", Precision::Q8)
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
        let path = std::env::temp_dir().join("rusting_brain_clip_config.json");
        std::fs::write(
            &path,
            r#"{"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 20,
                "intermediate_size": 5120, "max_position_embeddings": 77, "vocab_size": 49408,
                "layer_norm_eps": 1e-5, "hidden_act": "gelu", "projection_dim": 1280,
                "bos_token_id": 0, "eos_token_id": 2}"#,
        )
        .unwrap();
        let config = ClipTextConfig::from_file(&path).unwrap();
        std::fs::remove_file(path).ok();

        assert_eq!(config.d_model, 1280);
        assert_eq!(config.layers, 32);
        assert_eq!(config.head_dim(), 64);
        assert_eq!(config.projection_dim, Some(1280));
        // The legacy `bos 0, eos 2` means the vocabulary's last two entries.
        assert_eq!((config.bos_token, config.eos_token), (49406, 49407));
        assert!(!config.quick_gelu);
    }

    #[test]
    fn a_token_past_the_vocabulary_is_refused() {
        let config = tiny();
        let encoder = encoder(
            &config,
            "rusting_brain_clip_range.safetensors",
            Precision::F32,
        );

        assert!(encoder.forward(&[20, 900, 21]).is_err());
        assert!(encoder.forward(&[]).is_err());
        assert!(encoder.forward(&[20; 9]).is_err());
    }

    #[test]
    fn skipping_the_last_layer_reads_a_state_the_end_of_the_tower_does_not() {
        let config = ClipTextConfig {
            vocab_size: 320,
            bos_token: 300,
            eos_token: 301,
            ..tiny()
        };
        let encoder = encoder(
            &config,
            "rusting_brain_clip_skip.safetensors",
            Precision::F32,
        );
        let ids = encoder
            .tokenize(
                &crate::tokenizer::tests::tokenizer(&["h i"], &["hi"], None),
                "a lighthouse",
            )
            .unwrap();
        let (skipped, normalized) = encoder.forward_skipping(&ids, 1).unwrap();

        assert_eq!(skipped.rows, normalized.rows);
        assert_ne!(skipped.data, normalized.data);
        // Skipping nothing hands back the end of the tower twice.
        let (states, end) = encoder.forward_skipping(&ids, 0).unwrap();
        assert_eq!(states.data, end.data);
        assert_eq!(end.data, normalized.data);
    }
}
