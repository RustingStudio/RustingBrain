//! Reading and writing a [`TransformerLm`] in the layout Hugging Face publishes
//! LLaMA-family models in: a `config.json` beside a `model.safetensors`.
//!
//! The two models are already the same shape. Projections are stored
//! `[out, in]` on both sides, rotary positions use the same "rotate half"
//! pairing, and RMSNorm scales by its weight directly. So this module only
//! names each parameter the way the published checkpoints do.
//!
//! A model the layout cannot express is refused rather than approximated:
//! mixture-of-experts layers, a GELU feed-forward, bidirectional attention,
//! attached LoRA adapters (merge them first), and quantized weights. On the
//! reading side, a checkpoint with biases, per-head query and key norms, rope
//! scaling or a sliding window is refused for the same reason.

use crate::network::NetworkError;
use crate::param::Param;
use crate::safetensors::{self, ShardedSafeTensors};
use crate::transformer::TransformerLm;
use crate::transformer_block::FeedForward;
use std::collections::BTreeMap;
use std::path::Path;

/// Every parameter under its published name.
fn named_params(model: &mut TransformerLm) -> Result<Vec<(String, &mut Param)>, NetworkError> {
    let mut params = vec![(
        "model.embed_tokens.weight".to_string(),
        &mut model.embedding.weight,
    )];
    for (index, block) in model.blocks.iter_mut().enumerate() {
        let FeedForward::SwiGlu(mlp) = &mut block.feed_forward else {
            return Err(NetworkError::InvalidConfig(format!(
                "layer {index} is not a SwiGLU feed-forward, which the LLaMA layout requires"
            )));
        };
        let base = format!("model.layers.{index}");
        let attention = &mut block.attention;
        params.extend([
            (
                format!("{base}.input_layernorm.weight"),
                &mut block.attention_norm.weight,
            ),
            (
                format!("{base}.self_attn.q_proj.weight"),
                &mut attention.query.weight,
            ),
            (
                format!("{base}.self_attn.k_proj.weight"),
                &mut attention.key.weight,
            ),
            (
                format!("{base}.self_attn.v_proj.weight"),
                &mut attention.value.weight,
            ),
            (
                format!("{base}.self_attn.o_proj.weight"),
                &mut attention.output.weight,
            ),
            (
                format!("{base}.post_attention_layernorm.weight"),
                &mut block.feed_forward_norm.weight,
            ),
            (format!("{base}.mlp.gate_proj.weight"), &mut mlp.gate.weight),
            (format!("{base}.mlp.up_proj.weight"), &mut mlp.up.weight),
            (format!("{base}.mlp.down_proj.weight"), &mut mlp.down.weight),
        ]);
    }
    params.push(("model.norm.weight".into(), &mut model.final_norm.weight));
    if let Some(head) = &mut model.lm_head {
        params.push(("lm_head.weight".into(), &mut head.weight));
    }
    Ok(params)
}

fn refuse(reason: &str) -> NetworkError {
    NetworkError::InvalidConfig(format!("the LLaMA layout has no place for {reason}"))
}

impl TransformerLm {
    /// Writes `config.json` and an `F32` `model.safetensors` into `dir`, which
    /// `transformers` reads as `LlamaForCausalLM`.
    ///
    /// A model on a device is read back from it first, so this is safe to
    /// call in the middle of a run.
    pub fn save_hf(&mut self, dir: impl AsRef<Path>) -> Result<(), NetworkError> {
        if !self.config.causal {
            return Err(refuse("bidirectional attention"));
        }
        if self.has_lora() {
            return Err(refuse("LoRA adapters; call merge_lora first"));
        }
        if self.is_quantized() {
            return Err(refuse("quantized weights"));
        }
        #[cfg(feature = "cuda")]
        self.sync_from_device()?;

        let config = &self.config;
        let json = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "vocab_size": config.vocab_size,
            "hidden_size": config.d_model,
            "intermediate_size": config.d_ff,
            "num_hidden_layers": config.n_layers,
            "num_attention_heads": config.n_heads,
            "num_key_value_heads": config.n_kv_heads,
            "head_dim": config.head_dim,
            "max_position_embeddings": config.max_seq_len,
            "rope_theta": config.rope_base,
            "rms_norm_eps": config.rmsnorm_eps,
            "tie_word_embeddings": config.tie_embeddings,
            "hidden_act": "silu",
            "attention_bias": false,
            "mlp_bias": false,
            "torch_dtype": "float32",
        });

        let dir = dir.as_ref();
        let params = named_params(self)?;
        std::fs::create_dir_all(dir)?;
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&json)?,
        )?;
        // Norm weights are vectors in the published files.
        let shapes: Vec<Vec<usize>> = params
            .iter()
            .map(|(name, param)| match name.ends_with("norm.weight") {
                true => vec![param.value.data.len()],
                false => vec![param.value.rows, param.value.cols],
            })
            .collect();
        let tensors: Vec<(&str, &[usize], &[f32])> = params
            .iter()
            .zip(&shapes)
            .map(|((name, param), shape)| (name.as_str(), &shape[..], &param.value.data[..]))
            .collect();
        safetensors::write(
            dir.join("model.safetensors"),
            &tensors,
            &BTreeMap::from([("format".into(), "pt".into())]),
        )
    }

    /// Reads a LLaMA or Mistral checkpoint written by [`TransformerLm::save_hf`]
    /// or published on the Hugging Face Hub, single-file or sharded.
    ///
    /// `max_seq_len` replaces the checkpoint's `max_position_embeddings`, which
    /// is often far longer than a run on one card trains at.
    ///
    /// ponytail: the model is built with random weights and then overwritten,
    /// which costs one initialization pass. A builder that skips the
    /// initializer is the fix if loading a large model gets slow.
    pub fn load_hf(dir: impl AsRef<Path>, max_seq_len: usize) -> Result<Self, NetworkError> {
        let dir = dir.as_ref();
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let number = |key: &str| {
            json[key]
                .as_u64()
                .map(|value| value as usize)
                .ok_or_else(|| NetworkError::InvalidDataset(format!("config.json has no {key}")))
        };
        let float = |key: &str, default: f64| json[key].as_f64().unwrap_or(default) as f32;

        let model_type = json["model_type"].as_str().unwrap_or_default();
        if !matches!(model_type, "llama" | "mistral") {
            return Err(NetworkError::InvalidConfig(format!(
                "model_type {model_type:?} is not a LLaMA-layout model"
            )));
        }
        if !json["rope_scaling"].is_null() {
            return Err(refuse("rope scaling"));
        }
        if json["attention_bias"].as_bool() == Some(true)
            || json["mlp_bias"].as_bool() == Some(true)
        {
            return Err(refuse("biases"));
        }
        if json["hidden_act"].as_str().is_some_and(|act| act != "silu") {
            return Err(refuse("an activation other than SiLU"));
        }
        if json["sliding_window"]
            .as_u64()
            .is_some_and(|window| (window as usize) < max_seq_len)
        {
            return Err(refuse(
                "a sliding attention window shorter than max_seq_len",
            ));
        }

        let heads = number("num_attention_heads")?;
        let d_model = number("hidden_size")?;
        let mut model = TransformerLm::builder()
            .vocab_size(number("vocab_size")?)
            .d_model(d_model)
            .n_layers(number("num_hidden_layers")?)
            .heads(
                heads,
                number("num_key_value_heads").unwrap_or(heads),
                number("head_dim").unwrap_or(d_model / heads.max(1)),
            )
            .d_ff(number("intermediate_size")?)
            .moe_layers([])
            .max_seq_len(max_seq_len)
            .rope_base(float("rope_theta", 10_000.0))
            .rmsnorm_eps(float("rms_norm_eps", 1e-6))
            .tie_embeddings(json["tie_word_embeddings"].as_bool().unwrap_or(false))
            .build()?;

        let index = dir.join("model.safetensors.index.json");
        let mut file = ShardedSafeTensors::open(match index.exists() {
            true => index,
            false => dir.join("model.safetensors"),
        })?;
        if let Some(name) = file.names().find(|name| {
            name.ends_with(".bias") || name.contains("q_norm") || name.contains("k_norm")
        }) {
            return Err(refuse(&format!("the tensor {name}")));
        }
        for (name, param) in named_params(&mut model)? {
            let (values, shape) = file.tensor(&name)?;
            let (rows, cols) = (param.value.rows, param.value.cols);
            let fits = match shape[..] {
                [r, c] => (r, c) == (rows, cols),
                _ => values.len() == rows * cols,
            };
            if !fits {
                return Err(NetworkError::InvalidSnapshot(format!(
                    "{name} is {shape:?} where config.json implies [{rows}, {cols}]"
                )));
            }
            param.value.data = values;
        }
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    use crate::network::NetworkError;
    use crate::text_encoder::{TextEncoder, TextEncoderConfig};
    use crate::transformer::TransformerLm;
    use std::path::PathBuf;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("rb_hf_{name}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny(tie: bool) -> TransformerLm {
        let mut model = TransformerLm::builder()
            .vocab_size(37)
            .d_model(32)
            .n_layers(2)
            .heads(4, 2, 8)
            .d_ff(48)
            .moe_layers([])
            .max_seq_len(16)
            .rope_base(5000.0)
            .rmsnorm_eps(1e-5)
            .tie_embeddings(tie)
            .seed(3)
            .build()
            .unwrap();
        // Norm weights start at one, which would hide a norm read from the
        // wrong slot. Move every value off its initializer.
        for (index, param) in model.params_mut().into_iter().enumerate() {
            for (offset, value) in param.value.data.iter_mut().enumerate() {
                *value += 0.01 * ((index * 7 + offset) % 13) as f32 - 0.06;
            }
        }
        model
    }

    fn logits(model: &TransformerLm, ids: &[u32]) -> Matrix {
        model.forward_train(&[ids.to_vec()]).unwrap().0
    }

    const IDS: [u32; 6] = [3, 17, 0, 36, 9, 9];

    fn assert_close(got: &Matrix, expected: &Matrix, tolerance: f32) {
        assert_eq!((got.rows, got.cols), (expected.rows, expected.cols));
        for (a, b) in got.data.iter().zip(&expected.data) {
            assert!((a - b).abs() <= tolerance, "{a} against {b}");
        }
    }

    #[test]
    fn a_saved_model_loads_back_and_predicts_identically() {
        for tie in [false, true] {
            let dir = scratch(&format!("round_trip_{tie}"));
            let mut model = tiny(tie);
            model.save_hf(&dir).unwrap();
            let loaded = TransformerLm::load_hf(&dir, 16).unwrap();
            std::fs::remove_dir_all(&dir).ok();
            assert_eq!(loaded.config.tie_embeddings, tie);
            assert_close(&logits(&loaded, &IDS), &logits(&model, &IDS), 0.0);
        }
    }

    #[test]
    fn the_reference_llama_reader_agrees_with_what_was_saved() {
        // `TextEncoder` reads LLaMA-family checkpoints for the image pipeline
        // and was written against the published layout, not against this
        // module, so it checks the names, the orientation and `config.json`.
        let dir = scratch("reference");
        let mut model = tiny(false);
        model.save_hf(&dir).unwrap();
        let config = TextEncoderConfig::from_file(dir.join("config.json")).unwrap();
        let encoder = TextEncoder::load(dir.join("model.safetensors"), "model.", config).unwrap();
        std::fs::remove_dir_all(&dir).ok();

        let hidden = encoder.forward(&IDS).unwrap();
        let head = &model.lm_head.as_ref().unwrap().weight.value;
        let mut expected = Matrix::new(IDS.len(), head.rows);
        for row in 0..IDS.len() {
            for token in 0..head.rows {
                expected.data[row * head.rows + token] = hidden
                    .row(row)
                    .iter()
                    .zip(head.row(token))
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
        assert_close(&logits(&model, &IDS), &expected, 1e-4);
    }

    #[test]
    fn a_model_the_llama_layout_cannot_hold_is_refused() {
        let dir = scratch("refused");
        let mut moe = TransformerLm::builder()
            .vocab_size(16)
            .d_model(16)
            .n_layers(1)
            .heads(2, 2, 8)
            .d_ff(32)
            .experts(4, 2)
            .moe_layers([0])
            .max_seq_len(8)
            .build()
            .unwrap();
        let result = moe.save_hf(&dir);
        std::fs::remove_dir_all(&dir).ok();
        assert!(matches!(result, Err(NetworkError::InvalidConfig(_))));
    }
}
