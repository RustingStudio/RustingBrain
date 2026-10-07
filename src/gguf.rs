//! Writing a [`TransformerLm`] as a GGUF file, the format llama.cpp and
//! Ollama run.
//!
//! The model goes out as the `llama` architecture, so the same models
//! [`TransformerLm::save_hf`] accepts are accepted here and the rest are
//! refused for the same reasons. Weights are `F32`, or `Q8_0` (blocks of 32
//! int8 values with one FP16 scale) for every matrix whose rows split into
//! such blocks; norm weights stay `F32`, as llama.cpp's converter keeps them.
//!
//! NOTE: no tokenizer is written (`tokenizer.ggml.model` is `no_vocab`),
//! since the model does not own one. Ids in and ids out work; text needs the
//! vocabulary added, which is a follow-up once a tokenizer travels with the
//! model.

use crate::hf::named_params;
use crate::network::NetworkError;
use crate::rope::RopeScaling;
use crate::transformer::TransformerLm;
use std::io::Write;
use std::path::Path;

const ALIGNMENT: usize = 32;
const F32: u32 = 0;
const Q8_0: u32 = 8;
const Q8_BLOCK: usize = 32;

/// A metadata value of one of the GGUF types this writer uses.
enum Value {
    U32(u32),
    F32(f32),
    Str(String),
}

struct Tensor {
    name: String,
    /// Innermost first, as GGUF lists them.
    dims: Vec<u64>,
    kind: u32,
    bytes: Vec<u8>,
}

impl TransformerLm {
    /// Writes the model to `path` as a llama-architecture GGUF file, with its
    /// matrices in `Q8_0` when `q8_0` is set and in `F32` otherwise.
    ///
    /// A model on a device is read back from it first.
    ///
    /// ```no_run
    /// # use rusting_brain::TransformerLm;
    /// # let mut model = TransformerLm::load_bin("model.rbw")?;
    /// model.save_gguf("model-q8_0.gguf", true)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn save_gguf(&mut self, path: impl AsRef<Path>, q8_0: bool) -> Result<(), NetworkError> {
        self.prepare_llama_export()?;
        let config = self.config.clone();
        let mut metadata = vec![
            ("general.architecture", Value::Str("llama".into())),
            ("general.name", Value::Str("rusting_brain".into())),
            ("general.file_type", Value::U32(if q8_0 { 7 } else { 0 })),
            ("llama.vocab_size", Value::U32(config.vocab_size as u32)),
            (
                "llama.context_length",
                Value::U32(config.max_seq_len as u32),
            ),
            ("llama.embedding_length", Value::U32(config.d_model as u32)),
            ("llama.block_count", Value::U32(config.n_layers as u32)),
            ("llama.feed_forward_length", Value::U32(config.d_ff as u32)),
            (
                "llama.attention.head_count",
                Value::U32(config.n_heads as u32),
            ),
            (
                "llama.attention.head_count_kv",
                Value::U32(config.n_kv_heads as u32),
            ),
            (
                "llama.attention.key_length",
                Value::U32(config.head_dim as u32),
            ),
            (
                "llama.attention.value_length",
                Value::U32(config.head_dim as u32),
            ),
            (
                "llama.rope.dimension_count",
                Value::U32(config.head_dim as u32),
            ),
            ("llama.rope.freq_base", Value::F32(config.rope_base)),
            (
                "llama.attention.layer_norm_rms_epsilon",
                Value::F32(config.rmsnorm_eps),
            ),
            ("tokenizer.ggml.model", Value::Str("no_vocab".into())),
        ];

        let mut tensors = Vec::new();
        match config.rope_scaling {
            None => {}
            Some(RopeScaling::Linear { factor }) => {
                metadata.push(("llama.rope.scaling.type", Value::Str("linear".into())));
                metadata.push(("llama.rope.scaling.factor", Value::F32(factor)));
            }
            // llama.cpp divides each channel's angle by `rope_freqs`, which is
            // how its converter carries Llama 3.1's per-channel scaling.
            Some(scaling) => {
                let half = config.head_dim / 2;
                let factors: Vec<f32> = (0..half)
                    .map(|channel| {
                        let frequency = config
                            .rope_base
                            .powf(-2.0 * channel as f32 / config.head_dim as f32);
                        frequency / scaling.apply(frequency)
                    })
                    .collect();
                tensors.push(Tensor {
                    name: "rope_freqs.weight".into(),
                    dims: vec![half as u64],
                    kind: F32,
                    bytes: factors.iter().flat_map(|v| v.to_le_bytes()).collect(),
                });
            }
        }

        for (name, param) in named_params(self)? {
            let name = gguf_name(&name);
            let (rows, cols) = (param.value.rows, param.value.cols);
            let mut data = param.value.data.clone();
            // llama.cpp rotates adjacent channel pairs; this model pairs
            // channel j with j + head_dim / 2. Reordering each head's query and
            // key rows turns one convention into the other.
            if name.ends_with("attn_q.weight") {
                data = interleave_heads(&data, config.n_heads, cols);
            } else if name.ends_with("attn_k.weight") {
                data = interleave_heads(&data, config.n_kv_heads, cols);
            }
            let vector = name.ends_with("norm.weight");
            let dims = match vector {
                true => vec![data.len() as u64],
                false => vec![cols as u64, rows as u64],
            };
            let quantize = q8_0 && !vector && cols % Q8_BLOCK == 0;
            tensors.push(Tensor {
                name,
                dims,
                kind: if quantize { Q8_0 } else { F32 },
                bytes: match quantize {
                    true => quantize_q8_0(&data),
                    false => data.iter().flat_map(|v| v.to_le_bytes()).collect(),
                },
            });
        }

        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        file.write_all(&write_gguf(&metadata, &tensors))?;
        file.flush()?;
        Ok(())
    }
}

/// The GGUF name of a parameter under its Hugging Face LLaMA name, the map
/// llama.cpp's `convert_hf_to_gguf.py` applies.
fn gguf_name(hf: &str) -> String {
    let renamed = [
        ("model.embed_tokens.", "token_embd."),
        ("model.norm.", "output_norm."),
        ("lm_head.", "output."),
        ("model.layers.", "blk."),
        ("input_layernorm", "attn_norm"),
        ("self_attn.q_proj", "attn_q"),
        ("self_attn.k_proj", "attn_k"),
        ("self_attn.v_proj", "attn_v"),
        ("self_attn.o_proj", "attn_output"),
        ("post_attention_layernorm", "ffn_norm"),
        ("mlp.gate_proj", "ffn_gate"),
        ("mlp.up_proj", "ffn_up"),
        ("mlp.down_proj", "ffn_down"),
    ];
    renamed
        .iter()
        .fold(hf.to_string(), |name, (from, to)| name.replace(from, to))
}

/// Reorders the rows of each head from `[half, pair]` to `[pair, half]`: row
/// `t * head_dim / 2 + i` moves to `2 * i + t`, the `permute` of llama.cpp's
/// converter.
fn interleave_heads(data: &[f32], heads: usize, cols: usize) -> Vec<f32> {
    let head_dim = data.len() / cols / heads;
    let half = head_dim / 2;
    let mut out = vec![0.0; data.len()];
    for head in 0..heads {
        for t in 0..2 {
            for i in 0..half {
                let from = (head * head_dim + t * half + i) * cols;
                let to = (head * head_dim + 2 * i + t) * cols;
                out[to..to + cols].copy_from_slice(&data[from..from + cols]);
            }
        }
    }
    out
}

/// `Q8_0`: each block of 32 values becomes an FP16 scale `max|x| / 127` and
/// 32 int8 values `round(x / scale)`, rounded as ggml rounds them.
fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() / Q8_BLOCK * (2 + Q8_BLOCK));
    for block in data.chunks(Q8_BLOCK) {
        let scale = block.iter().fold(0.0f32, |max, v| max.max(v.abs())) / 127.0;
        let inverse = if scale == 0.0 { 0.0 } else { scale.recip() };
        out.extend_from_slice(&f16_bits(scale).to_le_bytes());
        out.extend(block.iter().map(|v| (v * inverse).round() as i8 as u8));
    }
    out
}

/// `value` as IEEE half precision, rounded to nearest even.
fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let biased = exponent - 127 + 15;
    if biased >= 0x1f {
        return sign | 0x7c00;
    }
    let (kept, shift) = match biased {
        // Subnormal: the implicit bit joins the mantissa, shifted further.
        ..=0 if biased < -10 => return sign,
        ..=0 => (mantissa | 0x80_0000, (14 - biased) as u32),
        _ => (((biased as u32) << 23) | mantissa, 13),
    };
    let half = kept >> shift;
    let rest = kept & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    let round = u32::from(rest > halfway || (rest == halfway && half & 1 == 1));
    // A carry out of the mantissa bumps the exponent, which is the right answer.
    sign | (half + round) as u16
}

fn put_string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

fn write_gguf(metadata: &[(&str, Value)], tensors: &[Tensor]) -> Vec<u8> {
    let mut out = b"GGUF".to_vec();
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        put_string(&mut out, key);
        match value {
            Value::U32(v) => {
                out.extend_from_slice(&4u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Value::F32(v) => {
                out.extend_from_slice(&6u32.to_le_bytes());
                out.extend_from_slice(&v.to_le_bytes());
            }
            Value::Str(v) => {
                out.extend_from_slice(&8u32.to_le_bytes());
                put_string(&mut out, v);
            }
        }
    }

    let padded = |len: usize| len.div_ceil(ALIGNMENT) * ALIGNMENT;
    let mut offset = 0;
    for tensor in tensors {
        put_string(&mut out, &tensor.name);
        out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.kind.to_le_bytes());
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        offset += padded(tensor.bytes.len());
    }
    out.resize(padded(out.len()), 0);
    for tensor in tensors {
        out.extend_from_slice(&tensor.bytes);
        out.resize(padded(out.len()), 0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_bits_round_like_ieee() {
        for (value, bits) in [
            (0.0, 0x0000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (65504.0, 0x7bff),
            (1e6, 0x7c00),
            (5.960_464_5e-8, 0x0001),
            (6.103_515_6e-5, 0x0400),
            (1.0 + 1.0 / 2048.0, 0x3c00),
            (1.0 + 3.0 / 2048.0, 0x3c02),
            (0.1, 0x2e66),
        ] {
            assert_eq!(f16_bits(value), bits, "{value}");
        }
    }

    #[test]
    fn q8_0_blocks_hold_a_scale_and_rounded_values() {
        let mut data: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        data[5] = 127.0;
        data.extend([0.0; 32]);
        let bytes = quantize_q8_0(&data);
        assert_eq!(bytes.len(), 2 * 34);
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), f16_bits(1.0));
        let first: Vec<i8> = bytes[2..34].iter().map(|&b| b as i8).collect();
        let expected: Vec<i8> = data[..32].iter().map(|&v| v as i8).collect();
        assert_eq!(first, expected);
        assert!(bytes[34..].iter().all(|&b| b == 0));
    }

    /// llama.cpp's rotation of adjacent pairs, applied to the interleaved
    /// rows, gives the interleaved rows of this model's rotate-half rotation:
    /// the export keeps every query-key dot product.
    #[test]
    fn interleaved_heads_turn_rotate_half_into_adjacent_pairs() {
        let (heads, head_dim) = (2, 8);
        let half = head_dim / 2;
        let rope = crate::rope::Rope::new(head_dim, 4, 10_000.0).unwrap();
        let row: Vec<f32> = (0..heads * head_dim)
            .map(|i| (i as f32 * 0.7).sin())
            .collect();
        let position = 3;
        let mut ours = crate::matrix::Matrix::from_vec(1, heads * head_dim, row.clone());
        rope.apply(&mut ours, heads, position).unwrap();
        let expected = interleave_heads(&ours.data, heads, 1);

        let mut theirs = interleave_heads(&row, heads, 1);
        for head in 0..heads {
            for i in 0..half {
                let angle = position as f32 * 10_000f32.powf(-2.0 * i as f32 / head_dim as f32);
                let (a, b) = (head * head_dim + 2 * i, head * head_dim + 2 * i + 1);
                let (x, y) = (theirs[a], theirs[b]);
                theirs[a] = x * angle.cos() - y * angle.sin();
                theirs[b] = x * angle.sin() + y * angle.cos();
            }
        }
        for (got, want) in theirs.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-5, "{theirs:?} vs {expected:?}");
        }
    }

    #[test]
    fn a_gguf_file_names_its_tensors_and_aligns_their_data() {
        let path = std::env::temp_dir().join(format!("rb-gguf-{}.gguf", std::process::id()));
        let mut model = TransformerLm::builder()
            .vocab_size(40)
            .d_model(32)
            .n_layers(2)
            .heads(4, 2, 8)
            .d_ff(64)
            .moe_layers([])
            .max_seq_len(16)
            .seed(3)
            .build()
            .unwrap();
        model.save_gguf(&path, true).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        // Embedding, final norm, nine per layer; the head is tied.
        assert_eq!(tensor_count, 2 + 2 * 9);
        let text = String::from_utf8_lossy(&bytes);
        for name in [
            "token_embd.weight",
            "blk.1.ffn_down.weight",
            "output_norm.weight",
        ] {
            assert!(text.contains(name), "{name}");
        }

        let mut moe = TransformerLm::builder().vocab_size(40).build().unwrap();
        assert!(moe.save_gguf(&path, false).is_err());
    }
}
