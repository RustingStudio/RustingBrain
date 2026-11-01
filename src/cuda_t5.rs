//! The T5 encoder on the device, the twin of [`T5Encoder::forward`].
//!
//! It runs on the BF16 kernels the language model trains with, not the FP16
//! image path: T5 1.1's feed-forward outputs pass FP16's 65504, which is why
//! the reference keeps that projection in FP32. Here the residual stream and
//! every GEMM accumulator are FP32, and only the weights and the operands a
//! GEMM reads are BF16.
//!
//! NOTE: attention is the three-kernel path (score GEMM, bias add and
//! softmax, value GEMM), FP32 and `[heads, tokens, tokens]` in memory. The
//! published pipelines encode 256 or 512 tokens, 64 MiB of scores at most;
//! fuse the bias into flash attention if a profile ever shows it.

use crate::gpu_model::{Act, Gpu};
use crate::gpu_transformer::{GpuContext, gemm_plain_batched, gemm_rhs_transposed_batched};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::t5::T5Encoder;
use cudarc::driver::CudaSlice;
use std::sync::Arc;

struct DeviceBlock {
    attention_norm: CudaSlice<f32>,
    /// Query, key and value stacked, `[3 * inner, d_model]`.
    qkv: Act,
    output: Act,
    mlp_norm: CudaSlice<f32>,
    /// The GELU half then the gate half, `[2 * mlp_dim, d_model]`.
    gate_up: Act,
    mlp_out: Act,
}

/// A T5 encoder resident on the device.
pub struct DeviceT5 {
    context: Arc<GpuContext>,
    blocks: Vec<DeviceBlock>,
    final_norm: CudaSlice<f32>,
    /// Whether the GEMMs read BF16 activations. Int8 weights read FP32 ones,
    /// as in [`crate::cuda_text_encoder`]: BF16 on top of Q8's rounding put
    /// umT5-XXL's embeddings at 0.0271 against the host's 0.0245.
    narrow: bool,
}

impl std::fmt::Debug for DeviceT5 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceT5").finish_non_exhaustive()
    }
}

impl DeviceT5 {
    /// Uploads `encoder`'s weights as BF16, or as int8 where it holds them at
    /// one byte. The host encoder stays the source of the embeddings and the
    /// position bias, so it is borrowed again by [`DeviceT5::forward`].
    pub fn upload(context: &Arc<GpuContext>, encoder: &T5Encoder) -> Result<Self, NetworkError> {
        if !encoder.config.gated {
            return Err(NetworkError::InvalidConfig(
                "the original ungated T5 has no device path; run it on the host".into(),
            ));
        }
        if context.half_accumulate {
            return Err(NetworkError::InvalidConfig(
                "T5 overflows FP16; use a BF16 context".into(),
            ));
        }
        let gpu = Gpu { context };
        let pack = |parts: &[&crate::conv::Dense]| -> Result<Act, NetworkError> {
            let inner = parts[0].in_dim();
            let mut values = Vec::new();
            for part in parts {
                values.extend_from_slice(&part.weight_f32().data);
            }
            let units = values.len() / inner;
            let wide = gpu.upload(&values)?;
            match parts.iter().any(|part| part.is_quantized()) {
                true => gpu.quantized(&wide.slice(..), units, inner),
                false => gpu.narrowed(&wide.slice(..), units * inner, true),
            }
        };
        let blocks = encoder
            .blocks
            .iter()
            .map(|block| {
                let gate = block.gate.as_ref().expect("checked gated above");
                Ok(DeviceBlock {
                    attention_norm: gpu.upload(&block.attention_norm)?,
                    qkv: pack(&[&block.query, &block.key, &block.value])?,
                    output: pack(&[&block.output])?,
                    mlp_norm: gpu.upload(&block.mlp_norm)?,
                    gate_up: pack(&[&block.mlp_in, gate])?,
                    mlp_out: pack(&[&block.mlp_out])?,
                })
            })
            .collect::<Result<_, NetworkError>>()?;
        Ok(Self {
            narrow: !encoder.blocks.iter().any(|block| block.query.is_quantized()),
            final_norm: gpu.upload(&encoder.final_norm)?,
            context: context.clone(),
            blocks,
        })
    }

    /// The hidden states for `ids`, `[tokens, d_model]`. `encoder` must be the
    /// one this was uploaded from.
    pub fn forward(&self, encoder: &T5Encoder, ids: &[u32]) -> Result<Matrix, NetworkError> {
        let gpu = Gpu {
            context: &self.context,
        };
        let config = &encoder.config;
        let (tokens, d_model) = (ids.len(), config.d_model);
        let (heads, head_dim) = (config.num_heads, config.head_dim);
        let inner = heads * head_dim;
        let width = config.mlp_dim;

        let mut hidden = gpu.upload(&encoder.embed(ids)?.data)?;
        let upload_bias = |layer: usize| gpu.upload(&encoder.bias(layer, tokens).concat());
        // umT5 has a table per block, T5 one for all of them.
        let shared = match encoder.position_bias.len() {
            1 => Some(upload_bias(0)?),
            _ => None,
        };
        let square = tokens * tokens;

        for (index, block) in self.blocks.iter().enumerate() {
            let own;
            let bias = match &shared {
                Some(bias) => bias,
                None => {
                    own = upload_bias(index)?;
                    &own
                }
            };
            let (normed, _, _) = gpu.rmsnorm(
                &hidden,
                &block.attention_norm.as_view(),
                tokens,
                d_model,
                config.eps,
                self.narrow,
                false,
            )?;
            let mut qkv = gpu.uninit(tokens * 3 * inner)?;
            gpu.project(
                &block.qkv,
                3 * inner,
                d_model,
                &normed.all(),
                self.narrow,
                &mut qkv.slice_mut(..),
                tokens,
                0.0,
            )?;
            // Every head at once: a head is `head_dim` columns along the row,
            // so the batch stride steps one head over. T5 does not scale.
            let mut scores = gpu.uninit(heads * square)?;
            gemm_rhs_transposed_batched(
                &self.context,
                &qkv.slice(..),
                3 * inner,
                &qkv.slice(inner..),
                3 * inner,
                &mut scores,
                tokens,
                tokens,
                tokens,
                head_dim,
                1.0,
                0.0,
                heads,
                head_dim,
                head_dim,
                square,
            )?;
            gpu.add(&mut scores, bias, 0, heads * square, true)?;
            let mut log_sum_exp = gpu.uninit(heads * tokens)?;
            gpu.attention_softmax(
                &mut scores,
                &mut log_sum_exp,
                heads * tokens,
                tokens,
                false,
                tokens,
                0,
            )?;
            let mut merged = gpu.uninit(tokens * inner)?;
            gemm_plain_batched(
                &self.context,
                &scores,
                tokens,
                &qkv.slice(2 * inner..),
                3 * inner,
                &mut merged,
                inner,
                tokens,
                head_dim,
                tokens,
                1.0,
                0.0,
                heads,
                square,
                head_dim,
                head_dim,
            )?;
            let merged = gpu.narrowed(&merged.slice(..), tokens * inner, true)?;
            gpu.project(
                &block.output,
                d_model,
                inner,
                &merged.all(),
                true,
                &mut hidden.slice_mut(..),
                tokens,
                1.0,
            )?;

            let (normed, _, _) = gpu.rmsnorm(
                &hidden,
                &block.mlp_norm.as_view(),
                tokens,
                d_model,
                config.eps,
                self.narrow,
                false,
            )?;
            let mut gate_up = gpu.uninit(tokens * 2 * width)?;
            gpu.project(
                &block.gate_up,
                2 * width,
                d_model,
                &normed.all(),
                self.narrow,
                &mut gate_up.slice_mut(..),
                tokens,
                0.0,
            )?;
            let wide = gpu.gated(&gate_up, tokens, width, self.narrow, true)?;
            gpu.project(
                &block.mlp_out,
                d_model,
                width,
                &wide.all(),
                self.narrow,
                &mut hidden.slice_mut(..),
                tokens,
                1.0,
            )?;
        }

        let (normed, _, _) = gpu.rmsnorm(
            &hidden,
            &self.final_norm.as_view(),
            tokens,
            d_model,
            config.eps,
            false,
            false,
        )?;
        let values = self
            .context
            .stream
            .clone_dtoh(&normed.wide())
            .map_err(|error| NetworkError::Cuda(format!("device to host copy: {error}")))?;
        Ok(Matrix::from_vec(tokens, d_model, values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_training::cuda_doctor;
    use crate::t5::tests::{checkpoint, checkpoint_with, tiny};
    use crate::transformer::Precision;

    fn cuda_or_skip() -> Option<Arc<GpuContext>> {
        match cuda_doctor(0, 8192) {
            Ok(_) => Some(GpuContext::new(0).expect("the CUDA doctor passed")),
            Err(NetworkError::Cuda(message))
                if message.contains("NO_DEVICE") || message.contains("no CUDA-capable device") =>
            {
                crate::cuda_training::skip_without_cuda();
                None
            }
            Err(error) => panic!("CUDA is present but the CUDA doctor failed: {error}"),
        }
    }

    #[test]
    fn a_device_t5_matches_the_host_or_skips_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = tiny();
        let path = std::env::temp_dir().join("rusting_brain_cuda_t5.safetensors");
        // T5's shared position table, then umT5's one per block.
        for (per_block, precision) in [
            (false, Precision::F32),
            (false, Precision::Q8),
            (true, Precision::F32),
        ] {
            checkpoint_with(&config, &path, per_block);
            let encoder = T5Encoder::load_at(&path, config.clone(), precision).unwrap();
            assert_eq!(encoder.position_bias.len() > 1, per_block);
            let device = DeviceT5::upload(&context, &encoder).unwrap();
            let ids = [5, 9, 3, 17, 2, 30, 1];
            let expected = encoder.forward(&ids).unwrap();
            let actual = device.forward(&encoder, &ids).unwrap();

            assert_eq!((actual.rows, actual.cols), (expected.rows, expected.cols));
            // BF16 operands carry eight bits of mantissa, a few tenths of a
            // percent each; two layers of them stay within 2% of the largest
            // state.
            let scale = expected.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let worst = actual
                .data
                .iter()
                .zip(&expected.data)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(
                worst < 2e-2 * scale,
                "{precision:?}, per block {per_block}: worst difference {worst} of {scale}"
            );
        }
        std::fs::remove_file(path).ok();
    }

    /// T5-base's shape over a 512-token prompt, host against device. Run by
    /// hand: `cargo test --release --features cuda --lib time_a_t5 -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn time_a_t5_base_sized_encoder() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = crate::t5::T5Config {
            d_model: 768,
            head_dim: 64,
            num_heads: 12,
            layers: 12,
            mlp_dim: 2048,
            ..tiny()
        };
        let path = std::env::temp_dir().join("rusting_brain_cuda_t5_base.safetensors");
        checkpoint(&config, &path);
        let encoder = T5Encoder::load_at(&path, config, Precision::F32).unwrap();
        std::fs::remove_file(path).ok();
        let device = DeviceT5::upload(&context, &encoder).unwrap();
        let ids: Vec<u32> = (0..512).map(|index| index % 32).collect();

        let started = std::time::Instant::now();
        let expected = encoder.forward(&ids).unwrap();
        let host = started.elapsed();
        device.forward(&encoder, &ids).unwrap();
        let mut device_times = Vec::new();
        let mut actual = None;
        for _ in 0..5 {
            let started = std::time::Instant::now();
            actual = Some(device.forward(&encoder, &ids).unwrap());
            device_times.push(started.elapsed());
        }
        let actual = actual.unwrap();
        let scale = expected.data.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = actual
            .data
            .iter()
            .zip(&expected.data)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        println!("host {host:?}, device {device_times:?}, worst {worst} of {scale}");
    }
}
