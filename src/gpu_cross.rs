//! Device cross-attention: one token set attending to another.
//!
//! [`crate::gpu_model`] holds a whole decoder layer on the device, but the
//! graph it drives is fixed — embedding, sixteen identical blocks, logits. The
//! models in `2Dto3DPlan.md` need a different one: a flow transformer whose
//! blocks alternate self-attention over latent tokens with cross-attention to
//! a frozen image encoder's output, and a shape decoder whose queries are
//! points in space. This module is the piece that graph does not already have.
//!
//! It is built from [`Gpu`]'s primitives rather than from new kernels. The
//! only kernel change cross-attention needed was a flag: the causal softmax,
//! its backward and the log-sum-exp rebuild now take `causal`, and clearing it
//! gives exactly the unmasked row softmax two independent sequence lengths
//! want.
//!
//! # What differs from self-attention
//!
//! - Queries and keys come from **different buffers with different widths**.
//!   ViT-B tokens are 768 wide and the denoiser reading them is 640.
//! - There are **two lengths**. `q_len` queries per sequence attend to all
//!   `kv_len` keys of the same sequence.
//! - **No rotary positions and no mask.** A latent token set has no order, and
//!   every query sees every key.
//!
//! # Precision
//!
//! NOTE: FP32 throughout. The fused BF16 path in [`crate::cuda_flash`] is
//! written around one sequence attending to itself, and the score matrix here
//! is small enough not to need it: 512 queries over 1024 image tokens at ten
//! heads and batch 4 is 84 MB, against the 21% of GPU time the flash kernel
//! was written to recover at `seq_len` 1024 in *every* block. Widen this to a
//! fused cross kernel when a profile says the score round trip costs more than
//! the rest of the block.

// Stage D of `2Dto3DPlan.md` is what calls this; until that block exists the
// only caller is the parity test below, and a `cargo build` would otherwise
// report the whole module as dead.
#![allow(dead_code)]

use crate::attention::MultiHeadAttention;
use crate::cuda_training::cuda_err;
use crate::gpu_model::{Act, Gpu};
use crate::gpu_transformer::{
    gemm_lhs_transposed_batched, gemm_plain_batched, gemm_rhs_transposed_batched,
};
use crate::network::NetworkError;
use cudarc::driver::{CudaSlice, CudaView};

/// Where one cross-attention's queries, keys and values live, and how they are
/// grouped into sequences.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CrossShape {
    /// Queries per sequence.
    pub(crate) q_len: usize,
    /// Keys and values per sequence.
    pub(crate) kv_len: usize,
    pub(crate) sequences: usize,
}

impl CrossShape {
    fn q_rows(&self) -> usize {
        self.q_len * self.sequences
    }

    fn kv_rows(&self) -> usize {
        self.kv_len * self.sequences
    }
}

/// What [`backward`] needs from [`forward`].
///
/// The probability matrix is not here: the forward pass keeps one log-sum-exp
/// per query and the backward pass rebuilds the probabilities from it, the
/// same trade [`crate::gpu_model`] makes for self-attention.
pub(crate) struct CrossCache {
    queries_in: Act,
    kv_in: Act,
    /// The query projection's output, `[q_rows, heads * head_dim]`.
    queries: Act,
    /// Key and value in one buffer, two slices of every row, because they read
    /// the same input and are one GEMM against their weights packed together.
    kv: Act,
    query_weight: Act,
    kv_weights: Act,
    output_weight: Act,
    merged: Act,
    log_sum_exp: CudaSlice<f32>,
    shape: CrossShape,
}

/// `attention.output(merge(softmax(Q Kᵀ / √d) V))`, with Q from `queries_in`
/// and K, V from `kv_in`.
///
/// Both inputs are FP32 and row-major: `queries_in` is `[q_rows, d_model]` and
/// `kv_in` is `[kv_rows, kv_features]`. The result is `[q_rows, d_model]`.
///
/// This is [`MultiHeadAttention::forward_train_cross`] on the device, and the
/// parity test at the bottom of this file holds the two to 1e-4.
pub(crate) fn forward(
    gpu: &Gpu<'_>,
    attention: &MultiHeadAttention,
    queries_in: &CudaView<'_, f32>,
    kv_in: &CudaView<'_, f32>,
    shape: CrossShape,
) -> Result<(CudaSlice<f32>, CrossCache), NetworkError> {
    let plan = Plan::new(attention, shape)?;
    let (q_rows, kv_rows) = (shape.q_rows(), shape.kv_rows());

    // Everything here stays FP32, so the packed weights and the activations
    // are wide `Act`s and the batched GEMMs can read them directly.
    let queries_in = gpu.narrowed(queries_in, q_rows * plan.d_model, false)?;
    let kv_in = gpu.narrowed(kv_in, kv_rows * plan.d_kv, false)?;

    let query_weight = gpu.pack(&[&attention.query], false)?;
    let kv_weights = gpu.pack(&[&attention.key, &attention.value], false)?;
    let output_weight = gpu.pack(&[&attention.output], false)?;

    let mut queries = gpu.act(q_rows * plan.query_width, false)?;
    gpu.linear_packed_act(
        &query_weight.all(),
        plan.query_width,
        plan.d_model,
        &queries_in.all(),
        false,
        &mut queries,
        q_rows,
    )?;

    // Key and value share `kv_in` and differ only in nothing at all, so one
    // GEMM at twice the width produces both.
    let mut kv = gpu.act(kv_rows * plan.kv_stride, false)?;
    gpu.linear_packed_act(
        &kv_weights.all(),
        plan.kv_stride,
        plan.d_kv,
        &kv_in.all(),
        false,
        &mut kv,
        kv_rows,
    )?;

    let mut probabilities = gpu.uninit(plan.heads * shape.sequences * plan.block_size)?;
    plan.scores(gpu, &queries, &kv, &mut probabilities, shape, plan.scale)?;
    let mut log_sum_exp = gpu.uninit(plan.heads * q_rows)?;
    gpu.softmax_lse(
        &mut probabilities,
        &mut log_sum_exp,
        plan.heads * shape.sequences * shape.q_len,
        shape.kv_len,
        false,
    )?;

    let mut merged = gpu.act(q_rows * plan.query_width, false)?;
    {
        let kv_view = kv.wide();
        let mut wide_merged = merged.wide_mut();
        for head in 0..plan.heads {
            let head_probabilities =
                probabilities.slice(head * shape.sequences * plan.block_size..);
            let value = kv_view.slice(plan.value_base + plan.kv_base(head)..);
            let mut head_merged = wide_merged.slice_mut(head * plan.head_dim..);
            gemm_plain_batched(
                gpu.context,
                &head_probabilities,
                shape.kv_len,
                &value,
                plan.kv_stride,
                &mut head_merged,
                plan.query_width,
                shape.q_len,
                plan.head_dim,
                shape.kv_len,
                1.0,
                0.0,
                shape.sequences,
                plan.block_size,
                shape.kv_len * plan.kv_stride,
                shape.q_len * plan.query_width,
            )?;
        }
    }

    let mut output = gpu.uninit(q_rows * plan.d_model)?;
    gpu.linear_packed(
        &output_weight.all(),
        plan.d_model,
        plan.query_width,
        &merged.all(),
        false,
        &mut output,
        q_rows,
        0.0,
    )?;

    Ok((
        output,
        CrossCache {
            queries_in,
            kv_in,
            queries,
            kv,
            query_weight,
            kv_weights,
            output_weight,
            merged,
            log_sum_exp,
            shape,
        },
    ))
}

/// Accumulates the four projections' gradients and returns
/// `(dL/dqueries_in, dL/dkv_in)`.
///
/// The second is what a caller drops when the key side is a frozen encoder.
/// Computing it anyway costs one GEMM and keeps the trainable case working.
pub(crate) fn backward(
    gpu: &Gpu<'_>,
    attention: &mut MultiHeadAttention,
    cache: &CrossCache,
    grad_output: &CudaSlice<f32>,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>), NetworkError> {
    let shape = cache.shape;
    let plan = Plan::new(attention, shape)?;
    let (q_rows, kv_rows) = (shape.q_rows(), shape.kv_rows());

    let grad_output = gpu.narrowed(&grad_output.slice(..), q_rows * plan.d_model, false)?;
    let mut grad_merged = gpu.act(q_rows * plan.query_width, false)?;
    gpu.linear_packed_backward_input(
        &cache.output_weight.all(),
        plan.d_model,
        plan.query_width,
        &grad_output.all(),
        false,
        false,
        grad_merged.destination(),
        q_rows,
        0.0,
    )?;
    gpu.accumulate_projection_grad(
        &mut attention.output,
        &grad_output.all(),
        &cache.merged.all(),
        false,
        q_rows,
    )?;

    // The forward pass kept only the log-sum-exp, so the probabilities are
    // rebuilt here and live for this backward pass alone.
    let mut probabilities = gpu.uninit(plan.heads * shape.sequences * plan.block_size)?;
    plan.scores(
        gpu,
        &cache.queries,
        &cache.kv,
        &mut probabilities,
        shape,
        plan.scale,
    )?;
    gpu.probs_from_lse(
        &mut probabilities,
        &cache.log_sum_exp,
        plan.heads * shape.sequences * shape.q_len,
        shape.kv_len,
        false,
    )?;

    let mut grad_scores = gpu.uninit(plan.heads * shape.sequences * plan.block_size)?;
    let mut grad_queries = gpu.act(q_rows * plan.query_width, false)?;
    let mut grad_kv = gpu.act(kv_rows * plan.kv_stride, false)?;
    {
        // Every query head writes its own slice of `grad_queries` and of
        // `grad_scores`, but the query heads of a group all accumulate into
        // one key and value head, so that buffer starts at zero.
        let mut wide_grad_kv = grad_kv.wide_mut();
        gpu.context
            .stream
            .memset_zeros(&mut wide_grad_kv)
            .map_err(cuda_err("gradient clear"))?;

        let kv_view = cache.kv.wide();
        let queries_view = cache.queries.wide();
        let grad_merged_view = grad_merged.wide();
        let mut wide_grad_queries = grad_queries.wide_mut();

        for head in 0..plan.heads {
            let kv_base = plan.kv_base(head);
            let upstream = grad_merged_view.slice(head * plan.head_dim..);
            let value = kv_view.slice(plan.value_base + kv_base..);
            let mut scores = grad_scores.slice_mut(head * shape.sequences * plan.block_size..);
            gemm_rhs_transposed_batched(
                gpu.context,
                &upstream,
                plan.query_width,
                &value,
                plan.kv_stride,
                &mut scores,
                shape.kv_len,
                shape.q_len,
                shape.kv_len,
                plan.head_dim,
                1.0,
                0.0,
                shape.sequences,
                shape.q_len * plan.query_width,
                shape.kv_len * plan.kv_stride,
                plan.block_size,
            )?;

            let head_probabilities =
                probabilities.slice(head * shape.sequences * plan.block_size..);
            let mut grad_value = wide_grad_kv.slice_mut(plan.value_base + kv_base..);
            gemm_lhs_transposed_batched(
                gpu.context,
                &head_probabilities,
                shape.kv_len,
                &upstream,
                plan.query_width,
                &mut grad_value,
                plan.kv_stride,
                shape.q_len,
                shape.kv_len,
                plan.head_dim,
                1.0,
                1.0,
                shape.sequences,
                plan.block_size,
                shape.q_len * plan.query_width,
                shape.kv_len * plan.kv_stride,
            )?;
        }

        gpu.softmax_backward(
            &mut grad_scores,
            &probabilities,
            plan.heads * shape.sequences * shape.q_len,
            shape.kv_len,
            false,
        )?;

        for head in 0..plan.heads {
            let kv_base = plan.kv_base(head);
            let scores = grad_scores.slice(head * shape.sequences * plan.block_size..);
            let key = kv_view.slice(plan.key_base + kv_base..);
            let mut grad_query = wide_grad_queries.slice_mut(head * plan.head_dim..);
            gemm_plain_batched(
                gpu.context,
                &scores,
                shape.kv_len,
                &key,
                plan.kv_stride,
                &mut grad_query,
                plan.query_width,
                shape.q_len,
                plan.head_dim,
                shape.kv_len,
                plan.scale,
                0.0,
                shape.sequences,
                plan.block_size,
                shape.kv_len * plan.kv_stride,
                shape.q_len * plan.query_width,
            )?;

            let query = queries_view.slice(head * plan.head_dim..);
            let mut grad_key = wide_grad_kv.slice_mut(plan.key_base + kv_base..);
            gemm_lhs_transposed_batched(
                gpu.context,
                &scores,
                shape.kv_len,
                &query,
                plan.query_width,
                &mut grad_key,
                plan.kv_stride,
                shape.q_len,
                shape.kv_len,
                plan.head_dim,
                plan.scale,
                1.0,
                shape.sequences,
                plan.block_size,
                shape.q_len * plan.query_width,
                shape.kv_len * plan.kv_stride,
            )?;
        }
    }

    let mut grad_queries_in = gpu.uninit(q_rows * plan.d_model)?;
    gpu.linear_packed_backward_input(
        &cache.query_weight.all(),
        plan.query_width,
        plan.d_model,
        &grad_queries.all(),
        false,
        false,
        &mut grad_queries_in,
        q_rows,
        0.0,
    )?;
    gpu.accumulate_packed_projection_grad(
        &mut [&mut attention.query],
        &grad_queries.all(),
        &cache.queries_in.all(),
        false,
        q_rows,
    )?;

    let mut grad_kv_in = gpu.uninit(kv_rows * plan.d_kv)?;
    gpu.linear_packed_backward_input(
        &cache.kv_weights.all(),
        plan.kv_stride,
        plan.d_kv,
        &grad_kv.all(),
        false,
        false,
        &mut grad_kv_in,
        kv_rows,
        0.0,
    )?;
    gpu.accumulate_packed_projection_grad(
        &mut [&mut attention.key, &mut attention.value],
        &grad_kv.all(),
        &cache.kv_in.all(),
        false,
        kv_rows,
    )?;

    Ok((grad_queries_in, grad_kv_in))
}

/// The widths and offsets both passes derive from the layer, in one place so
/// the two cannot disagree about where a head's keys start.
struct Plan {
    d_model: usize,
    d_kv: usize,
    heads: usize,
    group: usize,
    head_dim: usize,
    query_width: usize,
    /// Key and value packed: the stride of one row of the fused `kv` buffer.
    kv_stride: usize,
    key_base: usize,
    value_base: usize,
    block_size: usize,
    scale: f32,
}

impl Plan {
    fn new(attention: &MultiHeadAttention, shape: CrossShape) -> Result<Self, NetworkError> {
        if shape.q_len == 0 || shape.kv_len == 0 || shape.sequences == 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "cross-attention needs a non-empty shape, not {} x {} over {} sequences",
                shape.q_len, shape.kv_len, shape.sequences
            )));
        }
        if attention.is_rope_enabled() || attention.is_causal() {
            return Err(NetworkError::InvalidConfig(
                "device cross-attention wants a layer built with `MultiHeadAttention::cross`: \
                 rotary positions and the causal mask both have to be off"
                    .into(),
            ));
        }
        let head_dim = attention.head_dim();
        let kv_width = attention.num_kv_heads() * head_dim;
        Ok(Self {
            d_model: attention.d_model(),
            d_kv: attention.kv_features(),
            heads: attention.num_heads(),
            group: attention.num_heads() / attention.num_kv_heads(),
            head_dim,
            query_width: attention.num_heads() * head_dim,
            kv_stride: 2 * kv_width,
            key_base: 0,
            value_base: kv_width,
            block_size: shape.q_len * shape.kv_len,
            scale: (head_dim as f32).sqrt().recip(),
        })
    }

    /// Where head `head`'s key and value columns start inside their half of a
    /// fused `kv` row. Grouped-query heads share one.
    fn kv_base(&self, head: usize) -> usize {
        (head / self.group) * self.head_dim
    }

    /// `out = alpha * Q Kᵀ`, one batched GEMM per query head.
    ///
    /// Both passes need the same scores, so they share this rather than
    /// repeating eight arguments that have to agree.
    fn scores(
        &self,
        gpu: &Gpu<'_>,
        queries: &Act,
        kv: &Act,
        out: &mut CudaSlice<f32>,
        shape: CrossShape,
        alpha: f32,
    ) -> Result<(), NetworkError> {
        let queries = queries.wide();
        let kv = kv.wide();
        for head in 0..self.heads {
            let query = queries.slice(head * self.head_dim..);
            let key = kv.slice(self.key_base + self.kv_base(head)..);
            let mut head_scores = out.slice_mut(head * shape.sequences * self.block_size..);
            gemm_rhs_transposed_batched(
                gpu.context,
                &query,
                self.query_width,
                &key,
                self.kv_stride,
                &mut head_scores,
                shape.kv_len,
                shape.q_len,
                shape.kv_len,
                self.head_dim,
                alpha,
                0.0,
                shape.sequences,
                shape.q_len * self.query_width,
                shape.kv_len * self.kv_stride,
                self.block_size,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_training::cuda_doctor;
    use crate::gpu_transformer::GpuContext;
    use crate::matrix::Matrix;
    use crate::rope::Rope;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::sync::Arc;

    /// Same contract as the other CUDA tests: no device means the parity test
    /// reports success without running, a broken device is a failure.
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

    fn ramp(rows: usize, cols: usize, offset: usize) -> Matrix {
        Matrix::from_vec(
            rows,
            cols,
            (0..rows * cols)
                .map(|i| (((i + offset) * 37) % 23) as f32 / 11.0 - 1.0)
                .collect(),
        )
    }

    fn assert_close(label: &str, device: &[f32], host: &[f32], tolerance: f32) {
        assert_eq!(device.len(), host.len(), "{label}: length");
        for (index, (a, b)) in device.iter().zip(host).enumerate() {
            assert!(
                (a - b).abs() <= tolerance,
                "{label}[{index}]: {a} on the device vs {b} on the host"
            );
        }
    }

    /// The whole point of this module: the device path and
    /// [`MultiHeadAttention::forward_train_cross`] have to agree, output and
    /// gradients alike. Grouped heads, because sharing one key head between
    /// two query heads is where the batched GEMM strides are easiest to get
    /// wrong.
    #[test]
    fn cross_attention_matches_the_host_or_skips_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let gpu = Gpu { context: &context };

        let (d_model, d_kv, heads, kv_heads, head_dim) = (8, 6, 4, 2, 4);
        let (q_len, kv_len, sequences) = (3, 5, 2);
        let shape = CrossShape {
            q_len,
            kv_len,
            sequences,
        };

        let mut rng = StdRng::seed_from_u64(7);
        let rope = Rope::new(head_dim, 64, 10_000.0).unwrap();
        let mut host =
            MultiHeadAttention::cross(d_model, d_kv, heads, kv_heads, head_dim, rope, &mut rng)
                .unwrap();
        let mut device = host.clone();

        let queries_in = ramp(q_len * sequences, d_model, 0);
        let kv_in = ramp(kv_len * sequences, d_kv, 5);
        let grad_output = ramp(q_len * sequences, d_model, 11);

        let (host_output, host_cache) = host
            .forward_train_cross(&queries_in, &kv_in, q_len, kv_len)
            .unwrap();
        let (host_grad_queries, host_grad_kv) =
            host.backward_cross(&host_cache, &grad_output).unwrap();

        for param in device.params_mut() {
            param.move_to_cuda(&context).unwrap();
        }
        let device_queries_in = gpu.upload(&queries_in.data).unwrap();
        let device_kv_in = gpu.upload(&kv_in.data).unwrap();
        let device_grad_output = gpu.upload(&grad_output.data).unwrap();

        let (output, cache) = forward(
            &gpu,
            &device,
            &device_queries_in.slice(..),
            &device_kv_in.slice(..),
            shape,
        )
        .expect("the device forward pass");
        assert_close(
            "cross forward",
            &gpu.download(&output).unwrap(),
            &host_output.data,
            1e-4,
        );

        let (grad_queries, grad_kv) = backward(&gpu, &mut device, &cache, &device_grad_output)
            .expect("the device backward pass");
        assert_close(
            "cross grad_queries_in",
            &gpu.download(&grad_queries).unwrap(),
            &host_grad_queries.data,
            1e-4,
        );
        assert_close(
            "cross grad_kv_in",
            &gpu.download(&grad_kv).unwrap(),
            &host_grad_kv.data,
            1e-4,
        );

        // The weight gradients live on the device until they are asked for.
        for param in device.params_mut() {
            param.move_to_cpu().unwrap();
        }
        for (name, device, host) in [
            ("query", &device.query, &host.query),
            ("key", &device.key, &host.key),
            ("value", &device.value, &host.value),
            ("output", &device.output, &host.output),
        ] {
            assert_close(
                &format!("cross grad_{name}"),
                &device.weight.grad.data,
                &host.weight.grad.data,
                1e-4,
            );
        }
    }

    /// A self-attention layer reaching this path would silently lose its mask
    /// and its rotary positions, so it is refused instead.
    #[test]
    fn a_self_attention_layer_is_refused_or_skips_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let gpu = Gpu { context: &context };
        let mut rng = StdRng::seed_from_u64(1);
        let rope = Rope::new(4, 64, 10_000.0).unwrap();
        let layer = MultiHeadAttention::new(8, 2, 2, 4, rope, &mut rng).unwrap();

        let queries = gpu.zeros(2 * 8).unwrap();
        let keys = gpu.zeros(2 * 8).unwrap();
        let shape = CrossShape {
            q_len: 2,
            kv_len: 2,
            sequences: 1,
        };
        assert!(forward(&gpu, &layer, &queries.slice(..), &keys.slice(..), shape).is_err());
    }
}
