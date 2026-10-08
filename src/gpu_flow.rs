//! The flow transformer's blocks on the device.
//!
//! [`crate::gpu_model`] holds a decoder layer on the device and
//! [`crate::gpu_cross`] adds attention between two different token sets. This
//! module is the third piece `2Dto3DPlan.md` asks for: the block the flow
//! transformer actually has — self-attention over the latent slots,
//! cross-attention to the image tokens, a SwiGLU, each one wrapped in the
//! adaptive normalization of [`crate::adaln`].
//!
//! # What runs where
//!
//! Only the blocks run on the device. The timestep embedding, the shared
//! [`Modulation`](crate::adaln::Modulation), the input and output projections
//! and the final normalization stay on the host, because they are tiny: the
//! conditioning triple for a batch of eight at `d_model` 1024 is 98 KB against
//! the tens of megabytes a block moves, and keeping them on the host is two
//! transfers per step instead of a second copy of four host modules.
//!
//! That split decides the buffers. Every `AdaLayerNorm`'s shift, scale and
//! gate — the shared triple plus that sub-layer's offset, already summed —
//! are uploaded in one buffer per step, and their gradients come back in one
//! buffer per step, together with the RMSNorm scale gradients. A mid-pass
//! download drains the stream, which is the same reason
//! [`Gpu::accumulate_host_grad`] exists.
//!
//! # Precision
//!
//! Attention precision follows [`crate::gpu_cross`]: BF16 through the fused
//! kernels under mixed precision with heads 64 or 128 wide, FP32 otherwise.
//! The rest of the block is FP32. The self-attention here reuses that module
//! with the queries and the keys pointing at the same activations, which costs one extra copy of them and one extra
//! projection GEMM against a fused QKV. Fuse it when a profile says the block
//! is projection-bound rather than attention-bound.

use crate::adaln::{AdaLayerNorm, AdaLnCache, ModulationCache, TimestepCache};
use crate::flow_transformer::{FlowBlock, FlowTransformer};
use crate::gpu_cross::{self, CrossCache, CrossShape};
use crate::gpu_model::{Act, Gpu, SwiGluCache, backward_swiglu, forward_swiglu};
use crate::gpu_transformer::GpuContext;
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::param::Param;
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use std::sync::Arc;

/// How one call's activations are grouped.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FlowShape {
    /// Latent slots per sequence, which is also the self-attention length.
    pub(crate) latents: usize,
    /// Conditioning tokens per sequence.
    pub(crate) cond_tokens: usize,
    pub(crate) sequences: usize,
    pub(crate) d_model: usize,
}

impl FlowShape {
    fn rows(&self) -> usize {
        self.latents * self.sequences
    }

    /// One sub-layer's slice of the conditioning buffer.
    fn slab(&self) -> usize {
        self.sequences * 3 * self.d_model
    }

    /// Where block `index` starts in the conditioning buffer and in the
    /// RMSNorm scale-gradient buffer. Three sub-layers each, in the order
    /// self, cross, MLP.
    fn conditioning_at(&self, index: usize) -> (usize, usize) {
        (3 * index * self.slab(), 3 * index * self.d_model)
    }

    /// How long those two buffers are for a stack of `blocks` blocks.
    pub(crate) fn conditioning_len(&self, blocks: usize) -> (usize, usize) {
        (3 * blocks * self.slab(), 3 * blocks * self.d_model)
    }
}

/// A sub-layer's cache before its branch has run, which is the part
/// [`modulate`] can build.
struct Normed {
    input: CudaSlice<f32>,
    inverse_rms: CudaSlice<f32>,
    normalized: Act,
    weight: CudaSlice<f32>,
}

impl Normed {
    fn with(self, branch: CudaSlice<f32>) -> SubLayer {
        SubLayer {
            input: self.input,
            inverse_rms: self.inverse_rms,
            normalized: self.normalized,
            weight: self.weight,
            branch,
        }
    }
}

/// What one adaptively normalized sub-layer's backward pass needs.
struct SubLayer {
    /// The residual stream as it entered, which the norm differentiates
    /// against and the gated residual adds onto.
    input: CudaSlice<f32>,
    inverse_rms: CudaSlice<f32>,
    normalized: Act,
    weight: CudaSlice<f32>,
    /// What the branch produced, which is what the gate's gradient reads.
    branch: CudaSlice<f32>,
}

/// What [`block_backward`] needs from [`block_forward`].
pub(crate) struct GpuBlockCache {
    self_layer: SubLayer,
    self_attention: CrossCache,
    cross_layer: SubLayer,
    cross_attention: CrossCache,
    mlp_layer: SubLayer,
    mlp_modulated: Act,
    mlp: SwiGluCache,
}

/// Three sub-layers, each `hidden + gate * Branch(modulated(norm(hidden)))`.
///
/// `triples` holds every block's conditioning; block `index` reads the three
/// slabs at [`FlowShape::conditioning_at`], in the order self, cross, MLP.
/// `tokens` is the conditioning token set, `[sequences * cond_tokens,
/// cond_dim]`.
pub(crate) fn block_forward(
    gpu: &Gpu<'_>,
    block: &FlowBlock,
    hidden: CudaSlice<f32>,
    triples: &CudaSlice<f32>,
    index: usize,
    tokens: &CudaSlice<f32>,
    shape: FlowShape,
) -> Result<(CudaSlice<f32>, GpuBlockCache), NetworkError> {
    let (rows, width, slab) = (shape.rows(), shape.d_model, shape.slab());
    let self_shape = CrossShape {
        q_len: shape.latents,
        kv_len: shape.latents,
        sequences: shape.sequences,
    };
    let cross_shape = CrossShape {
        kv_len: shape.cond_tokens,
        ..self_shape
    };

    let (first, _) = shape.conditioning_at(index);
    let triple = triples.slice(first..first + slab);
    let (modulated, self_normed) =
        modulate(gpu, &block.self_norm, hidden, &triple, rows, shape.latents)?;
    // Queries and keys are the same activations here, which is what makes
    // `gpu_cross` serve self-attention: the layer is built with the mask and
    // the rotary positions off, so the two paths are the same arithmetic.
    let (branch, self_attention) = gpu_cross::forward(
        gpu,
        &block.self_attention,
        &modulated.slice(..),
        &modulated.slice(..),
        self_shape,
    )?;
    let hidden = gpu.gate_residual(
        &self_normed.input.slice(..),
        &branch.slice(..),
        &triple,
        rows,
        width,
        shape.latents,
    )?;
    let self_layer = self_normed.with(branch);

    let triple = triples.slice(first + slab..first + 2 * slab);
    let (modulated, cross_normed) =
        modulate(gpu, &block.cross_norm, hidden, &triple, rows, shape.latents)?;
    let (branch, cross_attention) = gpu_cross::forward(
        gpu,
        &block.cross_attention,
        &modulated.slice(..),
        &tokens.slice(..),
        cross_shape,
    )?;
    let hidden = gpu.gate_residual(
        &cross_normed.input.slice(..),
        &branch.slice(..),
        &triple,
        rows,
        width,
        shape.latents,
    )?;
    let cross_layer = cross_normed.with(branch);

    let triple = triples.slice(first + 2 * slab..first + 3 * slab);
    let (modulated, mlp_normed) =
        modulate(gpu, &block.mlp_norm, hidden, &triple, rows, shape.latents)?;
    let mlp_modulated = gpu.narrowed(&modulated.slice(..), rows * width, false)?;
    // The projection accumulates onto its destination, so the branch starts at
    // zero rather than at the residual: the gate below is what adds it back.
    let mut branch = gpu.zeros(rows * width)?;
    let mlp = forward_swiglu(gpu, &block.mlp, &mlp_modulated, &mut branch, rows)?;
    let output = gpu.gate_residual(
        &mlp_normed.input.slice(..),
        &branch.slice(..),
        &triple,
        rows,
        width,
        shape.latents,
    )?;
    let mlp_layer = mlp_normed.with(branch);

    Ok((
        output,
        GpuBlockCache {
            self_layer,
            self_attention,
            cross_layer,
            cross_attention,
            mlp_layer,
            mlp_modulated,
            mlp,
        },
    ))
}

/// Returns `dL/dhidden`, accumulating everything else where it belongs: the
/// weight gradients onto the device parameters, the conditioning gradient into
/// `grad_triples`, the RMSNorm scale gradients into `grad_norms` and the
/// conditioning tokens' gradient into `grad_tokens`.
///
/// Those three destinations are the caller's, zeroed once per step and
/// downloaded once at the end of it. `triples` is the same buffer
/// [`block_forward`] read, because the gate and the scale are needed again
/// here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_backward(
    gpu: &Gpu<'_>,
    block: &mut FlowBlock,
    cache: &GpuBlockCache,
    grad_output: &CudaSlice<f32>,
    triples: &CudaSlice<f32>,
    grad_triples: &mut CudaSlice<f32>,
    grad_norms: &mut CudaSlice<f32>,
    grad_tokens: &mut CudaSlice<f32>,
    index: usize,
    shape: FlowShape,
) -> Result<CudaSlice<f32>, NetworkError> {
    let (rows, width, slab) = (shape.rows(), shape.d_model, shape.slab());
    let (first, norm_first) = shape.conditioning_at(index);

    // The MLP, which is the last sub-layer the forward pass ran.
    let (at, norm_at) = (first + 2 * slab, norm_first + 2 * width);
    let grad_branch = gpu.gate_residual_backward(
        &cache.mlp_layer.branch.slice(..),
        &triples.slice(at..at + slab),
        &grad_output.slice(..),
        &mut grad_triples.slice_mut(at..at + slab),
        rows,
        width,
        shape.latents,
    )?;
    let grad_branch = gpu.narrowed(&grad_branch.slice(..), rows * width, false)?;
    let mut grad_modulated = gpu.uninit(rows * width)?;
    backward_swiglu(
        gpu,
        &mut block.mlp,
        &cache.mlp,
        &cache.mlp_modulated,
        &grad_branch,
        &mut grad_modulated,
        rows,
    )?;
    let grad_hidden = demodulate(
        gpu,
        &cache.mlp_layer,
        &grad_modulated,
        grad_output,
        &triples.slice(at..at + slab),
        &mut grad_triples.slice_mut(at..at + slab),
        &mut grad_norms.slice_mut(norm_at..norm_at + width),
        rows,
        shape.latents,
    )?;

    // The cross-attention. Its key side is the conditioning tokens, whose
    // gradient every block adds to.
    let (at, norm_at) = (first + slab, norm_first + width);
    let grad_branch = gpu.gate_residual_backward(
        &cache.cross_layer.branch.slice(..),
        &triples.slice(at..at + slab),
        &grad_hidden.slice(..),
        &mut grad_triples.slice_mut(at..at + slab),
        rows,
        width,
        shape.latents,
    )?;
    let (grad_modulated, grad_kv) = gpu_cross::backward(
        gpu,
        &mut block.cross_attention,
        &cache.cross_attention,
        &grad_branch,
    )?;
    let tokens = shape.sequences * shape.cond_tokens * block.cross_attention.kv_features();
    gpu.add(grad_tokens, &grad_kv, 0, tokens, true)?;
    let grad_hidden = demodulate(
        gpu,
        &cache.cross_layer,
        &grad_modulated,
        &grad_hidden,
        &triples.slice(at..at + slab),
        &mut grad_triples.slice_mut(at..at + slab),
        &mut grad_norms.slice_mut(norm_at..norm_at + width),
        rows,
        shape.latents,
    )?;

    // The self-attention, whose queries and keys were the same activations, so
    // its two input gradients are two halves of one.
    let (at, norm_at) = (first, norm_first);
    let grad_branch = gpu.gate_residual_backward(
        &cache.self_layer.branch.slice(..),
        &triples.slice(at..at + slab),
        &grad_hidden.slice(..),
        &mut grad_triples.slice_mut(at..at + slab),
        rows,
        width,
        shape.latents,
    )?;
    let (mut grad_modulated, grad_keys) = gpu_cross::backward(
        gpu,
        &mut block.self_attention,
        &cache.self_attention,
        &grad_branch,
    )?;
    gpu.add(&mut grad_modulated, &grad_keys, 0, rows * width, true)?;
    demodulate(
        gpu,
        &cache.self_layer,
        &grad_modulated,
        &grad_hidden,
        &triples.slice(at..at + slab),
        &mut grad_triples.slice_mut(at..at + slab),
        &mut grad_norms.slice_mut(norm_at..norm_at + width),
        rows,
        shape.latents,
    )
}

/// One `AdaLayerNorm`'s forward half: normalize, then modulate with this
/// sub-layer's slice of the conditioning.
fn modulate(
    gpu: &Gpu<'_>,
    norm: &AdaLayerNorm,
    input: CudaSlice<f32>,
    triple: &CudaView<'_, f32>,
    rows: usize,
    seq_len: usize,
) -> Result<(CudaSlice<f32>, Normed), NetworkError> {
    let width = norm.d_model();
    let weight = gpu.upload(&norm.norm.weight.value.data)?;
    let (normalized, inverse_rms, _) = gpu.rmsnorm(
        &input,
        &weight.as_view(),
        rows,
        width,
        norm.norm.eps,
        false,
        false,
    )?;
    let modulated = gpu.adaln_modulate(&normalized.wide(), triple, rows, width, seq_len)?;
    Ok((
        modulated,
        Normed {
            input,
            inverse_rms,
            normalized,
            weight,
        },
    ))
}

/// One `AdaLayerNorm`'s backward half.
///
/// `grad_residual` is what arrived at the sub-layer's output, which is also
/// what reaches its input through the residual path, so it is folded into the
/// norm's own kernel rather than added in a pass of its own.
#[allow(clippy::too_many_arguments)]
fn demodulate(
    gpu: &Gpu<'_>,
    layer: &SubLayer,
    grad_modulated: &CudaSlice<f32>,
    grad_residual: &CudaSlice<f32>,
    triple: &CudaView<'_, f32>,
    grad_triple: &mut CudaViewMut<'_, f32>,
    grad_weight: &mut CudaViewMut<'_, f32>,
    rows: usize,
    seq_len: usize,
) -> Result<CudaSlice<f32>, NetworkError> {
    let width = grad_weight.len();
    let grad_normalized = gpu.adaln_modulate_backward(
        &grad_modulated.slice(..),
        &layer.normalized.wide(),
        triple,
        grad_triple,
        rows,
        width,
        seq_len,
    )?;
    let (grad_input, _) = gpu.rmsnorm_backward(
        &layer.input,
        &grad_normalized,
        &layer.weight,
        &layer.inverse_rms,
        grad_weight,
        grad_residual,
        true,
        None,
        rows,
        width,
    )?;
    Ok(grad_input)
}

/// Moves the parts of the model the device path owns onto a device.
///
/// The attention projections and the feed-forwards, which is everything the
/// blocks multiply by. The normalization scales, the AdaLN offsets, the slot
/// identities and the four host modules stay where they are — see the module
/// documentation for why.
pub(crate) fn to_cuda(
    model: &mut FlowTransformer,
    context: &Arc<GpuContext>,
    memory_budget_mib: usize,
) -> Result<(), NetworkError> {
    // Value, gradient and the two Adam moments, all FP32, for the parameters
    // that are about to move. Activations are not in this number; the caller's
    // budget is meant to leave room for them.
    let weights: usize = device_params(model)
        .iter()
        .map(|param| param.value.data.len())
        .sum();
    let estimated_mib = (weights * 4 * 4).div_ceil(1024 * 1024);
    if memory_budget_mib > 0 && estimated_mib > memory_budget_mib {
        return Err(NetworkError::CudaMemoryBudget {
            estimated_mib,
            budget_mib: memory_budget_mib,
        });
    }
    for param in device_params(model) {
        param.move_to_cuda(context)?;
    }
    Ok(())
}

/// Brings them back, with whatever gradients they are holding.
pub(crate) fn to_cpu(model: &mut FlowTransformer) -> Result<(), NetworkError> {
    for param in device_params(model) {
        param.move_to_cpu()?;
    }
    Ok(())
}

fn device_params(model: &mut FlowTransformer) -> Vec<&mut Param> {
    let mut params = Vec::new();
    for block in &mut model.blocks {
        params.extend(block.self_attention.params_mut());
        params.extend(block.cross_attention.params_mut());
        params.extend(block.mlp.params_mut());
    }
    params
}

/// What [`backward`] needs from [`forward_train`].
pub(crate) struct GpuFlowCache {
    latent: Matrix,
    timestep: TimestepCache,
    modulation: ModulationCache,
    /// Every sub-layer's shift, scale and gate, on the device, because the
    /// backward pass reads them a second time.
    triples: CudaSlice<f32>,
    blocks: Vec<GpuBlockCache>,
    final_norm: AdaLnCache,
    final_modulated: Matrix,
    shape: FlowShape,
}

/// [`FlowTransformer::forward_train`] with the blocks on the device.
///
/// Same inputs, same outputs, same gradients — see the parity test at the
/// bottom of this file. The model has to have been through [`to_cuda`].
pub(crate) fn forward_train(
    model: &FlowTransformer,
    context: &Arc<GpuContext>,
    latent: &Matrix,
    tokens: &Matrix,
    times: &[f32],
) -> Result<(Matrix, GpuFlowCache), NetworkError> {
    let gpu = Gpu { context };
    let config = *model.config();
    let sequences = model.check(latent, tokens, times)?;
    let shape = FlowShape {
        latents: config.latents,
        cond_tokens: config.cond_tokens,
        sequences,
        d_model: config.d_model,
    };

    let (conditioning, timestep) = model.timestep.forward_train(times);
    let (triple, modulation) = model.modulation.forward_train(&conditioning);

    let mut embedded = model.latent_in.forward(latent);
    for row in 0..embedded.rows {
        let slot = model.slots.value.row(row % config.latents);
        for (value, identity) in embedded.row_mut(row).iter_mut().zip(slot) {
            *value += identity;
        }
    }

    let triples = gpu.upload(&combined_triples(model, &triple))?;
    let tokens_device = gpu.upload(&tokens.data)?;
    let mut hidden = gpu.upload(&embedded.data)?;
    let mut blocks = Vec::with_capacity(model.blocks.len());
    for (index, block) in model.blocks.iter().enumerate() {
        let (output, cache) =
            block_forward(&gpu, block, hidden, &triples, index, &tokens_device, shape)?;
        hidden = output;
        blocks.push(cache);
    }
    let hidden = Matrix::from_vec(shape.rows(), config.d_model, gpu.download(&hidden)?);

    let (final_modulated, _, final_norm) =
        model
            .final_norm
            .forward_train(&hidden, &triple, config.latents)?;
    let velocity = model.latent_out.forward(&final_modulated);

    Ok((
        velocity,
        GpuFlowCache {
            latent: latent.clone(),
            timestep,
            modulation,
            triples,
            blocks,
            final_norm,
            final_modulated,
            shape,
        },
    ))
}

/// [`FlowTransformer::backward`] with the blocks on the device.
pub(crate) fn backward(
    model: &mut FlowTransformer,
    context: &Arc<GpuContext>,
    cache: &GpuFlowCache,
    grad_velocity: &Matrix,
) -> Result<Matrix, NetworkError> {
    let gpu = Gpu { context };
    let config = *model.config();
    let shape = cache.shape;
    let (rows, width) = (shape.rows(), shape.d_model);

    let grad_final = model
        .latent_out
        .backward(&cache.final_modulated, grad_velocity);
    // No gate came out of the final layer, so none goes back into it.
    let zero_gate = Matrix::new(shape.sequences, width);
    let (grad_hidden, mut grad_triple) =
        model
            .final_norm
            .backward(&cache.final_norm, &grad_final, &zero_gate)?;

    let (triples_len, norms_len) = shape.conditioning_len(model.blocks.len());
    let mut grad_triples = gpu.zeros(triples_len)?;
    let mut grad_norms = gpu.zeros(norms_len)?;
    let mut grad_tokens = gpu.zeros(shape.sequences * config.cond_tokens * config.cond_dim)?;
    let mut grad_hidden_device = gpu.upload(&grad_hidden.data)?;

    for (index, (block, block_cache)) in
        model.blocks.iter_mut().zip(&cache.blocks).enumerate().rev()
    {
        grad_hidden_device = block_backward(
            &gpu,
            block,
            block_cache,
            &grad_hidden_device,
            &cache.triples,
            &mut grad_triples,
            &mut grad_norms,
            &mut grad_tokens,
            index,
            shape,
        )?;
    }

    // The three downloads of the backward pass, all of them after the whole
    // block stack is enqueued.
    let grad_hidden = Matrix::from_vec(rows, width, gpu.download(&grad_hidden_device)?);
    let triples = gpu.download(&grad_triples)?;
    let norms = gpu.download(&grad_norms)?;
    let grad_tokens = Matrix::from_vec(
        shape.sequences * config.cond_tokens,
        config.cond_dim,
        gpu.download(&grad_tokens)?,
    );

    let slab = shape.slab();
    for (index, block) in model.blocks.iter_mut().enumerate() {
        let (first, norm_first) = shape.conditioning_at(index);
        for (slot, norm) in [
            &mut block.self_norm,
            &mut block.cross_norm,
            &mut block.mlp_norm,
        ]
        .into_iter()
        .enumerate()
        {
            let scales = &norms[norm_first + slot * width..][..width];
            Gpu::accumulate_host_grad(&mut norm.norm.weight, scales);
            accumulate_conditioning(
                norm,
                &triples[first + slot * slab..][..slab],
                &mut grad_triple,
            );
        }
    }

    // Every sequence's rows share one set of slot identities, so the slot
    // gradient sums over the batch.
    if !model.slots.is_frozen() {
        for row in 0..grad_hidden.rows {
            let target = model.slots.grad.row_mut(row % config.latents);
            for (slot, value) in target.iter_mut().zip(grad_hidden.row(row)) {
                *slot += value;
            }
        }
    }
    model.latent_in.backward(&cache.latent, &grad_hidden);

    let grad_conditioning = model.modulation.backward(&cache.modulation, &grad_triple);
    model.timestep.backward(&cache.timestep, &grad_conditioning);

    Ok(grad_tokens)
}

/// Every sub-layer's `[sequences, 3 * d_model]` conditioning, the shared triple
/// with that sub-layer's offset already added, one block after another.
///
/// The host does this sum because the offsets live here: it is `3 * d_model`
/// values per sub-layer against the activations a block moves.
fn combined_triples(model: &FlowTransformer, triple: &Matrix) -> Vec<f32> {
    let mut combined = Vec::with_capacity(3 * model.blocks.len() * triple.data.len());
    for block in &model.blocks {
        for norm in [&block.self_norm, &block.cross_norm, &block.mlp_norm] {
            for row in 0..triple.rows {
                combined.extend(
                    triple
                        .row(row)
                        .iter()
                        .zip(&norm.offset.value.data)
                        .map(|(value, offset)| value + offset),
                );
            }
        }
    }
    combined
}

/// Splits one sub-layer's conditioning gradient between its offset, which sums
/// over the sequences, and the shared triple, which keeps them apart.
fn accumulate_conditioning(norm: &mut AdaLayerNorm, slab: &[f32], grad_triple: &mut Matrix) {
    let width = grad_triple.cols;
    for (row, values) in slab.chunks(width).enumerate() {
        for (slot, value) in grad_triple.row_mut(row).iter_mut().zip(values) {
            *slot += value;
        }
        if !norm.offset.is_frozen() {
            for (slot, value) in norm.offset.grad.data.iter_mut().zip(values) {
                *slot += value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_training::cuda_doctor;
    use crate::flow_transformer::FlowConfig;
    use rand::{Rng, SeedableRng, rngs::StdRng};

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

    fn tiny() -> FlowConfig {
        FlowConfig {
            d_model: 16,
            latents: 4,
            latent_dim: 3,
            cond_dim: 5,
            cond_tokens: 3,
            blocks: 2,
            num_heads: 2,
            head_dim: 8,
            d_ff: 16,
            d_cond: 8,
            frequencies: 4,
            eps: 1e-5,
        }
    }

    fn rows(count: usize, width: usize, salt: usize) -> Matrix {
        Matrix::from_vec(
            count,
            width,
            (0..count * width)
                .map(|i| ((i * 37 + salt * 11) % 23) as f32 / 23.0 - 0.5)
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

    /// The point of the module: a step on the device and the same step on the
    /// host have to reach the same velocity and the same gradients, for every
    /// parameter the model has.
    ///
    /// A fresh model is deliberately the identity — the AdaLN offsets and the
    /// output projection start at zero — so the weights are shaken awake
    /// first, or half of this would pass on zeros.
    #[test]
    fn a_step_matches_the_host_or_skips_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };

        let mut rng = StdRng::seed_from_u64(11);
        let mut host = FlowTransformer::new(tiny(), &mut rng).unwrap();
        for param in host.params_mut() {
            for value in &mut param.value.data {
                *value += rng.gen_range(-0.3..0.3);
            }
        }
        let mut device = host.clone();

        let config = *host.config();
        let sequences = 2;
        let latent = rows(sequences * config.latents, config.latent_dim, 1);
        let tokens = rows(sequences * config.cond_tokens, config.cond_dim, 2);
        let times = [0.8, 0.25];
        let grad_velocity = rows(latent.rows, config.latent_dim, 3);

        let (host_velocity, host_cache) = host.forward_train(&latent, &tokens, &times).unwrap();
        let host_grad_tokens = host.backward(&host_cache, &grad_velocity).unwrap();

        to_cuda(&mut device, &context, 0).unwrap();
        let (velocity, cache) = forward_train(&device, &context, &latent, &tokens, &times).unwrap();
        assert_close("velocity", &velocity.data, &host_velocity.data, 1e-4);

        let grad_tokens = backward(&mut device, &context, &cache, &grad_velocity).unwrap();
        assert_close(
            "grad_tokens",
            &grad_tokens.data,
            &host_grad_tokens.data,
            1e-4,
        );

        to_cpu(&mut device).unwrap();
        let expected: Vec<Vec<f32>> = host
            .params_mut()
            .iter()
            .map(|param| param.grad.data.clone())
            .collect();
        // The null embedding is the one parameter a forward pass never reads:
        // it is swapped in for the condition before the model sees it.
        let null_tokens = expected.len() - 1;
        for (index, (param, host)) in device.params_mut().iter().zip(&expected).enumerate() {
            assert!(
                index == null_tokens || host.iter().any(|value| value.abs() > 1e-6),
                "parameter {index} has no gradient to compare"
            );
            assert_close(
                &format!("grad of parameter {index}"),
                &param.grad.data,
                host,
                1e-4,
            );
        }
    }

    /// Stage 3.3's gate: the same tiny model trains to the same loss curve on
    /// the device as on the host.
    ///
    /// One step matching is not the same as a run matching — an error in how a
    /// gradient reaches the optimizer, or in which parameters moved, only
    /// shows once the weights start changing. Both models see the same data
    /// and the same random draws, so the two curves should track each other to
    /// what float arithmetic in a different order costs, and both should fall.
    #[test]
    fn a_training_run_tracks_the_host_curve_or_skips_without_device() {
        if cuda_or_skip().is_none() {
            return;
        }

        let mut rng = StdRng::seed_from_u64(12);
        let mut host = FlowTransformer::new(tiny(), &mut rng).unwrap();
        host.set_optimizer(crate::optimizers::Optimizer::adam(3e-3));
        let mut device = host.clone();
        device.to_cuda_with_precision(0, 0, false).unwrap();

        let config = *host.config();
        let clean = rows(config.latents, config.latent_dim, 20);
        let tokens = rows(config.cond_tokens, config.cond_dim, 21);

        let run = |model: &mut FlowTransformer, seed: u64| {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut losses = Vec::new();
            for _ in 0..60 {
                model.zero_grad();
                losses.push(model.train_step(&clean, &tokens, 0.1, &mut rng).unwrap());
                model.step_clipped(1.0, 1.0).unwrap();
            }
            losses
        };
        let host_losses = run(&mut host, 7);
        let device_losses = run(&mut device, 7);

        for (step, (device, host)) in device_losses.iter().zip(&host_losses).enumerate() {
            assert!(
                (device - host).abs() <= 1e-3 * host.max(1.0),
                "step {step}: {device} on the device vs {host} on the host"
            );
        }
        assert!(
            device_losses[59] < device_losses[0] * 0.5,
            "the device run did not learn: {} to {}",
            device_losses[0],
            device_losses[59]
        );

        // The weights the run produced, not just the losses it printed.
        device.to_cpu().unwrap();
        for (index, (param, expected)) in device
            .params_mut()
            .iter()
            .zip(host.params_mut())
            .enumerate()
        {
            assert_close(
                &format!("parameter {index} after 60 steps"),
                &param.value.data,
                &expected.value.data,
                1e-3,
            );
        }
    }
}
