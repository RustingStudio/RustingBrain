//! The shape autoencoder on the device.
//!
//! [`crate::gpu_flow`] does this for the flow transformer's blocks; this is the
//! same job for [`ShapeVae`], which is the stage the plan trains first. Both
//! halves move: the encoder reads a few thousand surface points and squeezes
//! them into the latents, the decoder answers a few thousand query points, and
//! every GEMM in between is large enough to be worth a launch.
//!
//! # What runs where
//!
//! The projections, the attentions and the feed-forwards run on the device.
//! The Fourier features of a point set are built on the host and uploaded, the
//! RMSNorm scales are uploaded per call with their gradients coming back in one
//! buffer, and the latent queries and the distance bias stay host parameters.
//! Those three are tiny — a norm scale is `d_model` numbers against the
//! `[points, d_model]` activations beside it — and keeping them on the host is
//! what keeps the optimizer, the checkpoint and the host path unchanged.
//!
//! The features are the one host computation that is not trivial: a sine and a
//! cosine per axis per octave over every point. ponytail: it is well under the
//! attention it feeds, and moving it would need a kernel plus a second copy of
//! the layout. Move it when a profile says the host is the ceiling.
//!
//! # Packed batches
//!
//! The ordinary API still accepts one shape, but the training driver packs
//! short equal-sized point clouds: they become independent `sequences` inside
//! one set of projection and attention launches. Very large query sets remain
//! shape-at-a-time because their combined attention workspace costs more than
//! the saved launches.
//!
//! # Precision
//!
//! Master weights, buffers and accumulators stay FP32. By default, cuBLAS
//! rounds GEMM multiplier inputs for BF16 tensor-core kernels; parity tests opt
//! back into the exact FP32 path.

use crate::gpu_cross::{self, CrossCache, CrossShape};
use crate::gpu_model::{Act, Gpu, SwiGluCache, backward_swiglu, forward_swiglu};
use crate::gpu_transformer::GpuContext;
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::norm::RmsNorm;
use crate::param::{Linear, Param};
use crate::shape_vae::{Block, ShapeVae, fourier_features, join, split};
use cudarc::driver::{CudaSlice, CudaView, CudaViewMut};
use std::sync::Arc;

/// An FP32 buffer as a GEMM operand.
///
/// The packed-weight GEMMs take their operands as bytes so that one code path
/// serves both precisions. Everything here is wide, so this is a view rather
/// than the copy [`Gpu::narrowed`] would make.
fn bytes(buffer: &CudaSlice<f32>) -> CudaView<'_, u8> {
    unsafe { buffer.transmute::<u8>(buffer.len() * 4) }
        .expect("an FP32 buffer is four bytes per element")
}

/// One projection's packed weight, with the two dimensions its GEMMs need.
pub(crate) struct Projection {
    weights: Act,
    units: usize,
    inner: usize,
}

impl Projection {
    fn new(gpu: &Gpu<'_>, linear: &Linear) -> Result<Self, NetworkError> {
        Ok(Self {
            weights: gpu.pack(&[linear], false)?,
            units: linear.out_features(),
            inner: linear.in_features(),
        })
    }

    /// `[rows, inner] -> [rows, units]`.
    fn forward(
        &self,
        gpu: &Gpu<'_>,
        input: &CudaView<'_, u8>,
        rows: usize,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let mut out = gpu.uninit(rows * self.units)?;
        gpu.linear_packed(
            &self.weights.all(),
            self.units,
            self.inner,
            input,
            false,
            &mut out,
            rows,
            0.0,
        )?;
        Ok(out)
    }

    /// `dL/dinput`, with the weight gradient accumulated onto the device
    /// parameter.
    fn backward(
        &self,
        gpu: &Gpu<'_>,
        linear: &mut Linear,
        grad_output: &CudaView<'_, u8>,
        input: &CudaView<'_, u8>,
        rows: usize,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let mut grad_input = gpu.uninit(rows * self.inner)?;
        gpu.linear_packed_backward_input(
            &self.weights.all(),
            self.units,
            self.inner,
            grad_output,
            false,
            false,
            &mut grad_input,
            rows,
            0.0,
        )?;
        gpu.accumulate_projection_grad(linear, grad_output, input, false, rows)?;
        Ok(grad_input)
    }
}

/// One RMSNorm's forward pass, and what its backward pass reads back.
struct Normed {
    /// The residual stream as it entered, which the norm differentiates
    /// against.
    input: CudaSlice<f32>,
    weight: CudaSlice<f32>,
    inverse_rms: CudaSlice<f32>,
    normed: Act,
}

/// `narrow` asks for the normed activation in BF16 rather than FP32.
///
/// It is only ever set for a norm whose output feeds a SwiGLU, because
/// [`forward_swiglu`] reads the flag off the activation and runs its whole
/// branch at the operand width it finds: the gate, the up and the down
/// projection between them are about three quarters of the decode's
/// arithmetic, and BF16 operands are what makes cuBLAS pick the kernel that
/// takes twice the `k` per instruction. The cache also holds half as many
/// bytes per row of it.
///
/// This is a real reduction in precision rather than a free one. The wide path
/// under `mixed_precision` asks cuBLAS for `32F_FAST_16BF`, which is a hint the
/// library may decline on a small GEMM, and it stores the activation itself at
/// full width; a narrow buffer rounds unconditionally. So the flag follows
/// `mixed_precision` and is never set on a norm whose consumer has not been
/// measured at this width.
fn normalize(
    gpu: &Gpu<'_>,
    norm: &RmsNorm,
    input: CudaSlice<f32>,
    rows: usize,
    width: usize,
    narrow: bool,
) -> Result<Normed, NetworkError> {
    let weight = gpu.upload(&norm.weight.value.data)?;
    let (normed, inverse_rms, _) =
        gpu.rmsnorm(&input, &weight, rows, width, norm.eps, narrow, false)?;
    Ok(Normed {
        input,
        weight,
        inverse_rms,
        normed,
    })
}

/// Returns `dL/dinput`, accumulating the scale gradient into `grad_weight`.
///
/// `residual`, when it is `Some`, is what arrived at the sub-layer's output and
/// so also reaches its input through the residual path; the norm's own kernel
/// folds it in rather than costing a second pass over the activation.
fn denormalize(
    gpu: &Gpu<'_>,
    layer: &Normed,
    grad_normed: &CudaSlice<f32>,
    grad_weight: &mut CudaViewMut<'_, f32>,
    residual: Option<&CudaSlice<f32>>,
    rows: usize,
    width: usize,
) -> Result<CudaSlice<f32>, NetworkError> {
    let (grad_input, _) = gpu.rmsnorm_backward(
        &layer.input,
        grad_normed,
        &layer.weight,
        &layer.inverse_rms,
        grad_weight,
        residual.unwrap_or(grad_normed),
        residual.is_some(),
        None,
        rows,
        width,
    )?;
    Ok(grad_input)
}

/// Pre-norm self-attention and a SwiGLU, both residual: one encoder block.
struct GpuBlockCache {
    attention_norm: Normed,
    attention: CrossCache,
    /// The input plus the attention branch is not kept separately: it is the
    /// feed-forward norm's own input.
    mlp_norm: Normed,
    mlp: SwiGluCache,
}

fn block_forward(
    gpu: &Gpu<'_>,
    block: &Block,
    hidden: CudaSlice<f32>,
    sequence_len: usize,
    sequences: usize,
    width: usize,
) -> Result<(CudaSlice<f32>, GpuBlockCache), NetworkError> {
    let rows = sequence_len * sequences;
    let shape = CrossShape {
        q_len: sequence_len,
        kv_len: sequence_len,
        sequences,
    };
    let attention_norm = normalize(gpu, &block.attention_norm, hidden, rows, width, false)?;
    // Queries and keys are the same activations here, which is what makes
    // `gpu_cross` serve self-attention: the layer is built with the mask and
    // the rotary positions off, so the two paths are the same arithmetic.
    let normed = attention_norm.normed.wide();
    let (attended, attention) = gpu_cross::forward(gpu, &block.attention, &normed, &normed, shape)?;
    let mut residual = attended;
    gpu.add(&mut residual, &attention_norm.input, 0, rows * width, true)?;

    let mlp_norm = normalize(
        gpu,
        &block.mlp_norm,
        residual,
        rows,
        width,
        gpu.context.mixed_precision,
    )?;
    // The down projection accumulates onto its destination, so the branch
    // starts at zero and the residual is added after it.
    let mut output = gpu.zeros(rows * width)?;
    let mlp = forward_swiglu(gpu, &block.mlp, &mlp_norm.normed, &mut output, rows)?;
    gpu.add(&mut output, &mlp_norm.input, 0, rows * width, true)?;

    Ok((
        output,
        GpuBlockCache {
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn block_backward(
    gpu: &Gpu<'_>,
    block: &mut Block,
    cache: &GpuBlockCache,
    grad_output: &CudaSlice<f32>,
    grad_norms: &mut CudaSlice<f32>,
    at: usize,
    sequence_len: usize,
    sequences: usize,
    width: usize,
) -> Result<CudaSlice<f32>, NetworkError> {
    let rows = sequence_len * sequences;
    let grad_branch = gpu.narrowed(
        &grad_output.slice(..),
        rows * width,
        cache.mlp_norm.normed.is_narrow(),
    )?;
    let mut grad_normed = gpu.uninit(rows * width)?;
    backward_swiglu(
        gpu,
        &mut block.mlp,
        &cache.mlp,
        &cache.mlp_norm.normed,
        &grad_branch,
        &mut grad_normed,
        rows,
    )?;
    let grad_residual = denormalize(
        gpu,
        &cache.mlp_norm,
        &grad_normed,
        &mut grad_norms.slice_mut(at + width..at + 2 * width),
        Some(grad_output),
        rows,
        width,
    )?;

    // The self-attention's queries and keys were the same activations, so its
    // two input gradients are two halves of one.
    let (mut grad_normed, grad_keys) =
        gpu_cross::backward(gpu, &mut block.attention, &cache.attention, &grad_residual)?;
    gpu.add(&mut grad_normed, &grad_keys, 0, rows * width, true)?;
    denormalize(
        gpu,
        &cache.attention_norm,
        &grad_normed,
        &mut grad_norms.slice_mut(at..at + width),
        Some(&grad_residual),
        rows,
        width,
    )
}

/// What [`encode_backward`] needs from [`encode_train`].
pub(crate) struct GpuEncoderCache {
    /// The Fourier features of the surface points, which is what the input
    /// projection's weight gradient is a product against.
    features: CudaSlice<f32>,
    cross: CrossCache,
    blocks: Vec<GpuBlockCache>,
    encoder_norm: Normed,
    to_moments: Projection,
    points: usize,
    sequences: usize,
}

/// [`ShapeVae::encode_train`] with everything but the features on the device.
pub(crate) fn encode_train(
    model: &ShapeVae,
    context: &Arc<GpuContext>,
    surface: &Matrix,
) -> Result<(Matrix, Matrix, GpuEncoderCache), NetworkError> {
    encode_train_batch(model, context, surface, 1)
}

/// Batched [`encode_train`]. Every sequence has the same number of surface
/// points and produces one contiguous block of latent rows.
pub(crate) fn encode_train_batch(
    model: &ShapeVae,
    context: &Arc<GpuContext>,
    surface: &Matrix,
    sequences: usize,
) -> Result<(Matrix, Matrix, GpuEncoderCache), NetworkError> {
    if sequences == 0 || surface.rows == 0 || surface.rows % sequences != 0 {
        return Err(NetworkError::InvalidConfig(format!(
            "{} surface rows cannot be split across {sequences} shapes",
            surface.rows
        )));
    }
    let gpu = Gpu { context };
    let config = *model.config();
    let (latents, width) = (config.latents, config.d_model);

    let features = model.surface_features(surface)?;
    let points = features.rows / sequences;
    let feature_rows = features.rows;
    let features = gpu.upload(&features.data)?;
    let surface_in = Projection::new(&gpu, &model.surface_in)?;
    let tokens = surface_in.forward(&gpu, &bytes(&features), feature_rows)?;

    // The learned queries are the whole compression: however many surface
    // points came in, exactly `latents` rows come out.
    let repeated_queries: Vec<f32> = (0..sequences)
        .flat_map(|_| model.latent_queries.value.data.iter().copied())
        .collect();
    let queries = gpu.upload(&repeated_queries)?;
    let (crossed, cross) = gpu_cross::forward(
        &gpu,
        &model.read_surface,
        &queries.slice(..),
        &tokens.slice(..),
        CrossShape {
            q_len: latents,
            kv_len: points,
            sequences,
        },
    )?;
    let mut hidden = crossed;
    let latent_rows = sequences * latents;
    gpu.add(&mut hidden, &queries, 0, latent_rows * width, true)?;

    let mut blocks = Vec::with_capacity(model.blocks.len());
    for block in &model.blocks {
        let (output, cache) = block_forward(&gpu, block, hidden, latents, sequences, width)?;
        hidden = output;
        blocks.push(cache);
    }

    let encoder_norm = normalize(&gpu, &model.encoder_norm, hidden, latent_rows, width, false)?;
    let to_moments = Projection::new(&gpu, &model.to_moments)?;
    let moments = to_moments.forward(&gpu, &encoder_norm.normed.all(), latent_rows)?;
    let moments = Matrix::from_vec(latent_rows, 2 * config.latent_dim, gpu.download(&moments)?);
    let (mean, log_variance) = split(&moments);

    Ok((
        mean,
        log_variance,
        GpuEncoderCache {
            features,
            cross,
            blocks,
            encoder_norm,
            to_moments,
            points,
            sequences,
        },
    ))
}

/// [`ShapeVae::encode_backward`] on the device.
pub(crate) fn encode_backward(
    model: &mut ShapeVae,
    context: &Arc<GpuContext>,
    cache: &GpuEncoderCache,
    grad_mean: &Matrix,
    grad_log_variance: &Matrix,
) -> Result<(), NetworkError> {
    let gpu = Gpu { context };
    let config = *model.config();
    let (latents, width) = (config.latents, config.d_model);
    let latent_rows = latents * cache.sequences;

    // One buffer for every norm scale in the stack, downloaded once at the end:
    // a download in the middle of the pass drains the stream.
    let norms = 1 + 2 * model.blocks.len();
    let mut grad_norms = gpu.zeros(norms * width)?;

    let grad_moments = join(grad_mean, grad_log_variance);
    let grad_moments = gpu.upload(&grad_moments.data)?;
    let grad_normed = cache.to_moments.backward(
        &gpu,
        &mut model.to_moments,
        &bytes(&grad_moments),
        &cache.encoder_norm.normed.all(),
        latent_rows,
    )?;
    let mut grad = denormalize(
        &gpu,
        &cache.encoder_norm,
        &grad_normed,
        &mut grad_norms.slice_mut(..width),
        None,
        latent_rows,
        width,
    )?;

    for (index, (block, block_cache)) in
        model.blocks.iter_mut().zip(&cache.blocks).enumerate().rev()
    {
        grad = block_backward(
            &gpu,
            block,
            block_cache,
            &grad,
            &mut grad_norms,
            width + 2 * index * width,
            latents,
            cache.sequences,
            width,
        )?;
    }

    let (grad_queries, grad_tokens) =
        gpu_cross::backward(&gpu, &mut model.read_surface, &cache.cross, &grad)?;
    // The queries sit on the residual path, so they take both shares.
    let mut grad_queries = grad_queries;
    gpu.add(&mut grad_queries, &grad, 0, latent_rows * width, true)?;

    // The projection wants its weight gradient; its input gradient is the
    // gradient with respect to the surface points, which nothing reads.
    gpu.accumulate_projection_grad(
        &mut model.surface_in,
        &bytes(&grad_tokens),
        &bytes(&cache.features),
        false,
        cache.points * cache.sequences,
    )?;

    let grad_queries = gpu.download(&grad_queries)?;
    let scales = gpu.download(&grad_norms)?;
    if !model.latent_queries.is_frozen() {
        for sequence in grad_queries.chunks_exact(latents * width) {
            for (slot, value) in model.latent_queries.grad.data.iter_mut().zip(sequence) {
                *slot += value;
            }
        }
    }
    Gpu::accumulate_host_grad(&mut model.encoder_norm.weight, &scales[..width]);
    for (index, block) in model.blocks.iter_mut().enumerate() {
        let at = width + 2 * index * width;
        Gpu::accumulate_host_grad(&mut block.attention_norm.weight, &scales[at..at + width]);
        Gpu::accumulate_host_grad(
            &mut block.mlp_norm.weight,
            &scales[at + width..at + 2 * width],
        );
    }
    Ok(())
}

/// What [`decode_backward`] needs from [`decode_train`].
pub(crate) struct GpuDecoderCache {
    /// The latent itself, which the widening projection's weight gradient is a
    /// product against.
    latent: CudaSlice<f32>,
    from_latent: Projection,
    features: CudaSlice<f32>,
    cross: CrossCache,
    decoder_norm: Normed,
    mlp: SwiGluCache,
    head_norm: Normed,
    head: Projection,
    color_head: Projection,
    rows: usize,
    sequences: usize,
}

/// [`ShapeVae::decode_train`] on the device, for one chunk of query points.
pub(crate) fn decode_train(
    model: &ShapeVae,
    context: &Arc<GpuContext>,
    latent: &Matrix,
    queries: &Matrix,
) -> Result<(Vec<f32>, GpuDecoderCache), NetworkError> {
    decode_train_batch(model, context, latent, queries, 1)
}

/// Batched [`decode_train`]. Latents and queries are grouped into the same
/// number of equal-sized sequences.
pub(crate) fn decode_train_batch(
    model: &ShapeVae,
    context: &Arc<GpuContext>,
    latent: &Matrix,
    queries: &Matrix,
    sequences: usize,
) -> Result<(Vec<f32>, GpuDecoderCache), NetworkError> {
    let gpu = Gpu { context };
    let config = *model.config();
    let (latents, width) = (config.latents, config.d_model);
    if sequences == 0
        || latent.rows != sequences * latents
        || latent.cols != config.latent_dim
        || queries.rows == 0
        || queries.rows % sequences != 0
    {
        return Err(NetworkError::InvalidConfig(format!(
            "the batch has {} latent rows and {} query rows for {sequences} shapes; expected {} latent rows and equal query counts",
            latent.rows,
            queries.rows,
            sequences * latents,
        )));
    }

    let latent_device = gpu.upload(&latent.data)?;
    let from_latent = Projection::new(&gpu, &model.from_latent)?;
    let latent_rows = sequences * latents;
    let kv = from_latent.forward(&gpu, &bytes(&latent_device), latent_rows)?;

    let features = fourier_features(queries, config.frequencies)?;
    let rows = features.rows;
    let query_rows = rows / sequences;
    let features = gpu.upload(&features.data)?;
    let query_in = Projection::new(&gpu, &model.query_in)?;
    let embedded = query_in.forward(&gpu, &bytes(&features), rows)?;

    let (crossed, cross) = gpu_cross::forward(
        &gpu,
        &model.read_latent,
        &embedded.slice(..),
        &kv.slice(..),
        CrossShape {
            q_len: query_rows,
            kv_len: latents,
            sequences,
        },
    )?;
    let mut residual = crossed;
    gpu.add(&mut residual, &embedded, 0, rows * width, true)?;

    let decoder_norm = normalize(
        &gpu,
        &model.decoder_norm,
        residual,
        rows,
        width,
        gpu.context.mixed_precision,
    )?;
    let mut hidden = gpu.zeros(rows * width)?;
    let mlp = forward_swiglu(
        &gpu,
        &model.decoder_mlp,
        &decoder_norm.normed,
        &mut hidden,
        rows,
    )?;
    gpu.add(&mut hidden, &decoder_norm.input, 0, rows * width, true)?;

    let head_norm = normalize(&gpu, &model.head_norm, hidden, rows, width, false)?;
    let head = Projection::new(&gpu, &model.head)?;
    let distances = head.forward(&gpu, &head_norm.normed.all(), rows)?;
    let bias = model.head_bias.value.data[0];
    let distances = gpu
        .download(&distances)?
        .iter()
        .map(|value| value + bias)
        .collect();

    Ok((
        distances,
        GpuDecoderCache {
            latent: latent_device,
            from_latent,
            features,
            cross,
            decoder_norm,
            mlp,
            head_norm,
            head,
            color_head: Projection::new(&gpu, &model.color_head)?,
            rows,
            sequences,
        },
    ))
}

/// [`ShapeVae::colors`] on the device: one projection off the cached hidden
/// state the distance was read from.
pub(crate) fn colors(model: &ShapeVae, cache: &GpuDecoderCache) -> Result<Matrix, NetworkError> {
    let context = model.device_context()?;
    let gpu = Gpu { context: &context };
    let colors = cache
        .color_head
        .forward(&gpu, &cache.head_norm.normed.all(), cache.rows)?;
    Ok(Matrix::from_vec(cache.rows, 3, gpu.download(&colors)?))
}

/// [`ShapeVae::decode_backward_colored`] on the device.
pub(crate) fn decode_backward(
    model: &mut ShapeVae,
    context: &Arc<GpuContext>,
    cache: &GpuDecoderCache,
    grad_distances: &[f32],
    grad_colors: Option<&Matrix>,
) -> Result<Matrix, NetworkError> {
    let gpu = Gpu { context };
    let config = *model.config();
    let (latents, width, rows) = (config.latents, config.d_model, cache.rows);
    let latent_rows = latents * cache.sequences;
    if grad_distances.len() != rows {
        return Err(NetworkError::InvalidConfig(format!(
            "{rows} query points were decoded and {} gradients came back",
            grad_distances.len()
        )));
    }
    if let Some(grad_colors) = grad_colors
        && (grad_colors.rows != rows || grad_colors.cols != 3)
    {
        return Err(NetworkError::InvalidConfig(format!(
            "the colour gradient is [{}, {}] and should be [{rows}, 3]",
            grad_colors.rows, grad_colors.cols
        )));
    }
    if !model.head_bias.is_frozen() {
        model.head_bias.grad.data[0] += grad_distances.iter().sum::<f32>();
    }

    let mut grad_norms = gpu.zeros(2 * width)?;
    let grad_output = gpu.upload(grad_distances)?;
    let mut grad_normed = cache.head.backward(
        &gpu,
        &mut model.head,
        &bytes(&grad_output),
        &cache.head_norm.normed.all(),
        rows,
    )?;
    if let Some(grad_colors) = grad_colors {
        let grad_colors = gpu.upload(&grad_colors.data)?;
        let from_colors = cache.color_head.backward(
            &gpu,
            &mut model.color_head,
            &bytes(&grad_colors),
            &cache.head_norm.normed.all(),
            rows,
        )?;
        gpu.add(&mut grad_normed, &from_colors, 0, rows * width, true)?;
    }
    let grad_hidden = denormalize(
        &gpu,
        &cache.head_norm,
        &grad_normed,
        &mut grad_norms.slice_mut(width..),
        None,
        rows,
        width,
    )?;

    let grad_branch = gpu.narrowed(
        &grad_hidden.slice(..),
        rows * width,
        cache.decoder_norm.normed.is_narrow(),
    )?;
    let mut grad_normed = gpu.uninit(rows * width)?;
    backward_swiglu(
        &gpu,
        &mut model.decoder_mlp,
        &cache.mlp,
        &cache.decoder_norm.normed,
        &grad_branch,
        &mut grad_normed,
        rows,
    )?;
    let grad_residual = denormalize(
        &gpu,
        &cache.decoder_norm,
        &grad_normed,
        &mut grad_norms.slice_mut(..width),
        Some(&grad_hidden),
        rows,
        width,
    )?;

    let (grad_embedded, grad_kv) =
        gpu_cross::backward(&gpu, &mut model.read_latent, &cache.cross, &grad_residual)?;
    // The embedded query is on the residual path, like the latent queries in
    // the encoder.
    let mut grad_embedded = grad_embedded;
    gpu.add(&mut grad_embedded, &grad_residual, 0, rows * width, true)?;
    // Only the weight gradient: nothing reads the gradient with respect to the
    // query points themselves.
    gpu.accumulate_projection_grad(
        &mut model.query_in,
        &bytes(&grad_embedded),
        &bytes(&cache.features),
        false,
        rows,
    )?;

    let grad_latent = cache.from_latent.backward(
        &gpu,
        &mut model.from_latent,
        &bytes(&grad_kv),
        &bytes(&cache.latent),
        latent_rows,
    )?;
    let grad_latent = Matrix::from_vec(latent_rows, config.latent_dim, gpu.download(&grad_latent)?);

    let scales = gpu.download(&grad_norms)?;
    Gpu::accumulate_host_grad(&mut model.decoder_norm.weight, &scales[..width]);
    Gpu::accumulate_host_grad(&mut model.head_norm.weight, &scales[width..]);
    Ok(grad_latent)
}

/// Moves the parts of the model the device path owns onto a device.
///
/// Every projection, which is everything the two passes multiply by. The
/// normalization scales, the latent queries and the distance bias stay on the
/// host — see the module documentation for why.
pub(crate) fn to_cuda(
    model: &mut ShapeVae,
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
pub(crate) fn to_cpu(model: &mut ShapeVae) -> Result<(), NetworkError> {
    for param in device_params(model) {
        param.move_to_cpu()?;
    }
    Ok(())
}

fn device_params(model: &mut ShapeVae) -> Vec<&mut Param> {
    let mut params = model.surface_in.params_mut();
    params.extend(model.read_surface.params_mut());
    for block in &mut model.blocks {
        params.extend(block.attention.params_mut());
        params.extend(block.mlp.params_mut());
    }
    params.extend(model.to_moments.params_mut());
    params.extend(model.from_latent.params_mut());
    params.extend(model.query_in.params_mut());
    params.extend(model.read_latent.params_mut());
    params.extend(model.decoder_mlp.params_mut());
    params.extend(model.head.params_mut());
    params.extend(model.color_head.params_mut());
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_training::cuda_doctor;
    use crate::losses;
    use crate::optimizers::{self, Optimizer};
    use crate::shape_vae::ShapeVaeConfig;
    use rand::{Rng, SeedableRng, rngs::StdRng};

    /// Same contract as the other CUDA tests: no device means the parity test
    /// reports success without running, a broken device is a failure.
    fn cuda_or_skip() -> bool {
        match cuda_doctor(0, 8192) {
            Ok(_) => true,
            Err(NetworkError::Cuda(message))
                if message.contains("NO_DEVICE") || message.contains("no CUDA-capable device") =>
            {
                false
            }
            Err(error) => panic!("CUDA is present but the CUDA doctor failed: {error}"),
        }
    }

    fn tiny() -> ShapeVaeConfig {
        ShapeVaeConfig {
            d_model: 16,
            latents: 4,
            latent_dim: 3,
            num_heads: 2,
            head_dim: 8,
            d_ff: 16,
            encoder_blocks: 2,
            frequencies: 2,
            eps: 1e-5,
            free_bits: 0.0,
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

    /// Relative once the numbers are large: a gradient summed over a few
    /// thousand points is tens of units wide, and the two paths add it up in a
    /// different order.
    fn assert_close(label: &str, device: &[f32], host: &[f32], tolerance: f32) {
        assert_eq!(device.len(), host.len(), "{label}: length");
        for (index, (a, b)) in device.iter().zip(host).enumerate() {
            assert!(
                (a - b).abs() <= tolerance * b.abs().max(1.0),
                "{label}[{index}]: {a} on the device vs {b} on the host"
            );
        }
    }

    /// One full step — encode, decode, both backward passes, colours included —
    /// has to reach the same numbers and the same gradients on both paths.
    #[test]
    fn a_step_matches_the_host_or_skips_without_device() {
        step_parity(false, 1e-4);
    }

    /// The same step with reduced precision on, which is what
    /// [`ShapeVae::to_cuda`] gives a real run.
    ///
    /// Under mixed precision the SwiGLU's operands are stored and handed to
    /// cuBLAS as BF16, so the tolerance is the one eight mantissa bits earn
    /// rather than the FP32 path's. It is loose here for a second reason: the
    /// model is sixteen channels wide, its weights are random, and its colour
    /// head ends in a sigmoid, so the sums cancel and a rounded operand arrives
    /// magnified. The factor is the same one the FP32 run shows — that path
    /// rounds about a thousand times less and lands about a thousand times
    /// closer — so the two tolerances describe one model, not two behaviours.
    /// What the test pins down is that the narrowed branch still computes the
    /// same function: a flag that disagrees with the buffer it describes reads
    /// those bytes at the wrong width, and that is garbage rather than a
    /// rounder number.
    #[test]
    fn a_reduced_precision_step_tracks_the_host_or_skips_without_device() {
        step_parity(true, 0.2);
    }

    /// `decode` is what marching a sampled shape calls, millions of points
    /// at a time, so the device has to answer it too and in more than one
    /// chunk.
    #[test]
    fn a_chunked_decode_matches_the_host_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let mut rng = StdRng::seed_from_u64(22);
        let mut host = ShapeVae::new(tiny(), &mut rng).unwrap();
        for param in host.params_mut() {
            for value in &mut param.value.data {
                *value += rng.gen_range(-0.3..0.3);
            }
        }
        let mut device = host.clone();
        device.to_cuda_with_precision(0, 0, false).unwrap();

        let latent = rows(4, 3, 4);
        let queries = rows(29, 3, 5);
        let expected = host.decode(&latent, &queries, 8).unwrap();
        let got = device.decode(&latent, &queries, 8).unwrap();
        assert_close("distances", &got, &expected, 1e-4);

        let expected = host.decode_colors(&latent, &queries, 8).unwrap();
        let got = device.decode_colors(&latent, &queries, 8).unwrap();
        assert_close("colours", got.as_flattened(), expected.as_flattened(), 1e-4);
    }

    fn step_parity(mixed_precision: bool, tolerance: f32) {
        if !cuda_or_skip() {
            return;
        }

        let mut rng = StdRng::seed_from_u64(21);
        let mut host = ShapeVae::new(tiny(), &mut rng).unwrap();
        // A fresh model's norm scales are all ones and its bias is zero, so the
        // weights are shaken awake first or half of this would pass on a
        // degenerate model.
        for param in host.params_mut() {
            for value in &mut param.value.data {
                *value += rng.gen_range(-0.3..0.3);
            }
        }
        let mut device = host.clone();

        let surface = rows(32, 6, 1);
        let queries = rows(12, 3, 2);
        let grad_distances: Vec<f32> = (0..queries.rows).map(|i| (i as f32 % 5.0) / 7.0).collect();
        let grad_colors = rows(queries.rows, 3, 3);

        let step = |model: &mut ShapeVae| {
            let (mean, log_variance, encoder) = model.encode_train(&surface).unwrap();
            let (distances, decoder) = model.decode_train(&mean, &queries).unwrap();
            let colors = model.colors(&decoder).unwrap();
            let grad_latent = model
                .decode_backward_colored(&decoder, &grad_distances, Some(&grad_colors))
                .unwrap();
            let grad_log_variance = Matrix::new(log_variance.rows, log_variance.cols);
            model
                .encode_backward(&encoder, &grad_latent, &grad_log_variance)
                .unwrap();
            (mean, log_variance, distances, colors)
        };

        let (host_mean, host_log_variance, host_distances, host_colors) = step(&mut host);
        device
            .to_cuda_with_precision(0, 0, mixed_precision)
            .unwrap();
        let (mean, log_variance, distances, colors) = step(&mut device);

        assert_close("mean", &mean.data, &host_mean.data, tolerance);
        assert_close(
            "log_variance",
            &log_variance.data,
            &host_log_variance.data,
            tolerance,
        );
        assert_close("distances", &distances, &host_distances, tolerance);
        assert_close("colors", &colors.data, &host_colors.data, tolerance);

        device.to_cpu().unwrap();
        let expected: Vec<Vec<f32>> = host
            .params_mut()
            .iter()
            .map(|param| param.grad.data.clone())
            .collect();
        for (index, (param, host)) in device.params_mut().iter().zip(&expected).enumerate() {
            assert!(
                host.iter().any(|value| value.abs() > 1e-6),
                "parameter {index} has no gradient to compare"
            );
            assert_close(
                &format!("grad of parameter {index}"),
                &param.grad.data,
                host,
                tolerance,
            );
        }
    }

    /// Packing shapes must only change launch geometry, not sequence
    /// boundaries or the gradients accumulated for shared parameters.
    #[test]
    fn a_packed_batch_matches_sequential_device_steps_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }

        let mut rng = StdRng::seed_from_u64(31);
        let base = ShapeVae::new(tiny(), &mut rng).unwrap();
        let mut sequential = base.clone();
        let mut packed = base;
        sequential.to_cuda_with_precision(0, 0, false).unwrap();
        packed.to_cuda_with_precision(0, 0, false).unwrap();

        let shapes = 2;
        let points = 20;
        let query_rows = 12;
        let surface = rows(shapes * points, 6, 32);
        let queries = rows(shapes * query_rows, 3, 33);
        let upstream: Vec<f32> = (0..shapes * query_rows)
            .map(|i| ((i * 7) % 11) as f32 / 13.0)
            .collect();

        let mut sequential_mean = Vec::new();
        let mut sequential_distances = Vec::new();
        for shape in 0..shapes {
            let take = |matrix: &Matrix, rows: usize| {
                Matrix::from_vec(
                    rows,
                    matrix.cols,
                    matrix.data[shape * rows * matrix.cols..(shape + 1) * rows * matrix.cols]
                        .to_vec(),
                )
            };
            let (mean, _, encoder) = sequential.encode_train(&take(&surface, points)).unwrap();
            let (distance, decoder) = sequential
                .decode_train(&mean, &take(&queries, query_rows))
                .unwrap();
            let grad_latent = sequential
                .decode_backward(
                    &decoder,
                    &upstream[shape * query_rows..(shape + 1) * query_rows],
                )
                .unwrap();
            let zero = Matrix::new(mean.rows, mean.cols);
            sequential
                .encode_backward(&encoder, &grad_latent, &zero)
                .unwrap();
            sequential_mean.extend(mean.data);
            sequential_distances.extend(distance);
        }

        let (mean, _, encoder) = packed.encode_train_batch(&surface, shapes).unwrap();
        let (distances, decoder) = packed.decode_train_batch(&mean, &queries, shapes).unwrap();
        let grad_latent = packed.decode_backward(&decoder, &upstream).unwrap();
        let zero = Matrix::new(mean.rows, mean.cols);
        packed
            .encode_backward(&encoder, &grad_latent, &zero)
            .unwrap();

        assert_close("packed means", &mean.data, &sequential_mean, 1e-4);
        assert_close("packed distances", &distances, &sequential_distances, 1e-4);

        sequential.to_cpu().unwrap();
        packed.to_cpu().unwrap();
        let expected: Vec<Vec<f32>> = sequential
            .params_mut()
            .iter()
            .map(|param| param.grad.data.clone())
            .collect();
        for (index, (param, expected)) in packed.params_mut().iter().zip(&expected).enumerate() {
            assert_close(
                &format!("packed gradient {index}"),
                &param.grad.data,
                expected,
                2e-4,
            );
        }
    }

    /// The gate stage one is trained under: the same tiny autoencoder reaches
    /// the same loss curve on the device as on the host, and both fall.
    ///
    /// One step matching is not a run matching — an error in how a gradient
    /// reaches the optimizer only shows once the weights start moving.
    #[test]
    fn a_training_run_tracks_the_host_curve_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }

        let mut rng = StdRng::seed_from_u64(22);
        let mut host = ShapeVae::new(tiny(), &mut rng).unwrap();
        let mut device = host.clone();
        device.to_cuda_with_precision(0, 0, false).unwrap();

        let surface = rows(32, 6, 5);
        let queries = rows(24, 3, 6);
        // A field the model can actually fit: the distance to the origin, less
        // a radius, which is a sphere.
        let targets: Vec<f32> = queries
            .data
            .chunks_exact(3)
            .map(|point| point.iter().map(|value| value * value).sum::<f32>().sqrt() - 0.3)
            .collect();

        let run = |model: &mut ShapeVae| {
            let optimizer = Optimizer::adam(3e-3);
            let mut losses = Vec::new();
            for step in 1..=40 {
                let (mean, _, encoder) = model.encode_train(&surface).unwrap();
                let (distances, decoder) = model.decode_train(&mean, &queries).unwrap();
                let (loss, grad) = losses::clamped_l1(&distances, &targets, 1.0);
                losses.push(loss);
                let grad_latent = model.decode_backward(&decoder, &grad).unwrap();
                let zero = Matrix::new(mean.rows, mean.cols);
                model
                    .encode_backward(&encoder, &grad_latent, &zero)
                    .unwrap();
                optimizers::step_clipped(&mut model.params_mut(), &optimizer, step, 1.0, 1.0)
                    .unwrap();
                optimizers::zero_grad(&mut model.params_mut());
            }
            losses
        };
        let host_losses = run(&mut host);
        let device_losses = run(&mut device);

        for (step, (device, host)) in device_losses.iter().zip(&host_losses).enumerate() {
            assert!(
                (device - host).abs() <= 1e-3 * host.max(1.0),
                "step {step}: {device} on the device vs {host} on the host"
            );
        }
        assert!(
            device_losses[39] < device_losses[0] * 0.5,
            "the device run did not learn: {} to {}",
            device_losses[0],
            device_losses[39]
        );

        device.to_cpu().unwrap();
        for (index, (param, expected)) in device
            .params_mut()
            .iter()
            .zip(host.params_mut())
            .enumerate()
        {
            assert_close(
                &format!("parameter {index} after 40 steps"),
                &param.value.data,
                &expected.value.data,
                1e-3,
            );
        }
    }
}
