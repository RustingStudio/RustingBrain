//! The device-resident training path: a whole decoder layer without a round trip.
//!
//! [`gpu_transformer`](crate::gpu_transformer) accelerates the matmul-bound
//! pieces of the *host* modules, which means every sublayer uploads its input
//! and downloads its output, because the host operation that follows needs the
//! values. This module is the other half: activations live in device buffers
//! from the embedding gather to the logits, and everything between two GEMMs -
//! RMSNorm, RoPE, the causal softmax, SwiGLU, and the MoE router's top-k and
//! gating - is a kernel from [`crate::cuda_training`]'s module.
//!
//! Host traffic per training step is therefore:
//!
//! * token ids up, logits down, `dL/dlogits` up. The loss stays on the host.
//! * the RMSNorm scales up and their gradients down, once per layer per step.
//!   Those weights stay host-resident so that cached decode keeps working.
//! * one `[rows * top_k]` routing table down per MoE layer, and the token lists
//!   back up. cuBLAS takes its GEMM dimensions on the host, so the number of
//!   tokens routed to each expert has to reach the host before the expert GEMMs
//!   can be shaped. This is one small copy per layer, not one per expert.
//!
//! Batching is what makes this pay. Every buffer is `[batch * seq_len, width]`,
//! the per-head attention GEMMs are strided-batched over the sequences, and an
//! expert sees the tokens routed to it from the *whole* batch in one GEMM.

use crate::batch::TokenBatch;
use crate::cuda_training::{cfg, cuda_alloc_err, cuda_err};
use crate::ffn::SwiGlu;
use crate::gpu_transformer::{
    DeviceParam, GpuContext, act_lhs_transposed, act_plain, act_rhs_transposed,
    gemm_lhs_transposed, gemm_lhs_transposed_batched, gemm_plain, gemm_plain_batched,
    gemm_rhs_transposed, gemm_rhs_transposed_batched,
};
use crate::matrix::Matrix;
use crate::moe::MoeLayer;
use crate::network::NetworkError;
use crate::param::{Linear, Lora, Param};
use crate::rope::Rope;
use crate::transformer::TransformerLm;
use crate::transformer_block::{FeedForward, TransformerBlock};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaView, CudaViewMut, DevicePtr, DevicePtrMut,
    PushKernelArg,
};
use half::bf16;
use std::sync::Arc;

/// Every fused kernel, resolved once per device.
pub(crate) struct ModelKernels {
    rmsnorm_forward: CudaFunction,
    rmsnorm_backward: CudaFunction,
    rope: CudaFunction,
    softmax_backward: CudaFunction,
    probs_from_lse: CudaFunction,
    softmax_lse_rows: CudaFunction,
    swiglu_forward: CudaFunction,
    swiglu_backward: CudaFunction,
    adaln_modulate: CudaFunction,
    adaln_modulate_backward: CudaFunction,
    gate_residual: CudaFunction,
    gate_residual_backward: CudaFunction,
    softmax_lse: CudaFunction,
    topk_gate: CudaFunction,
    gather: CudaFunction,
    gather_scaled: CudaFunction,
    scatter_scaled: CudaFunction,
    scatter: CudaFunction,
    scatter_negated: CudaFunction,
    row_dot: CudaFunction,
    stats: CudaFunction,
    grad_probabilities: CudaFunction,
    aux_grad: CudaFunction,
    grad_logits: CudaFunction,
    add: CudaFunction,
    cross_entropy: CudaFunction,
    cross_entropy_bf16: CudaFunction,
    to_bf16: CudaFunction,
    from_bf16: CudaFunction,
    accumulate_bf16: CudaFunction,
    cast_act: CudaFunction,
}

impl ModelKernels {
    pub(crate) fn load(module: &Arc<CudaModule>) -> Result<Self, NetworkError> {
        let get = |name: &str| {
            module
                .load_function(name)
                .map_err(cuda_err("CUDA kernel lookup"))
        };
        Ok(Self {
            rmsnorm_forward: get("rmsnorm_fwd")?,
            rmsnorm_backward: get("rmsnorm_bwd")?,
            rope: get("rope_rotate")?,
            softmax_backward: get("attention_softmax_bwd")?,
            probs_from_lse: get("attention_probs_from_lse")?,
            softmax_lse_rows: get("attention_softmax_lse")?,
            swiglu_forward: get("swiglu_fwd")?,
            swiglu_backward: get("swiglu_bwd")?,
            adaln_modulate: get("adaln_modulate")?,
            adaln_modulate_backward: get("adaln_modulate_bwd")?,
            gate_residual: get("gate_residual_fwd")?,
            gate_residual_backward: get("gate_residual_bwd")?,
            softmax_lse: get("softmax_lse")?,
            topk_gate: get("topk_gate")?,
            gather: get("gather_rows")?,
            gather_scaled: get("gather_scale_rows")?,
            scatter_scaled: get("scatter_add_scaled")?,
            scatter: get("scatter_add_rows")?,
            scatter_negated: get("scatter_rows_neg")?,
            row_dot: get("row_dot_scatter")?,
            stats: get("moe_stats")?,
            grad_probabilities: get("moe_grad_probs")?,
            aux_grad: get("moe_aux_grad")?,
            grad_logits: get("router_grad_logits")?,
            add: get("add_inplace")?,
            cross_entropy: get("ce_loss_grad")?,
            cross_entropy_bf16: get("ce_loss_grad_bf16")?,
            to_bf16: get("to_bf16")?,
            from_bf16: get("from_bf16")?,
            accumulate_bf16: get("accumulate_bf16")?,
            cast_act: get("cast_act")?,
        })
    }
}

/// Everything [`Gpu::flash_attention`] needs to locate one block's queries,
/// keys and values inside the fused projection output.
#[derive(Clone, Copy)]
pub(crate) struct FlashShape {
    pub(crate) rows: usize,
    pub(crate) seq_len: usize,
    pub(crate) sequences: usize,
    pub(crate) heads: usize,
    pub(crate) group: usize,
    pub(crate) qkv_width: usize,
    pub(crate) query_width: usize,
    pub(crate) key_base: usize,
    pub(crate) value_base: usize,
    pub(crate) causal: bool,
}

/// RoPE's `cos` and `sin` tables on the device.
///
/// Every layer is built from the same [`Rope`] configuration, so one upload
/// serves the whole model and is kept for the life of the context.
pub(crate) struct DeviceRope {
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    head_dim: usize,
    max_seq_len: usize,
}

/// What the backward pass over a stack of blocks needs from its forward pass,
/// all of it device-resident.
pub(crate) struct GpuStack {
    rows: usize,
    seq_len: usize,
    sequences: usize,
    valid_tokens: usize,
    blocks: Vec<BlockCache>,
    auxiliary_loss: f32,
}

impl std::fmt::Debug for GpuStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuStack")
            .field("rows", &self.rows)
            .field("seq_len", &self.seq_len)
            .finish()
    }
}

/// What [`backward`] needs from [`forward`]: the block stack's cache, plus the
/// token ids and the final norm around it.
pub(crate) struct GpuCache {
    stack: GpuStack,
    ids: CudaSlice<u32>,
    final_input: CudaSlice<f32>,
    final_inverse_rms: CudaSlice<f32>,
    final_weight: CudaSlice<f32>,
    final_output: Act,
}

impl std::fmt::Debug for GpuCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.stack.fmt(f)
    }
}

impl GpuCache {
    pub(crate) fn auxiliary_loss(&self) -> f32 {
        self.stack.auxiliary_loss
    }
}

/// A GEMM operand, held as untyped device bytes.
///
/// Every activation in a transformer block is read by exactly one thing: a
/// GEMM that, with reduced precision on, would have rounded it to BF16 inside
/// the tensor cores anyway. Storing it BF16 in the first place halves the
/// traffic and, more importantly, changes which cuBLAS kernel runs: BF16
/// operands land on `s16816 ... align8`, twice the `k` per tensor-core
/// instruction of the `s1688 ... align4` kernel FP32 operands get.
///
/// Casting the operands per call does not pay - measured at 0.65x to 0.91x by
/// `examples/gemm_probe.rs` - so the kernel that *produces* an activation has
/// to write it narrow. That is what the `narrow` flag threaded through
/// `rmsnorm_fwd`, `swiglu_fwd`, `swiglu_bwd` and `flash_attention_fwd` is for.
///
/// One byte buffer serves both precisions because a kernel launch only ever
/// passes a device pointer, and cuBLAS takes the element type as a runtime
/// argument. Buffers that a plain FP32 kernel still writes - the routed
/// feed-forward's router input - are held wide and read back through
/// [`Act::wide`].
pub(crate) struct Act {
    bytes: CudaSlice<u8>,
    narrow: bool,
    len: usize,
}

impl Act {
    pub(crate) fn element(narrow: bool) -> usize {
        if narrow { 2 } else { 4 }
    }

    pub(crate) fn is_narrow(&self) -> bool {
        self.narrow
    }

    pub(crate) fn flag(&self) -> i32 {
        i32::from(self.narrow)
    }

    /// The operand starting at element `offset`, which is how a routed layer
    /// hands cuBLAS one expert's slice of a grouped buffer.
    pub(crate) fn at(&self, offset: usize) -> CudaView<'_, u8> {
        self.bytes.slice(offset * Act::element(self.narrow)..)
    }

    pub(crate) fn all(&self) -> CudaView<'_, u8> {
        self.bytes.slice(..)
    }

    /// The same bytes as FP32, for the kernels that never learned to narrow.
    ///
    /// Only sound on a wide buffer, which is why every caller is on the routed
    /// path, where [`Gpu::act`] is asked for `narrow = false`.
    pub(crate) fn wide(&self) -> CudaView<'_, f32> {
        debug_assert!(!self.narrow, "a narrow activation is not FP32");
        unsafe { self.bytes.transmute::<f32>(self.len) }
            .expect("a device allocation is aligned for FP32")
    }

    pub(crate) fn wide_mut(&mut self) -> CudaViewMut<'_, f32> {
        debug_assert!(!self.narrow, "a narrow activation is not FP32");
        unsafe { self.bytes.transmute_mut::<f32>(self.len) }
            .expect("a device allocation is aligned for FP32")
    }

    /// The destination argument of the kernel that writes this buffer.
    pub(crate) fn destination(&mut self) -> &mut CudaSlice<u8> {
        &mut self.bytes
    }
}

struct BlockCache {
    input: CudaSlice<f32>,
    attention_weight: CudaSlice<f32>,
    attention_inverse_rms: CudaSlice<f32>,
    attention_normed: Act,
    /// Query, key and value in one buffer, three slices of every row, because
    /// they are produced by one GEMM against the three weights packed together.
    /// Narrow whenever the fused attention kernels run, because they are the
    /// only readers and they rounded it to BF16 on every tile load anyway.
    qkv: Act,
    /// Those three weights packed, which the input gradient reads again in the
    /// backward pass.
    qkv_weights: Act,
    /// The per-query log-sum-exp of the attention scores, which is all the
    /// backward pass needs of a probability matrix that never existed.
    log_sum_exp: CudaSlice<f32>,
    merged: Act,
    /// The attention output projection, narrowed to match `merged`.
    output_weight: Act,
    residual: CudaSlice<f32>,
    feed_forward_weight: CudaSlice<f32>,
    feed_forward_inverse_rms: CudaSlice<f32>,
    feed_forward_normed: Act,
    feed_forward: FfnCache,
}

/// The three buffers a SwiGLU backward pass cannot recover from its output.
pub(crate) struct SwiGluCache {
    /// Gate and up in one buffer, two halves of every row, because they are
    /// produced by one GEMM at twice the width.
    pub(crate) gate_up: CudaSlice<f32>,
    pub(crate) hidden: Act,
    /// The gate and up weights concatenated, which the input gradient reads
    /// again in the backward pass.
    pub(crate) weights: Act,
    /// The down projection, narrowed to match `hidden`. A routed layer has one
    /// per expert and keeps them in the parameters instead, so this is `None`
    /// there.
    pub(crate) down: Option<Act>,
}

/// The normed activations, the inverse RMS of each row that the backward pass
/// needs, and the optional FP32 duplicate of the input.
type NormOutput = (Act, CudaSlice<f32>, Option<CudaSlice<f32>>);

/// The dense variant stays unboxed despite being the larger of the two: a
/// cache is built for every layer of every step, and 272 bytes of stack beats
/// an allocation on that path. The routed variant is boxed because it is
/// larger again by an order of magnitude.
#[allow(clippy::large_enum_variant)]
enum FfnCache {
    Dense(SwiGluCache),
    Moe(Box<MoeGpuCache>),
}

struct MoeGpuCache {
    probabilities: CudaSlice<f32>,
    log_sum_exp: CudaSlice<f32>,
    expert_of: CudaSlice<i32>,
    /// One entry per routed token slot, grouped by expert: the token it came
    /// from, the flat assignment slot, and the gate it was scaled by.
    token_of: CudaSlice<u32>,
    slot_of: CudaSlice<u32>,
    gates: CudaSlice<f32>,
    /// Where each expert's group starts in those buffers, and how long it is.
    offsets: Vec<usize>,
    counts: Vec<usize>,
    routed: usize,
    input: Act,
    expert: SwiGluCache,
    output: CudaSlice<f32>,
    shared: Option<SwiGluCache>,
}

/// A borrowed device, plus the kernel launches, so the model code below reads
/// as a sequence of operations rather than a sequence of `unsafe` blocks.
pub(crate) struct Gpu<'a> {
    pub(crate) context: &'a Arc<GpuContext>,
}

impl Gpu<'_> {
    pub(crate) fn zeros(&self, len: usize) -> Result<CudaSlice<f32>, NetworkError> {
        self.context
            .stream
            .alloc_zeros::<f32>(len.max(1))
            .map_err(cuda_alloc_err("device allocation", len.max(1) * 4))
    }

    /// An allocation whose contents are whatever the driver last left there.
    ///
    /// Only for a buffer the very next operation overwrites completely: a GEMM
    /// with `beta = 0`, or a kernel that writes every element. The zeroing
    /// `zeros` does is a full-bandwidth pass over the buffer, which on the
    /// chunked LM head alone cost more than every memcpy in a step combined.
    pub(crate) fn uninit(&self, len: usize) -> Result<CudaSlice<f32>, NetworkError> {
        unsafe { self.context.stream.alloc::<f32>(len.max(1)) }
            .map_err(cuda_alloc_err("device allocation", len.max(1) * 4))
    }

    /// [`Gpu::uninit`] for a BF16 buffer.
    pub(crate) fn uninit_bf16(&self, len: usize) -> Result<CudaSlice<bf16>, NetworkError> {
        unsafe { self.context.stream.alloc::<bf16>(len.max(1)) }
            .map_err(cuda_alloc_err("device allocation", len.max(1) * 2))
    }

    /// [`Gpu::uninit`] for a GEMM operand of `len` elements, which the very
    /// next kernel writes in full.
    pub(crate) fn act(&self, len: usize, narrow: bool) -> Result<Act, NetworkError> {
        let bytes = unsafe {
            self.context
                .stream
                .alloc::<u8>(len.max(1) * Act::element(narrow))
        }
        .map_err(cuda_alloc_err(
            "device allocation",
            len.max(1) * Act::element(narrow),
        ))?;
        Ok(Act { bytes, narrow, len })
    }

    /// `dst[..n] = src[..n]`, narrowed on the way if `narrow`.
    ///
    /// For a buffer a GEMM or an FP32-only kernel produced and a GEMM has to
    /// read: the weights, and the gradients the RoPE backward pass rotates in
    /// place before anything multiplies by them.
    pub(crate) fn cast_into(
        &self,
        dst: &mut CudaViewMut<'_, u8>,
        src: &CudaView<'_, f32>,
        n: usize,
        narrow: bool,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.cast_act)
                .arg(dst)
                .arg(src)
                .arg(&(n as i32))
                .arg(&i32::from(narrow))
                .launch(cfg(n))
                .map_err(cuda_err("activation cast kernel"))?;
        }
        Ok(())
    }

    /// [`Gpu::cast_into`] into a fresh operand.
    pub(crate) fn narrowed(
        &self,
        src: &CudaView<'_, f32>,
        len: usize,
        narrow: bool,
    ) -> Result<Act, NetworkError> {
        let mut act = self.act(len, narrow)?;
        let mut dst = act.bytes.slice_mut(..);
        self.cast_into(&mut dst, src, len, narrow)?;
        Ok(act)
    }

    /// `dst[..n] = bf16(src[..n])`, rounded to nearest even.
    pub(crate) fn cast_to_bf16(
        &self,
        dst: &mut CudaSlice<bf16>,
        src: &CudaView<'_, f32>,
        n: usize,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.to_bf16)
                .arg(dst)
                .arg(src)
                .arg(&(n as i32))
                .launch(cfg(n))
                .map_err(cuda_err("BF16 cast kernel"))?;
        }
        Ok(())
    }

    /// `dst[..n] = f32(src[..n])`, which is exact.
    pub(crate) fn cast_from_bf16(
        &self,
        dst: &mut CudaViewMut<'_, f32>,
        src: &CudaSlice<bf16>,
        n: usize,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.from_bf16)
                .arg(dst)
                .arg(src)
                .arg(&(n as i32))
                .launch(cfg(n))
                .map_err(cuda_err("BF16 widening kernel"))?;
        }
        Ok(())
    }

    /// `acc[..n] += f32(src[..n])`, the FP32 half of a BF16 gradient GEMM.
    pub(crate) fn accumulate_bf16(
        &self,
        acc: &mut CudaSlice<f32>,
        src: &CudaSlice<bf16>,
        n: usize,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.accumulate_bf16)
                .arg(acc)
                .arg(src)
                .arg(&(n as i32))
                .launch(cfg(n))
                .map_err(cuda_err("BF16 accumulate kernel"))?;
        }
        Ok(())
    }

    pub(crate) fn upload(&self, data: &[f32]) -> Result<CudaSlice<f32>, NetworkError> {
        self.context.stream.clone_htod(data).map_err(cuda_alloc_err(
            "host to device copy",
            std::mem::size_of_val(data),
        ))
    }

    pub(crate) fn upload_indices(&self, data: &[u32]) -> Result<CudaSlice<u32>, NetworkError> {
        self.context.stream.clone_htod(data).map_err(cuda_alloc_err(
            "host to device copy",
            std::mem::size_of_val(data),
        ))
    }

    pub(crate) fn upload_flags(&self, data: &[i32]) -> Result<CudaSlice<i32>, NetworkError> {
        self.context.stream.clone_htod(data).map_err(cuda_alloc_err(
            "host to device copy",
            std::mem::size_of_val(data),
        ))
    }

    pub(crate) fn download(&self, source: &CudaSlice<f32>) -> Result<Vec<f32>, NetworkError> {
        self.context
            .stream
            .clone_dtoh(source)
            .map_err(cuda_err("device to host copy"))
    }

    /// [`Gpu::download`] for a wide activation, which is what the kernel tests
    /// compare against the host path.
    #[cfg(test)]
    pub(crate) fn download_act(&self, source: &Act) -> Result<Vec<f32>, NetworkError> {
        if !source.is_narrow() {
            return self
                .context
                .stream
                .clone_dtoh(&source.wide())
                .map_err(cuda_err("device to host copy"));
        }
        let bytes = self
            .context
            .stream
            .clone_dtoh(&source.all())
            .map_err(cuda_err("device to host copy"))?;
        Ok(bytes
            .chunks_exact(2)
            .map(|half| f32::from_bits(u32::from(u16::from_le_bytes([half[0], half[1]])) << 16))
            .collect())
    }

    pub(crate) fn upload_signed(&self, data: &[i32]) -> Result<CudaSlice<i32>, NetworkError> {
        self.context.stream.clone_htod(data).map_err(cuda_alloc_err(
            "host to device copy",
            std::mem::size_of_val(data),
        ))
    }

    pub(crate) fn download_signed(
        &self,
        source: &CudaSlice<i32>,
    ) -> Result<Vec<i32>, NetworkError> {
        self.context
            .stream
            .clone_dtoh(source)
            .map_err(cuda_err("device to host copy"))
    }

    /// `[rows, width]` back into a host matrix.
    pub(crate) fn matrix(
        &self,
        source: &CudaSlice<f32>,
        rows: usize,
        width: usize,
    ) -> Result<Matrix, NetworkError> {
        let mut matrix = Matrix::new(rows, width);
        self.context
            .stream
            .memcpy_dtoh(source, &mut matrix.data)
            .map_err(cuda_err("device to host copy"))?;
        Ok(matrix)
    }

    /// Adds one slice of a downloaded scale-gradient block into a host
    /// parameter's gradient.
    ///
    /// The RMSNorm scales are the only weights the device path does not own,
    /// so this is the one place a gradient crosses back in the plain
    /// (unnegated) convention. The whole backward pass writes its scale
    /// gradients into a single device buffer and downloads it once, because a
    /// download in the middle of the pass drains the stream: thirteen of them
    /// left the device idle for about 20 ms of every step.
    pub(crate) fn accumulate_host_grad(param: &mut Param, values: &[f32]) {
        for (slot, value) in param.grad.data.iter_mut().zip(values) {
            *slot += value;
        }
    }

    pub(crate) fn rope_tables(&self, rope: &Rope) -> Result<Arc<DeviceRope>, NetworkError> {
        let mut slot = self
            .context
            .rope
            .lock()
            .map_err(|_| NetworkError::Cuda("the rope table lock was poisoned".into()))?;
        if let Some(tables) = slot.as_ref()
            && tables.head_dim == rope.head_dim()
            && tables.max_seq_len == rope.max_seq_len()
        {
            return Ok(tables.clone());
        }
        let tables = Arc::new(DeviceRope {
            cos: self.upload(rope.cos())?,
            sin: self.upload(rope.sin())?,
            head_dim: rope.head_dim(),
            max_seq_len: rope.max_seq_len(),
        });
        *slot = Some(tables.clone());
        Ok(tables)
    }

    /// `out[rows, weight.rows] = x . weight^T`.
    pub(crate) fn linear<X: DevicePtr<f32>>(
        &self,
        weight: &DeviceParam,
        x: &X,
        rows: usize,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let mut out = self.uninit(rows * weight.rows())?;
        gemm_rhs_transposed(
            self.context,
            x,
            weight.cols(),
            weight.value(),
            weight.cols(),
            &mut out,
            weight.rows(),
            rows,
            weight.rows(),
            weight.cols(),
            1.0,
            0.0,
        )?;
        Ok(out)
    }

    /// `out = grad_output . weight + beta * out`, the input-gradient half of a
    /// linear layer.
    pub(crate) fn linear_backward_input<G: DevicePtr<f32>, O: DevicePtrMut<f32>>(
        &self,
        weight: &DeviceParam,
        grad_output: &G,
        out: &mut O,
        rows: usize,
        beta: f32,
    ) -> Result<(), NetworkError> {
        gemm_plain(
            self.context,
            grad_output,
            weight.rows(),
            weight.value(),
            weight.cols(),
            out,
            weight.cols(),
            rows,
            weight.cols(),
            weight.rows(),
            1.0,
            beta,
        )
    }

    /// `negated_grad -= grad_output^T . input`, which is the device gradient
    /// convention the shared optimizer kernels expect.
    pub(crate) fn accumulate_weight_grad<G: DevicePtr<f32>, X: DevicePtr<f32>>(
        &self,
        weight: &mut DeviceParam,
        grad_output: &G,
        input: &X,
        rows: usize,
    ) -> Result<(), NetworkError> {
        if weight.is_frozen() {
            return Ok(());
        }
        let (units, cols) = (weight.rows(), weight.cols());
        let beta = weight.grad_beta();
        gemm_lhs_transposed(
            self.context,
            grad_output,
            units,
            input,
            cols,
            weight.negated_grad_mut(),
            cols,
            rows,
            units,
            cols,
            -1.0,
            beta,
        )
    }

    /// `copy` asks for an FP32 duplicate of `input` alongside the normed
    /// output. The residual branch below a norm wants one, because its
    /// projection accumulates over the block input with `beta = 1`; this kernel
    /// has the row in registers already, so it is a store rather than a pass.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rmsnorm(
        &self,
        input: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        eps: f32,
        narrow: bool,
        copy: bool,
    ) -> Result<NormOutput, NetworkError> {
        let mut out = self.act(rows * cols, narrow)?;
        let mut inverse = self.uninit(rows)?;
        // A one-element stand-in keeps the launch arguments uniform when no
        // duplicate is asked for; `do_copy` is what decides whether it is written.
        let mut duplicate = self.uninit(if copy { rows * cols } else { 1 })?;
        let do_copy = i32::from(copy);
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.rmsnorm_forward)
                .arg(out.destination())
                .arg(&mut inverse)
                .arg(input)
                .arg(weight)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&eps)
                .arg(&i32::from(narrow))
                .arg(&mut duplicate)
                .arg(&do_copy)
                .launch(row_grid(rows))
                .map_err(cuda_err("RMSNorm kernel"))?;
        }
        Ok((out, inverse, copy.then_some(duplicate)))
    }

    /// Returns `dL/dinput` and accumulates the scale gradient on the device.
    #[allow(clippy::too_many_arguments)]
    /// `residual`, when `add_residual` is set, is added into the input
    /// gradient: a residual branch's upstream gradient passes through here
    /// instead of costing a separate pass over the whole activation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rmsnorm_backward(
        &self,
        input: &CudaSlice<f32>,
        grad_output: &CudaSlice<f32>,
        weight: &CudaSlice<f32>,
        inverse: &CudaSlice<f32>,
        grad_weight: &mut CudaViewMut<'_, f32>,
        residual: &CudaSlice<f32>,
        add_residual: bool,
        copy: Option<bool>,
        rows: usize,
        cols: usize,
    ) -> Result<(CudaSlice<f32>, Option<Act>), NetworkError> {
        let mut grad_input = self.uninit(rows * cols)?;
        let add_residual = i32::from(add_residual);
        let copy_mode = copy.map_or(0, |narrow| 1 + i32::from(narrow));
        // A one-byte stand-in keeps the launch arguments uniform when no copy
        // is asked for; `copy_mode` is what decides whether it is written.
        let mut copy = self.act(copy.map_or(1, |_| rows * cols), copy.unwrap_or(false))?;
        let smem = rmsnorm_smem(cols);
        let mut config = row_grid(rows);
        config.shared_mem_bytes = smem.unwrap_or(0);
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.rmsnorm_backward)
                .arg(&mut grad_input)
                .arg(grad_weight)
                .arg(input)
                .arg(grad_output)
                .arg(weight)
                .arg(inverse)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(i32::from(smem.is_some())))
                .arg(residual)
                .arg(&add_residual)
                .arg(&mut copy.bytes)
                .arg(&copy_mode)
                .launch(config)
                .map_err(cuda_err("RMSNorm backward kernel"))?;
        }
        Ok((grad_input, (copy_mode != 0).then_some(copy)))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope(
        &self,
        tensor: &mut Act,
        tables: &DeviceRope,
        rows: usize,
        heads: usize,
        head_dim: usize,
        seq_len: usize,
        direction: f32,
        width: usize,
        offset: usize,
    ) -> Result<(), NetworkError> {
        let elements = rows * heads * head_dim / 2;
        let narrow = tensor.flag();
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.rope)
                .arg(tensor.destination())
                .arg(&tables.cos)
                .arg(&tables.sin)
                .arg(&(rows as i32))
                .arg(&(heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(seq_len as i32))
                .arg(&direction)
                .arg(&(width as i32))
                .arg(&(offset as i32))
                .arg(&narrow)
                .launch(cfg(elements))
                .map_err(cuda_err("RoPE kernel"))?;
        }
        Ok(())
    }

    pub(crate) fn softmax_backward(
        &self,
        grad: &mut CudaSlice<f32>,
        probabilities: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        causal: bool,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.softmax_backward)
                .arg(grad)
                .arg(probabilities)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&i32::from(causal))
                .launch(warp_per_row_grid(rows))
                .map_err(cuda_err("attention softmax backward kernel"))?;
        }
        Ok(())
    }

    /// Causal softmax over the scores in place, writing the per-query
    /// log-sum-exp the backward pass keeps instead of the probabilities.
    pub(crate) fn softmax_lse(
        &self,
        scores: &mut CudaSlice<f32>,
        lse: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
        causal: bool,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.softmax_lse_rows)
                .arg(scores)
                .arg(lse)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&i32::from(causal))
                .launch(warp_per_row_grid(rows))
                .map_err(cuda_err("attention softmax kernel"))?;
        }
        Ok(())
    }

    /// Attention for one block in a single kernel: scores, causal softmax and
    /// the value matmul, with no `[seq_len, seq_len]` matrix ever written to
    /// global memory.
    ///
    /// Produces exactly the `merged` and `log_sum_exp` the three-kernel path
    /// produced, which is why the backward pass needs no change to go with it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attention(
        &self,
        flash: &crate::cuda_flash::FlashKernels,
        qkv: &Act,
        merged: &mut Act,
        log_sum_exp: &mut CudaSlice<f32>,
        shape: FlashShape,
        scale: f32,
    ) -> Result<(), NetworkError> {
        debug_assert!(qkv.is_narrow() && merged.is_narrow());
        let config = cudarc::driver::LaunchConfig {
            grid_dim: (
                shape.seq_len.div_ceil(crate::cuda_flash::TILE) as u32,
                shape.heads as u32,
                shape.sequences as u32,
            ),
            block_dim: (crate::cuda_flash::THREADS, 1, 1),
            shared_mem_bytes: crate::cuda_flash::SHARED_BYTES,
        };
        unsafe {
            self.context
                .stream
                .launch_builder(&flash.forward)
                .arg(&qkv.all())
                .arg(merged.destination())
                .arg(log_sum_exp)
                .arg(&(shape.seq_len as i32))
                .arg(&(shape.qkv_width as i32))
                .arg(&(shape.query_width as i32))
                .arg(&(shape.key_base as i32))
                .arg(&(shape.value_base as i32))
                .arg(&(shape.group as i32))
                .arg(&(shape.rows as i32))
                .arg(&scale)
                .arg(&i32::from(shape.causal))
                .launch(config)
                .map_err(cuda_err("fused attention kernel"))?;
        }
        Ok(())
    }

    /// The backward half of the same fusion: three kernels in place of two
    /// `[seq_len, seq_len]` buffers, six passes over them and four batched
    /// GEMMs.
    ///
    /// The query gradient is blocked over query tiles, the key and value
    /// gradients over key tiles, because each of those is the blocking under
    /// which a gradient is complete inside one block. The scores are rebuilt
    /// from the log-sum-exp in both, which costs one extra matmul and saves
    /// every byte of the round trip.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attention_backward(
        &self,
        flash: &crate::cuda_flash::FlashKernels,
        qkv: &Act,
        merged: &Act,
        grad_merged: &Act,
        log_sum_exp: &CudaSlice<f32>,
        delta: &mut CudaSlice<f32>,
        grad_qkv: &mut Act,
        shape: FlashShape,
        scale: f32,
    ) -> Result<(), NetworkError> {
        let tiles = shape.seq_len.div_ceil(crate::cuda_flash::TILE) as u32;
        let warps = shape.heads * shape.rows;
        unsafe {
            self.context
                .stream
                .launch_builder(&flash.delta)
                .arg(&merged.all())
                .arg(&grad_merged.all())
                .arg(&mut *delta)
                .arg(&(shape.rows as i32))
                .arg(&(shape.heads as i32))
                .arg(&(shape.query_width as i32))
                .arg(&(shape.rows as i32))
                .launch(warp_per_row_grid(warps))
                .map_err(cuda_err("attention delta kernel"))?;
        }

        let delta = &*delta;
        let grad_merged_view = grad_merged.all();
        // Both kernels write disjoint columns of `grad_qkv`, so they take it
        // as a shared view rather than one after the other.
        debug_assert!(qkv.is_narrow() && grad_qkv.is_narrow() && grad_merged.is_narrow());
        let qkv = qkv.all();
        let grad_qkv = grad_qkv.all();
        let launch = |function, grid_y, shared_mem_bytes| {
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (tiles, grid_y, shape.sequences as u32),
                block_dim: (crate::cuda_flash::THREADS, 1, 1),
                shared_mem_bytes,
            };
            unsafe {
                self.context
                    .stream
                    .launch_builder(function)
                    .arg(&qkv)
                    .arg(&grad_merged_view)
                    .arg(log_sum_exp)
                    .arg(&*delta)
                    .arg(&grad_qkv)
                    .arg(&(shape.seq_len as i32))
                    .arg(&(shape.qkv_width as i32))
                    .arg(&(shape.query_width as i32))
                    .arg(&(shape.key_base as i32))
                    .arg(&(shape.value_base as i32))
                    .arg(&(shape.group as i32))
                    .arg(&(shape.rows as i32))
                    .arg(&scale)
                    .arg(&i32::from(shape.causal))
                    .launch(config)
                    .map_err(cuda_err("fused attention backward kernel"))
            }
        };
        launch(
            &flash.grad_query,
            shape.heads as u32,
            crate::cuda_flash::DQ_SHARED_BYTES,
        )?;
        launch(
            &flash.grad_key_value,
            (shape.heads / shape.group) as u32,
            crate::cuda_flash::DKV_SHARED_BYTES,
        )?;
        Ok(())
    }

    /// Rebuild the attention probabilities in place from scaled scores and the
    /// stored log-sum-exp, which is what the backward GEMMs expect to read.
    pub(crate) fn probs_from_lse(
        &self,
        scores: &mut CudaSlice<f32>,
        lse: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        causal: bool,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.probs_from_lse)
                .arg(scores)
                .arg(lse)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&i32::from(causal))
                .launch(cfg(rows * cols))
                .map_err(cuda_err("attention probability kernel"))?;
        }
        Ok(())
    }

    /// `target += source[offset..offset + len]`, the offset being what lets one
    /// fused weight-gradient buffer be split back into two parameters.
    /// `keep` adds into `target`; without it `target` is overwritten, which is
    /// what a gradient buffer still holding the last step's value wants.
    pub(crate) fn add(
        &self,
        target: &mut CudaSlice<f32>,
        source: &CudaSlice<f32>,
        offset: usize,
        len: usize,
        keep: bool,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.add)
                .arg(target)
                .arg(source)
                .arg(&(offset as i32))
                .arg(&(len as i32))
                .arg(&i32::from(keep))
                .launch(cfg(if (len | offset) % 4 == 0 {
                    len / 4
                } else {
                    len
                }))
                .map_err(cuda_err("addition kernel"))?;
        }
        Ok(())
    }

    pub(crate) fn gather(
        &self,
        out: &mut CudaViewMut<'_, f32>,
        source: &CudaView<'_, f32>,
        rows_of: &CudaSlice<u32>,
        rows: usize,
        width: usize,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.gather)
                .arg(out)
                .arg(source)
                .arg(rows_of)
                .arg(&0i32)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .launch(cfg(rows * width))
                .map_err(cuda_err("row gather kernel"))?;
        }
        Ok(())
    }

    /// `out[r] = scale[r] * source[rows_of[r]]`, or without the scale, written
    /// at `out`'s precision.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gather_scaled(
        &self,
        out: &mut Act,
        source: &CudaView<'_, f32>,
        rows_of: &CudaSlice<u32>,
        scale: &CudaSlice<f32>,
        scaled: bool,
        rows: usize,
        width: usize,
    ) -> Result<(), NetworkError> {
        let narrow = out.flag();
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.gather_scaled)
                .arg(out.destination())
                .arg(source)
                .arg(rows_of)
                .arg(scale)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .arg(&i32::from(scaled))
                .arg(&narrow)
                .launch(cfg(rows * width))
                .map_err(cuda_err("scaled row gather kernel"))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scatter_scaled(
        &self,
        out: &mut CudaSlice<f32>,
        source: &CudaSlice<f32>,
        rows_of: &CudaSlice<u32>,
        scale: &CudaSlice<f32>,
        rows: usize,
        width: usize,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.scatter_scaled)
                .arg(out)
                .arg(source)
                .arg(rows_of)
                .arg(scale)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .launch(cfg(rows * width))
                .map_err(cuda_err("scaled row scatter kernel"))?;
        }
        Ok(())
    }

    pub(crate) fn scatter(
        &self,
        out: &mut CudaSlice<f32>,
        source: &CudaSlice<f32>,
        rows_of: &CudaSlice<u32>,
        rows: usize,
        width: usize,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.scatter)
                .arg(out)
                .arg(source)
                .arg(rows_of)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .launch(cfg(rows * width))
                .map_err(cuda_err("row scatter kernel"))?;
        }
        Ok(())
    }

    /// Scatters `-upstream` into a device gradient, which is the embedding
    /// table's backward pass under the negated-gradient convention.
    pub(crate) fn scatter_negated(
        &self,
        grad: &mut CudaSlice<f32>,
        upstream: &CudaSlice<f32>,
        rows_of: &CudaSlice<u32>,
        rows: usize,
        width: usize,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.scatter_negated)
                .arg(grad)
                .arg(upstream)
                .arg(rows_of)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .launch(cfg(rows * width))
                .map_err(cuda_err("embedding scatter kernel"))?;
        }
        Ok(())
    }

    pub(crate) fn swiglu(
        &self,
        gate_up: &CudaSlice<f32>,
        rows: usize,
        width: usize,
        narrow: bool,
    ) -> Result<Act, NetworkError> {
        let mut hidden = self.act(rows * width, narrow)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.swiglu_forward)
                .arg(hidden.destination())
                .arg(gate_up)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .arg(&i32::from(narrow))
                .launch(cfg(if width % 4 == 0 {
                    rows * width / 4
                } else {
                    rows * width
                }))
                .map_err(cuda_err("SwiGLU kernel"))?;
        }
        Ok(hidden)
    }

    /// The gate and up gradients, in the same two-halves-per-row layout the
    /// forward activations use, so the weight gradient is one wide GEMM too.
    pub(crate) fn swiglu_backward(
        &self,
        gate_up: &CudaSlice<f32>,
        grad_hidden: &CudaSlice<f32>,
        rows: usize,
        width: usize,
        narrow: bool,
    ) -> Result<Act, NetworkError> {
        let mut grad_gate_up = self.act(rows * 2 * width, narrow)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.swiglu_backward)
                .arg(grad_gate_up.destination())
                .arg(gate_up)
                .arg(grad_hidden)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .arg(&i32::from(narrow))
                .launch(cfg(if width % 4 == 0 {
                    rows * width / 4
                } else {
                    rows * width
                }))
                .map_err(cuda_err("SwiGLU backward kernel"))?;
        }
        Ok(grad_gate_up)
    }

    /// AdaLN's modulation: `normalized * (1 + scale) + shift`.
    ///
    /// `triple` is `[sequences, 3 * cols]` — shift, scale and gate in that
    /// order, one row per sequence, which is where
    /// [`AdaLayerNorm`](crate::adaln::AdaLayerNorm) puts them. `rows` is
    /// `sequences * seq_len`.
    pub(crate) fn adaln_modulate(
        &self,
        normalized: &CudaView<'_, f32>,
        triple: &CudaView<'_, f32>,
        rows: usize,
        cols: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let mut out = self.uninit(rows * cols)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.adaln_modulate)
                .arg(&mut out)
                .arg(normalized)
                .arg(triple)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(seq_len as i32))
                .launch(cfg(rows * cols))
                .map_err(cuda_err("AdaLN modulation kernel"))?;
        }
        Ok(out)
    }

    /// Returns `dL/dnormalized` and accumulates the shift and scale gradients
    /// into `grad_triple`, which the caller zeroed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn adaln_modulate_backward(
        &self,
        grad_modulated: &CudaView<'_, f32>,
        normalized: &CudaView<'_, f32>,
        triple: &CudaView<'_, f32>,
        grad_triple: &mut CudaViewMut<'_, f32>,
        rows: usize,
        cols: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let mut grad_normalized = self.uninit(rows * cols)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.adaln_modulate_backward)
                .arg(&mut grad_normalized)
                .arg(grad_triple)
                .arg(grad_modulated)
                .arg(normalized)
                .arg(triple)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(seq_len as i32))
                .launch(cfg(rows * cols))
                .map_err(cuda_err("AdaLN modulation backward kernel"))?;
        }
        Ok(grad_normalized)
    }

    /// `residual + gate * branch`, with the gate read from `triple`'s third
    /// slot.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gate_residual(
        &self,
        residual: &CudaView<'_, f32>,
        branch: &CudaView<'_, f32>,
        triple: &CudaView<'_, f32>,
        rows: usize,
        cols: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let mut out = self.uninit(rows * cols)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.gate_residual)
                .arg(&mut out)
                .arg(residual)
                .arg(branch)
                .arg(triple)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(seq_len as i32))
                .launch(cfg(rows * cols))
                .map_err(cuda_err("gated residual kernel"))?;
        }
        Ok(out)
    }

    /// Returns `dL/dbranch` and accumulates the gate gradient into
    /// `grad_triple`. `dL/dresidual` is the upstream gradient unchanged.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gate_residual_backward(
        &self,
        branch: &CudaView<'_, f32>,
        triple: &CudaView<'_, f32>,
        grad_output: &CudaView<'_, f32>,
        grad_triple: &mut CudaViewMut<'_, f32>,
        rows: usize,
        cols: usize,
        seq_len: usize,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let mut grad_branch = self.uninit(rows * cols)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.context.model.gate_residual_backward)
                .arg(&mut grad_branch)
                .arg(grad_triple)
                .arg(branch)
                .arg(triple)
                .arg(grad_output)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(seq_len as i32))
                .launch(cfg(rows * cols))
                .map_err(cuda_err("gated residual backward kernel"))?;
        }
        Ok(grad_branch)
    }

    /// Several weight matrices end to end in one buffer, which is the weight
    /// of the single wide projection that replaces them. cuBLAS is close to
    /// twice as fast on one wide shape as on two narrow ones, and the packing
    /// itself is a device-to-device copy of weight-sized buffers.
    pub(crate) fn pack(&self, parts: &[&Linear], narrow: bool) -> Result<Act, NetworkError> {
        let element = Act::element(narrow);
        let mut total = 0;
        for part in parts {
            let device = device_of(part)?;
            total += device.rows() * device.cols();
        }
        let mut packed = self.act(total, narrow)?;
        let mut base = 0;
        for part in parts {
            let device = device_of(part)?;
            let (units, inner) = (device.rows(), device.cols());
            let len = units * inner;
            let mut slot = packed
                .bytes
                .slice_mut(base * element..(base + len) * element);
            self.cast_into(&mut slot, &device.value().slice(..), len, narrow)?;
            if let Some(lora) = &part.lora {
                self.fold_lora(lora, &mut slot, units, inner, narrow)?;
            }
            base += len;
        }
        Ok(packed)
    }

    /// Adds `scale * up . down` to a packed weight, which is what makes a LoRA
    /// adapter free everywhere the packed weight is read.
    ///
    /// NOTE: the adapter is materialized into the weight once per forward
    /// pass rather than carried as two extra GEMMs at every reader. That costs
    /// one `[out, in]` GEMM of inner dimension `rank` per projection per step,
    /// against two `[rows, rank]` GEMMs in the forward pass and one more in the
    /// backward pass, and it leaves the projection GEMM and the input-gradient
    /// GEMM - packed, strided, BF16, flash-attention-fed - untouched. Move the
    /// adapter to the readers if a run is ever weight-bound rather than
    /// token-bound.
    pub(crate) fn fold_lora(
        &self,
        lora: &Lora,
        slot: &mut CudaViewMut<'_, u8>,
        units: usize,
        inner: usize,
        narrow: bool,
    ) -> Result<(), NetworkError> {
        let down = device_param(&lora.down)?;
        let up = device_param(&lora.up)?;
        let rank = down.rows();
        // Both adapter matrices are FP32 on the device, and cuBLAS has no GEMM
        // from FP32 operands into a narrow output, so a narrow weight gets
        // narrowed copies of the two. They are `rank` wide, so the casts are
        // small next to the weight they fold into.
        let up = self.narrowed(&up.value().slice(..), units * rank, narrow)?;
        let down = self.narrowed(&down.value().slice(..), rank * inner, narrow)?;
        act_plain(
            self.context,
            &up.all(),
            rank,
            &down.all(),
            inner,
            narrow,
            narrow,
            slot,
            inner,
            units,
            inner,
            rank,
            lora.scale,
            1.0,
        )
    }

    /// The gate and up weights of every given feed-forward, one after another,
    /// so a routed layer pays a single allocation for all of its experts.
    pub(crate) fn pack_gate_up<'a>(
        &self,
        ffns: impl Iterator<Item = &'a SwiGlu>,
        narrow: bool,
    ) -> Result<Act, NetworkError> {
        let mut parts = Vec::new();
        for ffn in ffns {
            parts.push(&ffn.gate);
            parts.push(&ffn.up);
        }
        self.pack(&parts, narrow)
    }

    /// The down projections of the given feed-forwards, one after another.
    ///
    /// The routed path packs them for the same reason the dense path does:
    /// a packed weight is where a LoRA adapter is folded in, so the readers
    /// of the routed expert's output projection need no adapter of their own.
    pub(crate) fn pack_down<'a>(
        &self,
        ffns: impl Iterator<Item = &'a SwiGlu>,
        narrow: bool,
    ) -> Result<Act, NetworkError> {
        let parts: Vec<&Linear> = ffns.map(|ffn| &ffn.down).collect();
        self.pack(&parts, narrow)
    }

    /// `out = x . weight^T + beta * out` for a weight that is a plain buffer
    /// rather than a parameter, which is what the packed gate-and-up is.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_packed<O: DevicePtrMut<f32>>(
        &self,
        weights: &CudaView<'_, u8>,
        units: usize,
        inner: usize,
        x: &CudaView<'_, u8>,
        narrow: bool,
        out: &mut O,
        rows: usize,
        beta: f32,
    ) -> Result<(), NetworkError> {
        act_rhs_transposed(
            self.context,
            x,
            inner,
            weights,
            inner,
            narrow,
            false,
            out,
            units,
            rows,
            units,
            inner,
            1.0,
            beta,
        )
    }

    /// [`Gpu::linear_packed`] straight into a narrow result.
    ///
    /// Only for a result whose readers all take narrow input: cuBLAS writes
    /// BF16 itself, so there is no cast kernel and no FP32 round trip.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_packed_act(
        &self,
        weights: &CudaView<'_, u8>,
        units: usize,
        inner: usize,
        x: &CudaView<'_, u8>,
        narrow: bool,
        out: &mut Act,
        rows: usize,
    ) -> Result<(), NetworkError> {
        let out_narrow = out.is_narrow();
        act_rhs_transposed(
            self.context,
            x,
            inner,
            weights,
            inner,
            narrow,
            out_narrow,
            out.destination(),
            units,
            rows,
            units,
            inner,
            1.0,
            0.0,
        )
    }

    /// The input-gradient half of [`Gpu::linear_packed`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn linear_packed_backward_input<T, O: DevicePtrMut<T>>(
        &self,
        weights: &CudaView<'_, u8>,
        units: usize,
        inner: usize,
        grad_output: &CudaView<'_, u8>,
        narrow: bool,
        out_narrow: bool,
        out: &mut O,
        rows: usize,
        beta: f32,
    ) -> Result<(), NetworkError> {
        act_plain(
            self.context,
            grad_output,
            units,
            weights,
            inner,
            narrow,
            out_narrow,
            out,
            inner,
            rows,
            inner,
            units,
            1.0,
            beta,
        )
    }

    /// [`Gpu::accumulate_weight_grad`] over [`Act`] operands, for the dense
    /// projections whose weights are packed narrow.
    pub(crate) fn accumulate_weight_grad_act(
        &self,
        weight: &mut DeviceParam,
        grad_output: &CudaView<'_, u8>,
        input: &CudaView<'_, u8>,
        narrow: bool,
        rows: usize,
    ) -> Result<(), NetworkError> {
        if weight.is_frozen() {
            return Ok(());
        }
        let (units, cols) = (weight.rows(), weight.cols());
        let beta = weight.grad_beta();
        act_lhs_transposed(
            self.context,
            grad_output,
            units,
            input,
            cols,
            narrow,
            weight.negated_grad_mut(),
            cols,
            rows,
            units,
            cols,
            -1.0,
            beta,
        )
    }

    /// One wide weight-gradient GEMM for a packed projection, split back into
    /// the separate parameters afterwards. The split is one add per parameter
    /// over a weight-sized buffer, which is nothing next to halving the GEMM
    /// time.
    pub(crate) fn accumulate_packed_grad(
        &self,
        parts: &mut [&mut DeviceParam],
        grad: &CudaView<'_, u8>,
        input: &CudaView<'_, u8>,
        narrow: bool,
        rows: usize,
    ) -> Result<(), NetworkError> {
        if parts.iter().all(|part| part.is_frozen()) {
            return Ok(());
        }
        debug_assert!(
            parts.iter().all(|part| !part.is_frozen()),
            "a packed projection freezes as a unit"
        );
        let inner = parts[0].cols();
        let units: usize = parts.iter().map(|part| part.rows()).sum();
        let mut scratch = self.uninit(units * inner)?;
        act_lhs_transposed(
            self.context,
            grad,
            units,
            input,
            inner,
            narrow,
            &mut scratch,
            inner,
            rows,
            units,
            inner,
            -1.0,
            0.0,
        )?;
        let mut base = 0;
        for part in parts {
            let len = part.rows() * inner;
            let keep = part.grad_beta() != 0.0;
            self.add(part.negated_grad_mut(), &scratch, base, len, keep)?;
            base += len;
        }
        Ok(())
    }

    /// One projection's weight gradients: the base weight's unless it is
    /// frozen, and its adapter's if it has one.
    ///
    /// `dL/dinput` is not this function's business either way. The packed
    /// weight the input-gradient GEMM reads already carries
    /// `scale * up . down`, folded in by [`Gpu::fold_lora`].
    pub(crate) fn accumulate_projection_grad(
        &self,
        linear: &mut Linear,
        grad_output: &CudaView<'_, u8>,
        input: &CudaView<'_, u8>,
        narrow: bool,
        rows: usize,
    ) -> Result<(), NetworkError> {
        let units = linear.out_features();
        self.accumulate_weight_grad_act(device_of_mut(linear)?, grad_output, input, narrow, rows)?;
        match &mut linear.lora {
            Some(lora) => self.accumulate_lora_grad(lora, grad_output, units, input, narrow, rows),
            None => Ok(()),
        }
    }

    /// [`Gpu::accumulate_projection_grad`] for projections that share one wide
    /// GEMM, each adapter reading its own column band of the fused gradient.
    pub(crate) fn accumulate_packed_projection_grad(
        &self,
        parts: &mut [&mut Linear],
        grad_output: &CudaView<'_, u8>,
        input: &CudaView<'_, u8>,
        narrow: bool,
        rows: usize,
    ) -> Result<(), NetworkError> {
        let stride: usize = parts.iter().map(|part| part.out_features()).sum();
        {
            let mut devices = Vec::with_capacity(parts.len());
            for part in parts.iter_mut() {
                devices.push(device_of_mut(part)?);
            }
            self.accumulate_packed_grad(&mut devices, grad_output, input, narrow, rows)?;
        }
        let element = Act::element(narrow);
        let mut base = 0;
        for part in parts.iter_mut() {
            let units = part.out_features();
            if let Some(lora) = &mut part.lora {
                self.accumulate_lora_grad(
                    lora,
                    &grad_output.slice(base * element..),
                    stride,
                    input,
                    narrow,
                    rows,
                )?;
            }
            base += units;
        }
        Ok(())
    }

    /// `-dL/dup` and `-dL/ddown` for one adapter.
    ///
    /// `grad_stride` is the row stride of the buffer `grad_output` points
    /// into, which is wider than the projection whenever several projections
    /// share one fused gradient. The adapter's own width comes from `up`.
    pub(crate) fn accumulate_lora_grad(
        &self,
        lora: &mut Lora,
        grad_output: &CudaView<'_, u8>,
        grad_stride: usize,
        input: &CudaView<'_, u8>,
        narrow: bool,
        rows: usize,
    ) -> Result<(), NetworkError> {
        let (rank, inner) = {
            let down = device_param(&lora.down)?;
            (down.rows(), down.cols())
        };
        let units = device_param(&lora.up)?.rows();
        // One cuBLAS call reads one operand type, and the activations here may
        // be BF16 while the adapter is always FP32, so the adapter matrices
        // are cast to match. They are rank-sized: the cast is noise next to
        // the GEMM that reads them.
        let down_act = self.narrowed(
            &device_param(&lora.down)?.value().slice(..),
            rank * inner,
            narrow,
        )?;
        let up_act = self.narrowed(
            &device_param(&lora.up)?.value().slice(..),
            units * rank,
            narrow,
        )?;

        // The same `[rows, rank]` intermediate the forward pass would have
        // produced, had the adapter not been folded into the weight. Recomputing
        // it costs one thin GEMM; caching it would cost a buffer per projection
        // per layer for the whole depth of the backward pass.
        let mut hidden = self.act(rows * rank, narrow)?;
        act_rhs_transposed(
            self.context,
            input,
            inner,
            &down_act.all(),
            inner,
            narrow,
            narrow,
            hidden.destination(),
            rank,
            rows,
            rank,
            inner,
            lora.scale,
            0.0,
        )?;
        {
            let up = device_param_mut(&mut lora.up)?;
            let beta = up.grad_beta();
            act_lhs_transposed(
                self.context,
                grad_output,
                grad_stride,
                &hidden.all(),
                rank,
                narrow,
                up.negated_grad_mut(),
                rank,
                rows,
                units,
                rank,
                -1.0,
                beta,
            )?;
        }

        let mut grad_hidden = self.act(rows * rank, narrow)?;
        act_plain(
            self.context,
            grad_output,
            grad_stride,
            &up_act.all(),
            rank,
            narrow,
            narrow,
            grad_hidden.destination(),
            rank,
            rows,
            rank,
            units,
            lora.scale,
            0.0,
        )?;
        let down = device_param_mut(&mut lora.down)?;
        let beta = down.grad_beta();
        act_lhs_transposed(
            self.context,
            &grad_hidden.all(),
            rank,
            input,
            inner,
            narrow,
            down.negated_grad_mut(),
            inner,
            rows,
            rank,
            inner,
            -1.0,
            beta,
        )
    }
}

/// The device mirror of a LoRA matrix, or an error naming what is missing.
fn device_param(param: &Param) -> Result<&DeviceParam, NetworkError> {
    param
        .device
        .as_ref()
        .ok_or_else(|| NetworkError::Cuda("a LoRA adapter is not resident on the device".into()))
}

fn device_param_mut(param: &mut Param) -> Result<&mut DeviceParam, NetworkError> {
    param
        .device
        .as_mut()
        .ok_or_else(|| NetworkError::Cuda("a LoRA adapter is not resident on the device".into()))
}

/// The device mirror of a projection, or an error naming the layer that is
/// still host-resident.
/// The output projection: an untied `lm_head`, or the embedding table when the
/// weights are tied.
fn head_of(model: &TransformerLm) -> Result<&DeviceParam, NetworkError> {
    match &model.lm_head {
        Some(head) => device_of(head),
        None => model.embedding.weight.device.as_ref().ok_or_else(|| {
            NetworkError::Cuda("the embedding table is not resident on the device".into())
        }),
    }
}

fn head_of_mut(model: &mut TransformerLm) -> Result<&mut DeviceParam, NetworkError> {
    match &mut model.lm_head {
        Some(head) => device_of_mut(head),
        None => model.embedding.weight.device.as_mut().ok_or_else(|| {
            NetworkError::Cuda("the embedding table is not resident on the device".into())
        }),
    }
}

pub(crate) fn device_of(linear: &Linear) -> Result<&DeviceParam, NetworkError> {
    linear
        .weight
        .device
        .as_ref()
        .ok_or_else(|| NetworkError::Cuda("a projection is not resident on the device".into()))
}

pub(crate) fn device_of_mut(linear: &mut Linear) -> Result<&mut DeviceParam, NetworkError> {
    linear
        .weight
        .device
        .as_mut()
        .ok_or_else(|| NetworkError::Cuda("a projection is not resident on the device".into()))
}

/// The device path implements the SwiGLU shape only, which is what
/// [`TransformerBuilder`](crate::transformer::TransformerBuilder) builds.
pub(crate) fn unsupported(kind: &str) -> NetworkError {
    NetworkError::UnsupportedCuda(format!(
        "the device training path implements SwiGLU feed-forwards and MoE layers, not {kind}"
    ))
}

/// Device-resident forward pass over a packed batch.
///
/// Returns the logits on the host, because the caller's loss is a host
/// computation, and a cache of device buffers for [`backward`]. The logits are
/// `[rows, vocab_size]`, which for a real vocabulary is the largest tensor in
/// the step by a wide margin; [`train_step`] exists so that the training path
/// never has to materialize it.
pub(crate) fn forward(
    model: &TransformerLm,
    context: &Arc<GpuContext>,
    batch: &TokenBatch,
) -> Result<(Matrix, GpuCache), NetworkError> {
    let cache = forward_hidden(model, context, batch)?;
    let gpu = Gpu { context };
    let rows = cache.stack.rows;
    let logits = gpu.linear(head_of(model)?, &cache.final_output.wide(), rows)?;
    let logits = gpu.matrix(&logits, rows, model.embedding.vocab_size())?;
    Ok((logits, cache))
}

/// Everything [`forward`] does except the language-model head, which the two
/// callers want in different shapes.
fn forward_hidden(
    model: &TransformerLm,
    context: &Arc<GpuContext>,
    batch: &TokenBatch,
) -> Result<GpuCache, NetworkError> {
    let gpu = Gpu { context };
    let rows = batch.rows();
    let seq_len = batch.seq_len();
    let sequences = batch.batch();
    let d_model = model.config.d_model;
    let vocab = model.embedding.vocab_size();

    for &id in batch.ids() {
        if id as usize >= vocab {
            return Err(NetworkError::TokenOutOfRange {
                id,
                vocab_size: vocab,
            });
        }
    }

    let layout = batch.layout();
    let flags: Vec<i32> = (0..rows)
        .map(|row| i32::from(layout.is_valid(row)))
        .collect();
    let valid_tokens = flags.iter().filter(|&&flag| flag != 0).count();
    let valid = gpu.upload_flags(&flags)?;
    let ids = gpu.upload_indices(batch.ids())?;

    let embedding = model.embedding.weight.device.as_ref().ok_or_else(|| {
        NetworkError::Cuda("the embedding table is not resident on the device".into())
    })?;
    let mut hidden = gpu.uninit(rows * d_model)?;
    gpu.gather(
        &mut hidden.slice_mut(..),
        &embedding.value().slice(..),
        &ids,
        rows,
        d_model,
    )?;

    let (hidden, stack) = forward_stack(
        &gpu,
        &model.blocks,
        hidden,
        &valid,
        valid_tokens,
        rows,
        seq_len,
        sequences,
    )?;

    let final_weight = gpu.upload(&model.final_norm.weight.value.data)?;
    // The head narrows its own operands, so the last norm stays FP32.
    let (final_output, final_inverse_rms, _) = gpu.rmsnorm(
        &hidden,
        &final_weight,
        rows,
        d_model,
        model.final_norm.eps,
        false,
        false,
    )?;

    Ok(GpuCache {
        stack,
        ids,
        final_input: hidden,
        final_inverse_rms,
        final_weight,
        final_output,
    })
}

/// Every block in order over `hidden`, returning the last block's output.
#[allow(clippy::too_many_arguments)]
fn forward_stack(
    gpu: &Gpu<'_>,
    blocks: &[TransformerBlock],
    mut hidden: CudaSlice<f32>,
    valid: &CudaSlice<i32>,
    valid_tokens: usize,
    rows: usize,
    seq_len: usize,
    sequences: usize,
) -> Result<(CudaSlice<f32>, GpuStack), NetworkError> {
    let mut caches = Vec::with_capacity(blocks.len());
    let mut auxiliary_loss = 0.0;
    for block in blocks {
        let (output, cache) = forward_block(
            gpu,
            block,
            hidden,
            valid,
            valid_tokens,
            rows,
            seq_len,
            sequences,
            &mut auxiliary_loss,
        )?;
        caches.push(cache);
        hidden = output;
    }
    Ok((
        hidden,
        GpuStack {
            rows,
            seq_len,
            sequences,
            valid_tokens,
            blocks: caches,
            auxiliary_loss,
        },
    ))
}

/// A stack of blocks whose input and output are hidden states on the host
/// rather than token ids and logits: a vision transformer's, whose patch
/// projection, norm, pooling and head are small enough to stay there.
///
/// Every row is a real position, so nothing is masked out.
pub(crate) fn stack_forward(
    blocks: &[TransformerBlock],
    context: &Arc<GpuContext>,
    input: &Matrix,
    seq_len: usize,
) -> Result<(Matrix, GpuStack), NetworkError> {
    let gpu = Gpu { context };
    let rows = input.rows;
    let valid = gpu.upload_flags(&vec![1; rows])?;
    let hidden = gpu.upload(&input.data)?;
    let (hidden, stack) = forward_stack(
        &gpu,
        blocks,
        hidden,
        &valid,
        rows,
        rows,
        seq_len,
        rows / seq_len,
    )?;
    Ok((gpu.matrix(&hidden, rows, input.cols)?, stack))
}

/// The backward half of [`stack_forward`]: accumulates every block's weight
/// and norm gradients and returns `dL/dinput` on the host.
pub(crate) fn stack_backward(
    blocks: &mut [TransformerBlock],
    context: &Arc<GpuContext>,
    stack: &GpuStack,
    grad_output: &Matrix,
) -> Result<Matrix, NetworkError> {
    let gpu = Gpu { context };
    let rows = stack.rows;
    let d_model = grad_output.cols;
    if grad_output.rows != rows {
        return Err(NetworkError::InvalidInput {
            expected: rows,
            actual: grad_output.rows,
        });
    }
    let grad = gpu.upload(&grad_output.data)?;
    // What the final norm's backward would have handed the top block: its own
    // operand copy in whichever precision that block's feed forward reads.
    let grad_act = match stack_operand(stack, blocks.len().checked_sub(1)) {
        Some(narrow) => Some(gpu.narrowed(&grad.slice(..), rows * d_model, narrow)?),
        None => None,
    };
    // Slot 0 is the final norm's, which this stack does not have.
    let mut grad_norms = gpu.zeros(norm_slots(blocks.len()) * d_model)?;
    let grad = backward_stack(&gpu, blocks, stack, grad, grad_act, &mut grad_norms)?;
    let values = gpu.download(&grad_norms)?;
    accumulate_block_norms(blocks, &values, d_model);
    gpu.matrix(&grad, rows, d_model)
}

/// The gradient a norm hands down is a GEMM operand for the layer below it,
/// so the norm writes that operand copy itself. Which precision it wants is
/// the receiving feed forward's; a routed one takes FP32 and narrows the
/// shared expert's operand on its own, so it asks for no copy at all.
fn stack_operand(stack: &GpuStack, index: Option<usize>) -> Option<bool> {
    stack
        .blocks
        .get(index?)
        .and_then(|block| match &block.feed_forward {
            FfnCache::Dense(_) => Some(block.feed_forward_normed.is_narrow()),
            FfnCache::Moe(_) => None,
        })
}

/// Every block from the top down, returning `dL/dinput` of the bottom one.
fn backward_stack(
    gpu: &Gpu<'_>,
    blocks: &mut [TransformerBlock],
    stack: &GpuStack,
    mut grad_hidden: CudaSlice<f32>,
    mut grad_hidden_act: Option<Act>,
    grad_norms: &mut CudaSlice<f32>,
) -> Result<CudaSlice<f32>, NetworkError> {
    for (index, (block, block_cache)) in blocks.iter_mut().zip(&stack.blocks).enumerate().rev() {
        (grad_hidden, grad_hidden_act) = backward_block(
            gpu,
            block,
            block_cache,
            &grad_hidden,
            grad_hidden_act.as_ref(),
            stack_operand(stack, index.checked_sub(1)),
            stack,
            grad_norms,
            index,
        )?;
    }
    Ok(grad_hidden)
}

/// Adds the downloaded norm-scale gradients to each block's host-resident
/// norm weights.
fn accumulate_block_norms(blocks: &mut [TransformerBlock], values: &[f32], d_model: usize) {
    let at = |slot: usize| &values[slot * d_model..(slot + 1) * d_model];
    for (index, block) in blocks.iter_mut().enumerate() {
        let (attention, feed_forward) = norm_slot(index);
        Gpu::accumulate_host_grad(&mut block.attention_norm.weight, at(attention));
        Gpu::accumulate_host_grad(&mut block.feed_forward_norm.weight, at(feed_forward));
    }
}

/// One decoder block: pre-norm attention, then a pre-norm feed-forward, both
/// residual. `input` is consumed into the cache, which is what the backward
/// pass needs it for.
#[allow(clippy::too_many_arguments)]
fn forward_block(
    gpu: &Gpu<'_>,
    block: &TransformerBlock,
    input: CudaSlice<f32>,
    valid: &CudaSlice<i32>,
    valid_tokens: usize,
    rows: usize,
    seq_len: usize,
    sequences: usize,
    auxiliary_loss: &mut f32,
) -> Result<(CudaSlice<f32>, BlockCache), NetworkError> {
    let attention = &block.attention;
    let d_model = attention.d_model();
    let heads = attention.num_heads();
    let kv_heads = attention.num_kv_heads();
    let head_dim = attention.head_dim();
    let causal = attention.is_causal();
    let scale = (head_dim as f32).sqrt().recip();
    let query_width = heads * head_dim;

    // The attention side of the block is always fused-GEMM territory, so its
    // activations follow the context flag. The feed-forward norm only narrows
    // when the feed-forward is dense: a routed layer's router and gather read
    // it as FP32, and the gather narrows the rows the experts read.
    let narrow = gpu.context.mixed_precision;
    let dense_ffn = matches!(block.feed_forward, FeedForward::SwiGlu(_));
    let ffn_narrow = narrow && dense_ffn;

    let attention_weight = gpu.upload(&block.attention_norm.weight.value.data)?;
    // The residual this norm's branch lands on is the duplicate it writes.
    let (attention_normed, attention_inverse_rms, residual) = gpu.rmsnorm(
        &input,
        &attention_weight,
        rows,
        d_model,
        block.attention_norm.eps,
        narrow,
        true,
    )?;
    let mut residual = residual.expect("a norm asked for a duplicate returns one");

    // Query, key and value read the same input and differ only in width, so
    // they are one GEMM against their three weight matrices packed end to end.
    // Every later reader takes a slice of the fused row instead of a buffer of
    // its own, which is why the strides below are `qkv_width` rather than the
    // width of the projection being read.
    let kv_width = kv_heads * head_dim;
    let qkv_width = query_width + 2 * kv_width;
    // Only the fused kernels read a narrow `qkv`; the batched-GEMM fallback
    // below wants FP32, so the projection's output type follows the same
    // choice the attention path does.
    let fused = gpu
        .context
        .flash
        .as_ref()
        .filter(|_| crate::cuda_flash::eligible(gpu.context.mixed_precision, head_dim));
    let qkv_weights = gpu.pack(
        &[&attention.query, &attention.key, &attention.value],
        narrow,
    )?;
    let mut qkv = gpu.act(rows * qkv_width, narrow && fused.is_some())?;
    gpu.linear_packed_act(
        &qkv_weights.all(),
        qkv_width,
        d_model,
        &attention_normed.all(),
        narrow,
        &mut qkv,
        rows,
    )?;
    let key_base = query_width;
    let value_base = query_width + kv_width;

    let tables = gpu.rope_tables(&attention.rope)?;
    if seq_len > tables.max_seq_len {
        return Err(NetworkError::SequenceTooLong {
            length: seq_len,
            max_seq_len: tables.max_seq_len,
        });
    }
    gpu.rope(
        &mut qkv, &tables, rows, heads, head_dim, seq_len, 1.0, qkv_width, 0,
    )?;
    gpu.rope(
        &mut qkv, &tables, rows, kv_heads, head_dim, seq_len, 1.0, qkv_width, key_base,
    )?;

    // Attention, one of two ways. The fused kernel keeps the scores in
    // registers and never writes the `[seq_len, seq_len]` matrix at all; the
    // three-kernel path below writes it, reads it back for the softmax and
    // reads it a third time for the value matmul, which profiling put at 21.3%
    // of GPU time. Both produce the same `merged` and `log_sum_exp`.
    let group = heads / kv_heads;
    let shape = FlashShape {
        rows,
        seq_len,
        sequences,
        heads,
        group,
        qkv_width,
        query_width,
        key_base,
        value_base,
        causal,
    };
    let mut log_sum_exp = gpu.uninit(heads * rows)?;
    // Only the fused kernel knows how to write a narrow `merged`; the batched
    // GEMM the fallback path ends with produces FP32.
    let mut merged = gpu.act(rows * query_width, narrow && fused.is_some())?;
    if let Some(flash) = fused {
        gpu.flash_attention(flash, &qkv, &mut merged, &mut log_sum_exp, shape, scale)?;
    } else {
        let mut wide_merged = merged.wide_mut();
        let qkv = qkv.wide();
        let block_size = seq_len * seq_len;
        let mut probabilities = gpu.uninit(heads * sequences * block_size)?;
        for head in 0..heads {
            let kv_base = (head / group) * head_dim;
            let query = qkv.slice(head * head_dim..);
            let key = qkv.slice(key_base + kv_base..);
            let mut scores = probabilities.slice_mut(head * sequences * block_size..);
            gemm_rhs_transposed_batched(
                gpu.context,
                &query,
                qkv_width,
                &key,
                qkv_width,
                &mut scores,
                seq_len,
                seq_len,
                seq_len,
                head_dim,
                scale,
                0.0,
                sequences,
                seq_len * qkv_width,
                seq_len * qkv_width,
                block_size,
            )?;
        }
        gpu.softmax_lse(
            &mut probabilities,
            &mut log_sum_exp,
            heads * sequences * seq_len,
            seq_len,
            causal,
        )?;
        for head in 0..heads {
            let kv_base = (head / group) * head_dim;
            let head_probabilities = probabilities.slice(head * sequences * block_size..);
            let value = qkv.slice(value_base + kv_base..);
            let mut head_merged = wide_merged.slice_mut(head * head_dim..);
            gemm_plain_batched(
                gpu.context,
                &head_probabilities,
                seq_len,
                &value,
                qkv_width,
                &mut head_merged,
                query_width,
                seq_len,
                head_dim,
                seq_len,
                1.0,
                0.0,
                sequences,
                block_size,
                seq_len * qkv_width,
                seq_len * query_width,
            )?;
        }
    }

    // The residual add is the output projection's GEMM with beta = 1 over the
    // duplicate the norm above left behind, so it costs no extra kernel.
    let output_weight = gpu.pack(&[&attention.output], merged.is_narrow())?;
    gpu.linear_packed(
        &output_weight.all(),
        d_model,
        query_width,
        &merged.all(),
        merged.is_narrow(),
        &mut residual,
        rows,
        1.0,
    )?;

    let feed_forward_weight = gpu.upload(&block.feed_forward_norm.weight.value.data)?;
    let (feed_forward_normed, feed_forward_inverse_rms, output) = gpu.rmsnorm(
        &residual,
        &feed_forward_weight,
        rows,
        d_model,
        block.feed_forward_norm.eps,
        ffn_narrow,
        true,
    )?;
    let mut output = output.expect("a norm asked for a duplicate returns one");
    let feed_forward = match &block.feed_forward {
        FeedForward::SwiGlu(ffn) => FfnCache::Dense(forward_swiglu(
            gpu,
            ffn,
            &feed_forward_normed,
            &mut output,
            rows,
        )?),
        FeedForward::Gelu(_) => return Err(unsupported("a GELU MLP")),
        FeedForward::Moe(moe) => {
            let (cache, losses) = forward_moe(
                gpu,
                moe,
                &feed_forward_normed,
                &mut output,
                valid,
                valid_tokens,
                rows,
            )?;
            *auxiliary_loss += losses;
            FfnCache::Moe(Box::new(cache))
        }
    };

    Ok((
        output,
        BlockCache {
            input,
            attention_weight,
            attention_inverse_rms,
            attention_normed,
            qkv,
            qkv_weights,
            log_sum_exp,
            merged,
            output_weight,
            residual,
            feed_forward_weight,
            feed_forward_inverse_rms,
            feed_forward_normed,
            feed_forward,
        },
    ))
}

/// `out += down(silu(gate(x)) * up(x))`.
pub(crate) fn forward_swiglu(
    gpu: &Gpu<'_>,
    ffn: &SwiGlu,
    input: &Act,
    out: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<SwiGluCache, NetworkError> {
    let width = ffn.d_ff();
    let inner = ffn.d_model();
    let narrow = input.is_narrow();
    let weights = gpu.pack_gate_up(std::iter::once(ffn), narrow)?;
    let mut gate_up = gpu.uninit(rows * 2 * width)?;
    gpu.linear_packed(
        &weights.all(),
        2 * width,
        inner,
        &input.all(),
        narrow,
        &mut gate_up,
        rows,
        0.0,
    )?;
    let hidden = gpu.swiglu(&gate_up, rows, width, narrow)?;
    let down = gpu.pack(&[&ffn.down], narrow)?;
    gpu.linear_packed(
        &down.all(),
        inner,
        width,
        &hidden.all(),
        narrow,
        out,
        rows,
        1.0,
    )?;
    Ok(SwiGluCache {
        gate_up,
        hidden,
        weights,
        down: Some(down),
    })
}

/// The routed feed-forward.
///
/// The router's softmax, top-k and gate renormalization are one kernel each.
/// The routing table then comes back to the host once, so the per-expert token
/// counts can shape the expert GEMMs, and the tokens are gathered into one
/// contiguous buffer grouped by expert: three GEMMs per non-empty expert for
/// the whole batch, not three per expert per sequence.
#[allow(clippy::too_many_arguments)]
fn forward_moe(
    gpu: &Gpu<'_>,
    moe: &MoeLayer,
    input: &Act,
    out: &mut CudaSlice<f32>,
    valid: &CudaSlice<i32>,
    valid_tokens: usize,
    rows: usize,
) -> Result<(MoeGpuCache, f32), NetworkError> {
    let experts = moe.config.num_experts;
    let top_k = moe.config.experts_per_token;
    let d_model = moe.d_model();
    let width = moe.config.d_ff;

    let logits = gpu.linear(device_of(&moe.router.projection)?, &input.wide(), rows)?;
    let mut probabilities = gpu.zeros(rows * experts)?;
    let mut log_sum_exp = gpu.zeros(rows)?;
    unsafe {
        gpu.context
            .stream
            .launch_builder(&gpu.context.model.softmax_lse)
            .arg(&mut probabilities)
            .arg(&mut log_sum_exp)
            .arg(&logits)
            .arg(&(rows as i32))
            .arg(&(experts as i32))
            .launch(cfg(rows))
            .map_err(cuda_err("router softmax kernel"))?;
    }

    let mut expert_of = gpu
        .context
        .stream
        .alloc_zeros::<i32>(rows * top_k)
        .map_err(cuda_err("device allocation"))?;
    let mut gate_of = gpu.zeros(rows * top_k)?;
    unsafe {
        gpu.context
            .stream
            .launch_builder(&gpu.context.model.topk_gate)
            .arg(&mut expert_of)
            .arg(&mut gate_of)
            .arg(&probabilities)
            .arg(valid)
            .arg(&(rows as i32))
            .arg(&(experts as i32))
            .arg(&(top_k as i32))
            .launch(cfg(rows))
            .map_err(cuda_err("router top-k kernel"))?;
    }

    // The one deliberate round trip: cuBLAS needs each expert's token count on
    // the host to shape its GEMM.
    let assigned = gpu.download_signed(&expert_of)?;
    let gates_host = gpu.download(&gate_of)?;

    let mut counts = vec![0usize; experts];
    for &expert in &assigned {
        if expert >= 0 {
            counts[expert as usize] += 1;
        }
    }
    let mut offsets = Vec::with_capacity(experts);
    let mut routed = 0;
    for &count in &counts {
        offsets.push(routed);
        routed += count;
    }

    let mut cursor = offsets.clone();
    let mut token_host = vec![0u32; routed.max(1)];
    let mut slot_host = vec![0u32; routed.max(1)];
    let mut gate_host = vec![0f32; routed.max(1)];
    for (slot, &expert) in assigned.iter().enumerate() {
        if expert < 0 {
            continue;
        }
        let position = &mut cursor[expert as usize];
        token_host[*position] = (slot / top_k) as u32;
        slot_host[*position] = slot as u32;
        gate_host[*position] = gates_host[slot];
        *position += 1;
    }

    let token_of = gpu.upload_indices(&token_host)?;
    let slot_of = gpu.upload_indices(&slot_host)?;
    let gates = gpu.upload(&gate_host)?;

    // Every one of these is written in full before it is read: `gather` and
    // `gather_scaled` touch every routed row, and the per-expert loops
    // partition the routed rows between them with `beta = 0` on the first
    // write of each slice.
    let narrow = gpu.context.mixed_precision;
    let mut gathered = gpu.act(routed * d_model, narrow)?;
    if routed > 0 {
        gpu.gather_scaled(
            &mut gathered,
            &input.wide(),
            &token_of,
            &gates,
            false,
            routed,
            d_model,
        )?;
    }

    let expert_weights = gpu.pack_gate_up(moe.experts.iter(), narrow)?;
    let packed_stride = 2 * width * d_model;
    let mut gate_up = gpu.uninit(routed * 2 * width)?;
    for (expert, _) in moe.experts.iter().enumerate() {
        let count = counts[expert];
        if count == 0 {
            continue;
        }
        let rows_in = gathered.at(offsets[expert] * d_model);
        let weights = expert_weights.at(expert * packed_stride);
        let mut out = gate_up.slice_mut(offsets[expert] * 2 * width..);
        gpu.linear_packed(
            &weights,
            2 * width,
            d_model,
            &rows_in,
            narrow,
            &mut out,
            count,
            0.0,
        )?;
    }
    let hidden = gpu.swiglu(&gate_up, routed, width, narrow)?;

    let down_weights = gpu.pack_down(moe.experts.iter(), narrow)?;
    let down_stride = d_model * width;
    let mut expert_output = gpu.uninit(routed * d_model)?;
    for (expert, _) in moe.experts.iter().enumerate() {
        let count = counts[expert];
        if count == 0 {
            continue;
        }
        let rows_in = hidden.at(offsets[expert] * width);
        let mut rows_out = expert_output.slice_mut(offsets[expert] * d_model..);
        gpu.linear_packed(
            &down_weights.at(expert * down_stride),
            d_model,
            width,
            &rows_in,
            narrow,
            &mut rows_out,
            count,
            0.0,
        )?;
    }
    if routed > 0 {
        gpu.scatter_scaled(out, &expert_output, &token_of, &gates, routed, d_model)?;
    }

    let shared = match &moe.shared {
        Some(module) => Some(forward_swiglu(gpu, module, input, out, rows)?),
        None => None,
    };

    // One download of `num_experts + 1` floats carries both auxiliary losses:
    // the per-expert probability mass, and the summed squared log-sum-exp.
    let mut sums = gpu.zeros(experts + 1)?;
    unsafe {
        gpu.context
            .stream
            .launch_builder(&gpu.context.model.stats)
            .arg(&mut sums)
            .arg(&probabilities)
            .arg(&log_sum_exp)
            .arg(valid)
            .arg(&(rows as i32))
            .arg(&(experts as i32))
            .launch(cfg(rows))
            .map_err(cuda_err("MoE statistics kernel"))?;
    }
    let sums = gpu.download(&sums)?;

    let mut auxiliary = 0.0;
    if valid_tokens > 0 {
        let slots = (valid_tokens * top_k) as f32;
        let balance: f32 = (0..experts)
            .map(|expert| (counts[expert] as f32 / slots) * (sums[expert] / valid_tokens as f32))
            .sum();
        auxiliary += experts as f32 * moe.config.aux_loss_weight * balance;
        auxiliary += moe.config.router_z_loss_weight * sums[experts] / valid_tokens as f32;
    }

    Ok((
        MoeGpuCache {
            probabilities,
            log_sum_exp,
            expert_of,
            token_of,
            slot_of,
            gates,
            offsets,
            counts,
            routed,
            input: gathered,
            expert: SwiGluCache {
                gate_up,
                hidden,
                weights: expert_weights,
                down: Some(down_weights),
            },
            output: expert_output,
            shared,
        },
        auxiliary,
    ))
}

/// Device-resident backward pass.
///
/// `grad_logits` arrives from the host loss and every gradient produced here
/// stays on the device, except the RMSNorm scales, which are host-owned so that
/// cached decode keeps working.
pub(crate) fn backward(
    model: &mut TransformerLm,
    context: &Arc<GpuContext>,
    cache: &GpuCache,
    grad_logits: &Matrix,
) -> Result<(), NetworkError> {
    let gpu = Gpu { context };
    let rows = cache.stack.rows;
    let d_model = model.config.d_model;

    if grad_logits.rows != rows {
        return Err(NetworkError::InvalidInput {
            expected: rows,
            actual: grad_logits.rows,
        });
    }

    let upstream = gpu.upload(&grad_logits.data)?;
    let mut grad_final = gpu.zeros(rows * d_model)?;
    {
        // Tied weights land in the same gradient buffer the embedding scatter
        // writes to at the end, which is what tying means.
        let head = head_of_mut(model)?;
        gpu.linear_backward_input(head, &upstream, &mut grad_final, rows, 0.0)?;
        gpu.accumulate_weight_grad(head, &upstream, &cache.final_output.wide(), rows)?;
    }

    backward_from_final(model, context, cache, &grad_final)
}

/// Forward, loss and backward with the logits never leaving the device.
///
/// The `[rows, vocab_size]` logits dominate a training step: at batch 256 and
/// sequence 128 over a 32k vocabulary they are 4 GiB, and the host loss paid
/// for them four times over - a 4 GiB download, a 4 GiB host allocation, a
/// single-threaded softmax over a billion elements, and a 4 GiB upload. On an
/// RTX 3060 that was 86% of the step while the GPU sat idle.
///
/// Here the head runs one chunk of rows at a time. Each chunk's logits are
/// turned into their own gradient in place by `ce_loss_grad` and consumed by
/// the head's backward before the next chunk is produced, so the peak cost is
/// one chunk rather than the whole matrix, and nothing crosses the bus except
/// the row targets going out and one loss scalar coming back. Capping the
/// chunk instead of the batch is also what lets a large batch run at all: the
/// full logits for batch 1024 do not fit in 12 GiB, one chunk always does.
///
/// Returns the mean language-modelling loss and the summed auxiliary losses,
/// the LM head's z-loss among them.
pub(crate) fn train_step(
    model: &mut TransformerLm,
    context: &Arc<GpuContext>,
    batch: &TokenBatch,
) -> Result<(f32, f32), NetworkError> {
    let vocab = model.embedding.vocab_size();
    let predicted = batch.predicted();
    if predicted == 0 {
        return Err(NetworkError::InvalidConfig(
            "a causal language-modelling loss needs at least two tokens".into(),
        ));
    }
    // The kernel carries targets as `int`, and -1 marks a row that predicts
    // nothing, so a vocabulary that does not fit in a positive `i32` is
    // refused rather than silently aliased onto that marker.
    if vocab > i32::MAX as usize {
        return Err(NetworkError::InvalidConfig(format!(
            "the device loss supports vocabularies up to {}, got {vocab}",
            i32::MAX
        )));
    }
    let targets = causal_targets(batch, vocab)?;

    let cache = forward_hidden(model, context, batch)?;
    let gpu = Gpu { context };
    let rows = cache.stack.rows;
    let d_model = model.config.d_model;

    let targets = gpu.upload_signed(&targets)?;
    // [cross-entropy, z-loss], both summed over the predicted rows.
    let mut loss = gpu.zeros(2)?;
    let mut grad_final = gpu.zeros(rows * d_model)?;
    let inverse_predicted = 1.0 / predicted as f32;
    let z_loss = model.z_loss;
    let logit_bytes = if gpu.context.mixed_precision { 2 } else { 4 };
    let chunk = head_chunk_rows(rows, vocab, logit_bytes);

    {
        let head = head_of_mut(model)?;
        let mut reduced = gpu
            .context
            .mixed_precision
            .then(|| HeadBf16::new(&gpu, head, chunk))
            .transpose()?;
        let final_output = cache.final_output.wide();
        let mut start = 0;
        while start < rows {
            let count = chunk.min(rows - start);
            let input = final_output.slice(start * d_model..(start + count) * d_model);
            let targets = targets.slice(start..start + count);
            let mut out = grad_final.slice_mut(start * d_model..(start + count) * d_model);
            if let Some(reduced) = reduced.as_mut() {
                reduced.chunk(
                    &gpu,
                    head,
                    &input,
                    &targets,
                    &mut loss,
                    &mut out,
                    count,
                    inverse_predicted,
                    z_loss,
                )?;
                start += count;
                continue;
            }
            let mut logits = gpu.linear(head, &input, count)?;
            unsafe {
                gpu.context
                    .stream
                    .launch_builder(&gpu.context.model.cross_entropy)
                    .arg(&mut logits)
                    .arg(&mut loss)
                    .arg(&targets)
                    .arg(&(count as i32))
                    .arg(&(vocab as i32))
                    .arg(&inverse_predicted)
                    .arg(&z_loss)
                    .launch(row_per_block(count))
                    .map_err(cuda_err("cross-entropy kernel"))?;
            }
            // `logits` now holds dL/dlogits for this chunk.
            gpu.linear_backward_input(head, &logits, &mut out, count, 0.0)?;
            gpu.accumulate_weight_grad(head, &logits, &input, count)?;
            start += count;
        }
    }

    // After the backward pass, not before: this download drains the stream,
    // and the backward pass does not need the value.
    backward_from_final(model, context, &cache, &grad_final)?;
    let loss = gpu.download(&loss)?;
    let lm_loss = loss[0] / predicted as f32;
    Ok((
        lm_loss,
        cache.stack.auxiliary_loss + loss[1] / predicted as f32,
    ))
}

/// The LM head run in BF16, behind [`crate::TransformerBuilder::mixed_precision`].
///
/// The head is the one part of the step where reduced precision is worth its
/// casts: its three GEMMs are about three quarters of all GEMM time, because
/// every one of them is `vocab`-wide. Weights, optimizer state and the
/// gradient that leaves for the rest of the network all stay FP32; only the
/// head's own multiplier inputs, its logits and its two gradient products are
/// BF16, and cuBLAS still accumulates those in FP32.
///
/// No loss scaling. BF16 has FP32's exponent range, so the smallest gradient
/// here, around `1 / predicted_tokens`, is some thirty orders of magnitude
/// above the smallest normal BF16 and nothing underflows that FP32 would have
/// kept.
struct HeadBf16 {
    /// The head weight, cast once per step.
    weight: CudaSlice<bf16>,
    /// One chunk's product `-dL/dW`, widened into the FP32 gradient after
    /// every chunk rather than accumulated in BF16.
    weight_grad: CudaSlice<bf16>,
    input: CudaSlice<bf16>,
    logits: CudaSlice<bf16>,
    grad_input: CudaSlice<bf16>,
}

impl HeadBf16 {
    fn new(gpu: &Gpu, head: &DeviceParam, chunk: usize) -> Result<Self, NetworkError> {
        let (vocab, d_model) = (head.rows(), head.cols());
        let mut weight = gpu.uninit_bf16(vocab * d_model)?;
        gpu.cast_to_bf16(&mut weight, &head.value().slice(..), vocab * d_model)?;
        Ok(Self {
            weight,
            // A frozen head takes no gradient, so the buffer that would carry
            // one is a placeholder.
            weight_grad: gpu.uninit_bf16(if head.is_frozen() { 1 } else { vocab * d_model })?,
            input: gpu.uninit_bf16(chunk * d_model)?,
            logits: gpu.uninit_bf16(chunk * vocab)?,
            grad_input: gpu.uninit_bf16(chunk * d_model)?,
        })
    }

    /// Forward, loss and both gradients for one chunk of rows.
    ///
    /// `grad_input` is overwritten, and `head`'s negated gradient accumulated
    /// into, exactly as the FP32 path does.
    #[allow(clippy::too_many_arguments)]
    fn chunk(
        &mut self,
        gpu: &Gpu,
        head: &mut DeviceParam,
        input: &CudaView<'_, f32>,
        targets: &CudaView<'_, i32>,
        loss: &mut CudaSlice<f32>,
        grad_input: &mut CudaViewMut<'_, f32>,
        count: usize,
        inverse_predicted: f32,
        z_loss: f32,
    ) -> Result<(), NetworkError> {
        let (vocab, d_model) = (head.rows(), head.cols());
        gpu.cast_to_bf16(&mut self.input, input, count * d_model)?;
        gemm_rhs_transposed(
            gpu.context,
            &self.input,
            d_model,
            &self.weight,
            d_model,
            &mut self.logits,
            vocab,
            count,
            vocab,
            d_model,
            bf16::ONE,
            bf16::ZERO,
        )?;
        unsafe {
            gpu.context
                .stream
                .launch_builder(&gpu.context.model.cross_entropy_bf16)
                .arg(&mut self.logits)
                .arg(loss)
                .arg(targets)
                .arg(&(count as i32))
                .arg(&(vocab as i32))
                .arg(&inverse_predicted)
                .arg(&z_loss)
                .launch(row_per_block(count))
                .map_err(cuda_err("cross-entropy kernel"))?;
        }
        // `self.logits` now holds dL/dlogits for this chunk.
        gemm_plain(
            gpu.context,
            &self.logits,
            vocab,
            &self.weight,
            d_model,
            &mut self.grad_input,
            d_model,
            count,
            d_model,
            vocab,
            bf16::ONE,
            bf16::ZERO,
        )?;
        gpu.cast_from_bf16(grad_input, &self.grad_input, count * d_model)?;
        if head.is_frozen() {
            return Ok(());
        }
        gemm_lhs_transposed(
            gpu.context,
            &self.logits,
            vocab,
            &self.input,
            d_model,
            &mut self.weight_grad,
            d_model,
            count,
            vocab,
            d_model,
            bf16::NEG_ONE,
            bf16::ZERO,
        )?;
        head.clear_grad()?;
        gpu.accumulate_bf16(head.negated_grad_mut(), &self.weight_grad, vocab * d_model)
    }
}

/// Next-token target per row, or -1 where a row predicts nothing.
///
/// Padding rows and the last position of every sequence predict nothing, which
/// is the same rule [`crate::causal_lm_loss_batch`] applies on the host.
fn causal_targets(batch: &TokenBatch, vocab: usize) -> Result<Vec<i32>, NetworkError> {
    let ids = batch.ids();
    let mut targets = vec![-1i32; batch.rows()];
    for row in 0..batch.rows() {
        // -1 is what the kernel reads as "this row predicts nothing", so a
        // masked position needs no separate machinery on the device: it is
        // spelled the same way as padding.
        if !batch.predicts(row) {
            continue;
        }
        let id = ids[row + 1];
        if id as usize >= vocab {
            return Err(NetworkError::TokenOutOfRange {
                id,
                vocab_size: vocab,
            });
        }
        targets[row] = id as i32;
    }
    Ok(targets)
}

/// How many rows of logits to hold at once.
///
/// Small enough that the chunk fits comfortably alongside the activations,
/// large enough that the head's GEMMs still have a tall `M`. A 32k vocabulary
/// gives about 2000 rows, which cuBLAS is already happy with.
fn head_chunk_rows(rows: usize, vocab: usize, logit_bytes: usize) -> usize {
    #[cfg(test)]
    {
        let forced = CHUNK_ROWS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        if forced > 0 {
            return forced.min(rows.max(1));
        }
    }
    // NOTE: one fixed budget, not a tuned schedule. Make it a builder knob
    // if a model ever wants a different trade against activation memory.
    const BUDGET_BYTES: usize = 256 << 20;
    (BUDGET_BYTES / (vocab * logit_bytes)).clamp(1, rows.max(1))
}

/// Forces a chunk size so a test can drive the loop over several chunks.
///
/// A real vocabulary needs thousands of rows before the budget splits the
/// head, which no parity-sized model reaches. Results must not depend on the
/// chunk size, so a test that leaves this set only slows its neighbours down.
#[cfg(test)]
pub(crate) static CHUNK_ROWS_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// One block per row, sized for `ce_loss_grad`'s shared-memory reduction.
fn row_per_block(rows: usize) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig {
        grid_dim: (rows.max(1) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Blocks per launch for the kernels that grid-stride over rows.
///
/// A row count in the hundreds of thousands does not need a block each: the
/// cap keeps enough blocks to fill the device several times over while
/// cutting the number of blocks that contend for a shared gradient column.
const ROW_GRID_BLOCKS: usize = 2048;

/// One warp per row, eight rows per block.
fn warp_per_row_grid(rows: usize) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig {
        grid_dim: (rows.div_ceil(8).clamp(1, ROW_GRID_BLOCKS) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn row_grid(rows: usize) -> cudarc::driver::LaunchConfig {
    cudarc::driver::LaunchConfig {
        grid_dim: (rows.clamp(1, ROW_GRID_BLOCKS) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Shared memory for `rmsnorm_bwd`'s per-block scale-gradient accumulator, or
/// `None` for a width that will not fit in the 48 KiB a block gets by default.
fn rmsnorm_smem(cols: usize) -> Option<u32> {
    let bytes = cols.checked_mul(4)?;
    (bytes <= 48 << 10).then_some(bytes as u32)
}

/// The rest of the backward pass, once the gradient with respect to the final
/// normalization's output is known.
///
/// [`backward`] gets there from a host `dL/dlogits`; [`train_step`] gets there
/// from the fused loss, one chunk of rows at a time.
fn backward_from_final(
    model: &mut TransformerLm,
    context: &Arc<GpuContext>,
    cache: &GpuCache,
    grad_final: &CudaSlice<f32>,
) -> Result<(), NetworkError> {
    let gpu = Gpu { context };
    let rows = cache.stack.rows;
    let d_model = model.config.d_model;

    // Every RMSNorm scale gradient in the model, in one buffer: slot 0 is the
    // final norm, then two slots per block. See [`Gpu::accumulate_host_grad`]
    // for why they are not downloaded where they are produced.
    let mut grad_norms = gpu.zeros(norm_slots(model.blocks.len()) * d_model)?;
    let (grad_hidden, grad_hidden_act) = gpu.rmsnorm_backward(
        &cache.final_input,
        grad_final,
        &cache.final_weight,
        &cache.final_inverse_rms,
        &mut grad_norms.slice_mut(0..d_model),
        grad_final,
        false,
        stack_operand(&cache.stack, model.blocks.len().checked_sub(1)),
        rows,
        d_model,
    )?;
    let grad_hidden = backward_stack(
        &gpu,
        &mut model.blocks,
        &cache.stack,
        grad_hidden,
        grad_hidden_act,
        &mut grad_norms,
    )?;

    {
        let embedding = model.embedding.weight.device.as_mut().ok_or_else(|| {
            NetworkError::Cuda("the embedding table is not resident on the device".into())
        })?;
        // The scatter is a pile of atomic adds, so it needs real zeros under it.
        // A frozen table has no buffer to scatter into and nothing that would
        // read one.
        if !embedding.is_frozen() {
            embedding.clear_grad()?;
            gpu.scatter_negated(
                embedding.negated_grad_mut(),
                &grad_hidden,
                &cache.ids,
                rows,
                d_model,
            )?;
        }
    }

    // The one host round trip of the backward pass, after everything else is
    // enqueued.
    let values = gpu.download(&grad_norms)?;
    Gpu::accumulate_host_grad(&mut model.final_norm.weight, &values[..d_model]);
    accumulate_block_norms(&mut model.blocks, &values, d_model);
    Ok(())
}

/// Slots in the scale-gradient block: the final norm, then two per block.
fn norm_slots(blocks: usize) -> usize {
    1 + 2 * blocks
}

/// The `(attention_norm, feed_forward_norm)` slots of block `index`.
fn norm_slot(index: usize) -> (usize, usize) {
    (1 + 2 * index, 2 + 2 * index)
}

#[allow(clippy::too_many_arguments)]
fn backward_block(
    gpu: &Gpu<'_>,
    block: &mut TransformerBlock,
    cache: &BlockCache,
    grad_output: &CudaSlice<f32>,
    grad_output_act: Option<&Act>,
    output_operand: Option<bool>,
    batch: &GpuStack,
    grad_norms: &mut CudaSlice<f32>,
    index: usize,
) -> Result<(CudaSlice<f32>, Option<Act>), NetworkError> {
    let rows = batch.rows;
    let d_model = block.attention.d_model();
    let (attention_slot, feed_forward_slot) = norm_slot(index);

    let mut grad_normed = match block.feed_forward {
        FeedForward::SwiGlu(_) => gpu.uninit(rows * d_model)?,
        _ => gpu.zeros(rows * d_model)?,
    };
    match (&mut block.feed_forward, &cache.feed_forward) {
        (FeedForward::SwiGlu(ffn), FfnCache::Dense(ffn_cache)) => backward_swiglu(
            gpu,
            ffn,
            ffn_cache,
            &cache.feed_forward_normed,
            grad_output_act.expect("a dense feed forward was handed its operand"),
            &mut grad_normed,
            rows,
        )?,
        (FeedForward::Moe(moe), FfnCache::Moe(moe_cache)) => backward_moe(
            gpu,
            moe,
            moe_cache,
            &cache.feed_forward_normed,
            grad_output,
            &mut grad_normed,
            batch.valid_tokens,
            rows,
        )?,
        (FeedForward::Gelu(_), _) => return Err(unsupported("a GELU MLP")),
        _ => {
            return Err(NetworkError::Cuda(
                "the device cache does not match the layer that produced it".into(),
            ));
        }
    }

    // The residual passes the upstream gradient through untouched alongside
    // the branch gradient, so it rides in the norm's backward kernel.
    let (grad_residual, grad_residual_act) = gpu.rmsnorm_backward(
        &cache.residual,
        &grad_normed,
        &cache.feed_forward_weight,
        &cache.feed_forward_inverse_rms,
        &mut grad_norms.slice_mut(feed_forward_slot * d_model..(feed_forward_slot + 1) * d_model),
        grad_output,
        true,
        Some(cache.merged.is_narrow()),
        rows,
        d_model,
    )?;
    let grad_residual_act = grad_residual_act.expect("a copy was asked for");

    let grad_attention_normed = backward_attention(gpu, block, cache, &grad_residual_act, batch)?;

    let (grad_input, grad_input_act) = gpu.rmsnorm_backward(
        &cache.input,
        &grad_attention_normed,
        &cache.attention_weight,
        &cache.attention_inverse_rms,
        &mut grad_norms.slice_mut(attention_slot * d_model..(attention_slot + 1) * d_model),
        &grad_residual,
        true,
        output_operand,
        rows,
        d_model,
    )?;

    Ok((grad_input, grad_input_act))
}

/// Attention backward, head by head, each head one batched GEMM per operand.
///
/// RoPE is orthogonal, so its backward pass is the same rotation applied with
/// the opposite sign, and the `1/sqrt(head_dim)` scale rides in the alpha of
/// the two GEMMs that read the score gradient.
fn backward_attention(
    gpu: &Gpu<'_>,
    block: &mut TransformerBlock,
    cache: &BlockCache,
    grad_output: &Act,
    batch: &GpuStack,
) -> Result<CudaSlice<f32>, NetworkError> {
    let rows = batch.rows;
    let seq_len = batch.seq_len;
    let sequences = batch.sequences;
    let attention = &mut block.attention;
    let d_model = attention.d_model();
    let heads = attention.num_heads();
    let kv_heads = attention.num_kv_heads();
    let head_dim = attention.head_dim();
    let causal = attention.is_causal();
    let group = heads / kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let query_width = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let qkv_width = query_width + 2 * kv_width;
    let key_base = query_width;
    let value_base = query_width + kv_width;
    let block_size = seq_len * seq_len;

    // The output projection reads the upstream gradient twice; the norm that
    // produced it already wrote the operand copy both GEMMs read.
    let merged_narrow = cache.merged.is_narrow();
    debug_assert_eq!(grad_output.is_narrow(), merged_narrow);
    let grad_output_act = grad_output;
    // The fused backward kernels are the only readers that take it narrow,
    // and they run exactly when the forward pass wrote a narrow `merged`.
    let mut grad_merged = gpu.act(rows * query_width, merged_narrow)?;
    gpu.linear_packed_backward_input(
        &cache.output_weight.all(),
        d_model,
        query_width,
        &grad_output_act.all(),
        merged_narrow,
        merged_narrow,
        grad_merged.destination(),
        rows,
        0.0,
    )?;
    gpu.accumulate_projection_grad(
        &mut attention.output,
        &grad_output_act.all(),
        &cache.merged.all(),
        merged_narrow,
        rows,
    )?;

    // The fused path, where the device can take it: the same three kernels
    // that replaced the forward softmax replace six passes over two
    // `[seq_len, seq_len]` buffers and four batched GEMMs here.
    // A narrow `qkv` is one only the fused kernels can read, and they write
    // one in turn, so the forward pass's choice decides the path here.
    let fused = gpu.context.flash.as_ref().filter(|_| cache.qkv.is_narrow());
    // The backward kernels read `qkv` and write `grad_qkv` in the same layout,
    // so the gradient is narrow exactly when the forward projection was.
    let narrow_qkv = cache.qkv.is_narrow();
    let mut grad_qkv = gpu.act(rows * qkv_width, narrow_qkv)?;
    if let Some(flash) = fused {
        let shape = FlashShape {
            rows,
            seq_len,
            sequences,
            heads,
            group,
            qkv_width,
            query_width,
            key_base,
            value_base,
            causal,
        };
        let mut delta = gpu.uninit(heads * rows)?;
        // Every column of `grad_qkv` is written rather than accumulated: the
        // query gradient by the query-tile kernel, the key and value gradients
        // by the key-tile one, so there is nothing to clear first.
        gpu.flash_attention_backward(
            flash,
            &cache.qkv,
            &cache.merged,
            &grad_merged,
            &cache.log_sum_exp,
            &mut delta,
            &mut grad_qkv,
            shape,
            scale,
        )?;
    } else {
        // The forward pass kept only the log-sum-exp, so the probabilities are
        // rebuilt here: one batched GEMM per head for the scores, then one
        // exponential per element. The matrix lives for this block's backward pass
        // alone rather than for the whole depth of the model.
        let qkv = cache.qkv.wide();
        let wide_grad_merged = grad_merged.wide();
        let mut probabilities = gpu.uninit(heads * sequences * block_size)?;
        for head in 0..heads {
            let kv_base = (head / group) * head_dim;
            let query = qkv.slice(head * head_dim..);
            let key = qkv.slice(key_base + kv_base..);
            let mut scores = probabilities.slice_mut(head * sequences * block_size..);
            gemm_rhs_transposed_batched(
                gpu.context,
                &query,
                qkv_width,
                &key,
                qkv_width,
                &mut scores,
                seq_len,
                seq_len,
                seq_len,
                head_dim,
                scale,
                0.0,
                sequences,
                seq_len * qkv_width,
                seq_len * qkv_width,
                block_size,
            )?;
        }
        gpu.probs_from_lse(
            &mut probabilities,
            &cache.log_sum_exp,
            heads * sequences * seq_len,
            seq_len,
            causal,
        )?;

        // Every head overwrites its own slice of `grad_scores`, so the buffer does
        // not need clearing. `grad_values` does: query heads in a group accumulate
        // into the same key/value head.
        let mut grad_scores = gpu.uninit(heads * sequences * block_size)?;
        // The query slice is overwritten head by head, but the key and value
        // slices accumulate over every query head in a group, so the whole fused
        // buffer starts at zero.
        let mut grad_qkv = grad_qkv.wide_mut();
        gpu.context
            .stream
            .memset_zeros(&mut grad_qkv)
            .map_err(cuda_err("gradient clear"))?;
        for head in 0..heads {
            let kv_base = (head / group) * head_dim;
            let upstream = wide_grad_merged.slice(head * head_dim..);
            let value = qkv.slice(value_base + kv_base..);
            let mut scores = grad_scores.slice_mut(head * sequences * block_size..);
            gemm_rhs_transposed_batched(
                gpu.context,
                &upstream,
                query_width,
                &value,
                qkv_width,
                &mut scores,
                seq_len,
                seq_len,
                seq_len,
                head_dim,
                1.0,
                0.0,
                sequences,
                seq_len * query_width,
                seq_len * qkv_width,
                block_size,
            )?;

            let head_probabilities = probabilities.slice(head * sequences * block_size..);
            let mut grad_value = grad_qkv.slice_mut(value_base + kv_base..);
            gemm_lhs_transposed_batched(
                gpu.context,
                &head_probabilities,
                seq_len,
                &upstream,
                query_width,
                &mut grad_value,
                qkv_width,
                seq_len,
                seq_len,
                head_dim,
                1.0,
                1.0,
                sequences,
                block_size,
                seq_len * query_width,
                seq_len * qkv_width,
            )?;
        }

        gpu.softmax_backward(
            &mut grad_scores,
            &probabilities,
            heads * sequences * seq_len,
            seq_len,
            causal,
        )?;

        for head in 0..heads {
            let kv_base = (head / group) * head_dim;
            let scores = grad_scores.slice(head * sequences * block_size..);
            let key = qkv.slice(key_base + kv_base..);
            let mut grad_query = grad_qkv.slice_mut(head * head_dim..);
            gemm_plain_batched(
                gpu.context,
                &scores,
                seq_len,
                &key,
                qkv_width,
                &mut grad_query,
                qkv_width,
                seq_len,
                head_dim,
                seq_len,
                scale,
                0.0,
                sequences,
                block_size,
                seq_len * qkv_width,
                seq_len * qkv_width,
            )?;

            let query = qkv.slice(head * head_dim..);
            let mut grad_key = grad_qkv.slice_mut(key_base + kv_base..);
            gemm_lhs_transposed_batched(
                gpu.context,
                &scores,
                seq_len,
                &query,
                qkv_width,
                &mut grad_key,
                qkv_width,
                seq_len,
                seq_len,
                head_dim,
                scale,
                1.0,
                sequences,
                block_size,
                seq_len * qkv_width,
                seq_len * qkv_width,
            )?;
        }
    }

    let tables = gpu.rope_tables(&attention.rope)?;
    gpu.rope(
        &mut grad_qkv,
        &tables,
        rows,
        heads,
        head_dim,
        seq_len,
        -1.0,
        qkv_width,
        0,
    )?;
    gpu.rope(
        &mut grad_qkv,
        &tables,
        rows,
        kv_heads,
        head_dim,
        seq_len,
        -1.0,
        qkv_width,
        key_base,
    )?;

    // The fused kernels already wrote a narrow gradient; the fallback path's
    // batched GEMMs produce FP32, so that one still pays for a cast.
    let narrow = cache.qkv_weights.is_narrow();
    let grad_qkv = if narrow == narrow_qkv {
        grad_qkv
    } else {
        gpu.narrowed(&grad_qkv.wide(), rows * qkv_width, narrow)?
    };
    let mut grad_input = gpu.uninit(rows * d_model)?;
    gpu.linear_packed_backward_input(
        &cache.qkv_weights.all(),
        qkv_width,
        d_model,
        &grad_qkv.all(),
        narrow,
        false,
        &mut grad_input,
        rows,
        0.0,
    )?;
    gpu.accumulate_packed_projection_grad(
        &mut [
            &mut attention.query,
            &mut attention.key,
            &mut attention.value,
        ],
        &grad_qkv.all(),
        &cache.attention_normed.all(),
        narrow,
        rows,
    )?;

    Ok(grad_input)
}

/// Accumulates the three projection gradients and adds `dL/dinput` into
/// `grad_input`, which the caller has already zeroed.
pub(crate) fn backward_swiglu(
    gpu: &Gpu<'_>,
    ffn: &mut SwiGlu,
    cache: &SwiGluCache,
    input: &Act,
    grad_output: &Act,
    grad_input: &mut CudaSlice<f32>,
    rows: usize,
) -> Result<(), NetworkError> {
    let width = ffn.d_ff();
    let inner = ffn.d_model();
    let narrow = input.is_narrow();
    debug_assert_eq!(grad_output.is_narrow(), narrow);
    let grad_output_act = grad_output;
    let down = cache
        .down
        .as_ref()
        .ok_or_else(|| NetworkError::Cuda("the SwiGLU cache has no down projection".into()))?;
    let mut grad_hidden = gpu.uninit(rows * width)?;
    gpu.linear_packed_backward_input(
        &down.all(),
        inner,
        width,
        &grad_output_act.all(),
        narrow,
        false,
        &mut grad_hidden,
        rows,
        0.0,
    )?;
    gpu.accumulate_projection_grad(
        &mut ffn.down,
        &grad_output_act.all(),
        &cache.hidden.all(),
        narrow,
        rows,
    )?;

    let grad_gate_up = gpu.swiglu_backward(&cache.gate_up, &grad_hidden, rows, width, narrow)?;

    // The fused projection *writes* `grad_input`, and in the MoE layer the
    // shared expert runs before anything scatters into the same buffer. That is
    // what lets the caller leave it uninitialized.
    gpu.linear_packed_backward_input(
        &cache.weights.all(),
        2 * width,
        inner,
        &grad_gate_up.all(),
        narrow,
        false,
        grad_input,
        rows,
        0.0,
    )?;
    gpu.accumulate_packed_projection_grad(
        &mut [&mut ffn.gate, &mut ffn.up],
        &grad_gate_up.all(),
        &input.all(),
        narrow,
        rows,
    )?;
    Ok(())
}

/// MoE backward.
///
/// Everything is one launch or one GEMM per expert over the whole batch: the
/// gate gradients are a row-wise dot product, the expert gradients run on the
/// same grouped buffers the forward pass built, and the router's top-k
/// renormalization, softmax, load-balancing loss and z-loss are three kernels
/// over `[rows, num_experts]`.
#[allow(clippy::too_many_arguments)]
fn backward_moe(
    gpu: &Gpu<'_>,
    moe: &mut MoeLayer,
    cache: &MoeGpuCache,
    input: &Act,
    grad_output: &CudaSlice<f32>,
    grad_input: &mut CudaSlice<f32>,
    valid_tokens: usize,
    rows: usize,
) -> Result<(), NetworkError> {
    let experts = moe.config.num_experts;
    let top_k = moe.config.experts_per_token;
    let d_model = moe.d_model();
    let width = moe.config.d_ff;
    let routed = cache.routed;

    if let (Some(shared), Some(shared_cache)) = (moe.shared.as_mut(), &cache.shared) {
        // The norm above a routed layer stays FP32, so it did not leave an
        // operand copy for the shared expert's GEMMs; it is made here.
        let upstream = gpu.narrowed(&grad_output.slice(..), rows * d_model, input.is_narrow())?;
        backward_swiglu(
            gpu,
            shared,
            shared_cache,
            input,
            &upstream,
            grad_input,
            rows,
        )?;
    }

    let mut grad_gates = gpu.zeros(rows * top_k)?;
    if routed > 0 {
        unsafe {
            gpu.context
                .stream
                .launch_builder(&gpu.context.model.row_dot)
                .arg(&mut grad_gates)
                .arg(&cache.slot_of)
                .arg(grad_output)
                .arg(&cache.token_of)
                .arg(&cache.output)
                .arg(&(routed as i32))
                .arg(&(d_model as i32))
                .launch(warp_per_row_grid(routed))
                .map_err(cuda_err("gate gradient kernel"))?;
        }

        let narrow = cache.input.is_narrow();
        let mut grad_expert_output = gpu.act(routed * d_model, narrow)?;
        gpu.gather_scaled(
            &mut grad_expert_output,
            &grad_output.slice(..),
            &cache.token_of,
            &cache.gates,
            true,
            routed,
            d_model,
        )?;

        let down = cache.expert.down.as_ref().ok_or_else(|| {
            NetworkError::Cuda("the routed feed-forward cache has no down projection".into())
        })?;
        let down_stride = d_model * width;
        let mut grad_hidden = gpu.uninit(routed * width)?;
        for (expert, module) in moe.experts.iter_mut().enumerate() {
            let count = cache.counts[expert];
            if count == 0 {
                continue;
            }
            let offset = cache.offsets[expert];
            let upstream = grad_expert_output.at(offset * d_model);
            let mut grad = grad_hidden.slice_mut(offset * width..);
            gpu.linear_packed_backward_input(
                &down.at(expert * down_stride),
                d_model,
                width,
                &upstream,
                narrow,
                false,
                &mut grad,
                count,
                0.0,
            )?;
            gpu.accumulate_projection_grad(
                &mut module.down,
                &upstream,
                &cache.expert.hidden.at(offset * width),
                narrow,
                count,
            )?;
        }

        let grad_gate_up =
            gpu.swiglu_backward(&cache.expert.gate_up, &grad_hidden, routed, width, narrow)?;

        let packed_stride = 2 * width * d_model;
        let mut grad_expert_input = gpu.uninit(routed * d_model)?;
        for (expert, module) in moe.experts.iter_mut().enumerate() {
            let count = cache.counts[expert];
            if count == 0 {
                continue;
            }
            let offset = cache.offsets[expert];
            let rows_in = cache.input.at(offset * d_model);
            let upstream = grad_gate_up.at(offset * 2 * width);
            let weights = cache.expert.weights.at(expert * packed_stride);
            let mut grad = grad_expert_input.slice_mut(offset * d_model..);
            gpu.linear_packed_backward_input(
                &weights,
                2 * width,
                d_model,
                &upstream,
                narrow,
                false,
                &mut grad,
                count,
                0.0,
            )?;
            gpu.accumulate_packed_projection_grad(
                &mut [&mut module.gate, &mut module.up],
                &upstream,
                &rows_in,
                narrow,
                count,
            )?;
        }

        gpu.scatter(
            grad_input,
            &grad_expert_input,
            &cache.token_of,
            routed,
            d_model,
        )?;
    }

    // Gate gradients flow back through the top-k renormalization, then through
    // the softmax. The load-balancing loss adds a term on the probabilities;
    // the z-loss acts on the logits directly.
    let mut grad_probabilities = gpu.zeros(rows * experts)?;
    unsafe {
        gpu.context
            .stream
            .launch_builder(&gpu.context.model.grad_probabilities)
            .arg(&mut grad_probabilities)
            .arg(&grad_gates)
            .arg(&cache.probabilities)
            .arg(&cache.expert_of)
            .arg(&(rows as i32))
            .arg(&(experts as i32))
            .arg(&(top_k as i32))
            .launch(cfg(rows))
            .map_err(cuda_err("router gate gradient kernel"))?;
    }

    if moe.config.aux_loss_weight != 0.0 && valid_tokens > 0 {
        let slots = (valid_tokens * top_k) as f32;
        let loads: Vec<f32> = cache
            .counts
            .iter()
            .map(|&count| count as f32 / slots)
            .collect();
        let loads = gpu.upload(&loads)?;
        let scale = moe.config.aux_loss_weight * experts as f32 / valid_tokens as f32;
        unsafe {
            gpu.context
                .stream
                .launch_builder(&gpu.context.model.aux_grad)
                .arg(&mut grad_probabilities)
                .arg(&loads)
                .arg(&cache.expert_of)
                .arg(&(rows as i32))
                .arg(&(experts as i32))
                .arg(&(top_k as i32))
                .arg(&scale)
                .launch(cfg(rows * experts))
                .map_err(cuda_err("load-balancing gradient kernel"))?;
        }
    }

    let z_factor = if moe.config.router_z_loss_weight != 0.0 {
        2.0 * moe.config.router_z_loss_weight / valid_tokens.max(1) as f32
    } else {
        0.0
    };
    let mut grad_logits = gpu.zeros(rows * experts)?;
    unsafe {
        gpu.context
            .stream
            .launch_builder(&gpu.context.model.grad_logits)
            .arg(&mut grad_logits)
            .arg(&grad_probabilities)
            .arg(&cache.probabilities)
            .arg(&cache.log_sum_exp)
            .arg(&cache.expert_of)
            .arg(&(rows as i32))
            .arg(&(experts as i32))
            .arg(&(top_k as i32))
            .arg(&z_factor)
            .launch(cfg(rows))
            .map_err(cuda_err("router logit gradient kernel"))?;
    }

    gpu.linear_backward_input(
        device_of(&moe.router.projection)?,
        &grad_logits,
        grad_input,
        rows,
        1.0,
    )?;
    gpu.accumulate_weight_grad(
        device_of_mut(&mut moe.router.projection)?,
        &grad_logits,
        &input.wide(),
        rows,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_training::cuda_doctor;
    use crate::norm::RmsNorm;

    /// Same contract as the other CUDA tests: no device means the parity tests
    /// report success without running, a broken device is a failure.
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

    fn ramp(len: usize) -> Vec<f32> {
        ramp_from(len, 0)
    }

    /// A second, differently phased ramp, so an operand and its upstream
    /// gradient are not the same numbers.
    fn ramp_from(len: usize, offset: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i + offset) * 37) % 23) as f32 / 11.0 - 1.0)
            .collect()
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

    #[test]
    fn an_allocation_that_cannot_fit_says_out_of_memory_rather_than_cuda() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let gpu = Gpu { context: &context };
        // Four tebibytes: larger than any card, so the driver refuses it
        // without disturbing whatever else is resident.
        match gpu.zeros(1 << 40) {
            Err(NetworkError::CudaOutOfMemory {
                requested_mib,
                free_mib,
            }) => {
                assert_eq!(requested_mib, 4 * 1024 * 1024);
                assert!(free_mib > 0, "a working device reports some free memory");
            }
            other => panic!("expected an out-of-memory error, got {other:?}"),
        }
    }

    #[test]
    fn the_rmsnorm_kernels_match_the_host_or_skip_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let gpu = Gpu { context: &context };
        let (rows, cols) = (5, 8);

        let mut norm = RmsNorm::new(cols, 1e-6);
        norm.weight.value.data = ramp(cols).iter().map(|v| 1.0 + v * 0.25).collect();
        let input = Matrix::from_vec(rows, cols, ramp(rows * cols));
        let grad_output = Matrix::from_vec(rows, cols, ramp_from(rows * cols, 3));

        let weight = gpu.upload(&norm.weight.value.data).unwrap();
        let device_input = gpu.upload(&input.data).unwrap();
        let (output, inverse, _) = gpu
            .rmsnorm(&device_input, &weight, rows, cols, norm.eps, false, false)
            .unwrap();
        assert_close(
            "rmsnorm forward",
            &gpu.download_act(&output).unwrap(),
            &norm.forward(&input).data,
            1e-5,
        );

        let device_grad = gpu.upload(&grad_output.data).unwrap();
        let mut grad_weight = gpu.zeros(cols).unwrap();
        let grad_input = gpu
            .rmsnorm_backward(
                &device_input,
                &device_grad,
                &weight,
                &inverse,
                &mut grad_weight.slice_mut(..),
                &device_grad,
                false,
                None,
                rows,
                cols,
            )
            .unwrap()
            .0;

        let host_grad_input = norm.backward(&input, &grad_output);
        assert_close(
            "rmsnorm grad_input",
            &gpu.download(&grad_input).unwrap(),
            &host_grad_input.data,
            1e-5,
        );
        assert_close(
            "rmsnorm grad_weight",
            &gpu.download(&grad_weight).unwrap(),
            &norm.weight.grad.data,
            1e-5,
        );
    }

    #[test]
    fn the_rope_kernel_matches_the_host_or_skips_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let gpu = Gpu { context: &context };
        let (sequences, seq_len, heads, head_dim) = (2, 3, 2, 4);
        let rows = sequences * seq_len;
        let width = heads * head_dim;

        let rope = Rope::new(head_dim, 16, 10_000.0).unwrap();
        let tables = gpu.rope_tables(&rope).unwrap();
        // The kernel rotates in place in whichever precision the buffer holds,
        // so both layouts run here; BF16 keeps eight mantissa bits, hence the
        // wider tolerance.
        for (narrow, tolerance) in [(false, 1e-5), (true, 1e-2)] {
            let mut host = Matrix::from_vec(rows, width, ramp(rows * width));
            let wide = gpu.upload(&host.data).unwrap();
            let mut device = gpu.narrowed(&wide.slice(..), rows * width, narrow).unwrap();

            gpu.rope(
                &mut device,
                &tables,
                rows,
                heads,
                head_dim,
                seq_len,
                1.0,
                width,
                0,
            )
            .unwrap();
            rope.apply_batched(&mut host, heads, seq_len).unwrap();
            assert_close(
                "rope",
                &gpu.download_act(&device).unwrap(),
                &host.data,
                tolerance,
            );

            // The inverse is the same kernel with the opposite sign, which is
            // what the backward pass uses.
            gpu.rope(
                &mut device,
                &tables,
                rows,
                heads,
                head_dim,
                seq_len,
                -1.0,
                width,
                0,
            )
            .unwrap();
            rope.apply_inverse_batched(&mut host, heads, seq_len)
                .unwrap();
            assert_close(
                "rope inverse",
                &gpu.download_act(&device).unwrap(),
                &host.data,
                tolerance,
            );
        }
    }

    #[test]
    fn the_causal_probability_kernels_mask_and_normalize_or_skip_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let gpu = Gpu { context: &context };
        let (sequences, seq_len) = (2, 4);
        let rows = sequences * seq_len;

        let scores = ramp(rows * seq_len);
        // The fused forward pass leaves exactly this statistic behind.
        let log_sum_exp: Vec<f32> = (0..rows)
            .map(|row| {
                let source = &scores[row * seq_len..row * seq_len + row % seq_len + 1];
                let max = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                max + source.iter().map(|v| (v - max).exp()).sum::<f32>().ln()
            })
            .collect();
        let mut device = gpu.upload(&scores).unwrap();
        let device_lse = gpu.upload(&log_sum_exp).unwrap();
        gpu.probs_from_lse(&mut device, &device_lse, rows, seq_len, true)
            .unwrap();
        let probabilities = gpu.download(&device).unwrap();

        for row in 0..rows {
            let visible = row % seq_len + 1;
            let slice = &probabilities[row * seq_len..(row + 1) * seq_len];

            let mut expected = vec![0.0f32; seq_len];
            let source = &scores[row * seq_len..row * seq_len + visible];
            let max = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total: f32 = source.iter().map(|v| (v - max).exp()).sum();
            for (slot, &value) in expected.iter_mut().zip(source) {
                *slot = (value - max).exp() / total;
            }

            assert_close(
                &format!("causal probabilities row {row}"),
                slice,
                &expected,
                1e-6,
            );
            // Everything past the diagonal is masked, not merely small.
            assert!(slice[visible..].iter().all(|&p| p == 0.0));
        }

        // The backward pass of a softmax whose upstream gradient is constant is
        // zero, which is a check the masking cannot pass by accident.
        let mut grad = gpu.upload(&vec![1.0; rows * seq_len]).unwrap();
        gpu.softmax_backward(&mut grad, &device, rows, seq_len, true)
            .unwrap();
        assert_close(
            "causal softmax backward",
            &gpu.download(&grad).unwrap(),
            &vec![0.0; rows * seq_len],
            1e-6,
        );
    }

    #[test]
    fn the_swiglu_kernel_matches_the_host_formula_or_skips_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let gpu = Gpu { context: &context };
        let (rows, width) = (4, 4);
        let len = rows * width;

        let gate = ramp(len);
        let up = ramp_from(len, 5);
        // Gate and up interleave by row, which is the layout the fused
        // projection produces.
        let mut fused = Vec::with_capacity(2 * len);
        for row in 0..rows {
            fused.extend_from_slice(&gate[row * width..(row + 1) * width]);
            fused.extend_from_slice(&up[row * width..(row + 1) * width]);
        }
        let gate_up = gpu.upload(&fused).unwrap();
        let hidden = gpu.swiglu(&gate_up, rows, width, false).unwrap();

        let silu = |x: f32| x / (1.0 + (-x).exp());
        let expected: Vec<f32> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
        assert_close(
            "swiglu",
            &gpu.download_act(&hidden).unwrap(),
            &expected,
            1e-6,
        );

        let grad_hidden = gpu.upload(&vec![1.0; len]).unwrap();
        let grad_fused = gpu
            .swiglu_backward(&gate_up, &grad_hidden, rows, width, false)
            .unwrap();
        let grad_fused = gpu.download_act(&grad_fused).unwrap();
        let mut grad_gate = Vec::with_capacity(len);
        let mut grad_up = Vec::with_capacity(len);
        for row in 0..rows {
            let base = row * 2 * width;
            grad_gate.extend_from_slice(&grad_fused[base..base + width]);
            grad_up.extend_from_slice(&grad_fused[base + width..base + 2 * width]);
        }

        // Compared against a central difference of the same formula.
        let epsilon = 1e-3;
        let numeric: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| (silu(g + epsilon) * u - silu(g - epsilon) * u) / (2.0 * epsilon))
            .collect();
        assert_close("swiglu grad_gate", &grad_gate, &numeric, 2e-3);
        assert_close(
            "swiglu grad_up",
            &grad_up,
            &gate.iter().map(|&g| silu(g)).collect::<Vec<_>>(),
            1e-6,
        );
    }

    #[test]
    fn the_router_kernels_select_and_renormalize_or_skip_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let gpu = Gpu { context: &context };
        let (rows, experts, top_k) = (4, 5, 2);

        let logits = ramp(rows * experts);
        let device_logits = gpu.upload(&logits).unwrap();
        let mut probabilities = gpu.zeros(rows * experts).unwrap();
        let mut log_sum_exp = gpu.zeros(rows).unwrap();
        unsafe {
            gpu.context
                .stream
                .launch_builder(&gpu.context.model.softmax_lse)
                .arg(&mut probabilities)
                .arg(&mut log_sum_exp)
                .arg(&device_logits)
                .arg(&(rows as i32))
                .arg(&(experts as i32))
                .launch(cfg(rows))
                .unwrap();
        }

        // The last row is padding, which must be routed nowhere.
        let flags: Vec<i32> = (0..rows).map(|row| i32::from(row + 1 < rows)).collect();
        let valid = gpu.upload_flags(&flags).unwrap();
        let mut expert_of = gpu.context.stream.alloc_zeros::<i32>(rows * top_k).unwrap();
        let mut gate_of = gpu.zeros(rows * top_k).unwrap();
        unsafe {
            gpu.context
                .stream
                .launch_builder(&gpu.context.model.topk_gate)
                .arg(&mut expert_of)
                .arg(&mut gate_of)
                .arg(&probabilities)
                .arg(&valid)
                .arg(&(rows as i32))
                .arg(&(experts as i32))
                .arg(&(top_k as i32))
                .launch(cfg(rows))
                .unwrap();
        }

        let probabilities = gpu.download(&probabilities).unwrap();
        let log_sum_exp = gpu.download(&log_sum_exp).unwrap();
        let assigned = gpu.download_signed(&expert_of).unwrap();
        let gates = gpu.download(&gate_of).unwrap();

        for row in 0..rows {
            let slice = &probabilities[row * experts..(row + 1) * experts];
            let sum: f32 = slice.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {row} softmax sums to {sum}");

            let source = &logits[row * experts..(row + 1) * experts];
            let max = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let expected = max + source.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
            assert!((log_sum_exp[row] - expected).abs() < 1e-5);

            let selected = &assigned[row * top_k..(row + 1) * top_k];
            if row + 1 == rows {
                assert!(selected.iter().all(|&expert| expert < 0), "padding routed");
                assert!(
                    gates[row * top_k..(row + 1) * top_k]
                        .iter()
                        .all(|&g| g == 0.0)
                );
                continue;
            }

            // The selected experts are the top-k, and their gates renormalize.
            let mut order: Vec<usize> = (0..experts).collect();
            order.sort_by(|&a, &b| slice[b].total_cmp(&slice[a]));
            assert_eq!(
                selected.iter().map(|&e| e as usize).collect::<Vec<_>>(),
                order[..top_k],
                "row {row} top-{top_k}"
            );
            let total: f32 = selected.iter().map(|&e| slice[e as usize]).sum();
            for (rank, &expert) in selected.iter().enumerate() {
                let expected = slice[expert as usize] / total;
                assert!((gates[row * top_k + rank] - expected).abs() < 1e-6);
            }
        }
    }

    /// The shape of one attention layer, so the host reference and the device
    /// path under test are described once instead of by eleven loose arguments.
    #[derive(Clone, Copy)]
    struct AttentionShape {
        sequences: usize,
        seq_len: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        scale: f32,
    }

    /// A plain quadratic reference for causal grouped-query attention. Returns
    /// the merged output and the probability matrix laid out the way the
    /// backward GEMMs read it, `[head][sequence][query, key]`.
    fn host_attention(
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        shape: AttentionShape,
    ) -> (Vec<f32>, Vec<f32>) {
        let AttentionShape {
            sequences,
            seq_len,
            heads,
            kv_heads,
            head_dim,
            scale,
        } = shape;
        let (group, query_width, kv_width) =
            (heads / kv_heads, heads * head_dim, kv_heads * head_dim);
        let mut out = vec![0.0; queries.len()];
        let mut probabilities = vec![0.0; heads * sequences * seq_len * seq_len];
        for s in 0..sequences {
            for h in 0..heads {
                let kv = h / group;
                for i in 0..seq_len {
                    let q = ((s * seq_len + i) * query_width) + h * head_dim;
                    let mut row = vec![0.0f32; i + 1];
                    for (j, p) in row.iter_mut().enumerate() {
                        let k = ((s * seq_len + j) * kv_width) + kv * head_dim;
                        *p = (0..head_dim)
                            .map(|d| queries[q + d] * keys[k + d])
                            .sum::<f32>()
                            * scale;
                    }
                    let top = row.iter().cloned().fold(f32::MIN, f32::max);
                    let total: f32 = row.iter().map(|p| (p - top).exp()).sum();
                    let base = ((h * sequences + s) * seq_len + i) * seq_len;
                    for (j, p) in row.iter().enumerate() {
                        let weight = (p - top).exp() / total;
                        probabilities[base + j] = weight;
                        let v = ((s * seq_len + j) * kv_width) + kv * head_dim;
                        for d in 0..head_dim {
                            out[q + d] += weight * values[v + d];
                        }
                    }
                }
            }
        }
        (out, probabilities)
    }

    #[test]
    fn the_attention_forward_matches_the_host_or_skips_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let gpu = Gpu { context: &context };
        let shape = AttentionShape {
            sequences: 2,
            seq_len: 11,
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            scale: 1.0 / 8.0f32.sqrt(),
        };
        let rows = shape.sequences * shape.seq_len;
        let wiggle = |len: usize, offset: usize| -> Vec<f32> {
            (0..len)
                .map(|i| ((i + offset) as f32 * 0.37).sin() * 0.8)
                .collect()
        };
        let queries = wiggle(rows * shape.heads * shape.head_dim, 0);
        let keys = wiggle(rows * shape.kv_heads * shape.head_dim, 5);
        let values = wiggle(rows * shape.kv_heads * shape.head_dim, 11);
        let (host_out, host_probabilities) = host_attention(&queries, &keys, &values, shape);

        let device_queries = gpu.upload(&queries).unwrap();
        let device_keys = gpu.upload(&keys).unwrap();
        let device_values = gpu.upload(&values).unwrap();

        let (group, block_size) = (shape.heads / shape.kv_heads, shape.seq_len * shape.seq_len);
        let (query_width, kv_width) = (
            shape.heads * shape.head_dim,
            shape.kv_heads * shape.head_dim,
        );
        let scores_of = |probabilities: &mut CudaSlice<f32>| {
            for head in 0..shape.heads {
                let query = device_queries.slice(head * shape.head_dim..);
                let key = device_keys.slice((head / group) * shape.head_dim..);
                let mut scores = probabilities.slice_mut(head * shape.sequences * block_size..);
                gemm_rhs_transposed_batched(
                    gpu.context,
                    &query,
                    query_width,
                    &key,
                    kv_width,
                    &mut scores,
                    shape.seq_len,
                    shape.seq_len,
                    shape.seq_len,
                    shape.head_dim,
                    shape.scale,
                    0.0,
                    shape.sequences,
                    shape.seq_len * query_width,
                    shape.seq_len * kv_width,
                    block_size,
                )
                .unwrap();
            }
        };

        let mut probabilities = gpu
            .zeros(shape.heads * shape.sequences * block_size)
            .unwrap();
        scores_of(&mut probabilities);
        let mut lse = gpu.zeros(shape.heads * rows).unwrap();
        gpu.softmax_lse(
            &mut probabilities,
            &mut lse,
            shape.heads * shape.sequences * shape.seq_len,
            shape.seq_len,
            true,
        )
        .unwrap();
        assert_close(
            "causal softmax probabilities",
            &gpu.download(&probabilities).unwrap(),
            &host_probabilities,
            2e-6,
        );

        let mut merged = gpu.zeros(queries.len()).unwrap();
        for head in 0..shape.heads {
            let head_probabilities = probabilities.slice(head * shape.sequences * block_size..);
            let value = device_values.slice((head / group) * shape.head_dim..);
            let mut head_merged = merged.slice_mut(head * shape.head_dim..);
            gemm_plain_batched(
                gpu.context,
                &head_probabilities,
                shape.seq_len,
                &value,
                kv_width,
                &mut head_merged,
                query_width,
                shape.seq_len,
                shape.head_dim,
                shape.seq_len,
                1.0,
                0.0,
                shape.sequences,
                block_size,
                shape.seq_len * kv_width,
                shape.seq_len * query_width,
            )
            .unwrap();
        }
        assert_close(
            "attention forward",
            &gpu.download(&merged).unwrap(),
            &host_out,
            2e-5,
        );

        // The backward pass throws the probabilities away and rebuilds them
        // from the scores and the log-sum-exp, so the rebuild has to reproduce
        // the same softmax.
        scores_of(&mut probabilities);
        gpu.probs_from_lse(
            &mut probabilities,
            &lse,
            shape.heads * shape.sequences * shape.seq_len,
            shape.seq_len,
            true,
        )
        .unwrap();
        assert_close(
            "probabilities rebuilt from the log-sum-exp",
            &gpu.download(&probabilities).unwrap(),
            &host_probabilities,
            2e-6,
        );
    }
}
