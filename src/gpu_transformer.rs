//! CUDA acceleration for the transformer path.
//!
//! Scope: the device *plumbing*, plus the per-module acceleration the cached
//! decode path uses. Every projection ([`Linear`](crate::param::Linear)), the
//! tied output projection, the embedding gather and scatter, and the two
//! attention score matmuls run on the device here, one host round trip per
//! call, because the host operation that follows needs the values.
//!
//! Training does not take that path. [`crate::gpu_model`] keeps the
//! activations device-resident for a whole batched forward and backward and
//! runs RoPE, RMSNorm, the causal softmax, SwiGLU and the MoE router as fused
//! kernels, using the [`GpuContext`], [`DeviceParam`] and GEMM helpers defined
//! below. What is left in this module is decode, where one token at a time
//! makes a fused path pointless.
//!
//! Two conventions are worth knowing before reading further.
//!
//! * Device gradients are stored **negated**, as `-dL/dw`. The `adam` and `sgd`
//!   PTX kernels are shared with [`crate::cuda_training`], which serves
//!   [`Network`](crate::network::Network) and its "add the gradient" update
//!   rule. Negating in the gradient GEMM's `alpha` costs nothing and lets both
//!   paths use one kernel.
//! * A device operation cannot return an error: `Linear::forward` and friends
//!   are infallible by signature. Instead the context is *poisoned*, the
//!   operation yields zeros, and the next fallible boundary
//!   ([`TransformerLm::forward_train`](crate::transformer::TransformerLm::forward_train),
//!   `backward`, `train_step`, `sync_from_device`) returns the recorded
//!   [`NetworkError::Cuda`]. Nothing falls back to the CPU silently.

use crate::activations::softmax;
use crate::cuda_training::{cfg, cuda_err, device_context};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::optimizers::Optimizer;
use cudarc::cublas::{
    CudaBlas, Gemm, GemmConfig, StridedBatchedConfig, result as cublas, sys as cublas_sys,
    sys::cublasOperation_t,
};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, CudaView, DevicePtr, DevicePtrMut, PushKernelArg,
};
use std::any::TypeId;
use std::sync::{Arc, Mutex};

/// One device, its stream, cuBLAS handle and the kernels the transformer uses.
///
/// Shared by every parameter of one model, so a session uploads weights once
/// and keeps them resident until [`TransformerLm::to_cpu`](crate::transformer::TransformerLm::to_cpu).
pub struct GpuContext {
    pub device: usize,
    pub(crate) stream: Arc<CudaStream>,
    pub(crate) blas: CudaBlas,
    /// Set by [`GpuContext::with_precision`]. Read by the training step, which
    /// runs the LM head in bf16 when it is on.
    pub(crate) mixed_precision: bool,
    /// Set by [`GpuContext::half_accumulate`]. Narrow operands are then FP16
    /// rather than BF16 and the accumulator is FP16 too.
    pub(crate) half_accumulate: bool,
    adam: CudaFunction,
    sgd: CudaFunction,
    scale: CudaFunction,
    gather: CudaFunction,
    scatter: CudaFunction,
    /// The fused kernels the device-resident training path uses.
    pub(crate) model: crate::gpu_model::ModelKernels,
    /// The fused attention kernel, present only on Ampere and later. `None`
    /// leaves attention on the three-kernel cuBLAS path.
    pub(crate) flash: Option<crate::cuda_flash::FlashKernels>,
    /// RoPE tables, uploaded on first use and shared by every layer.
    pub(crate) rope: Mutex<Option<Arc<crate::gpu_model::DeviceRope>>>,
    /// First CUDA failure seen by an infallible operation.
    poison: Mutex<Option<String>>,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("device", &self.device)
            .finish()
    }
}

impl GpuContext {
    /// Fails closed: no device, no cuBLAS or no kernels means an error, never a
    /// quiet CPU fallback.
    pub fn new(device: usize) -> Result<Arc<Self>, NetworkError> {
        Self::with_precision(device, false)
    }

    /// A context whose narrow GEMMs read FP16 operands and accumulate in FP16.
    ///
    /// The image path uses it. A consumer Ampere card runs its tensor cores at
    /// half rate when the accumulator is FP32, and FP16's eleven mantissa bits
    /// pay back what the narrow accumulator loses, so the same arithmetic is
    /// about 1.7x quicker for the same measured error. Training does not use
    /// it: BF16's exponent range is what keeps gradients from flushing to
    /// zero, and that is worth more there than the rate.
    pub(crate) fn half_accumulate(device: usize) -> Result<Arc<Self>, NetworkError> {
        let mut context = Self::with_precision(device, false)?;
        Arc::get_mut(&mut context)
            .expect("a context this function just built is unshared")
            .half_accumulate = true;
        Ok(context)
    }

    /// As [`GpuContext::new`], but `mixed_precision` lets cuBLAS run its GEMMs
    /// on the tensor cores in TF32. See
    /// [`TransformerBuilder::mixed_precision`](crate::transformer::TransformerBuilder::mixed_precision)
    /// for what that costs in accuracy.
    pub fn with_precision(device: usize, mixed_precision: bool) -> Result<Arc<Self>, NetworkError> {
        let (context, module) = device_context(device)?;
        // The stream holds its own reference to the context, so the context is
        // not kept here; the module is cached per process by `device_context`.
        let stream = context.new_stream().map_err(cuda_err("stream creation"))?;
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_err("cuBLAS initialization"))?;
        if mixed_precision {
            // Storage and accumulation stay FP32; only the multiplier inputs
            // are rounded to TF32's 10-bit mantissa inside the tensor cores.
            let status = unsafe {
                cudarc::cublas::sys::cublasSetMathMode(
                    *blas.handle(),
                    cudarc::cublas::sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                )
            };
            if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(NetworkError::Cuda(format!(
                    "cuBLAS rejected the TF32 math mode: {status:?}"
                )));
            }
        }
        let get = |name: &str| {
            module
                .load_function(name)
                .map_err(cuda_err("CUDA kernel lookup"))
        };
        Ok(Arc::new(Self {
            device,
            stream,
            blas,
            mixed_precision,
            half_accumulate: false,
            adam: get("adam")?,
            sgd: get("sgd")?,
            scale: get("scale_inplace")?,
            gather: get("gather_rows")?,
            scatter: get("scatter_rows_neg")?,
            model: crate::gpu_model::ModelKernels::load(&module)?,
            flash: crate::cuda_flash::flash_kernels(device, &context),
            rope: Mutex::new(None),
            poison: Mutex::new(None),
        }))
    }

    /// Records the first failure of an operation that has no way to return one.
    pub(crate) fn poison(&self, error: NetworkError) {
        if let Ok(mut slot) = self.poison.lock() {
            slot.get_or_insert_with(|| error.to_string());
        }
    }

    pub(crate) fn guard<T>(&self, result: Result<T, NetworkError>, fallback: T) -> T {
        match result {
            Ok(value) => value,
            Err(error) => {
                self.poison(error);
                fallback
            }
        }
    }

    /// `Ok(())` unless a device operation failed since the last check, in which
    /// case the failure is reported and cleared.
    /// Blocks until every launch on this context's stream has retired.
    ///
    /// Only timing code needs this: the training path is already ordered by
    /// the single stream, and its host round-trips synchronize implicitly.
    pub fn synchronize(&self) -> Result<(), NetworkError> {
        self.stream
            .synchronize()
            .map_err(cuda_err("stream synchronize"))
    }

    pub fn check(&self) -> Result<(), NetworkError> {
        let taken = self.poison.lock().ok().and_then(|mut slot| slot.take());
        match taken {
            None => Ok(()),
            Some(message) => Err(NetworkError::Cuda(message)),
        }
    }

    pub(crate) fn upload(&self, matrix: &Matrix) -> Result<CudaSlice<f32>, NetworkError> {
        self.stream
            .clone_htod(&matrix.data)
            .map_err(cuda_err("host to device copy"))
    }

    pub(crate) fn download(
        &self,
        source: &CudaSlice<f32>,
        target: &mut Matrix,
    ) -> Result<(), NetworkError> {
        self.stream
            .memcpy_dtoh(source, &mut target.data)
            .map_err(cuda_err("device to host copy"))
    }

    pub(crate) fn zeros(&self, len: usize) -> Result<CudaSlice<f32>, NetworkError> {
        self.stream
            .alloc_zeros::<f32>(len)
            .map_err(cuda_err("device allocation"))
    }

    /// Softmaxed attention weights per head plus the merged head outputs.
    ///
    /// The two score matmuls run on the device; the causal mask and the softmax
    /// between them run on the host, which is why every head round-trips. Row
    /// `i` of a head's weights may attend up to absolute position
    /// `position_offset + i`, and masked entries are left at zero exactly as the
    /// CPU path leaves them.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        queries: &Matrix,
        keys: &Matrix,
        values: &Matrix,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        position_offset: usize,
        keep_probabilities: bool,
    ) -> Result<(Vec<Matrix>, Matrix), NetworkError> {
        let tokens = queries.rows;
        let history = keys.rows;
        let group_size = num_heads / num_kv_heads;
        let scale = (head_dim as f32).sqrt().recip();

        let device_queries = self.upload(queries)?;
        let device_keys = self.upload(keys)?;
        let device_values = self.upload(values)?;
        let mut device_scores = self.zeros(tokens * history)?;
        let mut device_merged = self.zeros(tokens * num_heads * head_dim)?;

        let mut probabilities = Vec::with_capacity(if keep_probabilities { num_heads } else { 0 });
        let mut scores = Matrix::new(tokens, history);

        for head in 0..num_heads {
            let query_base = head * head_dim;
            let kv_base = (head / group_size) * head_dim;

            // scores[tokens, history] = scale * Q_head . K_head^T
            gemm_rhs_transposed(
                self,
                &device_queries.slice(query_base..),
                queries.cols,
                &device_keys.slice(kv_base..),
                keys.cols,
                &mut device_scores.slice_mut(0..),
                history,
                tokens,
                history,
                head_dim,
                scale,
                0.0,
            )?;
            self.download(&device_scores, &mut scores)?;

            for query in 0..tokens {
                let visible = position_offset + query + 1;
                let row = scores.row_mut(query);
                softmax(&mut row[..visible]);
                row[visible..].fill(0.0);
            }

            self.stream
                .memcpy_htod(&scores.data, &mut device_scores)
                .map_err(cuda_err("host to device copy"))?;

            // merged[:, head] = probabilities . V_head
            gemm_plain(
                self,
                &device_scores.slice(0..),
                history,
                &device_values.slice(kv_base..),
                values.cols,
                &mut device_merged.slice_mut(query_base..),
                num_heads * head_dim,
                tokens,
                head_dim,
                history,
                1.0,
                0.0,
            )?;

            if keep_probabilities {
                probabilities.push(scores.clone());
            }
        }

        let mut merged = Matrix::new(tokens, num_heads * head_dim);
        self.download(&device_merged, &mut merged)?;
        Ok((probabilities, merged))
    }
}

/// A [`Param`](crate::param::Param) mirrored on the device.
///
/// The host copy of the value goes stale as soon as the first device optimizer
/// step runs; [`TransformerLm::sync_from_device`](crate::transformer::TransformerLm::sync_from_device)
/// is what makes it current again.
pub struct DeviceParam {
    context: Arc<GpuContext>,
    rows: usize,
    cols: usize,
    value: CudaSlice<f32>,
    /// `-dL/dw`, see the module documentation.
    negated_grad: CudaSlice<f32>,
    /// Whether anything has written `negated_grad` since the last optimizer
    /// step. A step leaves the buffer holding its stale gradient rather than
    /// zeroing it, because the first writer of the next step can overwrite it;
    /// that spares the run a pass over every parameter once per step.
    grad_dirty: bool,
    moment1: CudaSlice<f32>,
    moment2: CudaSlice<f32>,
    /// Whether the parameter is held for its value alone. A frozen parameter
    /// is what a LoRA adapter leaves behind, and the point of freezing is that
    /// its gradient and both Adam moments never get allocated, so the base
    /// model costs one weight-sized device buffer to train with rather than
    /// four.
    frozen: bool,
}

impl std::fmt::Debug for DeviceParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceParam")
            .field("device", &self.context.device)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .finish()
    }
}

impl Clone for DeviceParam {
    /// Device-to-device, so a cloned model keeps its residency rather than
    /// silently dropping back to the host.
    fn clone(&self) -> Self {
        let copy = |source: &CudaSlice<f32>| {
            self.context
                .stream
                .clone_dtod(source)
                .map_err(cuda_err("device to device copy"))
        };
        let fallback = |len: usize| self.context.stream.alloc_zeros::<f32>(len).ok();
        let mut slices = Vec::with_capacity(4);
        for source in [
            &self.value,
            &self.negated_grad,
            &self.moment1,
            &self.moment2,
        ] {
            match copy(source) {
                Ok(slice) => slices.push(Some(slice)),
                Err(error) => {
                    self.context.poison(error);
                    slices.push(fallback(self.rows * self.cols));
                }
            }
        }
        let mut take = |index: usize| {
            slices[index]
                .take()
                .unwrap_or_else(|| panic!("CUDA allocation failed while cloning a parameter"))
        };
        Self {
            context: self.context.clone(),
            rows: self.rows,
            cols: self.cols,
            value: take(0),
            negated_grad: take(1),
            grad_dirty: self.grad_dirty,
            moment1: take(2),
            moment2: take(3),
            frozen: self.frozen,
        }
    }
}

impl DeviceParam {
    pub(crate) fn new(
        context: Arc<GpuContext>,
        value: &Matrix,
        frozen: bool,
    ) -> Result<Self, NetworkError> {
        // A frozen parameter never takes a gradient or an optimizer step, so
        // the three training buffers are placeholders rather than weights.
        let len = if frozen { 1 } else { value.data.len() };
        Ok(Self {
            rows: value.rows,
            cols: value.cols,
            value: context.upload(value)?,
            negated_grad: context.zeros(len)?,
            grad_dirty: false,
            moment1: context.zeros(len)?,
            moment2: context.zeros(len)?,
            frozen,
            context,
        })
    }

    /// Whether a gradient written here would be thrown away, which is what
    /// every accumulation site checks before doing its GEMM.
    pub(crate) fn is_frozen(&self) -> bool {
        self.frozen
    }

    pub(crate) fn context(&self) -> &Arc<GpuContext> {
        &self.context
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn cols(&self) -> usize {
        self.cols
    }

    /// The weight itself, for a caller that sequences its own GEMMs.
    pub(crate) fn value(&self) -> &CudaSlice<f32> {
        &self.value
    }

    /// `-dL/dw`, see the module documentation. A caller accumulating into this
    /// uses `alpha = -1.0`.
    pub(crate) fn negated_grad_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.negated_grad
    }

    /// The `beta` the next writer of `negated_grad` should use, and a claim on
    /// being that writer: zero while the buffer still holds the last step's
    /// gradient, so the writer overwrites it, and one for everyone after.
    pub(crate) fn grad_beta(&mut self) -> f32 {
        f32::from(u8::from(std::mem::replace(&mut self.grad_dirty, true)))
    }

    /// Zeroes `negated_grad` for a caller that can only accumulate into it,
    /// such as a scatter of atomic adds. Does nothing once the buffer holds
    /// this step's gradient.
    pub(crate) fn clear_grad(&mut self) -> Result<(), NetworkError> {
        if std::mem::replace(&mut self.grad_dirty, true) {
            return Ok(());
        }
        self.context
            .stream
            .memset_zeros(&mut self.negated_grad)
            .map_err(cuda_err("gradient reset"))
    }

    pub(crate) fn download_value(&self, target: &mut Matrix) -> Result<(), NetworkError> {
        self.context.download(&self.value, target)
    }

    /// The Adam moments, for a training run that persists optimizer state so a
    /// resumed run does not restart the optimizer from zero.
    pub(crate) fn download_moments(
        &self,
        first: &mut Matrix,
        second: &mut Matrix,
    ) -> Result<(), NetworkError> {
        if self.frozen {
            return Ok(());
        }
        self.context.download(&self.moment1, first)?;
        self.context.download(&self.moment2, second)
    }

    /// Restores moments read back from such a file.
    pub(crate) fn upload_moments(
        &mut self,
        first: &Matrix,
        second: &Matrix,
    ) -> Result<(), NetworkError> {
        if self.frozen {
            return Ok(());
        }
        self.moment1 = self.context.upload(first)?;
        self.moment2 = self.context.upload(second)?;
        Ok(())
    }

    /// The L2 norm of this parameter's gradient, computed on the device.
    ///
    /// The sign convention does not matter to a norm, so the negated buffer is
    /// read as it stands. A parameter that took no gradient this step reports
    /// zero rather than the stale buffer `grad_dirty` is guarding.
    pub(crate) fn grad_norm(&self) -> Result<f32, NetworkError> {
        if self.frozen || !self.grad_dirty {
            return Ok(0.0);
        }
        let (pointer, _guard) = self.negated_grad.device_ptr(&self.context.stream);
        let mut norm = 0.0f32;
        // cuBLAS scales internally, so a gradient that would overflow the
        // square of an f32 still gets a finite norm. Pointer mode is the
        // default host one, which makes this call synchronizing.
        let status = unsafe {
            cublas_sys::cublasSnrm2_v2(
                *self.context.blas.handle(),
                (self.rows * self.cols) as i32,
                pointer as *const f32,
                1,
                &mut norm,
            )
        };
        if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(NetworkError::Cuda(format!(
                "cuBLAS gradient norm failed: {status:?}"
            )));
        }
        Ok(norm)
    }

    /// The plain `dL/dw`, negating the device convention on the way out.
    pub(crate) fn download_grad(&self, target: &mut Matrix) -> Result<(), NetworkError> {
        if self.frozen || !self.grad_dirty {
            target.zeros();
            return Ok(());
        }
        self.context.download(&self.negated_grad, target)?;
        for slot in &mut target.data {
            *slot = -*slot;
        }
        Ok(())
    }

    pub(crate) fn zero_grad(&mut self) {
        self.grad_dirty = false;
    }

    /// `output[tokens, rows] = input[tokens, cols] . value^T`.
    pub(crate) fn matmul_rhs_transposed(&self, input: &Matrix) -> Matrix {
        let mut output = Matrix::new(input.rows, self.rows);
        let result = (|| {
            let device_input = self.context.upload(input)?;
            let mut device_output = self.context.zeros(output.data.len())?;
            gemm_rhs_transposed(
                &self.context,
                &device_input,
                self.cols,
                &self.value,
                self.cols,
                &mut device_output,
                self.rows,
                input.rows,
                self.rows,
                self.cols,
                1.0,
                0.0,
            )?;
            self.context.download(&device_output, &mut output)
        })();
        self.context.guard(result, ());
        output
    }

    /// `output[tokens, cols] = input[tokens, rows] . value`.
    pub(crate) fn matmul(&self, input: &Matrix) -> Matrix {
        let mut output = Matrix::new(input.rows, self.cols);
        let result = (|| {
            let device_input = self.context.upload(input)?;
            let mut device_output = self.context.zeros(output.data.len())?;
            gemm_plain(
                &self.context,
                &device_input,
                self.rows,
                &self.value,
                self.cols,
                &mut device_output,
                self.cols,
                input.rows,
                self.cols,
                self.rows,
                1.0,
                0.0,
            )?;
            self.context.download(&device_output, &mut output)
        })();
        self.context.guard(result, ());
        output
    }

    /// `negated_grad -= grad_output[tokens, rows]^T . input[tokens, cols]`.
    pub(crate) fn accumulate_grad(&mut self, grad_output: &Matrix, input: &Matrix) {
        if self.frozen {
            return;
        }
        let result = (|| {
            let device_grad_output = self.context.upload(grad_output)?;
            let device_input = self.context.upload(input)?;
            let beta = self.grad_beta();
            gemm_lhs_transposed(
                &self.context,
                &device_grad_output,
                self.rows,
                &device_input,
                self.cols,
                &mut self.negated_grad,
                self.cols,
                grad_output.rows,
                self.rows,
                self.cols,
                -1.0,
                beta,
            )
        })();
        self.context.guard(result, ());
    }

    /// `output[ids.len(), cols] = value[ids, ..]`, the embedding gather.
    pub(crate) fn gather(&self, ids: &[u32]) -> Matrix {
        let mut output = Matrix::new(ids.len(), self.cols);
        let result = (|| {
            let device_ids = self
                .context
                .stream
                .clone_htod(&ids.to_vec())
                .map_err(cuda_err("host to device copy"))?;
            let mut device_output = self.context.zeros(output.data.len())?;
            let elements = ids.len() * self.cols;
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.context.gather)
                    .arg(&mut device_output)
                    .arg(&self.value)
                    .arg(&device_ids)
                    .arg(&0i32)
                    .arg(&(ids.len() as i32))
                    .arg(&(self.cols as i32))
                    .launch(cfg(elements))
                    .map_err(cuda_err("embedding gather kernel"))?;
            }
            self.context.download(&device_output, &mut output)
        })();
        self.context.guard(result, ());
        output
    }

    /// Scatter-add of the gather's gradient, one row per token id.
    pub(crate) fn scatter_grad(&mut self, ids: &[u32], grad_output: &Matrix) {
        if self.frozen {
            return;
        }
        let result = (|| -> Result<(), NetworkError> {
            let device_ids = self
                .context
                .stream
                .clone_htod(&ids.to_vec())
                .map_err(cuda_err("host to device copy"))?;
            let device_grad = self.context.upload(grad_output)?;
            self.clear_grad()?;
            let elements = ids.len() * self.cols;
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.context.scatter)
                    .arg(&mut self.negated_grad)
                    .arg(&device_grad)
                    .arg(&device_ids)
                    .arg(&(ids.len() as i32))
                    .arg(&(self.cols as i32))
                    .launch(cfg(elements))
                    .map_err(cuda_err("embedding scatter kernel"))?;
            }
            Ok(())
        })();
        self.context.guard(result, ());
    }

    /// One optimizer step on the device, reusing `cuda_training`'s kernels.
    pub(crate) fn step(&mut self, optimizer: &Optimizer, step: usize, scale: f32) {
        if self.frozen {
            return;
        }
        let elements = self.rows * self.cols;
        let result = (|| -> Result<(), NetworkError> {
            // A parameter that took no gradient this step still takes a step,
            // on its moments alone, so the stale buffer has to become zeros.
            self.clear_grad()?;
            if scale != 1.0 {
                unsafe {
                    self.context
                        .stream
                        .launch_builder(&self.context.scale)
                        .arg(&mut self.negated_grad)
                        .arg(&(elements as i32))
                        .arg(&scale)
                        .launch(cfg(elements))
                        .map_err(cuda_err("gradient scaling kernel"))?;
                }
            }
            match *optimizer {
                Optimizer::Sgd { learning_rate } => unsafe {
                    self.context
                        .stream
                        .launch_builder(&self.context.sgd)
                        .arg(&mut self.value)
                        .arg(&self.negated_grad)
                        .arg(&(elements as i32))
                        .arg(&learning_rate)
                        .launch(cfg(elements))
                        .map_err(cuda_err("SGD update kernel"))?;
                },
                Optimizer::Adam {
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                } => {
                    let correction1 = 1.0 - beta1.powi(step as i32);
                    let correction2 = 1.0 - beta2.powi(step as i32);
                    unsafe {
                        self.context
                            .stream
                            .launch_builder(&self.context.adam)
                            .arg(&mut self.value)
                            .arg(&self.negated_grad)
                            .arg(&mut self.moment1)
                            .arg(&mut self.moment2)
                            .arg(&(elements as i32))
                            .arg(&learning_rate)
                            .arg(&beta1)
                            .arg(&beta2)
                            .arg(&epsilon)
                            .arg(&correction1)
                            .arg(&correction2)
                            .arg(&weight_decay)
                            .launch(cfg(elements))
                            .map_err(cuda_err("Adam update kernel"))?;
                    }
                }
                Optimizer::Lion { .. } => {
                    return Err(NetworkError::UnsupportedCuda(
                        "the Lion optimizer, which has no device kernel".into(),
                    ));
                }
            }
            self.grad_dirty = false;
            Ok(())
        })();
        self.context.guard(result, ());
    }
}

/// Attention entry point used by [`MultiHeadAttention`](crate::attention::MultiHeadAttention).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_heads(
    context: &Arc<GpuContext>,
    queries: &Matrix,
    keys: &Matrix,
    values: &Matrix,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    position_offset: usize,
    keep_probabilities: bool,
) -> (Vec<Matrix>, Matrix) {
    let fallback = (Vec::new(), Matrix::new(queries.rows, num_heads * head_dim));
    let result = context.attention(
        queries,
        keys,
        values,
        num_heads,
        num_kv_heads,
        head_dim,
        position_offset,
        keep_probabilities,
    );
    context.guard(result, fallback)
}

// The three GEMM shapes below are the row-major forms already proven by
// `cuda_training::{gemm, propagate_delta, grad_weights}`, generalized with
// explicit leading dimensions (so a head can be a strided view of a packed
// [tokens, heads * head_dim] buffer) and with alpha/beta exposed (so gradients
// accumulate in place and scores carry the 1/sqrt(head_dim) scale for free).
// cuBLAS is column-major, so a row-major `[r, c]` buffer with row stride `ld`
// is read as a column-major `[c, r]` with leading dimension `ld`.

/// Runs one GEMM, in BF16 when the context asks for reduced precision.
///
/// With reduced precision off, or for a caller whose operands are already
/// BF16 (the language-model head), this is exactly `blas.gemm`. Otherwise the
/// same operands go to `cublasGemmEx` with a `32F_FAST_16BF` compute type: A,
/// B and C stay FP32 in memory and cuBLAS still accumulates in FP32, and the
/// only change is that the multiplier inputs are rounded to BF16 inside the
/// tensor cores. Ampere runs BF16 tensor operations at twice the TF32 rate,
/// and on the default mixture-of-experts preset the transformer-block GEMMs
/// are more than half of all device time, so this is where the flag pays.
///
/// # Safety
///
/// Same contract as `Gemm::gemm`: the configuration must describe the three
/// buffers correctly.
unsafe fn gemm_dispatch<T: 'static, A: DevicePtr<T>, B: DevicePtr<T>, C: DevicePtrMut<T>>(
    context: &GpuContext,
    config: GemmConfig<T>,
    a: &A,
    b: &B,
    c: &mut C,
) -> Result<(), NetworkError>
where
    CudaBlas: Gemm<T>,
{
    if !context.mixed_precision || TypeId::of::<T>() != TypeId::of::<f32>() {
        return unsafe { context.blas.gemm(config, a, b, c) }.map_err(cuda_err("cuBLAS GEMM"));
    }
    let (a_pointer, _a_guard) = a.device_ptr(&context.stream);
    let (b_pointer, _b_guard) = b.device_ptr(&context.stream);
    let (c_pointer, _c_guard) = c.device_ptr_mut(&context.stream);
    unsafe {
        cublas::gemm_ex(
            *context.blas.handle(),
            config.transa,
            config.transb,
            config.m,
            config.n,
            config.k,
            &config.alpha as *const T as *const std::ffi::c_void,
            a_pointer as *const std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            config.lda,
            b_pointer as *const std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            config.ldb,
            &config.beta as *const T as *const std::ffi::c_void,
            c_pointer as *mut std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            config.ldc,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
    .map_err(cuda_err("cuBLAS BF16 GEMM"))
}

/// One GEMM whose operands are untyped bytes, either narrow or FP32.
///
/// [`Act`](crate::gpu_model::Act) stores the block activations narrow whenever
/// reduced precision is on, because every one of them is read only by a GEMM
/// that would have rounded it to BF16 inside the tensor cores anyway. Handing
/// cuBLAS BF16 operands instead of FP32 ones with a `32F_FAST_16BF` compute
/// type changes which kernel it picks: `s16816 ... align8`, twice the k per
/// instruction, rather than `s1688 ... align4`. The accumulator stays FP32
/// either way, so this is the same arithmetic on a better kernel.
///
/// A context built by [`GpuContext::half_accumulate`] reads those same narrow
/// bytes as FP16 and accumulates in FP16 too, which is the one case where the
/// arithmetic does change. See that function for why.
///
/// # Safety
///
/// Same contract as `Gemm::gemm`: the configuration must describe the three
/// buffers correctly, and `narrow` must say how the operand bytes are laid out.
unsafe fn act_dispatch<T, C: DevicePtrMut<T>>(
    context: &GpuContext,
    config: GemmConfig<f32>,
    a: &CudaView<'_, u8>,
    b: &CudaView<'_, u8>,
    narrow: bool,
    out_narrow: bool,
    c: &mut C,
) -> Result<(), NetworkError> {
    // Wide operands keep the `32F_FAST_16BF` compute type `gemm_dispatch`
    // uses, so a buffer that stayed FP32 (the routed feed-forward's, whose
    // gathers and scatters are FP32 kernels) runs on exactly the kernel it ran
    // on before.
    let (operand, compute) = if narrow && context.half_accumulate {
        (
            cublas_sys::cudaDataType_t::CUDA_R_16F,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
        )
    } else if narrow {
        (
            cublas_sys::cudaDataType_t::CUDA_R_16BF,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        )
    } else if context.mixed_precision {
        (
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
        )
    } else {
        (
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        )
    };
    // cuBLAS reads the two scalars at the compute type's width, so an FP16
    // accumulator wants them as halves and anything else as floats.
    let half = compute == cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F;
    let scalars = (
        half::f16::from_f32(config.alpha),
        half::f16::from_f32(config.beta),
    );
    let (alpha, beta): (*const std::ffi::c_void, *const std::ffi::c_void) = match half {
        true => (
            &scalars.0 as *const half::f16 as *const _,
            &scalars.1 as *const half::f16 as *const _,
        ),
        false => (
            &config.alpha as *const f32 as *const _,
            &config.beta as *const f32 as *const _,
        ),
    };
    let (a_pointer, _a_guard) = a.device_ptr(&context.stream);
    let (b_pointer, _b_guard) = b.device_ptr(&context.stream);
    let (c_pointer, _c_guard) = c.device_ptr_mut(&context.stream);
    unsafe {
        cublas::gemm_ex(
            *context.blas.handle(),
            config.transa,
            config.transb,
            config.m,
            config.n,
            config.k,
            alpha,
            a_pointer as *const std::ffi::c_void,
            operand,
            config.lda,
            b_pointer as *const std::ffi::c_void,
            operand,
            config.ldb,
            beta,
            c_pointer as *mut std::ffi::c_void,
            match (out_narrow, half) {
                (true, true) => cublas_sys::cudaDataType_t::CUDA_R_16F,
                (true, false) => cublas_sys::cudaDataType_t::CUDA_R_16BF,
                (false, _) => cublas_sys::cudaDataType_t::CUDA_R_32F,
            },
            config.ldc,
            compute,
            // The tensor-op hint is what makes cuBLAS prefer a tensor-core
            // kernel for operands it might otherwise run on the FP32 units.
            // An FP16 accumulator has no such kernels to fall back to, and
            // the hint measured slower there, so it is left off.
            match half {
                true => cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                false => cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            },
        )
    }
    .map_err(cuda_err("cuBLAS narrow GEMM"))
}

/// [`gemm_rhs_transposed`] over [`act_dispatch`] operands.
///
/// `out_narrow` narrows the result too, which saves the separate cast kernel
/// when the only reader of the result is a kernel that takes narrow input.
#[allow(clippy::too_many_arguments)]
pub(crate) fn act_rhs_transposed<T, O: DevicePtrMut<T>>(
    context: &GpuContext,
    x: &CudaView<'_, u8>,
    x_stride: usize,
    w: &CudaView<'_, u8>,
    w_stride: usize,
    narrow: bool,
    out_narrow: bool,
    out: &mut O,
    out_stride: usize,
    rows: usize,
    units: usize,
    inner: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), NetworkError> {
    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: units as i32,
        n: rows as i32,
        k: inner as i32,
        alpha,
        lda: w_stride as i32,
        ldb: x_stride as i32,
        beta,
        ldc: out_stride as i32,
    };
    unsafe { act_dispatch(context, config, w, x, narrow, out_narrow, out) }
}

/// [`gemm_plain`] over [`act_dispatch`] operands.
#[allow(clippy::too_many_arguments)]
pub(crate) fn act_plain<T, O: DevicePtrMut<T>>(
    context: &GpuContext,
    x: &CudaView<'_, u8>,
    x_stride: usize,
    w: &CudaView<'_, u8>,
    w_stride: usize,
    narrow: bool,
    out_narrow: bool,
    out: &mut O,
    out_stride: usize,
    rows: usize,
    cols: usize,
    inner: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), NetworkError> {
    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: cols as i32,
        n: rows as i32,
        k: inner as i32,
        alpha,
        lda: w_stride as i32,
        ldb: x_stride as i32,
        beta,
        ldc: out_stride as i32,
    };
    unsafe { act_dispatch(context, config, w, x, narrow, out_narrow, out) }
}

/// [`gemm_lhs_transposed`] over [`act_dispatch`] operands.
#[allow(clippy::too_many_arguments)]
pub(crate) fn act_lhs_transposed<O: DevicePtrMut<f32>>(
    context: &GpuContext,
    d: &CudaView<'_, u8>,
    d_stride: usize,
    x: &CudaView<'_, u8>,
    x_stride: usize,
    narrow: bool,
    out: &mut O,
    out_stride: usize,
    rows: usize,
    units: usize,
    cols: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), NetworkError> {
    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_T,
        m: cols as i32,
        n: units as i32,
        k: rows as i32,
        alpha,
        lda: x_stride as i32,
        ldb: d_stride as i32,
        beta,
        ldc: out_stride as i32,
    };
    unsafe { act_dispatch(context, config, x, d, narrow, false, out) }
}

/// `out[rows, units] = alpha * x[rows, inner] . w[units, inner]^T + beta * out`
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_rhs_transposed<
    T: 'static,
    X: DevicePtr<T>,
    W: DevicePtr<T>,
    O: DevicePtrMut<T>,
>(
    context: &GpuContext,
    x: &X,
    x_stride: usize,
    w: &W,
    w_stride: usize,
    out: &mut O,
    out_stride: usize,
    rows: usize,
    units: usize,
    inner: usize,
    alpha: T,
    beta: T,
) -> Result<(), NetworkError>
where
    CudaBlas: Gemm<T>,
{
    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: units as i32,
        n: rows as i32,
        k: inner as i32,
        alpha,
        lda: w_stride as i32,
        ldb: x_stride as i32,
        beta,
        ldc: out_stride as i32,
    };
    unsafe { gemm_dispatch(context, config, w, x, out) }
}

/// The batched twin of [`gemm_dispatch`]: the same reduced-precision compute
/// type on the same tensor cores, which the plain strided-batched entry point
/// cannot reach. Attention is the only batched caller and its matrices are the
/// narrowest in the model, so this is where the TF32 path was costing the most.
#[allow(clippy::too_many_arguments)]
unsafe fn gemm_strided_batched_dispatch<
    A: DevicePtr<f32>,
    B: DevicePtr<f32>,
    C: DevicePtrMut<f32>,
>(
    context: &GpuContext,
    config: StridedBatchedConfig<f32>,
    a: &A,
    b: &B,
    c: &mut C,
) -> Result<(), NetworkError> {
    if !context.mixed_precision {
        return unsafe { context.blas.gemm_strided_batched(config, a, b, c) }
            .map_err(cuda_err("cuBLAS batched GEMM"));
    }
    let (a_pointer, _a_guard) = a.device_ptr(&context.stream);
    let (b_pointer, _b_guard) = b.device_ptr(&context.stream);
    let (c_pointer, _c_guard) = c.device_ptr_mut(&context.stream);
    let gemm = config.gemm;
    unsafe {
        cublas::gemm_strided_batched_ex(
            *context.blas.handle(),
            gemm.transa,
            gemm.transb,
            gemm.m,
            gemm.n,
            gemm.k,
            &gemm.alpha as *const f32 as *const std::ffi::c_void,
            a_pointer as *const std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            gemm.lda,
            config.stride_a,
            b_pointer as *const std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            gemm.ldb,
            config.stride_b,
            &gemm.beta as *const f32 as *const std::ffi::c_void,
            c_pointer as *mut std::ffi::c_void,
            cublas_sys::cudaDataType_t::CUDA_R_32F,
            gemm.ldc,
            config.stride_c,
            config.batch_size,
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
            cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
    .map_err(cuda_err("cuBLAS batched BF16 GEMM"))
}

/// [`gemm_rhs_transposed`] over `count` independent matrices, one per sequence.
///
/// The strides are in elements and step from one sequence's block to the next,
/// which is what lets a whole batch of per-head attention GEMMs be one launch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_rhs_transposed_batched<
    X: DevicePtr<f32>,
    W: DevicePtr<f32>,
    O: DevicePtrMut<f32>,
>(
    context: &GpuContext,
    x: &X,
    x_stride: usize,
    w: &W,
    w_stride: usize,
    out: &mut O,
    out_stride: usize,
    rows: usize,
    units: usize,
    inner: usize,
    alpha: f32,
    beta: f32,
    count: usize,
    x_batch_stride: usize,
    w_batch_stride: usize,
    out_batch_stride: usize,
) -> Result<(), NetworkError> {
    let config = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: units as i32,
            n: rows as i32,
            k: inner as i32,
            alpha,
            lda: w_stride as i32,
            ldb: x_stride as i32,
            beta,
            ldc: out_stride as i32,
        },
        batch_size: count as i32,
        stride_a: w_batch_stride as i64,
        stride_b: x_batch_stride as i64,
        stride_c: out_batch_stride as i64,
    };
    unsafe { gemm_strided_batched_dispatch(context, config, w, x, out) }
}

/// [`gemm_plain`] over `count` independent matrices, one per sequence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_plain_batched<X: DevicePtr<f32>, W: DevicePtr<f32>, O: DevicePtrMut<f32>>(
    context: &GpuContext,
    x: &X,
    x_stride: usize,
    w: &W,
    w_stride: usize,
    out: &mut O,
    out_stride: usize,
    rows: usize,
    cols: usize,
    inner: usize,
    alpha: f32,
    beta: f32,
    count: usize,
    x_batch_stride: usize,
    w_batch_stride: usize,
    out_batch_stride: usize,
) -> Result<(), NetworkError> {
    let config = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: cols as i32,
            n: rows as i32,
            k: inner as i32,
            alpha,
            lda: w_stride as i32,
            ldb: x_stride as i32,
            beta,
            ldc: out_stride as i32,
        },
        batch_size: count as i32,
        stride_a: w_batch_stride as i64,
        stride_b: x_batch_stride as i64,
        stride_c: out_batch_stride as i64,
    };
    unsafe { gemm_strided_batched_dispatch(context, config, w, x, out) }
}

/// [`gemm_lhs_transposed`] over `count` independent matrices, one per sequence.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_lhs_transposed_batched<
    D: DevicePtr<f32>,
    X: DevicePtr<f32>,
    O: DevicePtrMut<f32>,
>(
    context: &GpuContext,
    d: &D,
    d_stride: usize,
    x: &X,
    x_stride: usize,
    out: &mut O,
    out_stride: usize,
    rows: usize,
    units: usize,
    cols: usize,
    alpha: f32,
    beta: f32,
    count: usize,
    d_batch_stride: usize,
    x_batch_stride: usize,
    out_batch_stride: usize,
) -> Result<(), NetworkError> {
    let config = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_T,
            m: cols as i32,
            n: units as i32,
            k: rows as i32,
            alpha,
            lda: x_stride as i32,
            ldb: d_stride as i32,
            beta,
            ldc: out_stride as i32,
        },
        batch_size: count as i32,
        stride_a: x_batch_stride as i64,
        stride_b: d_batch_stride as i64,
        stride_c: out_batch_stride as i64,
    };
    unsafe { gemm_strided_batched_dispatch(context, config, x, d, out) }
}

/// `out[rows, cols] = alpha * x[rows, inner] . w[inner, cols] + beta * out`
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_plain<T: 'static, X: DevicePtr<T>, W: DevicePtr<T>, O: DevicePtrMut<T>>(
    context: &GpuContext,
    x: &X,
    x_stride: usize,
    w: &W,
    w_stride: usize,
    out: &mut O,
    out_stride: usize,
    rows: usize,
    cols: usize,
    inner: usize,
    alpha: T,
    beta: T,
) -> Result<(), NetworkError>
where
    CudaBlas: Gemm<T>,
{
    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: cols as i32,
        n: rows as i32,
        k: inner as i32,
        alpha,
        lda: w_stride as i32,
        ldb: x_stride as i32,
        beta,
        ldc: out_stride as i32,
    };
    unsafe { gemm_dispatch(context, config, w, x, out) }
}

/// `out[units, cols] = alpha * d[rows, units]^T . x[rows, cols] + beta * out`
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_lhs_transposed<
    T: 'static,
    D: DevicePtr<T>,
    X: DevicePtr<T>,
    O: DevicePtrMut<T>,
>(
    context: &GpuContext,
    d: &D,
    d_stride: usize,
    x: &X,
    x_stride: usize,
    out: &mut O,
    out_stride: usize,
    rows: usize,
    units: usize,
    cols: usize,
    alpha: T,
    beta: T,
) -> Result<(), NetworkError>
where
    CudaBlas: Gemm<T>,
{
    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_T,
        m: cols as i32,
        n: units as i32,
        k: rows as i32,
        alpha,
        lda: x_stride as i32,
        ldb: d_stride as i32,
        beta,
        ldc: out_stride as i32,
    };
    unsafe { gemm_dispatch(context, config, x, d, out) }
}

#[cfg(test)]
mod tests {
    use crate::cuda_training::cuda_doctor;
    use crate::network::NetworkError;
    use crate::optimizers::Optimizer;
    use crate::transformer::{TransformerBuilder, TransformerLm};

    /// Same contract as the `cuda_training` tests: no device means the parity
    /// tests report success without running, a broken device is a failure.
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

    /// One dense block and one MoE block, so a parity run covers both feed
    /// forward shapes as well as grouped-query attention and tied embeddings.
    ///
    /// Reduced precision is switched off. It is the library default, but these
    /// are FP32 device-against-host parity tests, and TF32 and BF16 rounding
    /// would put them a full learning-rate step apart on any parameter whose
    /// true gradient is near zero — see the note on
    /// `mixed_precision_tracks_the_fp32_device_path_or_skips_without_device`.
    fn tiny() -> TransformerBuilder {
        TransformerLm::builder()
            .vocab_size(24)
            .d_model(16)
            .n_layers(2)
            .heads(4, 2, 4)
            .d_ff(32)
            .moe_d_ff(12)
            .experts(4, 2)
            .moe_layers([1])
            .shared_expert(true)
            .max_seq_len(32)
            .optimizer(Optimizer::adam(1e-2))
            .seed(1234)
            .mixed_precision(false)
    }

    fn assert_close(label: &str, gpu: &[f32], cpu: &[f32], tolerance: f32) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: length");
        for (index, (a, b)) in gpu.iter().zip(cpu).enumerate() {
            assert!(
                (a - b).abs() <= tolerance,
                "{label}[{index}]: {a} on the device vs {b} on the host"
            );
        }
    }

    #[test]
    fn gpu_grad_norm_matches_cpu_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let mut cpu = tiny().build().unwrap();
        let mut gpu = cpu.clone();
        gpu.to_cuda(0, 8192).unwrap();
        let batch = crate::batch::TokenBatch::new(&[[3u32, 8, 1, 5, 2, 7]]).unwrap();

        // A parameter that never took a gradient must read as zero rather than
        // as whatever the device buffer held before `zero_grad`.
        cpu.zero_grad();
        gpu.zero_grad();
        assert_eq!(gpu.grad_norm().unwrap(), 0.0);

        cpu.accumulate_step(&batch).unwrap();
        gpu.accumulate_step(&batch).unwrap();

        let (host, device) = (cpu.grad_norm().unwrap(), gpu.grad_norm().unwrap());
        assert!(host > 0.0);
        assert!(
            (device - host).abs() <= 1e-3 * host,
            "{device} on the device vs {host} on the host"
        );
    }

    #[test]
    fn gpu_forward_matches_cpu_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let cpu = tiny().build().unwrap();
        let mut gpu = cpu.clone();
        gpu.to_cuda(0, 8192).unwrap();
        let ids = [3u32, 8, 1, 5, 2, 7];

        let (host_logits, _) = cpu.forward_train(&[&ids[..]]).unwrap();
        let (device_logits, _) = gpu.forward_train(&[&ids[..]]).unwrap();

        assert_close("logits", &device_logits.data, &host_logits.data, 1e-3);
    }

    #[test]
    fn gpu_backward_matches_cpu_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let mut cpu = tiny().build().unwrap();
        let mut gpu = cpu.clone();
        gpu.to_cuda(0, 8192).unwrap();
        let ids = [3u32, 8, 1, 5, 2, 7];

        let host = cpu.train_step(&[&ids[..]]).unwrap();
        let device = gpu.train_step(&[&ids[..]]).unwrap();
        gpu.sync_from_device().unwrap();

        assert!((host.total() - device.total()).abs() < 1e-3);
        for (index, (device_param, host_param)) in
            gpu.params_mut().iter().zip(cpu.params_mut()).enumerate()
        {
            assert_close(
                &format!("parameter {index}"),
                &device_param.value.data,
                &host_param.value.data,
                1e-3,
            );
        }
    }

    #[test]
    fn a_gpu_lora_step_matches_the_host_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let mut cpu = tiny().build().unwrap();
        cpu.add_lora(crate::transformer::LoraConfig::new(4).alpha(8.0))
            .unwrap();
        let mut gpu = cpu.clone();
        gpu.to_cuda(0, 8192).unwrap();
        let ids = [3u32, 8, 1, 5, 2, 7];
        let before = cpu.blocks[0].attention.query.weight.value.clone();

        // Two steps, not one. `up` starts at zero, so the first step is the
        // only thing that moves it off zero and the second is the first one
        // where the adapter contributes to the forward pass at all.
        for step in 0..2 {
            let host = cpu.train_step(&[&ids[..]]).unwrap();
            let device = gpu.train_step(&[&ids[..]]).unwrap();
            assert!(
                (host.total() - device.total()).abs() < 1e-3,
                "step {step}: {} on the device vs {} on the host",
                device.total(),
                host.total()
            );
        }
        gpu.sync_from_device().unwrap();

        // The frozen base is the same matrix it started as, on both paths.
        assert_close(
            "frozen base weight",
            &gpu.blocks[0].attention.query.weight.value.data,
            &before.data,
            0.0,
        );
        let adapter = gpu.blocks[0]
            .attention
            .query
            .lora
            .as_ref()
            .expect("the query projection carries an adapter");
        assert!(
            adapter.up.value.data.iter().any(|&value| value != 0.0),
            "the device step left the adapter at its zero initialization"
        );

        for (index, (device_param, host_param)) in
            gpu.params_mut().iter().zip(cpu.params_mut()).enumerate()
        {
            assert_close(
                &format!("parameter {index}"),
                &device_param.value.data,
                &host_param.value.data,
                1e-3,
            );
        }
    }

    #[test]
    fn a_batched_gpu_forward_matches_the_host_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let cpu = tiny().build().unwrap();
        let mut gpu = cpu.clone();
        gpu.to_cuda(0, 8192).unwrap();
        let batch = [vec![3u32, 8, 1, 5, 2, 7], vec![9, 4, 4, 0, 11, 6]];

        let (host_logits, _) = cpu.forward_train(&batch).unwrap();
        let (device_logits, _) = gpu.forward_train(&batch).unwrap();

        assert_eq!(device_logits.rows, 12);
        assert_close(
            "batched logits",
            &device_logits.data,
            &host_logits.data,
            1e-3,
        );
    }

    /// Padding exercises the two places where a row is not independent: the
    /// router, which must leave a pad row unrouted and out of the balancing
    /// statistics, and the loss.
    #[test]
    fn a_padded_gpu_batch_matches_the_host_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let cpu = tiny().build().unwrap();
        let mut gpu = cpu.clone();
        gpu.to_cuda(0, 8192).unwrap();
        let batch = [vec![3u32, 8, 1, 5, 2, 7], vec![9, 4, 4], vec![2, 6]];

        let (host_logits, host_cache) = cpu.forward_train(&batch).unwrap();
        let (device_logits, device_cache) = gpu.forward_train(&batch).unwrap();

        assert_close(
            "padded logits",
            &device_logits.data,
            &host_logits.data,
            1e-3,
        );
        assert!(
            (device_cache.auxiliary_loss() - host_cache.auxiliary_loss()).abs() < 1e-4,
            "auxiliary loss {} on the device vs {} on the host",
            device_cache.auxiliary_loss(),
            host_cache.auxiliary_loss()
        );
    }

    #[test]
    fn a_batched_gpu_train_step_matches_the_host_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let mut cpu = tiny().build().unwrap();
        let mut gpu = cpu.clone();
        gpu.to_cuda(0, 8192).unwrap();
        let batch = [
            vec![3u32, 8, 1, 5, 2, 7],
            vec![9, 4, 4, 0, 11, 6],
            vec![2, 6, 1],
        ];

        let host = cpu.train_step(&batch).unwrap();
        let device = gpu.train_step(&batch).unwrap();
        gpu.sync_from_device().unwrap();

        assert!((host.lm_loss - device.lm_loss).abs() < 1e-3);
        assert!((host.auxiliary_loss - device.auxiliary_loss).abs() < 1e-4);
        for (index, (device_param, host_param)) in
            gpu.params_mut().iter().zip(cpu.params_mut()).enumerate()
        {
            assert_close(
                &format!("parameter {index}"),
                &device_param.value.data,
                &host_param.value.data,
                1e-3,
            );
        }
    }

    /// The fused head runs in row chunks, so a batch split across several of
    /// them has to give the same loss and the same weights as one that is not.
    #[test]
    fn a_chunked_gpu_head_matches_the_host_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let mut cpu = tiny().build().unwrap();
        let mut gpu = cpu.clone();
        gpu.to_cuda(0, 8192).unwrap();
        let batch = [
            vec![3u32, 8, 1, 5, 2, 7],
            vec![9, 4, 4, 0, 11, 6],
            vec![2, 6, 1],
        ];

        // 18 rows in 5-row chunks: four chunks, the last one short.
        crate::gpu_model::CHUNK_ROWS_OVERRIDE.store(5, std::sync::atomic::Ordering::Relaxed);
        let host = cpu.train_step(&batch).unwrap();
        let device = gpu.train_step(&batch).unwrap();
        crate::gpu_model::CHUNK_ROWS_OVERRIDE.store(0, std::sync::atomic::Ordering::Relaxed);
        gpu.sync_from_device().unwrap();

        assert!(
            (host.lm_loss - device.lm_loss).abs() < 1e-3,
            "chunked loss {} on the device vs {} on the host",
            device.lm_loss,
            host.lm_loss
        );
        for (index, (device_param, host_param)) in
            gpu.params_mut().iter().zip(cpu.params_mut()).enumerate()
        {
            assert_close(
                &format!("parameter {index}"),
                &device_param.value.data,
                &host_param.value.data,
                1e-3,
            );
        }
    }

    /// The reduced-precision path against the FP32 one, over two steps and
    /// several head chunks.
    ///
    /// Two roundings are in play. TF32 keeps FP32's exponent but truncates
    /// both GEMM inputs to a 10-bit mantissa, so a dot product carries about
    /// `2^-11` relative error per term where FP32 carries `2^-24`. The BF16
    /// head is coarser still at `2^-8`, and its logits, its cross-entropy and
    /// both of its gradient products are all stored that way. The tolerance
    /// here is therefore two orders of magnitude looser than the FP32 parity
    /// tests use.
    ///
    /// The chunk override splits the head over four chunks, the last one
    /// short, which is what exercises the BF16 weight gradient being widened
    /// and accumulated into the FP32 gradient once per chunk.
    ///
    /// Weights are deliberately not compared. Adam divides by the gradient's
    /// own magnitude, so a parameter whose true gradient is around zero gets a
    /// full learning-rate step in whichever direction the rounding noise
    /// points, and the two paths end up an entire `lr` apart on entries that
    /// barely affect the model. That is a property of the optimizer, not a
    /// sign that the flag computes a different function. What the flag must
    /// not do is change the function being computed, and on a 32k vocabulary
    /// the two loss curves stay together to four decimal places for twenty
    /// steps; `examples/bf16_loss_check.rs` is that check at a scale too slow
    /// for a unit test.
    #[test]
    fn mixed_precision_tracks_the_fp32_device_path_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let mut baseline = tiny().build().unwrap();
        let mut reduced = tiny().mixed_precision(true).build().unwrap();
        assert_eq!(
            baseline, reduced,
            "the two models start from the same weights"
        );
        baseline.to_cuda(0, 8192).unwrap();
        reduced.to_cuda(0, 8192).unwrap();
        let batch = [
            vec![3u32, 8, 1, 5, 2, 7],
            vec![9, 4, 4, 0, 11, 6],
            vec![2, 6, 1],
        ];

        crate::gpu_model::CHUNK_ROWS_OVERRIDE.store(5, std::sync::atomic::Ordering::Relaxed);
        for step in 0..2 {
            let exact = baseline.train_step(&batch).unwrap();
            let approximate = reduced.train_step(&batch).unwrap();
            assert!(
                (exact.lm_loss - approximate.lm_loss).abs() < 1e-2,
                "step {step}: reduced-precision loss {} vs FP32 {}",
                approximate.lm_loss,
                exact.lm_loss
            );
        }
        crate::gpu_model::CHUNK_ROWS_OVERRIDE.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Several steps in a row, which is where a stale cache or a gradient that
    /// is not cleared would show up.
    #[test]
    fn repeated_gpu_batched_steps_stay_in_step_with_the_host_or_skip_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let mut cpu = tiny().build().unwrap();
        let mut gpu = cpu.clone();
        gpu.to_cuda(0, 8192).unwrap();
        let batch = [vec![3u32, 8, 1, 5], vec![9, 4, 4, 0]];

        for step in 0..4 {
            let host = cpu.train_step(&batch).unwrap();
            let device = gpu.train_step(&batch).unwrap();
            assert!(
                (host.total() - device.total()).abs() < 2e-3,
                "step {step}: {} on the device vs {} on the host",
                device.total(),
                host.total()
            );
        }

        gpu.sync_from_device().unwrap();
        for (index, (device_param, host_param)) in
            gpu.params_mut().iter().zip(cpu.params_mut()).enumerate()
        {
            assert_close(
                &format!("parameter {index}"),
                &device_param.value.data,
                &host_param.value.data,
                2e-3,
            );
        }
    }

    #[test]
    fn a_cached_decode_on_the_device_matches_the_host_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let cpu = tiny().build().unwrap();
        let mut gpu = cpu.clone();
        gpu.to_cuda(0, 8192).unwrap();
        let ids = [3u32, 8, 1, 5];

        let mut host_caches = cpu.new_kv_caches();
        let mut device_caches = gpu.new_kv_caches();
        let host = cpu.forward_cached(&ids, &mut host_caches).unwrap();
        let device = gpu.forward_cached(&ids, &mut device_caches).unwrap();

        assert_close("prefill logits", &device.data, &host.data, 1e-3);
    }

    #[test]
    fn to_cuda_fails_closed_without_a_device() {
        if cuda_or_skip() {
            return;
        }
        let mut model = tiny().build().unwrap();
        assert!(matches!(model.to_cuda(0, 8192), Err(NetworkError::Cuda(_))));
        assert!(!model.on_device());
    }

    /// Runs with or without a device: the budget is checked before the driver
    /// is touched, so an impossible budget never reaches the hardware.
    #[test]
    fn an_impossible_memory_budget_is_refused_before_any_upload() {
        // Wide enough that value, gradient and both Adam moments do not fit in
        // a one mebibyte budget.
        let mut model = tiny()
            .d_model(64)
            .heads(4, 2, 16)
            .d_ff(256)
            .moe_d_ff(64)
            .build()
            .unwrap();
        assert!(matches!(
            model.to_cuda(0, 1),
            Err(NetworkError::CudaMemoryBudget { .. })
        ));
        assert!(!model.on_device());
    }

    /// The fused attention kernel against the three-kernel path it replaces.
    ///
    /// It only runs on a head dimension of 64 under reduced precision, which no
    /// other test in this file uses, so without this one it is never executed.
    /// The sequence is 100 tokens: longer than one 64-wide tile, and not a
    /// multiple of it, so the run covers an off-diagonal tile, the masked
    /// diagonal, and a ragged tail.
    ///
    /// `DISABLED` is process-wide. Nothing else here is eligible for the fused
    /// path, so flipping it cannot disturb a test running alongside.
    #[test]
    fn fused_attention_matches_the_three_kernel_path_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let model = || {
            TransformerLm::builder()
                .vocab_size(32)
                .d_model(128)
                .n_layers(2)
                .heads(2, 1, 64)
                .d_ff(64)
                .moe_layers([0usize; 0])
                .max_seq_len(128)
                .optimizer(Optimizer::adam(1e-2))
                .seed(99)
                .mixed_precision(true)
                .build()
                .unwrap()
        };
        let ids: Vec<u32> = (0..100).map(|index| (index * 7 % 32) as u32).collect();

        let (host_logits, _) = model().forward_train(&[&ids[..]]).unwrap();

        let mut fused = model();
        fused.to_cuda(0, 8192).unwrap();
        let (fused_logits, _) = fused.forward_train(&[&ids[..]]).unwrap();

        crate::cuda_flash::DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut split = model();
        split.to_cuda(0, 8192).unwrap();
        let split_forward = split.forward_train(&[&ids[..]]);
        crate::cuda_flash::DISABLED.store(false, std::sync::atomic::Ordering::Relaxed);
        let (split_logits, _) = split_forward.unwrap();

        // Both device paths round their matmul operands to BF16, so neither is
        // the reference and a fixed band between them would only be a band
        // around BF16 epsilon. The FP32 host run is the reference, and the
        // question the fusion has to answer is whether it loses accuracy the
        // path it replaces did not: measured here the split path lands 2.5e-3
        // from the host and the fused path 3.7e-3, on logits whose magnitude is
        // 0.475. A real fault in the tile loop, the mask or the running maximum
        // is off by far more than the factor of two below.
        let error = |logits: &crate::matrix::Matrix| {
            logits
                .data
                .iter()
                .zip(&host_logits.data)
                .map(|(device, host)| (device - host).abs())
                .fold(0.0f32, f32::max)
        };
        let (fused_error, split_error) = (error(&fused_logits), error(&split_logits));
        assert!(
            fused_error <= 2.0 * split_error,
            "the fused path is {fused_error} from the host where the three-kernel path is {split_error}"
        );
        assert!(
            fused_error < 1e-2,
            "both device paths drifted: {fused_error}"
        );
    }

    /// The same comparison for the backward half, which is a separate pair of
    /// kernels: the query gradient is blocked over query tiles, the key and
    /// value gradients over key tiles, and neither rebuilds the score matrix.
    /// A full training step is what puts every gradient into one number.
    #[test]
    fn fused_attention_backward_matches_the_three_kernel_path_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let model = || {
            TransformerLm::builder()
                .vocab_size(32)
                .d_model(256)
                .n_layers(2)
                .heads(4, 2, 64)
                .d_ff(64)
                .moe_layers([0usize; 0])
                .max_seq_len(128)
                .optimizer(Optimizer::sgd(1.0))
                .seed(99)
                .mixed_precision(true)
                .build()
                .unwrap()
        };
        let ids: Vec<u32> = (0..100).map(|index| (index * 7 % 32) as u32).collect();

        let mut host = model();
        host.train_step(&[&ids[..]]).unwrap();
        let (host_logits, _) = host.forward_train(&[&ids[..]]).unwrap();

        let mut fused = model();
        fused.to_cuda(0, 8192).unwrap();
        fused.train_step(&[&ids[..]]).unwrap();
        let (fused_logits, _) = fused.forward_train(&[&ids[..]]).unwrap();

        crate::cuda_flash::DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
        let mut split = model();
        split.to_cuda(0, 8192).unwrap();
        let stepped = split
            .train_step(&[&ids[..]])
            .and_then(|_| split.forward_train(&[&ids[..]]));
        crate::cuda_flash::DISABLED.store(false, std::sync::atomic::Ordering::Relaxed);
        let (split_logits, _) = stepped.unwrap();

        // A wrong gradient moves a parameter the wrong way, and plain gradient
        // descent at a rate of 1 makes that visible in the next forward pass
        // in proportion to the error. Adam would not: it normalizes the step,
        // so a parameter whose true gradient is near zero takes a full step in
        // whichever direction the rounding noise pointed, and both device
        // paths would land a learning rate apart from the host for reasons
        // that have nothing to do with the fusion.
        //
        // The bar is the forward test's: no worse against the FP32 host than
        // the path being replaced. Measured, the fused path runs 1.0 to 1.4
        // times the three-kernel path's error here, the difference being that
        // it takes the softmax row sum from the output gradient and the output
        // rather than from the probabilities it no longer has.
        let error = |logits: &crate::matrix::Matrix| {
            logits
                .data
                .iter()
                .zip(&host_logits.data)
                .map(|(device, host)| (device - host).abs())
                .fold(0.0f32, f32::max)
        };
        let (fused_error, split_error) = (error(&fused_logits), error(&split_logits));
        assert!(
            fused_error <= 2.0 * split_error,
            "the fused path is {fused_error} from the host where the three-kernel path is {split_error}"
        );
        // A second bound, in case both paths break the same way. A full
        // gradient-descent step at a rate of 1 moves logits of this shape by
        // 2.5 to 4.2, and BF16 rounding of the attention matmuls leaves both
        // device paths about 0.04 from the host afterwards.
        assert!(
            fused_error < 0.1,
            "both device paths drifted: {fused_error}"
        );
    }
}
