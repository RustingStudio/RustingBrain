//! Fail-closed FP32 Metal fitting, for Apple Silicon and any other Metal GPU.
//!
//! This is the same training loop as [`crate::cuda_training`], expressed in
//! Metal: nine element-wise kernels plus three tiled GEMMs that stand in for
//! cuBLAS, a persistent session that keeps the dataset and every tensor on the
//! device, and Adam applied on the GPU. Two things differ, and both come from
//! the hardware rather than from taste:
//!
//! * Apple GPUs share memory with the CPU, so every buffer is
//!   `StorageModeShared` and "upload" and "download" are memcpys. The byte
//!   counters in [`AcceleratorStats`] still record them, so a Mac and a CUDA
//!   box report comparable telemetry.
//! * Metal has no per-dispatch launch call. A whole epoch is encoded into one
//!   command buffer and committed once, which is why the resident-dataset path
//!   is the fast one: it needs no host writes in the middle of a batch loop.
use crate::{
    Activation, CudaTrainingCheckpoint, Dataset, Loss, Network, NetworkError, Optimizer,
    TrainConfig, TrainingHistory,
    accelerator::{AcceleratorDoctorReport, AcceleratorStats, MIB, estimate_tensor_memory_mib},
};
use metal::objc::rc::autoreleasepool;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::{cell::RefCell, collections::HashMap, ffi::c_void, rc::Rc, time::Instant};

/// Threads per threadgroup for the element-wise kernels.
const LANES: u64 = 256;
/// Tile edge for the GEMM kernels; must match `TS` in [`KERNELS`].
const TILE: u64 = 16;
/// Threads in the single-threadgroup loss reduction; must match
/// `REDUCE_WIDTH` in [`KERNELS`], and must be a power of two.
const REDUCE_WIDTH: u64 = 256;

const KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define TS 16
#define REDUCE_WIDTH 256

inline float apply_act(float v, int a) {
    if (a == 1) return v > 0.0f ? v : 0.0f;
    if (a == 2) return 1.0f / (1.0f + exp(-v));
    if (a == 3) return tanh(v);
    return v;
}

kernel void bias_act(device float *x [[buffer(0)]],
                     const device float *b [[buffer(1)]],
                     constant int &rows [[buffer(2)]],
                     constant int &cols [[buffer(3)]],
                     constant int &a [[buffer(4)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= uint(rows) * uint(cols)) return;
    x[i] = apply_act(x[i] + b[i % uint(cols)], a);
}

kernel void output_delta(device float *d [[buffer(0)]],
                         const device float *y [[buffer(1)]],
                         const device float *t [[buffer(2)]],
                         constant int &n [[buffer(3)]],
                         constant int &a [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= uint(n)) return;
    float v = t[i] - y[i];
    float o = y[i];
    if (a == 1) v *= o > 0.0f ? 1.0f : 0.0f;
    else if (a == 2) v *= o * (1.0f - o);
    else if (a == 3) v *= 1.0f - o * o;
    d[i] = v;
}

kernel void act_derivative(device float *d [[buffer(0)]],
                           const device float *o [[buffer(1)]],
                           constant int &n [[buffer(2)]],
                           constant int &a [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    if (i >= uint(n)) return;
    float v = o[i];
    if (a == 1) d[i] *= v > 0.0f ? 1.0f : 0.0f;
    else if (a == 2) d[i] *= v * (1.0f - v);
    else if (a == 3) d[i] *= 1.0f - v * v;
}

kernel void grad_b(device float *g [[buffer(0)]],
                   const device float *d [[buffer(1)]],
                   constant int &batch [[buffer(2)]],
                   constant int &out [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= uint(out)) return;
    float s = 0.0f;
    for (uint b = 0; b < uint(batch); ++b) s += d[b * uint(out) + i];
    g[i] = s / float(batch);
}

kernel void sgd(device float *x [[buffer(0)]],
                const device float *g [[buffer(1)]],
                constant int &n [[buffer(2)]],
                constant float &lr [[buffer(3)]],
                uint i [[thread_position_in_grid]]) {
    if (i < uint(n)) x[i] += lr * g[i];
}

kernel void adam(device float *x [[buffer(0)]],
                 const device float *g [[buffer(1)]],
                 device float *m [[buffer(2)]],
                 device float *v [[buffer(3)]],
                 constant int &n [[buffer(4)]],
                 constant float &lr [[buffer(5)]],
                 constant float &b1 [[buffer(6)]],
                 constant float &b2 [[buffer(7)]],
                 constant float &eps [[buffer(8)]],
                 constant float &c1 [[buffer(9)]],
                 constant float &c2 [[buffer(10)]],
                 constant float &wd [[buffer(11)]],
                 uint i [[thread_position_in_grid]]) {
    if (i >= uint(n)) return;
    float q = g[i];
    float mi = b1 * m[i] + (1.0f - b1) * q;
    float vi = b2 * v[i] + (1.0f - b2) * q * q;
    m[i] = mi;
    v[i] = vi;
    float w = x[i];
    w -= lr * wd * w;
    w += lr * (mi / c1) / (sqrt(vi / c2) + eps);
    x[i] = w;
}

// One threadgroup of REDUCE_WIDTH threads, always. A tree reduction in
// threadgroup memory replaces CUDA's `atomicAdd`: float atomics need a Metal 3
// device, and this runs once per batch over at most a few thousand elements.
kernel void mse_epoch_sum(device float *out [[buffer(0)]],
                          const device float *y [[buffer(1)]],
                          const device float *t [[buffer(2)]],
                          constant int &n [[buffer(3)]],
                          constant float &scale [[buffer(4)]],
                          threadgroup float *partial [[threadgroup(0)]],
                          uint tid [[thread_position_in_threadgroup]]) {
    float acc = 0.0f;
    for (uint i = tid; i < uint(n); i += REDUCE_WIDTH) {
        float z = y[i] - t[i];
        acc += z * z;
    }
    partial[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = REDUCE_WIDTH / 2; s > 0; s >>= 1) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[0] += partial[0] * scale;
}

kernel void gather_rows(device float *out [[buffer(0)]],
                        const device float *all [[buffer(1)]],
                        const device uint *order [[buffer(2)]],
                        constant int &start [[buffer(3)]],
                        constant int &rows [[buffer(4)]],
                        constant int &width [[buffer(5)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= uint(rows) * uint(width)) return;
    uint r = i / uint(width);
    uint c = i % uint(width);
    out[i] = all[order[uint(start) + r] * uint(width) + c];
}

// C[M, N] = A[M, K] . B[N, K]^T -- the forward pass, where B is a weight
// matrix stored row-major as [units, inputs].
kernel void gemm_nt(device float *c [[buffer(0)]],
                    const device float *a [[buffer(1)]],
                    const device float *b [[buffer(2)]],
                    constant int &M [[buffer(3)]],
                    constant int &N [[buffer(4)]],
                    constant int &K [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float sa[TS][TS];
    threadgroup float sb[TS][TS];
    uint row = tg.y * TS + lid.y;
    uint col = tg.x * TS + lid.x;
    float acc = 0.0f;
    uint tiles = (uint(K) + TS - 1) / TS;
    for (uint t = 0; t < tiles; ++t) {
        uint ak = t * TS + lid.x;
        uint bk = t * TS + lid.y;
        sa[lid.y][lid.x] = (row < uint(M) && ak < uint(K)) ? a[row * uint(K) + ak] : 0.0f;
        sb[lid.y][lid.x] = (col < uint(N) && bk < uint(K)) ? b[col * uint(K) + bk] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TS; ++k) acc += sa[lid.y][k] * sb[k][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < uint(M) && col < uint(N)) c[row * uint(N) + col] = acc;
}

// C[M, N] = alpha * A[K, M]^T . B[K, N] -- the weight gradient, where K is the
// batch size and alpha is 1/batch.
kernel void gemm_tn(device float *c [[buffer(0)]],
                    const device float *a [[buffer(1)]],
                    const device float *b [[buffer(2)]],
                    constant int &M [[buffer(3)]],
                    constant int &N [[buffer(4)]],
                    constant int &K [[buffer(5)]],
                    constant float &alpha [[buffer(6)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float sa[TS][TS];
    threadgroup float sb[TS][TS];
    uint row = tg.y * TS + lid.y;
    uint col = tg.x * TS + lid.x;
    float acc = 0.0f;
    uint tiles = (uint(K) + TS - 1) / TS;
    for (uint t = 0; t < tiles; ++t) {
        uint ak = t * TS + lid.x;
        uint bk = t * TS + lid.y;
        sa[lid.y][lid.x] = (row < uint(M) && ak < uint(K)) ? a[ak * uint(M) + row] : 0.0f;
        sb[lid.y][lid.x] = (col < uint(N) && bk < uint(K)) ? b[bk * uint(N) + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TS; ++k) acc += sa[lid.y][k] * sb[k][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < uint(M) && col < uint(N)) c[row * uint(N) + col] = alpha * acc;
}

// C[M, N] = A[M, K] . B[K, N] -- propagating a delta back through a layer.
kernel void gemm_nn(device float *c [[buffer(0)]],
                    const device float *a [[buffer(1)]],
                    const device float *b [[buffer(2)]],
                    constant int &M [[buffer(3)]],
                    constant int &N [[buffer(4)]],
                    constant int &K [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint2 lid [[thread_position_in_threadgroup]]) {
    threadgroup float sa[TS][TS];
    threadgroup float sb[TS][TS];
    uint row = tg.y * TS + lid.y;
    uint col = tg.x * TS + lid.x;
    float acc = 0.0f;
    uint tiles = (uint(K) + TS - 1) / TS;
    for (uint t = 0; t < tiles; ++t) {
        uint ak = t * TS + lid.x;
        uint bk = t * TS + lid.y;
        sa[lid.y][lid.x] = (row < uint(M) && ak < uint(K)) ? a[row * uint(K) + ak] : 0.0f;
        sb[lid.y][lid.x] = (col < uint(N) && bk < uint(K)) ? b[bk * uint(N) + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TS; ++k) acc += sa[lid.y][k] * sb[k][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < uint(M) && col < uint(N)) c[row * uint(N) + col] = acc;
}
"#;

fn metal_err<E: std::fmt::Display>(stage: &'static str) -> impl FnOnce(E) -> NetworkError {
    move |e| NetworkError::Metal(format!("{stage} failed: {e}"))
}

fn act(a: Activation) -> Result<i32, NetworkError> {
    match a {
        Activation::Linear => Ok(0),
        Activation::Relu => Ok(1),
        Activation::Sigmoid => Ok(2),
        Activation::Tanh => Ok(3),
        Activation::Softmax => Err(NetworkError::UnsupportedMetal("Softmax activation".into())),
    }
}

/// Every compute pipeline resolved once.  Compiling [`KERNELS`] costs a few
/// hundred milliseconds and building a pipeline state is a driver call, so a
/// campaign running dozens of sessions on one thread pays for both once.
struct Pipelines {
    bias_act: ComputePipelineState,
    output_delta: ComputePipelineState,
    act_derivative: ComputePipelineState,
    grad_b: ComputePipelineState,
    sgd: ComputePipelineState,
    adam: ComputePipelineState,
    mse_epoch_sum: ComputePipelineState,
    gather_rows: ComputePipelineState,
    gemm_nt: ComputePipelineState,
    gemm_tn: ComputePipelineState,
    gemm_nn: ComputePipelineState,
}

struct Gpu {
    device: Device,
    pipelines: Pipelines,
}

// Metal objects are Objective-C pointers and are not `Send`, so the cache is
// per-thread rather than a `OnceLock` as on the CUDA side. A campaign drives
// one session at a time from one thread, so this still compiles the shader
// library exactly once.
thread_local! {
    static GPUS: RefCell<HashMap<usize, Rc<Gpu>>> = RefCell::new(HashMap::new());
}

fn devices() -> Vec<Device> {
    let all = Device::all();
    if all.is_empty() {
        Device::system_default().into_iter().collect()
    } else {
        all
    }
}

fn open_device(index: usize) -> Result<Device, NetworkError> {
    let all = devices();
    all.into_iter().nth(index).ok_or_else(|| {
        NetworkError::Metal(format!(
            "no Metal device at index {index}; this machine reports {} device(s)",
            devices().len()
        ))
    })
}

fn build_pipelines(device: &Device) -> Result<Pipelines, NetworkError> {
    let library = device
        .new_library_with_source(KERNELS, &CompileOptions::new())
        .map_err(|e| NetworkError::Metal(format!("Metal shader compilation failed: {e}")))?;
    let state = |name: &str| -> Result<ComputePipelineState, NetworkError> {
        let function = library
            .get_function(name, None)
            .map_err(metal_err("Metal kernel lookup"))?;
        device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(metal_err("Metal pipeline creation"))
    };
    Ok(Pipelines {
        bias_act: state("bias_act")?,
        output_delta: state("output_delta")?,
        act_derivative: state("act_derivative")?,
        grad_b: state("grad_b")?,
        sgd: state("sgd")?,
        adam: state("adam")?,
        mse_epoch_sum: state("mse_epoch_sum")?,
        gather_rows: state("gather_rows")?,
        gemm_nt: state("gemm_nt")?,
        gemm_tn: state("gemm_tn")?,
        gemm_nn: state("gemm_nn")?,
    })
}

fn gpu(index: usize) -> Result<Rc<Gpu>, NetworkError> {
    GPUS.with(|cache| {
        if let Some(existing) = cache.borrow().get(&index) {
            return Ok(existing.clone());
        }
        let device = open_device(index)?;
        let pipelines = build_pipelines(&device)?;
        let entry = Rc::new(Gpu { device, pipelines });
        cache.borrow_mut().insert(index, entry.clone());
        Ok(entry)
    })
}

/// Probes a Metal device without training on it.
pub fn metal_doctor(
    device: usize,
    requested_budget_mib: usize,
) -> Result<AcceleratorDoctorReport, NetworkError> {
    autoreleasepool(|| {
        let gpu = gpu(device)?;
        let handle = &gpu.device;
        let probe = handle.new_buffer(1024, MTLResourceOptions::StorageModeShared);
        let allocation_test = !probe.contents().is_null();
        let total = handle.recommended_max_working_set_size() as usize / MIB;
        let used = handle.current_allocated_size() as usize / MIB;
        Ok(AcceleratorDoctorReport {
            backend: "metal",
            device,
            name: handle.name().to_owned(),
            revision: if handle.has_unified_memory() {
                "unified memory".into()
            } else {
                "discrete".into()
            },
            total_memory_mib: total,
            free_memory_mib: Some(total.saturating_sub(used)),
            requested_budget_mib,
            allocation_test,
            gemm_available: true,
            kernel_available: true,
        })
    })
}

/// A `StorageModeShared` buffer plus the element count, because `Buffer` only
/// remembers bytes and every kernel argument here is counted in elements.
struct Buf {
    buffer: Buffer,
    len: usize,
}

impl Buf {
    fn zeros(device: &Device, len: usize) -> Self {
        let bytes = (len.max(1) * size_of::<f32>()) as u64;
        let buffer = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        // `newBufferWithLength:` does not promise zeroed memory.
        unsafe { std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes as usize) };
        Self { buffer, len }
    }

    fn from_slice(device: &Device, values: &[f32]) -> Self {
        let buffer = Self::zeros(device, values.len());
        buffer.write(values);
        Self {
            buffer: buffer.buffer,
            len: values.len(),
        }
    }

    fn write(&self, values: &[f32]) {
        assert!(values.len() <= self.len.max(1));
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.buffer.contents().cast::<f32>(),
                values.len(),
            );
        }
    }

    fn read(&self) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(self.buffer.contents().cast::<f32>(), self.len) }
            .to_vec()
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// The shuffle order, the one non-`f32` device buffer.
struct OrderBuf {
    buffer: Buffer,
    len: usize,
}

impl OrderBuf {
    fn zeros(device: &Device, len: usize) -> Self {
        let bytes = (len.max(1) * size_of::<u32>()) as u64;
        let buffer = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, bytes as usize) };
        Self { buffer, len }
    }

    fn write(&self, values: &[u32]) {
        assert!(values.len() <= self.len.max(1));
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.buffer.contents().cast::<u32>(),
                values.len(),
            );
        }
    }
}

struct DevLayer {
    w: Buf,
    b: Buf,
    mw: Buf,
    vw: Buf,
    mb: Buf,
    vb: Buf,
    gw: Buf,
    gb: Buf,
    a: Buf,
    d: Buf,
}

struct State {
    gpu: Rc<Gpu>,
    queue: CommandQueue,
    layers: Vec<DevLayer>,
    input: Buf,
    target: Buf,
    loss: Buf,
    all_input: Option<Buf>,
    all_target: Option<Buf>,
    order: Option<OrderBuf>,
}

fn set_i32(encoder: &ComputeCommandEncoderRef, index: u64, value: i32) {
    encoder.set_bytes(
        index,
        size_of::<i32>() as u64,
        std::ptr::addr_of!(value).cast::<c_void>(),
    );
}

fn set_f32(encoder: &ComputeCommandEncoderRef, index: u64, value: f32) {
    encoder.set_bytes(
        index,
        size_of::<f32>() as u64,
        std::ptr::addr_of!(value).cast::<c_void>(),
    );
}

/// One thread per element, `LANES` threads to a threadgroup.
fn dispatch_1d(encoder: &ComputeCommandEncoderRef, elements: usize) {
    if elements == 0 {
        return;
    }
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (elements as u64).div_ceil(LANES),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: LANES,
            height: 1,
            depth: 1,
        },
    );
}

/// One thread per output element of an `[m, n]` matrix, in `TILE` x `TILE`
/// threadgroups so the tiles the kernel stages in threadgroup memory line up.
fn dispatch_tiles(encoder: &ComputeCommandEncoderRef, m: usize, n: usize) {
    if m == 0 || n == 0 {
        return;
    }
    encoder.dispatch_thread_groups(
        MTLSize {
            width: (n as u64).div_ceil(TILE),
            height: (m as u64).div_ceil(TILE),
            depth: 1,
        },
        MTLSize {
            width: TILE,
            height: TILE,
            depth: 1,
        },
    );
}

pub(crate) fn fit_metal(
    net: &mut Network,
    dataset: &Dataset,
    config: TrainConfig,
    device: usize,
    budget: usize,
) -> Result<TrainingHistory, NetworkError> {
    let mut session = MetalTrainingSession::new(net, dataset, config, device, budget)?;
    let mut history = Vec::with_capacity(config.epochs);
    for _ in 0..config.epochs {
        history.push(session.train_epoch()?);
    }
    session.synchronize_network(net)?;
    Ok(TrainingHistory { losses: history })
}

/// Persistent Metal state for epoch-at-a-time training and explicit snapshots.
pub struct MetalTrainingSession {
    state: State,
    model: Network,
    /// Flat row-major copies, kept only when the dataset did not fit the
    /// budget. When it did, the device buffers hold the only copy.
    host: Option<HostDataset>,
    rows: usize,
    input_width: usize,
    target_width: usize,
    order: Vec<usize>,
    config: TrainConfig,
    batch_size: usize,
    epoch: usize,
    stats: AcceleratorStats,
    staged_input: Vec<f32>,
    staged_target: Vec<f32>,
}

struct HostDataset {
    inputs: Vec<f32>,
    targets: Vec<f32>,
}

impl MetalTrainingSession {
    pub fn new(
        net: &Network,
        dataset: &Dataset,
        config: TrainConfig,
        device: usize,
        budget_mib: usize,
    ) -> Result<Self, NetworkError> {
        let started = Instant::now();
        if dataset.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }
        if net.loss != Loss::Mse {
            return Err(NetworkError::UnsupportedMetal(
                "only MSE loss is implemented".into(),
            ));
        }
        for layer in &net.layers {
            act(layer.activation)?;
        }
        for (x, y) in dataset.inputs.iter().zip(&dataset.targets) {
            net.validate_input(x)?;
            net.validate_target(y)?;
        }
        let mut batch = config.batch_size.max(1).min(dataset.len());
        while estimate_tensor_memory_mib(net, batch)? > budget_mib {
            if batch == 1 {
                return Err(NetworkError::MetalMemoryBudget {
                    estimated_mib: estimate_tensor_memory_mib(net, 1)?,
                    budget_mib,
                });
            }
            batch = batch.div_ceil(2);
        }
        let xs: Vec<f32> = dataset.inputs.iter().flatten().copied().collect();
        let ys: Vec<f32> = dataset.targets.iter().flatten().copied().collect();
        let dataset_bytes =
            (xs.len() + ys.len()) * size_of::<f32>() + dataset.len() * size_of::<u32>();
        let resident = estimate_tensor_memory_mib(net, batch)?
            .saturating_mul(MIB)
            .saturating_add(dataset_bytes)
            <= budget_mib.saturating_mul(MIB);
        let state = State::new(
            net,
            batch,
            device,
            resident.then_some((&xs, &ys, dataset.len())),
        )?;
        let parameter_bytes = net
            .layers
            .iter()
            .map(|l| (l.weights.data.len() + l.biases.data.len()) * 4 * size_of::<f32>())
            .sum::<usize>();
        let peak = state.allocated_bytes();
        if peak > budget_mib.saturating_mul(MIB) {
            return Err(NetworkError::MetalMemoryBudget {
                estimated_mib: peak.div_ceil(MIB),
                budget_mib,
            });
        }
        Ok(Self {
            state,
            model: net.clone(),
            host: (!resident).then(|| HostDataset {
                inputs: xs,
                targets: ys,
            }),
            rows: dataset.len(),
            input_width: net.input_size,
            target_width: net.output_size(),
            order: (0..dataset.len()).collect(),
            config,
            batch_size: batch,
            epoch: 0,
            staged_input: Vec::with_capacity(batch * net.input_size),
            staged_target: Vec::with_capacity(batch * net.output_size()),
            stats: AcceleratorStats {
                peak_allocated_bytes: peak,
                setup_time: started.elapsed(),
                host_to_device_bytes: parameter_bytes
                    + if resident {
                        dataset_bytes - dataset.len() * size_of::<u32>()
                    } else {
                        0
                    },
                dataset_resident: resident,
                ..Default::default()
            },
        })
    }

    /// Restores host checkpoint state and reconstructs the deterministic
    /// shuffle position before creating fresh device resources.
    pub fn from_checkpoint(
        checkpoint: CudaTrainingCheckpoint,
        dataset: &Dataset,
        mut config: TrainConfig,
        device: usize,
        budget_mib: usize,
    ) -> Result<Self, NetworkError> {
        let completed_epochs = checkpoint.epoch;
        config.seed = checkpoint.shuffle_seed;
        let mut model = Network::builder()
            .input_size(checkpoint.input_size)
            .dense(1, Activation::Linear)
            .build();
        model.restore_cuda_checkpoint(checkpoint)?;
        let mut session = Self::new(&model, dataset, config, device, budget_mib)?;
        if config.shuffle {
            for epoch in 0..completed_epochs {
                if let Some(seed) = config.seed {
                    session
                        .order
                        .shuffle(&mut StdRng::seed_from_u64(seed + epoch as u64));
                } else {
                    return Err(NetworkError::InvalidCudaCheckpoint(
                        "cannot deterministically resume an unseeded shuffle".into(),
                    ));
                }
            }
        }
        session.epoch = completed_epochs;
        Ok(session)
    }

    pub fn train_epoch(&mut self) -> Result<f32, NetworkError> {
        let started = Instant::now();
        if self.config.shuffle {
            if let Some(seed) = self.config.seed {
                self.order
                    .shuffle(&mut StdRng::seed_from_u64(seed + self.epoch as u64));
            } else {
                self.order.shuffle(&mut rand::thread_rng());
            }
        }
        let batches = self.rows.div_ceil(self.batch_size);
        let loss = autoreleasepool(|| -> Result<f32, NetworkError> {
            // The previous epoch's command buffer has completed, so writing
            // these shared buffers from the CPU now races with nothing.
            self.state.loss.write(&[0.0]);
            if let Some(order) = &self.state.order {
                let indices: Vec<u32> = self.order.iter().map(|&i| i as u32).collect();
                order.write(&indices);
                self.stats.host_to_device_bytes += indices.len() * size_of::<u32>();
            }
            if self.host.is_some() {
                // Staging rewrites the shared input buffer between batches, so
                // each batch has to have finished on the GPU before the next
                // one is written. This path only runs when the dataset did not
                // fit the memory budget.
                for start in (0..self.rows).step_by(self.batch_size) {
                    let rows = self.batch_size.min(self.rows - start);
                    self.stage_batch(start, rows);
                    let command = self.state.queue.new_command_buffer();
                    let encoder = command.new_compute_command_encoder();
                    self.state
                        .train_batch(encoder, &mut self.model, rows, batches)?;
                    encoder.end_encoding();
                    command.commit();
                    command.wait_until_completed();
                    self.stats.batches += 1;
                }
            } else {
                // Everything the epoch needs is already on the device, so the
                // whole epoch is one command buffer. Dispatches inside a
                // compute encoder run in order, which is what makes the
                // read-modify-write chain across batches safe.
                let command = self.state.queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                for start in (0..self.rows).step_by(self.batch_size) {
                    let rows = self.batch_size.min(self.rows - start);
                    self.state.gather_batch(
                        encoder,
                        start,
                        rows,
                        self.input_width,
                        self.target_width,
                    );
                    self.state
                        .train_batch(encoder, &mut self.model, rows, batches)?;
                    self.stats.batches += 1;
                }
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
            }
            Ok(self.state.loss.read()[0])
        })?;
        self.stats.device_to_host_bytes += size_of::<f32>();
        if !loss.is_finite() {
            return Err(NetworkError::Metal(
                "numerical preflight failed: non-finite loss".into(),
            ));
        }
        self.epoch += 1;
        self.stats.epochs += 1;
        self.stats.training_time += started.elapsed();
        Ok(loss)
    }

    fn stage_batch(&mut self, start: usize, rows: usize) {
        let host = self.host.as_ref().expect("host staging invariant");
        let ids = &self.order[start..start + rows];
        self.staged_input.clear();
        self.staged_target.clear();
        for &i in ids {
            let x = i * self.input_width;
            let y = i * self.target_width;
            self.staged_input
                .extend_from_slice(&host.inputs[x..x + self.input_width]);
            self.staged_target
                .extend_from_slice(&host.targets[y..y + self.target_width]);
        }
        self.state.input.write(&self.staged_input);
        self.state.target.write(&self.staged_target);
        self.stats.host_to_device_bytes +=
            (self.staged_input.len() + self.staged_target.len()) * size_of::<f32>();
    }

    pub fn checkpoint(&mut self) -> Result<CudaTrainingCheckpoint, NetworkError> {
        let started = Instant::now();
        self.state.copy_back(&mut self.model);
        self.stats.device_to_host_bytes += self.state.checkpoint_bytes();
        self.stats.checkpoint_time += started.elapsed();
        Ok(self.model.cuda_checkpoint(self.epoch, self.config.seed))
    }

    pub fn synchronize_network(&mut self, net: &mut Network) -> Result<(), NetworkError> {
        net.restore_cuda_checkpoint(self.checkpoint()?)
    }

    pub fn stats(&self) -> &AcceleratorStats {
        &self.stats
    }

    pub fn epoch(&self) -> usize {
        self.epoch
    }
}

impl State {
    fn new(
        net: &Network,
        batch: usize,
        device: usize,
        resident: Option<(&[f32], &[f32], usize)>,
    ) -> Result<Self, NetworkError> {
        let gpu = gpu(device)?;
        let queue = gpu.device.new_command_queue();
        let handle = &gpu.device;
        let mut layers = Vec::new();
        for (i, l) in net.layers.iter().enumerate() {
            let a = batch * l.weights.rows;
            layers.push(DevLayer {
                w: Buf::from_slice(handle, &l.weights.data),
                b: Buf::from_slice(handle, &l.biases.data),
                mw: Buf::from_slice(handle, &net.adam_m_weights[i].data),
                vw: Buf::from_slice(handle, &net.adam_v_weights[i].data),
                mb: Buf::from_slice(handle, &net.adam_m_biases[i].data),
                vb: Buf::from_slice(handle, &net.adam_v_biases[i].data),
                gw: Buf::zeros(handle, l.weights.data.len()),
                gb: Buf::zeros(handle, l.biases.data.len()),
                a: Buf::zeros(handle, a),
                d: Buf::zeros(handle, a),
            });
        }
        let (all_input, all_target, order) = match resident {
            Some((xs, ys, rows)) => (
                Some(Buf::from_slice(handle, xs)),
                Some(Buf::from_slice(handle, ys)),
                Some(OrderBuf::zeros(handle, rows)),
            ),
            None => (None, None, None),
        };
        let input = Buf::zeros(handle, batch * net.input_size);
        let target = Buf::zeros(handle, batch * net.output_size());
        let loss = Buf::zeros(handle, 1);
        Ok(Self {
            gpu: gpu.clone(),
            queue,
            layers,
            input,
            target,
            loss,
            all_input,
            all_target,
            order,
        })
    }

    fn gather_batch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        start: usize,
        rows: usize,
        input_width: usize,
        target_width: usize,
    ) {
        let all_input = self.all_input.as_ref().expect("resident input invariant");
        let all_target = self.all_target.as_ref().expect("resident target invariant");
        let order = self.order.as_ref().expect("resident order invariant");
        for (destination, source, width) in [
            (&self.input, all_input, input_width),
            (&self.target, all_target, target_width),
        ] {
            encoder.set_compute_pipeline_state(&self.gpu.pipelines.gather_rows);
            encoder.set_buffer(0, Some(&destination.buffer), 0);
            encoder.set_buffer(1, Some(&source.buffer), 0);
            encoder.set_buffer(2, Some(&order.buffer), 0);
            set_i32(encoder, 3, start as i32);
            set_i32(encoder, 4, rows as i32);
            set_i32(encoder, 5, width as i32);
            dispatch_1d(encoder, rows * width);
        }
    }

    fn checkpoint_bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|l| l.w.len() + l.b.len() + l.mw.len() + l.vw.len() + l.mb.len() + l.vb.len())
            .sum::<usize>()
            * size_of::<f32>()
    }

    fn allocated_bytes(&self) -> usize {
        let layers = self
            .layers
            .iter()
            .map(|l| {
                l.w.len()
                    + l.b.len()
                    + l.mw.len()
                    + l.vw.len()
                    + l.mb.len()
                    + l.vb.len()
                    + l.gw.len()
                    + l.gb.len()
                    + l.a.len()
                    + l.d.len()
            })
            .sum::<usize>()
            * size_of::<f32>();
        layers
            + (self.input.len()
                + self.target.len()
                + self.loss.len()
                + self.all_input.as_ref().map_or(0, Buf::len)
                + self.all_target.as_ref().map_or(0, Buf::len))
                * size_of::<f32>()
            + self.order.as_ref().map_or(0, |o| o.len) * size_of::<u32>()
    }

    fn gemm_nt(
        &self,
        encoder: &ComputeCommandEncoderRef,
        a: &Buf,
        w: &Buf,
        out: &Buf,
        batch: usize,
        input: usize,
        units: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.gpu.pipelines.gemm_nt);
        encoder.set_buffer(0, Some(&out.buffer), 0);
        encoder.set_buffer(1, Some(&a.buffer), 0);
        encoder.set_buffer(2, Some(&w.buffer), 0);
        set_i32(encoder, 3, batch as i32);
        set_i32(encoder, 4, units as i32);
        set_i32(encoder, 5, input as i32);
        dispatch_tiles(encoder, batch, units);
    }

    /// `gw[out, in] = (1/batch) * d[batch, out]^T . x[batch, in]`, row-major.
    fn grad_weights(
        &self,
        encoder: &ComputeCommandEncoderRef,
        d: &Buf,
        x: &Buf,
        gw: &Buf,
        batch: usize,
        out: usize,
        input: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.gpu.pipelines.gemm_tn);
        encoder.set_buffer(0, Some(&gw.buffer), 0);
        encoder.set_buffer(1, Some(&d.buffer), 0);
        encoder.set_buffer(2, Some(&x.buffer), 0);
        set_i32(encoder, 3, out as i32);
        set_i32(encoder, 4, input as i32);
        set_i32(encoder, 5, batch as i32);
        set_f32(encoder, 6, 1.0 / batch as f32);
        dispatch_tiles(encoder, out, input);
    }

    /// `prev[batch, in] = next[batch, out] . w[out, in]`, row-major. The
    /// activation derivative is applied afterwards by `act_derivative`.
    fn propagate_delta(
        &self,
        encoder: &ComputeCommandEncoderRef,
        next: &Buf,
        w: &Buf,
        prev: &Buf,
        batch: usize,
        out: usize,
        input: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.gpu.pipelines.gemm_nn);
        encoder.set_buffer(0, Some(&prev.buffer), 0);
        encoder.set_buffer(1, Some(&next.buffer), 0);
        encoder.set_buffer(2, Some(&w.buffer), 0);
        set_i32(encoder, 3, batch as i32);
        set_i32(encoder, 4, input as i32);
        set_i32(encoder, 5, out as i32);
        dispatch_tiles(encoder, batch, input);
    }

    fn train_batch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        net: &mut Network,
        b: usize,
        epoch_batches: usize,
    ) -> Result<(), NetworkError> {
        for i in 0..self.layers.len() {
            let input_units = if i == 0 {
                net.input_size
            } else {
                net.layers[i - 1].weights.rows
            };
            let out_units = net.layers[i].weights.rows;
            let source = if i == 0 {
                &self.input
            } else {
                &self.layers[i - 1].a
            };
            let l = &self.layers[i];
            self.gemm_nt(encoder, source, &l.w, &l.a, b, input_units, out_units);
            encoder.set_compute_pipeline_state(&self.gpu.pipelines.bias_act);
            encoder.set_buffer(0, Some(&l.a.buffer), 0);
            encoder.set_buffer(1, Some(&l.b.buffer), 0);
            set_i32(encoder, 2, b as i32);
            set_i32(encoder, 3, out_units as i32);
            set_i32(encoder, 4, act(net.layers[i].activation)?);
            dispatch_1d(encoder, b * out_units);
        }
        let last = self.layers.len() - 1;
        let outputs = b * net.output_size();
        {
            let l = &self.layers[last];
            encoder.set_compute_pipeline_state(&self.gpu.pipelines.output_delta);
            encoder.set_buffer(0, Some(&l.d.buffer), 0);
            encoder.set_buffer(1, Some(&l.a.buffer), 0);
            encoder.set_buffer(2, Some(&self.target.buffer), 0);
            set_i32(encoder, 3, outputs as i32);
            set_i32(encoder, 4, act(net.layers[last].activation)?);
            dispatch_1d(encoder, outputs);
        }
        for i in (0..self.layers.len()).rev() {
            let out = net.layers[i].weights.rows;
            let input = net.layers[i].weights.cols;
            let source = if i == 0 {
                &self.input
            } else {
                &self.layers[i - 1].a
            };
            let l = &self.layers[i];
            self.grad_weights(encoder, &l.d, source, &l.gw, b, out, input);
            encoder.set_compute_pipeline_state(&self.gpu.pipelines.grad_b);
            encoder.set_buffer(0, Some(&l.gb.buffer), 0);
            encoder.set_buffer(1, Some(&l.d.buffer), 0);
            set_i32(encoder, 2, b as i32);
            set_i32(encoder, 3, out as i32);
            dispatch_1d(encoder, out);
            if i > 0 {
                let previous = &self.layers[i - 1];
                self.propagate_delta(encoder, &l.d, &l.w, &previous.d, b, out, input);
                encoder.set_compute_pipeline_state(&self.gpu.pipelines.act_derivative);
                encoder.set_buffer(0, Some(&previous.d.buffer), 0);
                encoder.set_buffer(1, Some(&previous.a.buffer), 0);
                set_i32(encoder, 2, (b * input) as i32);
                set_i32(encoder, 3, act(net.layers[i - 1].activation)?);
                dispatch_1d(encoder, b * input);
            }
        }
        self.update(encoder, net);
        encoder.set_compute_pipeline_state(&self.gpu.pipelines.mse_epoch_sum);
        encoder.set_buffer(0, Some(&self.loss.buffer), 0);
        encoder.set_buffer(1, Some(&self.layers[last].a.buffer), 0);
        encoder.set_buffer(2, Some(&self.target.buffer), 0);
        set_i32(encoder, 3, outputs as i32);
        set_f32(encoder, 4, 1.0 / (outputs * epoch_batches) as f32);
        encoder.set_threadgroup_memory_length(0, REDUCE_WIDTH * size_of::<f32>() as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: REDUCE_WIDTH,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    fn update(&self, encoder: &ComputeCommandEncoderRef, net: &mut Network) {
        let optimizer = net.optimizer.clone();
        if matches!(optimizer, Optimizer::Adam { .. }) {
            net.adam_step += 1;
        }
        for l in &self.layers {
            match optimizer {
                Optimizer::Sgd { learning_rate } => {
                    for (parameter, gradient) in [(&l.w, &l.gw), (&l.b, &l.gb)] {
                        encoder.set_compute_pipeline_state(&self.gpu.pipelines.sgd);
                        encoder.set_buffer(0, Some(&parameter.buffer), 0);
                        encoder.set_buffer(1, Some(&gradient.buffer), 0);
                        set_i32(encoder, 2, parameter.len() as i32);
                        set_f32(encoder, 3, learning_rate);
                        dispatch_1d(encoder, parameter.len());
                    }
                }
                Optimizer::Adam {
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                } => {
                    let step = net.adam_step as i32;
                    let c1 = 1. - beta1.powi(step);
                    let c2 = 1. - beta2.powi(step);
                    // Biases are excluded from decay, exactly as on CUDA.
                    for (parameter, gradient, m, v, decay) in [
                        (&l.w, &l.gw, &l.mw, &l.vw, weight_decay),
                        (&l.b, &l.gb, &l.mb, &l.vb, 0.0),
                    ] {
                        encoder.set_compute_pipeline_state(&self.gpu.pipelines.adam);
                        encoder.set_buffer(0, Some(&parameter.buffer), 0);
                        encoder.set_buffer(1, Some(&gradient.buffer), 0);
                        encoder.set_buffer(2, Some(&m.buffer), 0);
                        encoder.set_buffer(3, Some(&v.buffer), 0);
                        set_i32(encoder, 4, parameter.len() as i32);
                        set_f32(encoder, 5, learning_rate);
                        set_f32(encoder, 6, beta1);
                        set_f32(encoder, 7, beta2);
                        set_f32(encoder, 8, epsilon);
                        set_f32(encoder, 9, c1);
                        set_f32(encoder, 10, c2);
                        set_f32(encoder, 11, decay);
                        dispatch_1d(encoder, parameter.len());
                    }
                }
            }
        }
    }

    fn copy_back(&self, net: &mut Network) {
        for (i, l) in self.layers.iter().enumerate() {
            net.layers[i].weights.data = l.w.read();
            net.layers[i].biases.data = l.b.read();
            net.adam_m_weights[i].data = l.mw.read();
            net.adam_v_weights[i].data = l.vw.read();
            net.adam_m_biases[i].data = l.mb.read();
            net.adam_v_biases[i].data = l.vb.read();
        }
    }
}
