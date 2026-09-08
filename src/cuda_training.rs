//! Fail-closed FP32 CUDA fitting.  All state below is owned by `CudaSlice`s;
//! host pointers are only used by cudarc's checked copy operations.
use crate::{
    Activation, CudaTrainingCheckpoint, Dataset, Loss, Network, NetworkError, Optimizer,
    TrainConfig, TrainingHistory,
};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{Ptx, compile_ptx};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

const MIB: usize = 1024 * 1024;
const KERNELS: &str = r#"
extern "C" __global__ void bias_act(float *x,const float*b,int rows,int cols,int act){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*cols){float v=x[i]+b[i%cols];if(act==1)v=v>0?v:0;else if(act==2)v=1.f/(1.f+expf(-v));else if(act==3)v=tanhf(v);x[i]=v;}}
extern "C" __global__ void output_delta(float*d,const float*y,const float*t,int n,int act){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=t[i]-y[i];float a=y[i];if(act==1)v*=a>0;else if(act==2)v*=a*(1-a);else if(act==3)v*=1-a*a;d[i]=v;}}
extern "C" __global__ void act_derivative(float*d,const float*a,int n,int act){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=a[i];if(act==1)d[i]*=v>0;else if(act==2)d[i]*=v*(1-v);else if(act==3)d[i]*=1-v*v;}}
extern "C" __global__ void grad_b(float*g,const float*d,int batch,int out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<out){float s=0;for(int b=0;b<batch;b++)s+=d[b*out+i];g[i]=s/(float)batch;}}
extern "C" __global__ void sgd(float*x,const float*g,int n,float lr){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=lr*g[i];}
extern "C" __global__ void adam(float*x,const float*g,float*m,float*v,int n,float lr,float b1,float b2,float eps,float c1,float c2,float wd){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float q=g[i];float mi=b1*m[i]+(1-b1)*q;float vi=b2*v[i]+(1-b2)*q*q;m[i]=mi;v[i]=vi;x[i]-=lr*wd*x[i];x[i]+=lr*(mi/c1)/(sqrtf(vi/c2)+eps);}}
extern "C" __global__ void mse_sum(float *out,const float*y,const float*t,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=y[i]-t[i];atomicAdd(out,z*z/(float)n);}}
extern "C" __global__ void mse_epoch_sum(float *out,const float*y,const float*t,int n,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=y[i]-t[i];atomicAdd(out,z*z*scale);}}
extern "C" __global__ void gather_rows(float*out,const float*all,const unsigned int*order,int start,int rows,int width){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*width){int r=i/width,c=i%width;out[i]=all[order[start+r]*width+c];}}
"#;

/// Measurements for the lifetime of a persistent CUDA training session.
#[derive(Clone, Debug, Default)]
pub struct CudaTrainingStats {
    /// Bytes owned by CUDA allocations at the high-water mark (driver overhead excluded).
    pub peak_allocated_bytes: usize,
    pub setup_time: Duration,
    pub training_time: Duration,
    pub checkpoint_time: Duration,
    pub epochs: usize,
    pub batches: usize,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub dataset_resident: bool,
}

/// Conservative tensor allocation estimate (MiB, rounded up). This is public
/// so applications can reject an unsuitable configuration before touching CUDA.
pub fn estimate_tensor_memory_mib(
    network: &Network,
    batch_size: usize,
) -> Result<usize, NetworkError> {
    if batch_size == 0 {
        return Err(NetworkError::Cuda("batch size must be non-zero".into()));
    }
    let mut floats = 0usize;
    let input = network.input_size;
    floats = floats
        .checked_add(
            batch_size
                .checked_mul(input)
                .ok_or_else(|| NetworkError::Cuda("tensor size overflow".into()))?,
        )
        .ok_or_else(|| NetworkError::Cuda("tensor size overflow".into()))?;
    for l in &network.layers {
        let p = l.weights.data.len() + l.biases.data.len();
        // parameter, gradient, and both Adam moments (also allocate for SGD for a stable upper bound)
        floats = floats
            .checked_add(
                p.checked_mul(4)
                    .ok_or_else(|| NetworkError::Cuda("tensor size overflow".into()))?,
            )
            .ok_or_else(|| NetworkError::Cuda("tensor size overflow".into()))?;
        let a = batch_size
            .checked_mul(l.weights.rows)
            .ok_or_else(|| NetworkError::Cuda("tensor size overflow".into()))?;
        // activation and delta
        floats = floats
            .checked_add(
                a.checked_mul(2)
                    .ok_or_else(|| NetworkError::Cuda("tensor size overflow".into()))?,
            )
            .ok_or_else(|| NetworkError::Cuda("tensor size overflow".into()))?;
    }
    floats = floats
        .checked_add(1)
        .ok_or_else(|| NetworkError::Cuda("tensor size overflow".into()))?;
    Ok(floats
        .checked_mul(4)
        .ok_or_else(|| NetworkError::Cuda("tensor size overflow".into()))?
        .div_ceil(MIB))
}

#[derive(Clone, Debug)]
pub struct CudaDoctorReport {
    pub device: usize,
    pub name: String,
    pub compute_capability: String,
    pub total_memory_mib: usize,
    pub free_memory_mib: Option<usize>,
    pub requested_budget_mib: usize,
    pub allocation_test: bool,
    pub cublas_available: bool,
    pub kernel_available: bool,
}

pub fn cuda_doctor(
    device: usize,
    requested_budget_mib: usize,
) -> Result<CudaDoctorReport, NetworkError> {
    let ctx = CudaContext::new(device).map_err(|e| {
        NetworkError::Cuda(format!(
            "CUDA driver/device {device} initialization failed: {e}"
        ))
    })?;
    let stream = ctx.new_stream().map_err(cuda_err("stream creation"))?;
    let _blas = CudaBlas::new(stream.clone()).map_err(cuda_err("cuBLAS initialization"))?;
    let ptx = compile_ptx(KERNELS)
        .map_err(|e| NetworkError::Cuda(format!("CUDA kernel compilation failed: {e}")))?;
    let module = ctx
        .load_module(ptx)
        .map_err(cuda_err("CUDA module loading"))?;
    let _ = module
        .load_function("sgd")
        .map_err(cuda_err("CUDA kernel lookup"))?;
    let allocation_test = stream.alloc_zeros::<f32>(256).is_ok();
    let total = unsafe { cudarc::driver::result::device::total_mem(ctx.cu_device()) }
        .map_err(cuda_err("CUDA memory query"))?
        / MIB;
    let cc = ctx
        .compute_capability()
        .map(|(a, b)| format!("{a}.{b}"))
        .unwrap_or_else(|_| "unknown".into());
    let free_memory_mib = cudarc::runtime::result::get_mem_info()
        .ok()
        .map(|(free, _)| free / MIB);
    Ok(CudaDoctorReport {
        device,
        name: ctx.name().unwrap_or_else(|_| "NVIDIA CUDA device".into()),
        compute_capability: cc,
        total_memory_mib: total,
        free_memory_mib,
        requested_budget_mib,
        allocation_test,
        cublas_available: true,
        kernel_available: true,
    })
}

fn cuda_err<E: std::fmt::Display>(stage: &'static str) -> impl FnOnce(E) -> NetworkError {
    move |e| NetworkError::Cuda(format!("{stage} failed: {e}"))
}

/// nvrtc compilation of [`KERNELS`] costs a few hundred milliseconds and the
/// source never varies, so a campaign running dozens of sessions compiles once.
fn kernel_ptx() -> Result<&'static Ptx, NetworkError> {
    static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| compile_ptx(KERNELS).map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| NetworkError::Cuda(format!("CUDA kernel compilation failed: {e}")))
}

/// Creating a `CudaContext` and loading a module are per-process costs, not
/// per-session ones. Device buffers still belong to their session and are freed
/// when it drops; only the context and the compiled module are shared.
fn device_context(device: usize) -> Result<(Arc<CudaContext>, Arc<CudaModule>), NetworkError> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, (Arc<CudaContext>, Arc<CudaModule>)>>> =
        OnceLock::new();
    let mut contexts = CONTEXTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| NetworkError::Cuda("CUDA context cache lock was poisoned".into()))?;
    if let Some(entry) = contexts.get(&device) {
        return Ok(entry.clone());
    }
    let ctx =
        CudaContext::new(device).map_err(cuda_err("CUDA driver/device initialization"))?;
    let module = ctx
        .load_module(kernel_ptx()?.clone())
        .map_err(cuda_err("CUDA module loading"))?;
    contexts.insert(device, (ctx.clone(), module.clone()));
    Ok((ctx, module))
}

/// Every kernel handle resolved once. `CudaModule::load_function` is a driver
/// lookup; calling it inside the batch loop cost more than the kernels it found.
struct Kernels {
    bias_act: CudaFunction,
    output_delta: CudaFunction,
    act_derivative: CudaFunction,
    grad_b: CudaFunction,
    sgd: CudaFunction,
    adam: CudaFunction,
    mse_epoch_sum: CudaFunction,
    gather_rows: CudaFunction,
}

impl Kernels {
    fn load(module: &Arc<CudaModule>) -> Result<Self, NetworkError> {
        let get = |name: &str| {
            module
                .load_function(name)
                .map_err(cuda_err("CUDA kernel lookup"))
        };
        Ok(Self {
            bias_act: get("bias_act")?,
            output_delta: get("output_delta")?,
            act_derivative: get("act_derivative")?,
            grad_b: get("grad_b")?,
            sgd: get("sgd")?,
            adam: get("adam")?,
            mse_epoch_sum: get("mse_epoch_sum")?,
            gather_rows: get("gather_rows")?,
        })
    }
}
fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(256), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn act(a: Activation) -> Result<i32, NetworkError> {
    match a {
        Activation::Linear => Ok(0),
        Activation::Relu => Ok(1),
        Activation::Sigmoid => Ok(2),
        Activation::Tanh => Ok(3),
        Activation::Softmax => Err(NetworkError::UnsupportedCuda("Softmax activation".into())),
    }
}

struct DevLayer {
    w: CudaSlice<f32>,
    b: CudaSlice<f32>,
    mw: CudaSlice<f32>,
    vw: CudaSlice<f32>,
    mb: CudaSlice<f32>,
    vb: CudaSlice<f32>,
    gw: CudaSlice<f32>,
    gb: CudaSlice<f32>,
    a: CudaSlice<f32>,
    d: CudaSlice<f32>,
}
struct State {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    kernels: Kernels,
    layers: Vec<DevLayer>,
    input: CudaSlice<f32>,
    target: CudaSlice<f32>,
    loss: CudaSlice<f32>,
    all_input: Option<CudaSlice<f32>>,
    all_target: Option<CudaSlice<f32>>,
    order: Option<CudaSlice<u32>>,
}

pub(crate) fn fit_cuda(
    net: &mut Network,
    dataset: &Dataset,
    config: TrainConfig,
    device: usize,
    budget: usize,
) -> Result<TrainingHistory, NetworkError> {
    let mut session = CudaTrainingSession::new(net, dataset, config, device, budget)?;
    let mut history = Vec::with_capacity(config.epochs);
    for _ in 0..config.epochs {
        history.push(session.train_epoch()?);
    }
    session.synchronize_network(net)?;
    Ok(TrainingHistory { losses: history })
}

/// Persistent CUDA state for epoch-at-a-time training and explicit snapshots.
pub struct CudaTrainingSession {
    state: State,
    model: Network,
    /// Flat row-major copies, kept only when the dataset did not fit on the
    /// device. When it did, the device owns the only copy and holding a second
    /// one on the host wasted as much memory as the dataset itself.
    host: Option<HostDataset>,
    rows: usize,
    input_width: usize,
    target_width: usize,
    order: Vec<usize>,
    config: TrainConfig,
    batch_size: usize,
    epoch: usize,
    stats: CudaTrainingStats,
    staged_input: Vec<f32>,
    staged_target: Vec<f32>,
}

struct HostDataset {
    inputs: Vec<f32>,
    targets: Vec<f32>,
}

impl CudaTrainingSession {
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
            return Err(NetworkError::UnsupportedCuda(
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
                return Err(NetworkError::CudaMemoryBudget {
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
            return Err(NetworkError::CudaMemoryBudget {
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
            stats: CudaTrainingStats {
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
        if let Some(device_order) = &mut self.state.order {
            let order: Vec<u32> = self.order.iter().map(|&i| i as u32).collect();
            self.state
                .stream
                .memcpy_htod(&order, device_order)
                .map_err(cuda_err("shuffle order upload"))?;
            self.stats.host_to_device_bytes += order.len() * size_of::<u32>();
        }
        self.state
            .stream
            .memset_zeros(&mut self.state.loss)
            .map_err(cuda_err("epoch loss reset"))?;
        let batches = self.rows.div_ceil(self.batch_size);
        for start in (0..self.rows).step_by(self.batch_size) {
            let rows = self.batch_size.min(self.rows - start);
            if let Some(host) = &self.host {
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
                self.state
                    .stream
                    .memcpy_htod(&self.staged_input, &mut self.state.input)
                    .map_err(cuda_err("input upload"))?;
                self.state
                    .stream
                    .memcpy_htod(&self.staged_target, &mut self.state.target)
                    .map_err(cuda_err("target upload"))?;
                self.stats.host_to_device_bytes +=
                    (self.staged_input.len() + self.staged_target.len()) * size_of::<f32>();
            } else {
                self.state
                    .gather_batch(start, rows, self.input_width, self.target_width)?;
            }
            self.state.train_batch(&mut self.model, rows, batches)?;
            self.stats.batches += 1;
        }
        self.state
            .stream
            .synchronize()
            .map_err(cuda_err("epoch synchronization"))?;
        let loss = self
            .state
            .stream
            .clone_dtoh(&self.state.loss)
            .map_err(cuda_err("epoch loss download"))?[0];
        self.stats.device_to_host_bytes += size_of::<f32>();
        if !loss.is_finite() {
            return Err(NetworkError::Cuda(
                "numerical preflight failed: non-finite loss".into(),
            ));
        }
        self.epoch += 1;
        self.stats.epochs += 1;
        self.stats.training_time += started.elapsed();
        Ok(loss)
    }

    pub fn checkpoint(&mut self) -> Result<CudaTrainingCheckpoint, NetworkError> {
        let started = Instant::now();
        self.state.copy_back(&mut self.model)?;
        self.stats.device_to_host_bytes += self.state.checkpoint_bytes();
        self.stats.checkpoint_time += started.elapsed();
        Ok(self.model.cuda_checkpoint(self.epoch, self.config.seed))
    }
    pub fn synchronize_network(&mut self, net: &mut Network) -> Result<(), NetworkError> {
        net.restore_cuda_checkpoint(self.checkpoint()?)
    }
    pub fn stats(&self) -> &CudaTrainingStats {
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
        let (ctx, module) = device_context(device)?;
        let stream = ctx.new_stream().map_err(cuda_err("stream creation"))?;
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_err("cuBLAS initialization"))?;
        let kernels = Kernels::load(&module)?;
        let mut layers = Vec::new();
        for (i, l) in net.layers.iter().enumerate() {
            let a = batch * l.weights.rows;
            let z = |v: &[f32]| {
                stream
                    .clone_htod(v)
                    .map_err(cuda_err("device allocation/upload"))
            };
            let zeros = |n| {
                stream
                    .alloc_zeros::<f32>(n)
                    .map_err(cuda_err("device allocation"))
            };
            layers.push(DevLayer {
                w: z(&l.weights.data)?,
                b: z(&l.biases.data)?,
                mw: z(&net.adam_m_weights[i].data)?,
                vw: z(&net.adam_v_weights[i].data)?,
                mb: z(&net.adam_m_biases[i].data)?,
                vb: z(&net.adam_v_biases[i].data)?,
                gw: zeros(l.weights.data.len())?,
                gb: zeros(l.biases.data.len())?,
                a: zeros(a)?,
                d: zeros(a)?,
            });
        }
        let (all_input, all_target, order) = if let Some((xs, ys, rows)) = resident {
            (
                Some(stream.clone_htod(xs).map_err(cuda_err("dataset upload"))?),
                Some(stream.clone_htod(ys).map_err(cuda_err("dataset upload"))?),
                Some(
                    stream
                        .alloc_zeros::<u32>(rows)
                        .map_err(cuda_err("order allocation"))?,
                ),
            )
        } else {
            (None, None, None)
        };
        Ok(Self {
            _ctx: ctx,
            stream: stream.clone(),
            blas,
            kernels,
            layers,
            input: stream
                .alloc_zeros(batch * net.input_size)
                .map_err(cuda_err("device allocation"))?,
            target: stream
                .alloc_zeros(batch * net.output_size())
                .map_err(cuda_err("device allocation"))?,
            loss: stream
                .alloc_zeros(1)
                .map_err(cuda_err("device allocation"))?,
            all_input,
            all_target,
            order,
        })
    }
    fn gather_batch(
        &mut self,
        start: usize,
        rows: usize,
        input_width: usize,
        target_width: usize,
    ) -> Result<(), NetworkError> {
        let f = &self.kernels.gather_rows;
        let all_input = self.all_input.as_ref().expect("resident input invariant");
        let all_target = self.all_target.as_ref().expect("resident target invariant");
        let order = self.order.as_ref().expect("resident order invariant");
        unsafe {
            self.stream
                .launch_builder(f)
                .arg(&mut self.input)
                .arg(all_input)
                .arg(order)
                .arg(&(start as i32))
                .arg(&(rows as i32))
                .arg(&(input_width as i32))
                .launch(cfg(rows * input_width))
                .map_err(cuda_err("input gather kernel"))?;
            self.stream
                .launch_builder(f)
                .arg(&mut self.target)
                .arg(all_target)
                .arg(order)
                .arg(&(start as i32))
                .arg(&(rows as i32))
                .arg(&(target_width as i32))
                .launch(cfg(rows * target_width))
                .map_err(cuda_err("target gather kernel"))?;
        }
        Ok(())
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
                + self.all_input.as_ref().map_or(0, CudaSlice::len)
                + self.all_target.as_ref().map_or(0, CudaSlice::len))
                * size_of::<f32>()
            + self.order.as_ref().map_or(0, CudaSlice::len) * size_of::<u32>()
    }
    fn train_batch(
        &mut self,
        net: &mut Network,
        b: usize,
        epoch_batches: usize,
    ) -> Result<(), NetworkError> {
        for i in 0..self.layers.len() {
            let units = if i == 0 {
                net.input_size
            } else {
                net.layers[i - 1].weights.rows
            };
            let out_units = net.layers[i].weights.rows;
            if i == 0 {
                let l = &mut self.layers[0];
                gemm(&self.blas, &self.input, &l.w, &mut l.a, b, units, out_units)?;
            } else {
                let (left, right) = self.layers.split_at_mut(i);
                let previous = &left[i - 1].a;
                let l = &mut right[0];
                gemm(&self.blas, previous, &l.w, &mut l.a, b, units, out_units)?;
            }
            let f = &self.kernels.bias_act;
            let l = &mut self.layers[i];
            unsafe {
                self.stream
                    .launch_builder(f)
                    .arg(&mut l.a)
                    .arg(&l.b)
                    .arg(&(b as i32))
                    .arg(&(net.layers[i].weights.rows as i32))
                    .arg(&act(net.layers[i].activation)?)
                    .launch(cfg(b * net.layers[i].weights.rows))
                    .map_err(cuda_err("bias/activation kernel"))?;
            }
        }
        let last = self.layers.len() - 1;
        let f = &self.kernels.output_delta;
        {
            let l = &mut self.layers[last];
            unsafe {
                self.stream
                    .launch_builder(f)
                    .arg(&mut l.d)
                    .arg(&l.a)
                    .arg(&self.target)
                    .arg(&((b * net.output_size()) as i32))
                    .arg(&act(net.layers[last].activation)?)
                    .launch(cfg(b * net.output_size()))
                    .map_err(cuda_err("output delta kernel"))?;
            }
        }
        for i in (0..self.layers.len()).rev() {
            let out = net.layers[i].weights.rows;
            let inp = net.layers[i].weights.cols;
            let fb = &self.kernels.grad_b;
            // GW[out, inp] = (1/b) * D^T . X. The old hand-written kernel gave
            // each of out*inp threads a serial loop over the batch with strided
            // reads; cuBLAS does the same reduction as a real GEMM.
            if i == 0 {
                let l = &mut self.layers[0];
                grad_weights(&self.blas, &l.d, &self.input, &mut l.gw, b, out, inp)?;
                unsafe {
                    self.stream
                        .launch_builder(fb)
                        .arg(&mut l.gb)
                        .arg(&l.d)
                        .arg(&(b as i32))
                        .arg(&(out as i32))
                        .launch(cfg(out))
                        .map_err(cuda_err("bias gradient kernel"))?;
                }
            } else {
                let (left, right) = self.layers.split_at_mut(i);
                let source = &left[i - 1].a;
                let l = &mut right[0];
                grad_weights(&self.blas, &l.d, source, &mut l.gw, b, out, inp)?;
                unsafe {
                    self.stream
                        .launch_builder(fb)
                        .arg(&mut l.gb)
                        .arg(&l.d)
                        .arg(&(b as i32))
                        .arg(&(out as i32))
                        .launch(cfg(out))
                        .map_err(cuda_err("bias gradient kernel"))?;
                }
            }
            if i > 0 {
                let fd = &self.kernels.act_derivative;
                let (left, right) = self.layers.split_at_mut(i);
                let prev = &mut left[i - 1];
                let cur = &right[0];
                // PD[b, inp] = D_next[b, out] . W[out, inp], then scaled in
                // place by the previous layer's activation derivative.
                propagate_delta(&self.blas, &cur.d, &cur.w, &mut prev.d, b, out, inp)?;
                unsafe {
                    self.stream
                        .launch_builder(fd)
                        .arg(&mut prev.d)
                        .arg(&prev.a)
                        .arg(&((b * inp) as i32))
                        .arg(&act(net.layers[i - 1].activation)?)
                        .launch(cfg(b * inp))
                        .map_err(cuda_err("activation derivative kernel"))?;
                }
            }
        }
        self.update(net)?;
        let fl = &self.kernels.mse_epoch_sum;
        let la = &self.layers[last].a;
        let n = b * net.output_size();
        let scale = 1.0 / (n * epoch_batches) as f32;
        unsafe {
            self.stream
                .launch_builder(fl)
                .arg(&mut self.loss)
                .arg(la)
                .arg(&self.target)
                .arg(&(n as i32))
                .arg(&scale)
                .launch(cfg(b * net.output_size()))
                .map_err(cuda_err("loss kernel"))?;
        }
        Ok(())
    }
    fn update(&mut self, net: &mut Network) -> Result<(), NetworkError> {
        let (f, adam) = match net.optimizer.clone() {
            Optimizer::Sgd { .. } => (&self.kernels.sgd, false),
            Optimizer::Adam { .. } => (&self.kernels.adam, true),
        };
        if adam {
            net.adam_step += 1;
        }
        for l in &mut self.layers {
            let nw = l.w.len();
            let nb = l.b.len();
            unsafe {
                if let Optimizer::Sgd { learning_rate } = net.optimizer {
                    self.stream
                        .launch_builder(f)
                        .arg(&mut l.w)
                        .arg(&l.gw)
                        .arg(&(nw as i32))
                        .arg(&learning_rate)
                        .launch(cfg(nw))
                        .map_err(cuda_err("SGD update kernel"))?;
                    self.stream
                        .launch_builder(f)
                        .arg(&mut l.b)
                        .arg(&l.gb)
                        .arg(&(nb as i32))
                        .arg(&learning_rate)
                        .launch(cfg(nb))
                        .map_err(cuda_err("SGD update kernel"))?;
                } else if let Optimizer::Adam {
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                } = net.optimizer
                {
                    let c1 = 1. - beta1.powi(net.adam_step as i32);
                    let c2 = 1. - beta2.powi(net.adam_step as i32);
                    let no_decay = 0.0f32;
                    self.stream
                        .launch_builder(f)
                        .arg(&mut l.w)
                        .arg(&l.gw)
                        .arg(&mut l.mw)
                        .arg(&mut l.vw)
                        .arg(&(nw as i32))
                        .arg(&learning_rate)
                        .arg(&beta1)
                        .arg(&beta2)
                        .arg(&epsilon)
                        .arg(&c1)
                        .arg(&c2)
                        .arg(&weight_decay)
                        .launch(cfg(nw))
                        .map_err(cuda_err("Adam update kernel"))?;
                    self.stream
                        .launch_builder(f)
                        .arg(&mut l.b)
                        .arg(&l.gb)
                        .arg(&mut l.mb)
                        .arg(&mut l.vb)
                        .arg(&(nb as i32))
                        .arg(&learning_rate)
                        .arg(&beta1)
                        .arg(&beta2)
                        .arg(&epsilon)
                        .arg(&c1)
                        .arg(&c2)
                        .arg(&no_decay)
                        .launch(cfg(nb))
                        .map_err(cuda_err("Adam update kernel"))?;
                }
            }
        }
        Ok(())
    }
    fn copy_back(&self, net: &mut Network) -> Result<(), NetworkError> {
        for (i, l) in self.layers.iter().enumerate() {
            net.layers[i].weights.data = self
                .stream
                .clone_dtoh(&l.w)
                .map_err(cuda_err("weight download"))?;
            net.layers[i].biases.data = self
                .stream
                .clone_dtoh(&l.b)
                .map_err(cuda_err("bias download"))?;
            net.adam_m_weights[i].data = self
                .stream
                .clone_dtoh(&l.mw)
                .map_err(cuda_err("moment download"))?;
            net.adam_v_weights[i].data = self
                .stream
                .clone_dtoh(&l.vw)
                .map_err(cuda_err("moment download"))?;
            net.adam_m_biases[i].data = self
                .stream
                .clone_dtoh(&l.mb)
                .map_err(cuda_err("moment download"))?;
            net.adam_v_biases[i].data = self
                .stream
                .clone_dtoh(&l.vb)
                .map_err(cuda_err("moment download"))?;
        }
        Ok(())
    }
}
fn gemm(
    blas: &CudaBlas,
    a: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    batch: usize,
    input: usize,
    units: usize,
) -> Result<(), NetworkError> {
    let c = GemmConfig {
        // `w` is stored row-major as [units, input]. cuBLAS reads the same
        // bytes as column-major [input, units] (W^T), so OP_T is required to
        // compute Z^T = W * X^T. The old OP_N path was only shape-compatible;
        // it multiplied a scrambled interpretation of W for non-square layers.
        transa: cublasOperation_t::CUBLAS_OP_T,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: units as i32,
        n: batch as i32,
        k: input as i32,
        alpha: 1.,
        lda: input as i32,
        ldb: input as i32,
        beta: 0.,
        ldc: units as i32,
    };
    unsafe { blas.gemm(c, w, a, out).map_err(cuda_err("cuBLAS GEMM")) }
}

/// `gw[out, in] = (1/batch) * d[batch, out]^T . x[batch, in]`, all row-major.
///
/// cuBLAS is column-major, so a row-major `[r, c]` buffer is read as `[c, r]`.
/// The result is written as its own transpose: column-major `[in, out]` with
/// leading dimension `in` is exactly row-major `[out, in]`.
fn grad_weights(
    blas: &CudaBlas,
    d: &CudaSlice<f32>,
    x: &CudaSlice<f32>,
    gw: &mut CudaSlice<f32>,
    batch: usize,
    out: usize,
    input: usize,
) -> Result<(), NetworkError> {
    let c = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_T,
        m: input as i32,
        n: out as i32,
        k: batch as i32,
        alpha: 1.0 / batch as f32,
        lda: input as i32,
        ldb: out as i32,
        beta: 0.,
        ldc: input as i32,
    };
    unsafe {
        blas.gemm(c, x, d, gw)
            .map_err(cuda_err("cuBLAS weight-gradient GEMM"))
    }
}

/// `prev[batch, in] = next[batch, out] . w[out, in]`, all row-major. The
/// activation derivative is applied afterwards by the `act_derivative` kernel.
fn propagate_delta(
    blas: &CudaBlas,
    next: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    prev: &mut CudaSlice<f32>,
    batch: usize,
    out: usize,
    input: usize,
) -> Result<(), NetworkError> {
    let c = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: input as i32,
        n: batch as i32,
        k: out as i32,
        alpha: 1.,
        lda: input as i32,
        ldb: out as i32,
        beta: 0.,
        ldc: input as i32,
    };
    unsafe {
        blas.gemm(c, w, next, prev)
            .map_err(cuda_err("cuBLAS hidden-delta GEMM"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Network, Optimizer};

    #[test]
    fn memory_estimate_is_nonzero_and_budget_is_checked_before_cuda() {
        let model = Network::builder()
            .input_size(2)
            .dense(3, Activation::Relu)
            .dense(1, Activation::Linear)
            .optimizer(Optimizer::sgd(0.1))
            .build();
        assert!(estimate_tensor_memory_mib(&model, 4).unwrap() >= 1);
        let data = Dataset::new(vec![vec![0.0, 0.0]], vec![vec![0.0]]);
        let error = fit_cuda(
            &mut model.clone(),
            &data,
            TrainConfig {
                epochs: 1,
                batch_size: 1,
                shuffle: false,
                seed: Some(1),
            },
            usize::MAX,
            0,
        )
        .unwrap_err();
        assert!(matches!(error, NetworkError::CudaMemoryBudget { .. }));
    }

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

    fn assert_update_parity(optimizer: Optimizer, tolerance: f32) {
        if !cuda_or_skip() {
            return;
        }
        let data = Dataset::new(
            vec![vec![0.2, -0.3], vec![0.4, 0.1]],
            vec![vec![0.7], vec![-0.2]],
        );
        let base = Network::builder()
            .input_size(2)
            .dense(3, Activation::Tanh)
            .dense(1, Activation::Linear)
            .loss(Loss::Mse)
            .optimizer(optimizer)
            .seed(19)
            .build();
        let mut cpu = base.clone();
        let mut gpu = base;
        cpu.train_batch(&data.inputs, &data.targets).unwrap();
        gpu.fit_with_backend(
            &data,
            TrainConfig {
                epochs: 1,
                batch_size: 2,
                shuffle: false,
                seed: Some(3),
            },
            crate::TrainingBackend::Cuda {
                device: 0,
                memory_budget_mib: 8192,
            },
        )
        .unwrap();
        for (a, b) in cpu.layers.iter().zip(&gpu.layers) {
            for (x, y) in a
                .weights
                .data
                .iter()
                .chain(&a.biases.data)
                .zip(b.weights.data.iter().chain(&b.biases.data))
            {
                assert!(
                    (x - y).abs() <= tolerance,
                    "CPU/GPU update differs: {x} vs {y}"
                );
            }
        }
    }

    #[test]
    fn cuda_sgd_one_batch_update_matches_cpu_or_skips_without_device() {
        assert_update_parity(Optimizer::sgd(0.01), 1e-5);
    }

    #[test]
    fn cuda_adam_one_batch_update_matches_cpu_or_skips_without_device() {
        assert_update_parity(Optimizer::adam(0.01), 2e-5);
    }

    #[test]
    fn cuda_adam_weight_decay_one_batch_update_matches_cpu_or_skips_without_device() {
        assert_update_parity(Optimizer::adam_with_weight_decay(0.01, 0.01), 2e-5);
    }

    /// This deliberately exercises the failure mode that a one-layer/one-batch
    /// test misses: rectangular dense layers, a short final batch, shuffling,
    /// and a complete optimizer epoch. Keep this fixture aligned with the
    /// production universe feature width when RustingTrade changes it.
    fn assert_full_epoch_parity(optimizer: Optimizer, tolerance: f32) {
        if !cuda_or_skip() {
            return;
        }
        let inputs: Vec<Vec<f32>> = (0..19)
            .map(|row| {
                (0..8)
                    .map(|col| ((row * 13 + col * 7) as f32 - 80.0) / 37.0)
                    .collect()
            })
            .collect();
        let targets: Vec<Vec<f32>> = inputs
            .iter()
            .map(|x| vec![0.3 * x[0] - 0.2 * x[3] + 0.1 * x[6] + 0.05])
            .collect();
        let data = Dataset::new(inputs, targets);
        let base = Network::builder()
            .input_size(8)
            .dense(13, Activation::Relu)
            .dense(7, Activation::Tanh)
            .dense(1, Activation::Linear)
            .loss(Loss::Mse)
            .optimizer(optimizer)
            .seed(0x5eed)
            .build();
        let config = TrainConfig {
            epochs: 1,
            batch_size: 6,
            shuffle: true,
            seed: Some(0x1234),
        };
        let mut cpu = base.clone();
        let mut gpu = base;
        let cpu_history = cpu.fit(&data, config).unwrap();
        let gpu_history = gpu
            .fit_with_backend(
                &data,
                config,
                crate::TrainingBackend::Cuda {
                    device: 0,
                    memory_budget_mib: 8192,
                },
            )
            .unwrap();
        assert!(
            (cpu_history.losses[0] - gpu_history.losses[0]).abs() <= tolerance,
            "full-epoch loss differs: CPU={} GPU={}",
            cpu_history.losses[0],
            gpu_history.losses[0]
        );
        for (cpu_layer, gpu_layer) in cpu.layers.iter().zip(&gpu.layers) {
            for (left, right) in cpu_layer
                .weights
                .data
                .iter()
                .chain(&cpu_layer.biases.data)
                .zip(gpu_layer.weights.data.iter().chain(&gpu_layer.biases.data))
            {
                assert!(
                    (left - right).abs() <= tolerance,
                    "full-epoch parameter mismatch: CPU={left} GPU={right}"
                );
            }
        }
    }

    #[test]
    fn cuda_full_network_sgd_epoch_matches_cpu_or_skips_without_device() {
        assert_full_epoch_parity(Optimizer::sgd(0.01), 1e-5);
    }

    #[test]
    fn cuda_full_network_adam_epoch_matches_cpu_or_skips_without_device() {
        assert_full_epoch_parity(Optimizer::adam(0.005), 2e-5);
    }

    #[test]
    fn persistent_session_checkpoint_resume_matches_uninterrupted_or_skips() {
        if !cuda_or_skip() {
            return;
        }
        let data = Dataset::new(
            (0..17)
                .map(|i| vec![i as f32 / 17.0, (i % 3) as f32])
                .collect(),
            (0..17).map(|i| vec![(i % 5) as f32 / 5.0]).collect(),
        );
        let base = Network::builder()
            .input_size(2)
            .dense(5, Activation::Tanh)
            .dense(1, Activation::Linear)
            .optimizer(Optimizer::adam(0.003))
            .seed(7)
            .build();
        let config = TrainConfig {
            epochs: 2,
            batch_size: 6,
            shuffle: true,
            seed: Some(91),
        };
        let mut uninterrupted = CudaTrainingSession::new(&base, &data, config, 0, 8192).unwrap();
        uninterrupted.train_epoch().unwrap();
        uninterrupted.train_epoch().unwrap();
        let expected = uninterrupted.checkpoint().unwrap();

        let mut first = CudaTrainingSession::new(&base, &data, config, 0, 8192).unwrap();
        first.train_epoch().unwrap();
        assert_eq!(first.stats().device_to_host_bytes, size_of::<f32>());
        let checkpoint = first.checkpoint().unwrap();
        let mut resumed =
            CudaTrainingSession::from_checkpoint(checkpoint, &data, config, 0, 8192).unwrap();
        resumed.train_epoch().unwrap();
        let actual = resumed.checkpoint().unwrap();
        assert_eq!(actual.optimizer_step, expected.optimizer_step);
        for (a, b) in actual.layers.iter().zip(&expected.layers) {
            for (x, y) in a
                .weights
                .data
                .iter()
                .chain(&a.biases.data)
                .zip(b.weights.data.iter().chain(&b.biases.data))
            {
                assert!(
                    (x - y).abs() <= 2e-5,
                    "resumed parameter differs: {x} vs {y}"
                );
            }
        }
        assert!(resumed.stats().peak_allocated_bytes > 0);
        assert_eq!(resumed.stats().epochs, 1);
    }
}
