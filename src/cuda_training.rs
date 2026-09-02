//! Fail-closed FP32 CUDA fitting.  All state below is owned by `CudaSlice`s;
//! host pointers are only used by cudarc's checked copy operations.
use crate::{
    Activation, Dataset, Loss, Network, NetworkError, Optimizer, TrainConfig, TrainingHistory,
};
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::sync::Arc;

const MIB: usize = 1024 * 1024;
const KERNELS: &str = r#"
extern "C" __global__ void bias_act(float *x,const float*b,int rows,int cols,int act){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*cols){float v=x[i]+b[i%cols];if(act==1)v=v>0?v:0;else if(act==2)v=1.f/(1.f+expf(-v));else if(act==3)v=tanhf(v);x[i]=v;}}
extern "C" __global__ void output_delta(float*d,const float*y,const float*t,int n,int act){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=t[i]-y[i];float a=y[i];if(act==1)v*=a>0;else if(act==2)v*=a*(1-a);else if(act==3)v*=1-a*a;d[i]=v;}}
extern "C" __global__ void hidden_delta(float*d,const float*next,const float*w,const float*a,int batch,int units,int nextunits,int act){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<batch*units){int r=i/units,c=i%units;float s=0;for(int j=0;j<nextunits;j++)s+=next[r*nextunits+j]*w[j*units+c];float v=a[i];if(act==1)s*=v>0;else if(act==2)s*=v*(1-v);else if(act==3)s*=1-v*v;d[i]=s;}}
extern "C" __global__ void grad_w(float*g,const float*d,const float*x,int batch,int out,int in){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<out*in){int r=i/in,c=i%in;float s=0;for(int b=0;b<batch;b++)s+=d[b*out+r]*x[b*in+c];g[i]=s/(float)batch;}}
extern "C" __global__ void grad_b(float*g,const float*d,int batch,int out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<out){float s=0;for(int b=0;b<batch;b++)s+=d[b*out+i];g[i]=s/(float)batch;}}
extern "C" __global__ void sgd(float*x,const float*g,int n,float lr){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=lr*g[i];}
extern "C" __global__ void adam(float*x,const float*g,float*m,float*v,int n,float lr,float b1,float b2,float eps,float c1,float c2){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float q=g[i];float mi=b1*m[i]+(1-b1)*q;float vi=b2*v[i]+(1-b2)*q*q;m[i]=mi;v[i]=vi;x[i]+=lr*(mi/c1)/(sqrtf(vi/c2)+eps);}}
extern "C" __global__ void mse_sum(float *out,const float*y,const float*t,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=y[i]-t[i];atomicAdd(out,z*z/(float)n);}}
"#;

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
    module: Arc<CudaModule>,
    layers: Vec<DevLayer>,
    input: CudaSlice<f32>,
    target: CudaSlice<f32>,
    loss: CudaSlice<f32>,
}

pub(crate) fn fit_cuda(
    net: &mut Network,
    dataset: &Dataset,
    config: TrainConfig,
    device: usize,
    budget: usize,
) -> Result<TrainingHistory, NetworkError> {
    if dataset.is_empty() {
        return Err(NetworkError::EmptyDataset);
    };
    if net.loss != Loss::Mse {
        return Err(NetworkError::UnsupportedCuda(
            "only MSE loss is implemented".into(),
        ));
    }
    for l in &net.layers {
        act(l.activation)?;
    }
    for (x, y) in dataset.inputs.iter().zip(&dataset.targets) {
        net.validate_input(x)?;
        net.validate_target(y)?;
    }
    let mut batch = config.batch_size.max(1).min(dataset.len());
    while estimate_tensor_memory_mib(net, batch)? > budget {
        if batch == 1 {
            let e = estimate_tensor_memory_mib(net, 1)?;
            return Err(NetworkError::CudaMemoryBudget {
                estimated_mib: e,
                budget_mib: budget,
            });
        }
        batch = (batch + 1) / 2;
    }
    let mut s = State::new(net, batch, device)?; // all fallible CUDA work occurs before training
    let mut working = dataset.clone();
    let mut history = Vec::with_capacity(config.epochs);
    for epoch in 0..config.epochs {
        if config.shuffle {
            working.shuffle(config.seed.map(|seed| seed + epoch as u64));
        }
        let mut total = 0.;
        let mut count = 0;
        for b in working.batches(batch) {
            total += s.train_batch(net, b.inputs, b.targets)?;
            count += 1;
        }
        history.push(total / count as f32);
    }
    s.copy_back(net)?;
    Ok(TrainingHistory { losses: history })
}

impl State {
    fn new(net: &Network, batch: usize, device: usize) -> Result<Self, NetworkError> {
        let ctx =
            CudaContext::new(device).map_err(cuda_err("CUDA driver/device initialization"))?;
        let stream = ctx.new_stream().map_err(cuda_err("stream creation"))?;
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_err("cuBLAS initialization"))?;
        let ptx = compile_ptx(KERNELS)
            .map_err(|e| NetworkError::Cuda(format!("CUDA kernel compilation failed: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(cuda_err("CUDA module loading"))?;
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
        Ok(Self {
            _ctx: ctx,
            stream: stream.clone(),
            blas,
            module,
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
        })
    }
    fn fun(&self, n: &str) -> Result<CudaFunction, NetworkError> {
        self.module
            .load_function(n)
            .map_err(cuda_err("CUDA kernel lookup"))
    }
    fn train_batch(
        &mut self,
        net: &mut Network,
        xs: &[Vec<f32>],
        ys: &[Vec<f32>],
    ) -> Result<f32, NetworkError> {
        let b = xs.len();
        let mut x = Vec::with_capacity(b * net.input_size);
        let mut y = Vec::with_capacity(b * net.output_size());
        for r in xs {
            x.extend_from_slice(r)
        }
        for r in ys {
            y.extend_from_slice(r)
        }
        self.stream
            .memcpy_htod(&x, &mut self.input)
            .map_err(cuda_err("input upload"))?;
        self.stream
            .memcpy_htod(&y, &mut self.target)
            .map_err(cuda_err("target upload"))?;
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
            let f = self.fun("bias_act")?;
            let l = &mut self.layers[i];
            unsafe {
                self.stream
                    .launch_builder(&f)
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
        let f = self.fun("output_delta")?;
        {
            let l = &mut self.layers[last];
            unsafe {
                self.stream
                    .launch_builder(&f)
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
            let fw = self.fun("grad_w")?;
            let fb = self.fun("grad_b")?;
            if i == 0 {
                let l = &mut self.layers[0];
                unsafe {
                    self.stream
                        .launch_builder(&fw)
                        .arg(&mut l.gw)
                        .arg(&l.d)
                        .arg(&self.input)
                        .arg(&(b as i32))
                        .arg(&(out as i32))
                        .arg(&(inp as i32))
                        .launch(cfg(out * inp))
                        .map_err(cuda_err("weight gradient kernel"))?;
                    self.stream
                        .launch_builder(&fb)
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
                unsafe {
                    self.stream
                        .launch_builder(&fw)
                        .arg(&mut l.gw)
                        .arg(&l.d)
                        .arg(source)
                        .arg(&(b as i32))
                        .arg(&(out as i32))
                        .arg(&(inp as i32))
                        .launch(cfg(out * inp))
                        .map_err(cuda_err("weight gradient kernel"))?;
                    self.stream
                        .launch_builder(&fb)
                        .arg(&mut l.gb)
                        .arg(&l.d)
                        .arg(&(b as i32))
                        .arg(&(out as i32))
                        .launch(cfg(out))
                        .map_err(cuda_err("bias gradient kernel"))?;
                }
            }
            if i > 0 {
                let fh = self.fun("hidden_delta")?;
                let (left, right) = self.layers.split_at_mut(i);
                let prev = &mut left[i - 1];
                let cur = &right[0];
                unsafe {
                    self.stream
                        .launch_builder(&fh)
                        .arg(&mut prev.d)
                        .arg(&cur.d)
                        .arg(&cur.w)
                        .arg(&prev.a)
                        .arg(&(b as i32))
                        .arg(&(inp as i32))
                        .arg(&(out as i32))
                        .arg(&act(net.layers[i - 1].activation)?)
                        .launch(cfg(b * inp))
                        .map_err(cuda_err("hidden delta kernel"))?;
                }
            }
        }
        self.update(net)?;
        let fl = self.fun("mse_sum")?;
        self.stream
            .memset_zeros(&mut self.loss)
            .map_err(cuda_err("loss reset"))?;
        let la = &self.layers[last].a;
        unsafe {
            self.stream
                .launch_builder(&fl)
                .arg(&mut self.loss)
                .arg(la)
                .arg(&self.target)
                .arg(&((b * net.output_size()) as i32))
                .launch(cfg(b * net.output_size()))
                .map_err(cuda_err("loss kernel"))?;
        }
        self.stream
            .synchronize()
            .map_err(cuda_err("CUDA synchronization"))?;
        let h = self
            .stream
            .clone_dtoh(&self.loss)
            .map_err(cuda_err("loss download"))?;
        if !h[0].is_finite() {
            return Err(NetworkError::Cuda(
                "numerical preflight failed: non-finite loss".into(),
            ));
        }
        Ok(h[0])
    }
    fn update(&mut self, net: &mut Network) -> Result<(), NetworkError> {
        let (f, adam) = match net.optimizer.clone() {
            Optimizer::Sgd { .. } => (self.fun("sgd")?, false),
            Optimizer::Adam { .. } => (self.fun("adam")?, true),
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
                        .launch_builder(&f)
                        .arg(&mut l.w)
                        .arg(&l.gw)
                        .arg(&(nw as i32))
                        .arg(&learning_rate)
                        .launch(cfg(nw))
                        .map_err(cuda_err("SGD update kernel"))?;
                    self.stream
                        .launch_builder(&f)
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
                } = net.optimizer
                {
                    let c1 = 1. - beta1.powi(net.adam_step as i32);
                    let c2 = 1. - beta2.powi(net.adam_step as i32);
                    self.stream
                        .launch_builder(&f)
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
                        .launch(cfg(nw))
                        .map_err(cuda_err("Adam update kernel"))?;
                    self.stream
                        .launch_builder(&f)
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
}
