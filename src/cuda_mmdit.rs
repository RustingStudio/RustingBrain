//! The MMDiT denoiser on the device, the twin of [`Dit::forward_with`].
//!
//! Like the T5 encoder in [`crate::cuda_t5`], it runs on the BF16 kernels the
//! language model trains with: the residual streams and every GEMM
//! accumulator are FP32, and only the weights and the operands a GEMM reads
//! are BF16. Attention is the fused BF16 kernel, unmasked.
//!
//! The conditioning vector is built on the host, because it is a handful of
//! small layers over one vector. Every block's modulation is then one GEMV:
//! the modulation projections of all blocks are stacked into one weight and
//! run once per call.
//!
//! NOTE: the norms, the head norms and the bias adds write FP32 and a
//! separate cast narrows the GEMM operands. Fusing the cast into those
//! kernels saves one pass per GEMM if a profile ever shows it.

use crate::conv::Dense;
use crate::cuda_training::{cfg, cuda_err};
use crate::dit_lora::{Adapter, DitLora};
use crate::gpu_model::{Act, FlashShape, Gpu};
use crate::gpu_transformer::{
    DeviceParam, GpuContext, act_lhs_transposed, act_plain, act_rhs_transposed,
};
use crate::matrix::Matrix;
use crate::mmdit::{Conditioning, Dit, Positions};
use crate::network::NetworkError;
use crate::param::Param;
use cudarc::driver::{CudaFunction, CudaSlice, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{Ptx, compile_ptx};
use std::sync::{Arc, OnceLock};

const KERNELS: &str = r#"
#define ROW_THREADS 256
__device__ __forceinline__ float row_sum(float v,float*red){
  int tid=threadIdx.x;red[tid]=v;__syncthreads();
  for(int s=ROW_THREADS/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
  float r=red[0];__syncthreads();return r;
}
// Layer norm with no learned scale, then `x * (1 + scale) + shift`, with the
// shift at `t[c]` and the scale at `t[cols + c]`. One block per row.
extern "C" __global__ void layer_norm_modulate(float*out,const float*x,const float*t,int rows,int cols,float eps){
  __shared__ float red[ROW_THREADS];int tid=threadIdx.x;
  for(int r=blockIdx.x;r<rows;r+=gridDim.x){
    const float*s=x+(size_t)r*cols;float*d=out+(size_t)r*cols;
    float a=0.f;for(int c=tid;c<cols;c+=ROW_THREADS)a+=s[c];
    float mean=row_sum(a,red)/cols;
    float b=0.f;for(int c=tid;c<cols;c+=ROW_THREADS){float v=s[c]-mean;b+=v*v;}
    float inv=rsqrtf(row_sum(b,red)/cols+eps);
    for(int c=tid;c<cols;c+=ROW_THREADS)d[c]=(s[c]-mean)*inv*(1.f+t[cols+c])+t[c];
  }
}
// The per-head RMS norm on a query or key, then the rotary position: pairs
// are adjacent dimensions, FLUX's order, not the language model's halves.
// One warp per (row, head), a lane per pair, so a warp's loads are one
// contiguous run (a thread per head read 128 strided floats and ran at a
// twentieth of the bandwidth). `first` is the row's index in the tables.
extern "C" __global__ void head_norm_rope(float*x,const float*scale,const float*cs,const float*sn,int rows,int heads,int head_dim,int stride,int off,int first){
  int w=(blockIdx.x*blockDim.x+threadIdx.x)>>5,lane=threadIdx.x&31;if(w>=rows*heads)return;
  int r=w/heads,h=w-r*heads,half=head_dim/2;
  float2*v=(float2*)(x+(size_t)r*stride+off+h*head_dim);const float2*k=(const float2*)scale;
  // The sum of squares runs in dimension order through shuffles, the order
  // the thread-per-head loop used, so the result is the same to the bit.
  float s=0.f;
  for(int b=0;b<half;b+=32){
    float2 a=b+lane<half?v[b+lane]:make_float2(0.f,0.f);int n=min(32,half-b);
    for(int j=0;j<n;j++){float x=__shfl_sync(0xffffffffu,a.x,j),y=__shfl_sync(0xffffffffu,a.y,j);s=fmaf(x,x,s);s=fmaf(y,y,s);}
  }
  float inv=rsqrtf(s/head_dim+1e-6f);
  const float*c=cs+(size_t)(first+r)*half;const float*n=sn+(size_t)(first+r)*half;
  for(int p=lane;p<half;p+=32){
    float2 e=v[p],g=k[p];float a=e.x*inv*g.x,b=e.y*inv*g.y;
    v[p]=make_float2(a*c[p]-b*n[p],a*n[p]+b*c[p]);
  }
}
// The backward of `layer_norm_modulate`: adds `dL/dx` to `gx` given `dL/dout`
// in `gy`; the shift has no part in it. One block per row.
extern "C" __global__ void layer_norm_modulate_bwd(float*gx,const float*gy,const float*x,const float*t,int rows,int cols,float eps){
  __shared__ float red[ROW_THREADS];int tid=threadIdx.x;
  for(int r=blockIdx.x;r<rows;r+=gridDim.x){
    const float*s=x+(size_t)r*cols;const float*g=gy+(size_t)r*cols;float*d=gx+(size_t)r*cols;
    float a=0.f;for(int c=tid;c<cols;c+=ROW_THREADS)a+=s[c];
    float mean=row_sum(a,red)/cols;
    float b=0.f;for(int c=tid;c<cols;c+=ROW_THREADS){float v=s[c]-mean;b+=v*v;}
    float inv=rsqrtf(row_sum(b,red)/cols+eps);
    float p=0.f,q=0.f;
    for(int c=tid;c<cols;c+=ROW_THREADS){float u=g[c]*(1.f+t[cols+c]);p+=u;q+=u*(s[c]-mean);}
    float gm=row_sum(p,red)/cols,pr=row_sum(q,red)*inv/cols;
    for(int c=tid;c<cols;c+=ROW_THREADS){float u=g[c]*(1.f+t[cols+c]);d[c]+=inv*(u-gm-(s[c]-mean)*inv*pr);}
  }
}
// A BF16 value in its 16 bits, or an FP32 one, by `narrow`.
__device__ __forceinline__ float ld(const void*p,size_t i,int narrow){
  return narrow?__uint_as_float(((unsigned)((const unsigned short*)p)[i])<<16):((const float*)p)[i];
}
__device__ __forceinline__ void st(void*p,size_t i,float v,int narrow){
  if(!narrow){((float*)p)[i]=v;return;}
  unsigned u=__float_as_uint(v);((unsigned short*)p)[i]=(unsigned short)((u+0x7fffu+((u>>16)&1u))>>16);
}
// The backward of `head_norm_rope`, in place on the gradient `g`: the inverse
// rotation, then the head norm's backward from `raw`, the values before the
// norm, in the same layout. Both are whole buffers; row `r` here is row
// `first + r` of them. The rotation is redone in the second pass rather
// than stored, so a BF16 gradient is rounded once.
extern "C" __global__ void head_norm_rope_bwd(void*g,const float*raw,const float*scale,const float*cs,const float*sn,int rows,int heads,int head_dim,int stride,int off,int first,int narrow){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*heads)return;int r=i/heads,h=i-r*heads;
  size_t at=(size_t)(first+r)*stride+off+h*head_dim;const float*x=raw+at;int half=head_dim/2;
  const float*c=cs+(size_t)(first+r)*half;const float*n=sn+(size_t)(first+r)*half;
  float s=0.f,dot=0.f;
  for(int p=0;p<half;p++){
    float a=ld(g,at+2*p,narrow),b=ld(g,at+2*p+1,narrow);
    float u=a*c[p]+b*n[p],w=b*c[p]-a*n[p];
    s+=x[2*p]*x[2*p]+x[2*p+1]*x[2*p+1];dot+=x[2*p]*u*scale[2*p]+x[2*p+1]*w*scale[2*p+1];
  }
  float inv=rsqrtf(s/head_dim+1e-6f);float k=inv*inv*inv*dot/head_dim;
  for(int p=0;p<half;p++){
    float a=ld(g,at+2*p,narrow),b=ld(g,at+2*p+1,narrow);
    float u=a*c[p]+b*n[p],w=b*c[p]-a*n[p];
    st(g,at+2*p,inv*u*scale[2*p]-x[2*p]*k,narrow);st(g,at+2*p+1,inv*w*scale[2*p+1]-x[2*p+1]*k,narrow);
  }
}
// `out = g * gate[c]`, the branch's gradient through a gated residual.
extern "C" __global__ void gate_rows(float*out,const float*g,const float*gate,int rows,int cols){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;out[i]=g[i]*gate[i%cols];
}
// `g *= gelu'(pre)`, the tanh GELU.
extern "C" __global__ void gelu_bwd(float*g,const float*pre,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float x=pre[i];
  float th=tanhf(0.7978846f*(x+0.044715f*x*x*x));
  g[i]*=0.5f*(1.f+th)+0.5f*x*(1.f-th*th)*0.7978846f*(1.f+3.f*0.044715f*x*x);
}
// `dst = act(src[:, off..off + cols] + bias[off..])`, where the activation is
// the tanh GELU (`gelu` 1) or none (0). `dst` is `[rows, cols]`, BF16 if
// `narrow`, `src` has row stride `stride`. With `gelu` 2 it is SwiGLU instead:
// `silu(x[:, c]) * x[:, cols + c]` of the biased `x`, so `src` is twice as
// wide as `dst`.
extern "C" __global__ void bias_gelu(void*dst,const float*src,const float*bias,int rows,int cols,int stride,int off,int gelu,int has_bias,int narrow){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;int r=i/cols,c=i-r*cols;
  const float*x=src+(size_t)r*stride+off;float v=x[c];
  if(has_bias)v+=bias[off+c];
  if(gelu==2){float u=x[cols+c];if(has_bias)u+=bias[off+cols+c];v=v/(1.f+expf(-v))*u;}
  else if(gelu)v=0.5f*v*(1.f+tanhf(0.7978846f*(v+0.044715f*v*v*v)));
  st(dst,i,v,narrow);
}
"#;

fn ptx() -> Result<&'static Ptx, NetworkError> {
    static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| compile_ptx(KERNELS).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| NetworkError::Cuda(format!("MMDiT kernel compilation failed: {error}")))
}

/// A layer's weight on the device, and its bias if it has one.
struct Linear {
    weight: Act,
    bias: Option<CudaSlice<f32>>,
    units: usize,
    inner: usize,
}

struct DeviceStream {
    qkv: Linear,
    query_norm: CudaSlice<f32>,
    key_norm: CudaSlice<f32>,
    projection: Linear,
    mlp_in: Linear,
    mlp_out: Linear,
}

struct DeviceSingle {
    fused_in: Linear,
    /// `fused_out`'s columns split at the attention's width: the attention
    /// half and the feed-forward half, two GEMMs into one output.
    attention_out: Linear,
    mlp_out: Linear,
    query_norm: CudaSlice<f32>,
    key_norm: CudaSlice<f32>,
}

/// What [`DeviceDit::forward_train`] keeps for [`DeviceDit::backward`].
pub struct DeviceTape {
    modulation: CudaSlice<f32>,
    cos: CudaSlice<f32>,
    sin: CudaSlice<f32>,
    text_rows: usize,
    image_rows: usize,
    double: Vec<DoubleTape>,
    single: Vec<SingleTape>,
    /// The stream after the last block, text then image rows.
    hidden: CudaSlice<f32>,
}

impl std::fmt::Debug for DeviceTape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceTape").finish_non_exhaustive()
    }
}

/// A double block's attention and each stream's own activations, text then
/// image.
struct DoubleTape {
    streams: Vec<StreamTape>,
    attention: AttentionTape,
}

struct StreamTape {
    input: CudaSlice<f32>,
    normed: Act,
    mid: CudaSlice<f32>,
    normed_mlp: Act,
    pre: CudaSlice<f32>,
    activated: Act,
}

/// The queries, keys and values before their head norms, the narrow buffer
/// attention read, its output and its log-sum-exp.
struct AttentionTape {
    raw: CudaSlice<f32>,
    qkv: Act,
    merged: Act,
    log_sum_exp: CudaSlice<f32>,
}

struct SingleTape {
    input: CudaSlice<f32>,
    normed: Act,
    attention: AttentionTape,
    pre: CudaSlice<f32>,
    activated: Act,
}

/// An adapter factor's device mirror.
fn resident(param: &Param) -> Result<&DeviceParam, NetworkError> {
    param.device.as_ref().ok_or_else(not_resident)
}

fn resident_mut(param: &mut Param) -> Result<&mut DeviceParam, NetworkError> {
    param.device.as_mut().ok_or_else(not_resident)
}

fn not_resident() -> NetworkError {
    NetworkError::Cuda("a DiT adapter is not on the device: call DitLora::move_to_cuda".into())
}

/// Self-attention over one unmasked sequence of `rows`, the shape of every
/// MMDiT attention.
fn flash_shape(rows: usize, heads: usize, width: usize) -> FlashShape {
    FlashShape {
        rows,
        seq_len: rows,
        kv_len: rows,
        sequences: 1,
        heads,
        group: 1,
        qkv_width: 3 * width,
        kv_width: 3 * width,
        query_width: width,
        key_base: width,
        value_base: 2 * width,
        causal: false,
    }
}

/// An MMDiT denoiser resident on the device.
pub struct DeviceDit {
    context: Arc<GpuContext>,
    layer_norm_modulate: CudaFunction,
    head_norm_rope: CudaFunction,
    bias_gelu: CudaFunction,
    layer_norm_modulate_bwd: CudaFunction,
    head_norm_rope_bwd: CudaFunction,
    gelu_bwd: CudaFunction,
    gate_rows: CudaFunction,
    /// Whether any weight is int8, which the backward pass cannot read.
    quantized: bool,
    image_in: Linear,
    text_in: Linear,
    /// Every distinct modulation projection stacked, in the order
    /// [`DeviceDit::forward`] first reads them: each double block's image
    /// then text stream, each single block, then the final layer.
    modulation: Linear,
    /// Where in the stacked modulation each of those readers' rows start,
    /// in that order. FLUX.2's blocks share three projections, so several
    /// readers point at the same rows.
    slots: Vec<usize>,
    /// Whether the feed-forwards are SwiGLU, as FLUX.2's are.
    swiglu: bool,
    double: Vec<(DeviceStream, DeviceStream)>,
    single: Vec<DeviceSingle>,
    final_out: Linear,
}

impl std::fmt::Debug for DeviceDit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceDit").finish_non_exhaustive()
    }
}

impl DeviceDit {
    /// Uploads `dit`'s weights as BF16, or as int8 where it holds them at one
    /// byte. The host model stays the source of the conditioning vector, so
    /// it is borrowed again by [`DeviceDit::forward_with`].
    pub fn upload(context: &Arc<GpuContext>, dit: &Dit) -> Result<Self, NetworkError> {
        if context.half_accumulate {
            return Err(NetworkError::InvalidConfig(
                "the denoiser needs a BF16 context, not an FP16 one".into(),
            ));
        }
        let head_dim = dit.config().head_dim();
        if context.flash.is_none() || !crate::cuda_flash::eligible(true, head_dim) {
            return Err(NetworkError::InvalidConfig(format!(
                "the device denoiser needs the fused attention kernel, which takes \
                 heads of 64 or 128 on Ampere or later; this one has {head_dim}"
            )));
        }
        let gpu = Gpu { context };
        let module = context
            .stream
            .context()
            .load_module(ptx()?.clone())
            .map_err(cuda_err("MMDiT module loading"))?;
        let get = |name: &str| {
            module
                .load_function(name)
                .map_err(cuda_err("MMDiT kernel lookup"))
        };
        let quantized = std::cell::Cell::new(false);
        let linear = |weight: Matrix, bias: Option<&[f32]>, int8: bool| {
            quantized.set(quantized.get() || int8);
            let (units, inner) = (weight.rows, weight.cols);
            let wide = gpu.upload(&weight.data)?;
            Ok::<_, NetworkError>(Linear {
                weight: match int8 {
                    true => gpu.quantized(&wide.slice(..), units, inner)?,
                    false => gpu.narrowed(&wide.slice(..), units * inner, true)?,
                },
                bias: bias.map(|bias| gpu.upload(bias)).transpose()?,
                units,
                inner,
            })
        };
        let dense = |layer: &Dense| {
            linear(
                layer.weight_f32().into_owned(),
                layer.bias.as_deref(),
                layer.is_quantized(),
            )
        };
        let stream = |stream: &crate::mmdit::Stream| {
            Ok::<_, NetworkError>(DeviceStream {
                qkv: dense(&stream.qkv)?,
                query_norm: gpu.upload(&stream.query_norm.scale)?,
                key_norm: gpu.upload(&stream.key_norm.scale)?,
                projection: dense(&stream.projection)?,
                mlp_in: dense(&stream.mlp_in)?,
                mlp_out: dense(&stream.mlp_out)?,
            })
        };

        let readers: Vec<&Dense> = dit
            .double
            .iter()
            .flat_map(|block| {
                [
                    &block.image.modulation.projection,
                    &block.text.modulation.projection,
                ]
            })
            .chain(dit.single.iter().map(|block| &block.modulation.projection))
            .chain([&dit.final_modulation])
            .collect();
        // A projection several blocks share is stacked once.
        let mut modulations: Vec<&Dense> = Vec::new();
        let mut slots = Vec::new();
        for &layer in &readers {
            let index = match modulations.iter().position(|seen| std::ptr::eq(*seen, layer)) {
                Some(index) => index,
                None => {
                    modulations.push(layer);
                    modulations.len() - 1
                }
            };
            slots.push(modulations[..index].iter().map(|seen| seen.out_dim()).sum());
        }
        let (mut weights, mut biases) = (Vec::new(), Vec::new());
        for layer in &modulations {
            weights.extend_from_slice(&layer.weight_f32().data);
            match &layer.bias {
                Some(bias) => biases.extend_from_slice(bias),
                None => biases.resize(biases.len() + layer.out_dim(), 0.0),
            }
        }
        let inner = dit.config().d_model;
        let modulation = linear(
            Matrix::from_vec(weights.len() / inner, inner, weights),
            Some(&biases),
            modulations.iter().any(|layer| layer.is_quantized()),
        )?;

        let width = dit.config().d_model;
        let single = dit
            .single
            .iter()
            .map(|block| {
                let fused_out = block.fused_out.weight_f32();
                let mlp_dim = fused_out.cols - width;
                let columns = |from: usize, count: usize| {
                    let mut part = Matrix::new(fused_out.rows, count);
                    for row in 0..fused_out.rows {
                        part.row_mut(row)
                            .copy_from_slice(&fused_out.row(row)[from..from + count]);
                    }
                    part
                };
                let quantized = block.fused_out.is_quantized();
                Ok(DeviceSingle {
                    fused_in: dense(&block.fused_in)?,
                    attention_out: linear(
                        columns(0, width),
                        block.fused_out.bias.as_deref(),
                        quantized,
                    )?,
                    mlp_out: linear(columns(width, mlp_dim), None, quantized)?,
                    query_norm: gpu.upload(&block.query_norm.scale)?,
                    key_norm: gpu.upload(&block.key_norm.scale)?,
                })
            })
            .collect::<Result<_, NetworkError>>()?;

        Ok(Self {
            layer_norm_modulate: get("layer_norm_modulate")?,
            head_norm_rope: get("head_norm_rope")?,
            bias_gelu: get("bias_gelu")?,
            layer_norm_modulate_bwd: get("layer_norm_modulate_bwd")?,
            head_norm_rope_bwd: get("head_norm_rope_bwd")?,
            gelu_bwd: get("gelu_bwd")?,
            gate_rows: get("gate_rows")?,
            quantized: quantized.get(),
            slots,
            swiglu: dit.swiglu,
            image_in: dense(&dit.image_in)?,
            text_in: dense(&dit.text_in)?,
            modulation,
            double: dit
                .double
                .iter()
                .map(|block| Ok((stream(&block.image)?, stream(&block.text)?)))
                .collect::<Result<_, NetworkError>>()?,
            single,
            final_out: dense(&dit.final_out)?,
            context: context.clone(),
        })
    }

    /// The prediction for `patches` at `timestep` under `conditioning`,
    /// `[patches, out_channels]`. `dit` must be the model this was uploaded
    /// from.
    pub fn forward_with(
        &self,
        dit: &Dit,
        patches: &Matrix,
        timestep: f32,
        conditioning: &Conditioning,
    ) -> Result<Matrix, NetworkError> {
        let none = DitLora::default();
        Ok(self
            .run(dit, &none, patches, timestep, conditioning, false)?
            .0)
    }

    /// [`DeviceDit::forward_with`] with `lora`'s adapters added to the block
    /// projections, the device twin of [`DitLora::forward`]. The adapters
    /// must be on the device, see [`DitLora::move_to_cuda`].
    pub fn forward_adapted(
        &self,
        dit: &Dit,
        lora: &DitLora,
        patches: &Matrix,
        timestep: f32,
        conditioning: &Conditioning,
    ) -> Result<Matrix, NetworkError> {
        Ok(self
            .run(dit, lora, patches, timestep, conditioning, false)?
            .0)
    }

    /// [`DeviceDit::forward_adapted`], and what [`DeviceDit::backward`]
    /// needs of it.
    pub fn forward_train(
        &self,
        dit: &Dit,
        lora: &DitLora,
        patches: &Matrix,
        timestep: f32,
        conditioning: &Conditioning,
    ) -> Result<(Matrix, DeviceTape), NetworkError> {
        self.run(dit, lora, patches, timestep, conditioning, true)
    }

    /// The forward pass; the tape's blocks are only kept if `keep`.
    fn run(
        &self,
        dit: &Dit,
        lora: &DitLora,
        patches: &Matrix,
        timestep: f32,
        conditioning: &Conditioning,
        keep: bool,
    ) -> Result<(Matrix, DeviceTape), NetworkError> {
        if patches.rows != conditioning.height * conditioning.width {
            return Err(NetworkError::InvalidTarget {
                expected: conditioning.height * conditioning.width,
                actual: patches.rows,
            });
        }
        lora.check(dit)?;
        let gpu = Gpu {
            context: &self.context,
        };
        let config = dit.config();
        let (width, heads, eps) = (config.d_model, config.num_heads, config.eps);
        let head_dim = config.head_dim();
        let (text_rows, image_rows) = (conditioning.text.rows, patches.rows);
        let rows = text_rows + image_rows;
        let copy = || cuda_err("device to device copy");

        let vector: Vec<f32> = dit
            .vector(timestep, conditioning)?
            .iter()
            .map(|value| crate::ffn::silu(*value))
            .collect();
        let vector = gpu.narrowed(&gpu.upload(&vector)?.slice(..), width, true)?;
        let modulation = self.apply(&gpu, &self.modulation, &[], &vector.all(), 1)?;
        let positions = Positions::new(config, text_rows, conditioning.height, conditioning.width);
        let (cos, sin) = (gpu.upload(&positions.cos)?, gpu.upload(&positions.sin)?);

        let mut text = self.input(&gpu, &self.text_in, &conditioning.text)?;
        let mut image = self.input(&gpu, &self.image_in, patches)?;
        let mut slots = self.slots.iter();
        let mut triple = |count: usize| {
            let at = *slots.next().expect("one slot per modulation reader");
            modulation.slice(at..at + count * width)
        };

        let mut double = Vec::new();
        for (index, (image_stream, text_stream)) in self.double.iter().enumerate() {
            let image_mod = triple(6);
            let text_mod = triple(6);
            let streams = [
                (text_stream, &text_mod, 0, text_rows, "text"),
                (image_stream, &image_mod, text_rows, image_rows, "image"),
            ];
            let adapted =
                |side: &str, name: &str| lora.get(&format!("double.{index}.{side}.{name}"));
            let mut qkv = gpu.uninit(rows * 3 * width)?;
            let mut normed_streams = Vec::new();
            for &(stream, modulation, first, count, side) in &streams {
                let hidden = if first == 0 { &text } else { &image };
                let normed = self.norm(&gpu, hidden, &modulation.slice(..), count, width, eps)?;
                let mut part = gpu.uninit(count * 3 * width)?;
                self.project(
                    &gpu,
                    &stream.qkv,
                    &normed,
                    count,
                    &mut part.slice_mut(..),
                    0.0,
                )?;
                self.adapt(
                    &gpu,
                    adapted(side, "qkv"),
                    &[(normed.all(), width)],
                    count,
                    &mut part.slice_mut(..),
                    3 * width,
                )?;
                let mut target = qkv.slice_mut(first * 3 * width..(first + count) * 3 * width);
                self.bias(&part, &stream.qkv, count, 0, 3 * width, false, &mut target)?;
                normed_streams.push(normed);
            }
            let raw = match keep {
                true => Some(self.context.stream.clone_dtod(&qkv).map_err(copy())?),
                false => None,
            };
            for &(stream, _, first, count, _) in &streams {
                for (scale, offset) in [(&stream.query_norm, 0), (&stream.key_norm, width)] {
                    self.head_norm_rope(
                        &mut qkv, scale, &cos, &sin, first, count, heads, head_dim, offset,
                    )?;
                }
            }
            let (merged, log_sum_exp, qkv) = self.attention(&gpu, &qkv, rows, heads, width)?;
            let mut saved = Vec::new();
            for (&(stream, modulation, first, count, side), normed) in
                streams.iter().zip(normed_streams)
            {
                let hidden = if first == 0 { &mut text } else { &mut image };
                let output = self.apply(
                    &gpu,
                    &stream.projection,
                    adapted(side, "projection"),
                    &merged.at(first * width),
                    count,
                )?;
                let gated = gpu.gate_residual(
                    &hidden.slice(..),
                    &output.slice(..),
                    &modulation.slice(..),
                    count,
                    width,
                    count,
                )?;
                let input = std::mem::replace(hidden, gated);
                let modulation = modulation.slice(3 * width..);
                let normed_mlp = self.norm(&gpu, hidden, &modulation, count, width, eps)?;
                let pre = self.apply(
                    &gpu,
                    &stream.mlp_in,
                    adapted(side, "mlp_in"),
                    &normed_mlp.all(),
                    count,
                )?;
                let units = stream.mlp_in.units;
                let activated = self.gelu(&gpu, &pre, None, count, units, units, 0)?;
                let output = self.apply(
                    &gpu,
                    &stream.mlp_out,
                    adapted(side, "mlp_out"),
                    &activated.all(),
                    count,
                )?;
                let gated = gpu.gate_residual(
                    &hidden.slice(..),
                    &output.slice(..),
                    &modulation,
                    count,
                    width,
                    count,
                )?;
                let mid = std::mem::replace(hidden, gated);
                saved.push(StreamTape {
                    input,
                    normed,
                    mid,
                    normed_mlp,
                    pre,
                    activated,
                });
            }
            if let Some(raw) = raw {
                double.push(DoubleTape {
                    streams: saved,
                    attention: AttentionTape {
                        raw,
                        qkv,
                        merged,
                        log_sum_exp,
                    },
                });
            }
        }

        let mut hidden = gpu.uninit(rows * width)?;
        let stream = &self.context.stream;
        stream
            .memcpy_dtod(&text, &mut hidden.slice_mut(..text_rows * width))
            .map_err(copy())?;
        stream
            .memcpy_dtod(&image, &mut hidden.slice_mut(text_rows * width..))
            .map_err(copy())?;
        let mut single = Vec::new();
        for (index, block) in self.single.iter().enumerate() {
            let adapted = |name: &str| lora.get(&format!("single.{index}.{name}"));
            let modulation = triple(3);
            let normed = self.norm(&gpu, &hidden, &modulation, rows, width, eps)?;
            let fused_width = block.fused_in.units;
            let mlp_dim = fused_width - 3 * width;
            let mut fused = gpu.uninit(rows * fused_width)?;
            self.project(
                &gpu,
                &block.fused_in,
                &normed,
                rows,
                &mut fused.slice_mut(..),
                0.0,
            )?;
            self.adapt(
                &gpu,
                adapted("fused_in"),
                &[(normed.all(), width)],
                rows,
                &mut fused.slice_mut(..),
                fused_width,
            )?;
            let mut qkv = gpu.uninit(rows * 3 * width)?;
            let linear = &block.fused_in;
            self.bias(
                &fused,
                linear,
                rows,
                0,
                3 * width,
                false,
                &mut qkv.slice_mut(..),
            )?;
            // The activation reads its half of `fused` in place; the biased
            // copy is made only for the backward pass.
            let activated = self.gelu(
                &gpu,
                &fused,
                linear.bias.as_ref(),
                rows,
                mlp_dim,
                fused_width,
                3 * width,
            )?;
            let pre = match keep {
                true => {
                    let mut pre = gpu.uninit(rows * mlp_dim)?;
                    let cols = &mut pre.slice_mut(..);
                    self.bias(&fused, linear, rows, 3 * width, mlp_dim, false, cols)?;
                    Some(pre)
                }
                false => None,
            };
            drop(fused);
            let raw = match keep {
                true => Some(stream.clone_dtod(&qkv).map_err(copy())?),
                false => None,
            };
            for (scale, offset) in [(&block.query_norm, 0), (&block.key_norm, width)] {
                self.head_norm_rope(
                    &mut qkv, scale, &cos, &sin, 0, rows, heads, head_dim, offset,
                )?;
            }
            let (merged, log_sum_exp, qkv) = self.attention(&gpu, &qkv, rows, heads, width)?;
            let mlp_width = block.mlp_out.inner;
            let mut output = gpu.uninit(rows * width)?;
            self.project(
                &gpu,
                &block.mlp_out,
                &activated,
                rows,
                &mut output.slice_mut(..),
                0.0,
            )?;
            self.project(
                &gpu,
                &block.attention_out,
                &merged,
                rows,
                &mut output.slice_mut(..),
                1.0,
            )?;
            self.adapt(
                &gpu,
                adapted("fused_out"),
                &[(merged.all(), width), (activated.all(), mlp_width)],
                rows,
                &mut output.slice_mut(..),
                width,
            )?;
            let attention_out = &block.attention_out;
            let biased = match attention_out.bias {
                Some(_) => {
                    let mut biased = gpu.uninit(rows * width)?;
                    let cols = &mut biased.slice_mut(..);
                    self.bias(&output, attention_out, rows, 0, width, false, cols)?;
                    biased
                }
                None => output,
            };
            let gated = gpu.gate_residual(
                &hidden.slice(..),
                &biased.slice(..),
                &modulation,
                rows,
                width,
                rows,
            )?;
            let input = std::mem::replace(&mut hidden, gated);
            if let (Some(raw), Some(pre)) = (raw, pre) {
                single.push(SingleTape {
                    input,
                    normed,
                    attention: AttentionTape {
                        raw,
                        qkv,
                        merged,
                        log_sum_exp,
                    },
                    pre,
                    activated,
                });
            }
        }

        // The final layer's modulation is a shift then a scale, which is the
        // layout the norm kernel reads.
        let image = hidden.slice(text_rows * width..);
        let mut normed = gpu.uninit(image_rows * width)?;
        self.launch_norm(&image, &triple(2), &mut normed, image_rows, width, eps)?;
        let normed = gpu.narrowed(&normed.slice(..), image_rows * width, true)?;
        let output = self.apply(&gpu, &self.final_out, &[], &normed.all(), image_rows)?;
        let output = Matrix::from_vec(image_rows, self.final_out.units, gpu.download(&output)?);
        let tape = DeviceTape {
            modulation,
            cos,
            sin,
            text_rows,
            image_rows,
            double,
            single,
            hidden,
        };
        Ok((output, tape))
    }

    /// Adds `dL/d(adapter)` for every adapter in `lora` to its device
    /// gradient, given `dL/d(prediction)`: the device twin of
    /// [`DitLora::backward`]. `tape` is [`DeviceDit::forward_train`]'s with
    /// the same `lora`; the base weights stay frozen.
    pub fn backward(
        &self,
        dit: &Dit,
        lora: &mut DitLora,
        tape: &DeviceTape,
        grad_output: &Matrix,
    ) -> Result<(), NetworkError> {
        if self.quantized {
            return Err(NetworkError::InvalidConfig(
                "the device backward needs BF16 base weights, not int8 ones".into(),
            ));
        }
        if self.swiglu {
            return Err(NetworkError::InvalidConfig(
                "the device backward runs GELU denoisers only, not SwiGLU (FLUX.2) ones".into(),
            ));
        }
        if tape.double.len() != self.double.len() || tape.single.len() != self.single.len() {
            return Err(NetworkError::InvalidConfig(
                "the tape is not from DeviceDit::forward_train".into(),
            ));
        }
        if grad_output.rows != tape.image_rows || grad_output.cols != self.final_out.units {
            return Err(NetworkError::InvalidTarget {
                expected: tape.image_rows * self.final_out.units,
                actual: grad_output.data.len(),
            });
        }
        let gpu = Gpu {
            context: &self.context,
        };
        let config = dit.config();
        let (width, heads, eps) = (config.d_model, config.num_heads, config.eps);
        let head_dim = config.head_dim();
        let (text_rows, image_rows) = (tape.text_rows, tape.image_rows);
        let rows = text_rows + image_rows;
        let modulation = &tape.modulation;
        let single_base = self.double.len() * 12 * width;

        let grad_output = gpu.upload(&grad_output.data)?;
        let grad_output = gpu.narrowed(&grad_output.slice(..), grad_output.len(), true)?;
        let grad_normed = self.grad_of(&gpu, &self.final_out, &grad_output, image_rows)?;
        let mut grad = gpu.zeros(rows * width)?;
        let image = text_rows * width..;
        self.norm_backward(
            &mut grad.slice_mut(image.clone()),
            &grad_normed.slice(..),
            &tape.hidden.slice(image),
            &modulation.slice(single_base + self.single.len() * 3 * width..),
            image_rows,
            width,
            eps,
        )?;

        for (index, (block, saved)) in self.single.iter().zip(&tape.single).enumerate().rev() {
            let key = |name: &str| format!("single.{index}.{name}");
            let modulation = modulation.slice(single_base + index * 3 * width..);
            let mlp_dim = block.mlp_out.inner;
            let branch = self.gate(
                &gpu,
                &grad.slice(..),
                &modulation.slice(2 * width..),
                rows,
                width,
            )?;
            let mut grad_merged = self.grad_of(&gpu, &block.attention_out, &branch, rows)?;
            let mut grad_pre = self.grad_of(&gpu, &block.mlp_out, &branch, rows)?;
            let attention = &saved.attention;
            for adapter in lora.adapters_mut(&key("fused_out")) {
                self.adapter_backward(
                    &gpu,
                    adapter,
                    &branch.at(adapter.rows.start),
                    width,
                    &[
                        (attention.merged.all(), width),
                        (saved.activated.all(), mlp_dim),
                    ],
                    &mut [grad_merged.slice_mut(..), grad_pre.slice_mut(..)],
                    rows,
                )?;
            }
            self.gelu_backward(
                &mut grad_pre.slice_mut(..),
                &saved.pre.slice(..),
                rows * mlp_dim,
            )?;
            let grad_mlp = gpu.narrowed(&grad_pre.slice(..), rows * mlp_dim, true)?;
            let mut grad_qkv =
                self.attention_backward(&gpu, attention, &grad_merged, rows, heads, width)?;
            for (scale, offset) in [(&block.query_norm, 0), (&block.key_norm, width)] {
                self.head_norm_rope_backward(
                    &mut grad_qkv,
                    &attention.raw,
                    scale,
                    &tape.cos,
                    &tape.sin,
                    0,
                    rows,
                    heads,
                    head_dim,
                    offset,
                )?;
            }
            let mut grad_normed = gpu.uninit(rows * width)?;
            let fused_in = &block.fused_in;
            let mut target = grad_normed.slice_mut(..);
            self.input_grad(
                &gpu,
                fused_in,
                0,
                3 * width,
                &grad_qkv.all(),
                rows,
                &mut target,
                0.0,
            )?;
            let first = 3 * width;
            self.input_grad(
                &gpu,
                fused_in,
                first,
                mlp_dim,
                &grad_mlp.all(),
                rows,
                &mut target,
                1.0,
            )?;
            for adapter in lora.adapters_mut(&key("fused_in")) {
                let (band, stride) = match (adapter.rows.start, adapter.rows.end) {
                    (start, _) if start >= first => (grad_mlp.at(start - first), mlp_dim),
                    (start, end) if end <= first => (grad_qkv.at(start), first),
                    _ => {
                        return Err(NetworkError::InvalidConfig(format!(
                            "an adapter on {} spans the attention and the feed-forward",
                            key("fused_in")
                        )));
                    }
                };
                let inputs = [(saved.normed.all(), width)];
                let mut grads = [grad_normed.slice_mut(..)];
                self.adapter_backward(&gpu, adapter, &band, stride, &inputs, &mut grads, rows)?;
            }
            self.norm_backward(
                &mut grad.slice_mut(..),
                &grad_normed.slice(..),
                &saved.input.slice(..),
                &modulation,
                rows,
                width,
                eps,
            )?;
        }

        for (index, ((image_stream, text_stream), saved)) in
            self.double.iter().zip(&tape.double).enumerate().rev()
        {
            let base = index * 12 * width;
            let streams = [
                (text_stream, base + 6 * width, 0, text_rows, "text"),
                (image_stream, base, text_rows, image_rows, "image"),
            ];
            let key = |side: &str, name: &str| format!("double.{index}.{side}.{name}");
            let attention = &saved.attention;
            // The feed-forward half, then the attention's projection, per stream.
            let mut grad_merged = gpu.uninit(rows * width)?;
            for (&(stream, at, first, count, side), saved) in streams.iter().zip(&saved.streams) {
                let modulation = modulation.slice(at..);
                let span = first * width..(first + count) * width;
                let mlp_dim = stream.mlp_in.units;
                let gate = modulation.slice(5 * width..);
                let branch = self.gate(&gpu, &grad.slice(span.clone()), &gate, count, width)?;
                let mut grad_pre = self.grad_of(&gpu, &stream.mlp_out, &branch, count)?;
                for adapter in lora.adapters_mut(&key(side, "mlp_out")) {
                    self.adapter_backward(
                        &gpu,
                        adapter,
                        &branch.at(adapter.rows.start),
                        width,
                        &[(saved.activated.all(), mlp_dim)],
                        &mut [grad_pre.slice_mut(..)],
                        count,
                    )?;
                }
                self.gelu_backward(
                    &mut grad_pre.slice_mut(..),
                    &saved.pre.slice(..),
                    count * mlp_dim,
                )?;
                let grad_mlp = gpu.narrowed(&grad_pre.slice(..), count * mlp_dim, true)?;
                let mut grad_normed = self.grad_of(&gpu, &stream.mlp_in, &grad_mlp, count)?;
                for adapter in lora.adapters_mut(&key(side, "mlp_in")) {
                    self.adapter_backward(
                        &gpu,
                        adapter,
                        &grad_mlp.at(adapter.rows.start),
                        mlp_dim,
                        &[(saved.normed_mlp.all(), width)],
                        &mut [grad_normed.slice_mut(..)],
                        count,
                    )?;
                }
                self.norm_backward(
                    &mut grad.slice_mut(span.clone()),
                    &grad_normed.slice(..),
                    &saved.mid.slice(..),
                    &modulation.slice(3 * width..),
                    count,
                    width,
                    eps,
                )?;
                let gate = modulation.slice(2 * width..);
                let branch = self.gate(&gpu, &grad.slice(span.clone()), &gate, count, width)?;
                let projection = &stream.projection;
                let mut target = grad_merged.slice_mut(span.clone());
                self.input_grad(
                    &gpu,
                    projection,
                    0,
                    width,
                    &branch.all(),
                    count,
                    &mut target,
                    0.0,
                )?;
                for adapter in lora.adapters_mut(&key(side, "projection")) {
                    self.adapter_backward(
                        &gpu,
                        adapter,
                        &branch.at(adapter.rows.start),
                        width,
                        &[(attention.merged.at(first * width), width)],
                        &mut [grad_merged.slice_mut(span.clone())],
                        count,
                    )?;
                }
            }
            // The attention, then each stream's queries, keys and values back
            // to the block's input.
            let mut grad_qkv =
                self.attention_backward(&gpu, attention, &grad_merged, rows, heads, width)?;
            for (&(stream, at, first, count, side), saved) in streams.iter().zip(&saved.streams) {
                for (scale, offset) in [(&stream.query_norm, 0), (&stream.key_norm, width)] {
                    self.head_norm_rope_backward(
                        &mut grad_qkv,
                        &attention.raw,
                        scale,
                        &tape.cos,
                        &tape.sin,
                        first,
                        count,
                        heads,
                        head_dim,
                        offset,
                    )?;
                }
                let qkv_width = 3 * width;
                let mut grad_normed = gpu.uninit(count * width)?;
                let band = grad_qkv.at(first * qkv_width);
                let mut target = grad_normed.slice_mut(..);
                let linear = &stream.qkv;
                self.input_grad(&gpu, linear, 0, qkv_width, &band, count, &mut target, 0.0)?;
                for adapter in lora.adapters_mut(&key(side, "qkv")) {
                    self.adapter_backward(
                        &gpu,
                        adapter,
                        &grad_qkv.at(first * qkv_width + adapter.rows.start),
                        qkv_width,
                        &[(saved.normed.all(), width)],
                        &mut [grad_normed.slice_mut(..)],
                        count,
                    )?;
                }
                self.norm_backward(
                    &mut grad.slice_mut(first * width..(first + count) * width),
                    &grad_normed.slice(..),
                    &saved.input.slice(..),
                    &modulation.slice(at..),
                    count,
                    width,
                    eps,
                )?;
            }
        }
        Ok(())
    }

    /// One rectified-flow training step on `lora`: `latent` (packed, like the
    /// sampler's) is moved to noise level `sigma` along the straight path to
    /// `noise`, the model regresses the velocity `noise - latent`, and the
    /// optimizer steps on the adapters with the gradient clipped to a norm of
    /// 1. Returns the mean squared error before the step.
    #[allow(clippy::too_many_arguments)]
    pub fn flow_step(
        &self,
        dit: &Dit,
        lora: &mut DitLora,
        latent: &Matrix,
        noise: &Matrix,
        sigma: f32,
        conditioning: &Conditioning,
        optimizer: &crate::optimizers::Optimizer,
        step: usize,
    ) -> Result<f32, NetworkError> {
        let noisy = Matrix::from_vec(
            latent.rows,
            latent.cols,
            latent
                .data
                .iter()
                .zip(&noise.data)
                .map(|(x, n)| (1.0 - sigma) * x + sigma * n)
                .collect(),
        );
        let (prediction, tape) = self.forward_train(dit, lora, &noisy, sigma, conditioning)?;
        let count = prediction.data.len() as f32;
        let error: Vec<f32> = prediction
            .data
            .iter()
            .zip(latent.data.iter().zip(&noise.data))
            .map(|(p, (x, n))| p - (n - x))
            .collect();
        let loss = error.iter().map(|e| e * e).sum::<f32>() / count;
        let grad = Matrix::from_vec(
            prediction.rows,
            prediction.cols,
            error.iter().map(|e| 2.0 * e / count).collect(),
        );
        self.backward(dit, lora, &tape, &grad)?;
        crate::optimizers::step_clipped(&mut lora.params_mut(), optimizer, step, 1.0, 1.0)?;
        crate::optimizers::zero_grad(&mut lora.params_mut());
        Ok(loss)
    }

    /// A host matrix through an input layer, FP32 out.
    fn input(
        &self,
        gpu: &Gpu<'_>,
        linear: &Linear,
        tokens: &Matrix,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let tokens_on_device = gpu.upload(&tokens.data)?;
        let narrow = gpu.narrowed(&tokens_on_device.slice(..), tokens.data.len(), true)?;
        self.apply(gpu, linear, &[], &narrow.all(), tokens.rows)
    }

    /// `linear` and its `adapters` over `rows` narrow rows, its bias added,
    /// FP32 out.
    fn apply(
        &self,
        gpu: &Gpu<'_>,
        linear: &Linear,
        adapters: &[Adapter],
        x: &CudaView<'_, u8>,
        rows: usize,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let mut raw = gpu.uninit(rows * linear.units)?;
        gpu.project(
            &linear.weight,
            linear.units,
            linear.inner,
            x,
            true,
            &mut raw.slice_mut(..),
            rows,
            0.0,
        )?;
        let inputs = [(x.slice(..), linear.inner)];
        let mut target = raw.slice_mut(..);
        self.adapt(gpu, adapters, &inputs, rows, &mut target, linear.units)?;
        let mut out = gpu.uninit(rows * linear.units)?;
        self.bias(
            &raw,
            linear,
            rows,
            0,
            linear.units,
            false,
            &mut out.slice_mut(..),
        )?;
        Ok(out)
    }

    /// Adds every adapter's `scale * (x down^T) up^T` to its rows of `out`,
    /// the projection's raw output, `units` wide; `inputs` are the column
    /// blocks of `x` and their widths.
    fn adapt(
        &self,
        gpu: &Gpu<'_>,
        adapters: &[Adapter],
        inputs: &[(CudaView<'_, u8>, usize)],
        rows: usize,
        out: &mut CudaViewMut<'_, f32>,
        units: usize,
    ) -> Result<(), NetworkError> {
        for adapter in adapters {
            let hidden = self.hidden(gpu, adapter, inputs, rows)?;
            let up = resident(&adapter.lora.up)?;
            let rank = up.cols();
            let up = gpu.narrowed(&up.value().slice(..), up.rows() * rank, true)?;
            act_rhs_transposed(
                &self.context,
                &hidden.all(),
                rank,
                &up.all(),
                rank,
                true,
                false,
                &mut out.slice_mut(adapter.rows.start..),
                units,
                rows,
                adapter.rows.len(),
                rank,
                1.0,
                1.0,
            )?;
        }
        Ok(())
    }

    /// An adapter's `scale * x down^T`, narrow.
    fn hidden(
        &self,
        gpu: &Gpu<'_>,
        adapter: &Adapter,
        inputs: &[(CudaView<'_, u8>, usize)],
        rows: usize,
    ) -> Result<Act, NetworkError> {
        let down = resident(&adapter.lora.down)?;
        let (rank, inner) = (down.rows(), down.cols());
        let down = gpu.narrowed(&down.value().slice(..), rank * inner, true)?;
        let mut hidden = gpu.uninit(rows * rank)?;
        let mut column = 0;
        for (index, (x, cols)) in inputs.iter().enumerate() {
            let beta = if index == 0 { 0.0 } else { 1.0 };
            act_rhs_transposed(
                &self.context,
                x,
                *cols,
                &down.at(column),
                inner,
                true,
                false,
                &mut hidden,
                rank,
                rows,
                rank,
                *cols,
                adapter.lora.scale,
                beta,
            )?;
            column += cols;
        }
        gpu.narrowed(&hidden.slice(..), rows * rank, true)
    }

    /// Adds `-dL/d(up)` and `-dL/d(down)` to the adapter's device gradients
    /// and `dL/dx` to `grads`, one per block of `inputs`. `band` is
    /// `dL/d(output)` at the adapter's first row, in rows of `stride`.
    #[allow(clippy::too_many_arguments)]
    fn adapter_backward(
        &self,
        gpu: &Gpu<'_>,
        adapter: &mut Adapter,
        band: &CudaView<'_, u8>,
        stride: usize,
        inputs: &[(CudaView<'_, u8>, usize)],
        grads: &mut [CudaViewMut<'_, f32>],
        rows: usize,
    ) -> Result<(), NetworkError> {
        let hidden = self.hidden(gpu, adapter, inputs, rows)?;
        let (units, scale) = (adapter.rows.len(), adapter.lora.scale);
        let up = resident_mut(&mut adapter.lora.up)?;
        let rank = up.cols();
        let up_narrow = gpu.narrowed(&up.value().slice(..), units * rank, true)?;
        let beta = up.grad_beta();
        act_lhs_transposed(
            &self.context,
            band,
            stride,
            &hidden.all(),
            rank,
            true,
            false,
            up.negated_grad_mut(),
            rank,
            rows,
            units,
            rank,
            -1.0,
            beta,
        )?;
        let mut grad_hidden = gpu.act(rows * rank, true)?;
        act_plain(
            &self.context,
            band,
            stride,
            &up_narrow.all(),
            rank,
            true,
            true,
            grad_hidden.destination(),
            rank,
            rows,
            rank,
            units,
            scale,
            0.0,
        )?;
        let down = resident_mut(&mut adapter.lora.down)?;
        let inner = down.cols();
        let down_narrow = gpu.narrowed(&down.value().slice(..), rank * inner, true)?;
        // One beta for every block: they are disjoint columns of one gradient.
        let beta = down.grad_beta();
        let mut column = 0;
        for ((x, cols), grad) in inputs.iter().zip(grads.iter_mut()) {
            act_lhs_transposed(
                &self.context,
                &grad_hidden.all(),
                rank,
                x,
                *cols,
                true,
                false,
                &mut down.negated_grad_mut().slice_mut(column..),
                inner,
                rows,
                rank,
                *cols,
                -1.0,
                beta,
            )?;
            act_plain(
                &self.context,
                &grad_hidden.all(),
                rank,
                &down_narrow.at(column),
                inner,
                true,
                false,
                grad,
                *cols,
                rows,
                *cols,
                rank,
                1.0,
                1.0,
            )?;
            column += cols;
        }
        Ok(())
    }

    /// `dL/dx` of `linear` over `rows`, FP32.
    fn grad_of(
        &self,
        gpu: &Gpu<'_>,
        linear: &Linear,
        grad: &Act,
        rows: usize,
    ) -> Result<CudaSlice<f32>, NetworkError> {
        let mut out = gpu.uninit(rows * linear.inner)?;
        let mut target = out.slice_mut(..);
        self.input_grad(
            gpu,
            linear,
            0,
            linear.units,
            &grad.all(),
            rows,
            &mut target,
            0.0,
        )?;
        Ok(out)
    }

    /// `out = grad W[first..first + units] + beta out`, the input gradient
    /// through `units` of `linear`'s output rows.
    #[allow(clippy::too_many_arguments)]
    fn input_grad(
        &self,
        gpu: &Gpu<'_>,
        linear: &Linear,
        first: usize,
        units: usize,
        grad: &CudaView<'_, u8>,
        rows: usize,
        out: &mut CudaViewMut<'_, f32>,
        beta: f32,
    ) -> Result<(), NetworkError> {
        let weight = linear.weight.at(first * linear.inner);
        gpu.linear_packed_backward_input(
            &weight,
            units,
            linear.inner,
            grad,
            true,
            false,
            out,
            rows,
            beta,
        )
    }

    /// `grad * gate[column]`, narrow: a gated residual's branch gradient.
    fn gate(
        &self,
        gpu: &Gpu<'_>,
        grad: &CudaView<'_, f32>,
        gate: &CudaView<'_, f32>,
        rows: usize,
        width: usize,
    ) -> Result<Act, NetworkError> {
        let mut out = gpu.uninit(rows * width)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.gate_rows)
                .arg(&mut out)
                .arg(grad)
                .arg(gate)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .launch(cfg(rows * width))
                .map_err(cuda_err("MMDiT gate kernel"))?;
        }
        gpu.narrowed(&out.slice(..), rows * width, true)
    }

    /// The tanh GELU, narrow, of the `cols` columns of `source` at `offset`
    /// plus `bias`, or for a SwiGLU model SwiGLU, which is half as wide. The
    /// kernel writes BF16 itself, so there is no FP32 copy to narrow.
    #[allow(clippy::too_many_arguments)]
    fn gelu(
        &self,
        gpu: &Gpu<'_>,
        source: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        rows: usize,
        cols: usize,
        stride: usize,
        offset: usize,
    ) -> Result<Act, NetworkError> {
        let (cols, mode) = match self.swiglu {
            true => (cols / 2, 2i32),
            false => (cols, 1),
        };
        let mut out = gpu.act(rows * cols, true)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.bias_gelu)
                .arg(out.destination())
                .arg(source)
                .arg(bias.unwrap_or(source))
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(stride as i32))
                .arg(&(offset as i32))
                .arg(&mode)
                .arg(&i32::from(bias.is_some()))
                .arg(&1i32)
                .launch(cfg(rows * cols))
                .map_err(cuda_err("MMDiT GELU kernel"))?;
        }
        Ok(out)
    }

    fn project(
        &self,
        gpu: &Gpu<'_>,
        linear: &Linear,
        x: &Act,
        rows: usize,
        out: &mut CudaViewMut<'_, f32>,
        beta: f32,
    ) -> Result<(), NetworkError> {
        gpu.project(
            &linear.weight,
            linear.units,
            linear.inner,
            &x.all(),
            true,
            out,
            rows,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn bias(
        &self,
        source: &CudaSlice<f32>,
        linear: &Linear,
        rows: usize,
        offset: usize,
        cols: usize,
        gelu: bool,
        out: &mut CudaViewMut<'_, f32>,
    ) -> Result<(), NetworkError> {
        // A layer without a bias still goes through the kernel for the GELU
        // and the column slice; the source stands in as the unread bias.
        let bias = linear.bias.as_ref().unwrap_or(source);
        unsafe {
            self.context
                .stream
                .launch_builder(&self.bias_gelu)
                .arg(out)
                .arg(source)
                .arg(bias)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .arg(&(linear.units as i32))
                .arg(&(offset as i32))
                .arg(&i32::from(gelu))
                .arg(&i32::from(linear.bias.is_some()))
                .arg(&0i32)
                .launch(cfg(rows * cols))
                .map_err(cuda_err("MMDiT bias kernel"))?;
        }
        Ok(())
    }

    /// The modulated layer norm, narrowed for the GEMM that reads it.
    fn norm(
        &self,
        gpu: &Gpu<'_>,
        hidden: &CudaSlice<f32>,
        modulation: &CudaView<'_, f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<Act, NetworkError> {
        let mut normed = gpu.uninit(rows * width)?;
        self.launch_norm(&hidden.slice(..), modulation, &mut normed, rows, width, eps)?;
        gpu.narrowed(&normed.slice(..), rows * width, true)
    }

    fn launch_norm(
        &self,
        hidden: &CudaView<'_, f32>,
        modulation: &CudaView<'_, f32>,
        out: &mut CudaSlice<f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.layer_norm_modulate)
                .arg(out)
                .arg(hidden)
                .arg(modulation)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (rows.clamp(1, 1024) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(cuda_err("MMDiT layer norm kernel"))?;
        }
        Ok(())
    }

    /// The head norm and the rotation on `count` rows of `qkv` from row
    /// `first` on, at column `offset`.
    #[allow(clippy::too_many_arguments)]
    fn head_norm_rope(
        &self,
        qkv: &mut CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        first: usize,
        count: usize,
        heads: usize,
        head_dim: usize,
        offset: usize,
    ) -> Result<(), NetworkError> {
        let stride = 3 * heads * head_dim;
        let mut rows = qkv.slice_mut(first * stride..);
        unsafe {
            self.context
                .stream
                .launch_builder(&self.head_norm_rope)
                .arg(&mut rows)
                .arg(scale)
                .arg(cos)
                .arg(sin)
                .arg(&(count as i32))
                .arg(&(heads as i32))
                .arg(&(head_dim as i32))
                .arg(&(stride as i32))
                .arg(&(offset as i32))
                .arg(&(first as i32))
                .launch(cfg(count * heads * 32))
                .map_err(cuda_err("MMDiT head norm kernel"))?;
        }
        Ok(())
    }

    /// Adds `dL/d(hidden)` of [`DeviceDit::launch_norm`] to `grad_hidden`,
    /// given `dL/d(normed)`.
    #[allow(clippy::too_many_arguments)]
    fn norm_backward(
        &self,
        grad_hidden: &mut CudaViewMut<'_, f32>,
        grad_normed: &CudaView<'_, f32>,
        hidden: &CudaView<'_, f32>,
        modulation: &CudaView<'_, f32>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.layer_norm_modulate_bwd)
                .arg(grad_hidden)
                .arg(grad_normed)
                .arg(hidden)
                .arg(modulation)
                .arg(&(rows as i32))
                .arg(&(width as i32))
                .arg(&eps)
                .launch(LaunchConfig {
                    grid_dim: (rows.clamp(1, 1024) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(cuda_err("MMDiT layer norm backward kernel"))?;
        }
        Ok(())
    }

    /// The backward of [`DeviceDit::head_norm_rope`], in place on `grad`, a
    /// gradient laid out as the qkv buffer, narrow or wide; `raw` is that
    /// buffer before the norm.
    #[allow(clippy::too_many_arguments)]
    fn head_norm_rope_backward(
        &self,
        grad: &mut Act,
        raw: &CudaSlice<f32>,
        scale: &CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        first: usize,
        count: usize,
        heads: usize,
        head_dim: usize,
        offset: usize,
    ) -> Result<(), NetworkError> {
        let narrow = grad.flag();
        unsafe {
            self.context
                .stream
                .launch_builder(&self.head_norm_rope_bwd)
                .arg(grad.destination())
                .arg(raw)
                .arg(scale)
                .arg(cos)
                .arg(sin)
                .arg(&(count as i32))
                .arg(&(heads as i32))
                .arg(&(head_dim as i32))
                .arg(&((3 * heads * head_dim) as i32))
                .arg(&(offset as i32))
                .arg(&(first as i32))
                .arg(&narrow)
                .launch(cfg(count * heads))
                .map_err(cuda_err("MMDiT head norm backward kernel"))?;
        }
        Ok(())
    }

    /// `grad *= gelu'(pre)`.
    fn gelu_backward(
        &self,
        grad: &mut CudaViewMut<'_, f32>,
        pre: &CudaView<'_, f32>,
        len: usize,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.gelu_bwd)
                .arg(grad)
                .arg(pre)
                .arg(&(len as i32))
                .launch(cfg(len))
                .map_err(cuda_err("MMDiT GELU backward kernel"))?;
        }
        Ok(())
    }

    /// Unmasked attention over the whole `[rows, 3 * width]` sequence: its
    /// `[rows, width]` narrow output, its log-sum-exp, and the narrow qkv it
    /// read.
    fn attention(
        &self,
        gpu: &Gpu<'_>,
        qkv: &CudaSlice<f32>,
        rows: usize,
        heads: usize,
        width: usize,
    ) -> Result<(Act, CudaSlice<f32>, Act), NetworkError> {
        let flash = self.context.flash.as_ref().expect("checked at upload");
        let qkv = gpu.narrowed(&qkv.slice(..), rows * 3 * width, true)?;
        let mut merged = gpu.act(rows * width, true)?;
        let mut log_sum_exp = gpu.uninit(heads * rows)?;
        let shape = flash_shape(rows, heads, width);
        let scale = ((width / heads) as f32).sqrt().recip();
        gpu.flash_attention(
            flash,
            &qkv,
            &qkv,
            &mut merged,
            &mut log_sum_exp,
            shape,
            scale,
        )?;
        Ok((merged, log_sum_exp, qkv))
    }

    /// `dL/d(qkv)` of [`DeviceDit::attention`], narrow, given `dL/d(merged)`.
    fn attention_backward(
        &self,
        gpu: &Gpu<'_>,
        saved: &AttentionTape,
        grad_merged: &CudaSlice<f32>,
        rows: usize,
        heads: usize,
        width: usize,
    ) -> Result<Act, NetworkError> {
        let flash = self.context.flash.as_ref().expect("checked at upload");
        let grad_merged = gpu.narrowed(&grad_merged.slice(..), rows * width, true)?;
        let grad_qkv = gpu.act(rows * 3 * width, true)?;
        let mut delta = gpu.uninit(heads * rows)?;
        gpu.flash_attention_backward(
            flash,
            &saved.qkv,
            &saved.qkv,
            &saved.merged,
            &grad_merged,
            &saved.log_sum_exp,
            &mut delta,
            &grad_qkv,
            &grad_qkv,
            flash_shape(rows, heads, width),
            ((width / heads) as f32).sqrt().recip(),
        )?;
        Ok(grad_qkv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_training::cuda_doctor;
    use crate::mmdit::DitConfig;
    use crate::mmdit::tests::{checkpoint, conditioning, tiny};
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

    /// The tiny model with heads the fused attention takes.
    fn config() -> DitConfig {
        DitConfig {
            d_model: 128,
            num_heads: 2,
            axes_dim: vec![32, 32],
            ..tiny()
        }
    }

    #[test]
    fn a_device_dit_matches_the_host_or_skips_without_device() {
        let config = config();
        let path = std::env::temp_dir().join("rusting_brain_cuda_mmdit.safetensors");
        checkpoint(&config, &path);
        device_matches_host(&config, &path);
    }

    /// FLUX.2's shape: shared modulations, SwiGLU, four rotary axes, no
    /// biases and no pooled prompt.
    #[test]
    fn a_flux2_device_dit_matches_the_host_or_skips_without_device() {
        let config = DitConfig {
            pooled_dim: 0,
            axes_dim: vec![16, 16, 16, 16],
            ..config()
        };
        let path = std::env::temp_dir().join("rusting_brain_cuda_mmdit_flux2.safetensors");
        crate::mmdit::tests::flux2_checkpoint(&config, &path);
        device_matches_host(&config, &path);
    }

    fn device_matches_host(config: &DitConfig, path: &std::path::Path) {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let (height, width) = (3, 5);
        let patches = Matrix::from_vec(
            height * width,
            config.in_channels,
            (0..height * width * config.in_channels)
                .map(|index| (index as f32 * 0.23).cos())
                .collect(),
        );
        let conditioning = conditioning(config, height, width);
        for precision in [Precision::F32, Precision::Q8] {
            let dit = Dit::load_at(path, "", config.clone(), precision).unwrap();
            let device = DeviceDit::upload(&context, &dit).unwrap();
            let expected = dit.forward_with(&patches, 0.4, &conditioning).unwrap();
            let actual = device
                .forward_with(&dit, &patches, 0.4, &conditioning)
                .unwrap();

            assert_eq!((actual.rows, actual.cols), (expected.rows, expected.cols));
            // BF16 weights alone move this random model's output by about 2%
            // (0.8% at Q8, whose weights both sides read exactly), so the
            // whole-model check only catches a misplaced block. The kernels'
            // arithmetic is checked tightly by `the_mmdit_kernels_match_the_host`.
            let error: f32 = actual
                .data
                .iter()
                .zip(&expected.data)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum::<f32>()
                .sqrt();
            let norm: f32 = expected.data.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(
                error < 3e-2 * norm,
                "{precision:?}: error {error} of {norm}"
            );
        }
        std::fs::remove_file(path).ok();
    }

    /// Random adapters on every block projection, `to_q`, `to_k` and `to_v`
    /// as three on one, as a diffusers PEFT file places them.
    /// The test checkpoint with random weights at a `1 / sqrt(fan in)`
    /// scale: the repeating pattern of [`checkpoint`] leaves attention nearly
    /// blind to its queries and keys, so their adapter gradients would sit
    /// under the BF16 rounding.
    fn random_checkpoint(config: &DitConfig, path: &std::path::Path) {
        let mut tensors = crate::mmdit::tests::tensors(config);
        let mut state = 0x9e37_79b9u32;
        for (name, (shape, values)) in &mut tensors {
            let fan_in = *shape.last().unwrap() as f32;
            for value in values.iter_mut() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let uniform = state as f32 / u32::MAX as f32 * 2.0 - 1.0;
                *value = if name.ends_with("scale") {
                    1.0 + 0.1 * uniform
                } else if shape.len() == 1 {
                    0.02 * uniform
                } else {
                    uniform * (3.0 / fan_in).sqrt()
                };
            }
        }
        crate::safetensors::write_checkpoint(path, &tensors);
    }

    fn adapters(dit: &Dit) -> DitLora {
        let config = dit.config();
        let (d, mlp, rank) = (config.d_model, config.mlp_dim(), 4);
        // Each list holds the bounds of the row bands that get an adapter.
        let thirds = vec![0, d, 2 * d, 3 * d];
        let mut keys = Vec::new();
        for index in 0..config.double_blocks {
            for side in ["image", "text"] {
                let key = |name: &str| format!("double.{index}.{side}.{name}");
                keys.push((key("qkv"), thirds.clone()));
                keys.push((key("projection"), vec![0, d]));
                keys.push((key("mlp_in"), vec![0, mlp]));
                keys.push((key("mlp_out"), vec![0, d]));
            }
        }
        for index in 0..config.single_blocks {
            let mut fused = thirds.clone();
            fused.push(3 * d + mlp);
            keys.push((format!("single.{index}.fused_in"), fused));
            keys.push((format!("single.{index}.fused_out"), vec![0, d]));
        }
        let mut seed = 0.0f32;
        let mut random = |rows: usize, cols: usize| {
            seed += 1.0;
            let values = (0..rows * cols).map(|i| (i as f32 * 0.37 + seed).sin() * 0.1);
            crate::param::Param::new(Matrix::from_vec(rows, cols, values.collect()))
        };
        let mut lora = DitLora::default();
        for (key, bounds) in keys {
            let inner = crate::dit_lora::projection(dit, &key).unwrap().in_dim();
            let list = bounds
                .windows(2)
                .map(|rows| crate::dit_lora::Adapter {
                    lora: crate::param::Lora {
                        down: random(rank, inner),
                        up: random(rows[1] - rows[0], rank),
                        scale: 2.0,
                    },
                    rows: rows[0]..rows[1],
                })
                .collect();
            lora.adapters.insert(key, list);
        }
        lora
    }

    #[test]
    fn device_adapters_match_the_host_or_skip_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = config();
        let path = std::env::temp_dir().join("rusting_brain_cuda_mmdit_lora.safetensors");
        random_checkpoint(&config, &path);
        let dit = Dit::load(&path, "", config.clone()).unwrap();
        std::fs::remove_file(&path).ok();
        let (height, width) = (3, 5);
        let patches = Matrix::from_vec(
            height * width,
            config.in_channels,
            (0..height * width * config.in_channels)
                .map(|index| (index as f32 * 0.23).cos())
                .collect(),
        );
        let conditioning = conditioning(&config, height, width);
        let mut lora = adapters(&dit);
        let plain = dit.forward_with(&patches, 0.4, &conditioning).unwrap();
        let (expected, _) = lora.forward(&dit, &patches, 0.4, &conditioning).unwrap();
        let device = DeviceDit::upload(&context, &dit).unwrap();
        lora.move_to_cuda(&context).unwrap();
        let actual = device
            .forward_adapted(&dit, &lora, &patches, 0.4, &conditioning)
            .unwrap();

        let distance = |from: &Matrix| -> f32 {
            let pairs = from.data.iter().zip(&expected.data);
            pairs.map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt()
        };
        let norm: f32 = expected.data.iter().map(|v| v * v).sum::<f32>().sqrt();
        let (moved, error) = (distance(&plain), distance(&actual));
        // The same BF16 bound as the plain model, and well inside how far the
        // adapters move the output, so leaving them out cannot pass.
        assert!(error < 3e-2 * norm, "error {error} of {norm}");
        assert!(
            error < moved / 5.0,
            "error {error}, the adapters move it {moved}"
        );
    }

    #[test]
    fn flow_steps_fit_one_latent_or_skip_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = config();
        let path = std::env::temp_dir().join("rusting_brain_cuda_mmdit_flow.safetensors");
        random_checkpoint(&config, &path);
        let dit = Dit::load(&path, "", config.clone()).unwrap();
        std::fs::remove_file(&path).ok();
        let (height, width) = (3, 5);
        let wave = |phase: f32| {
            Matrix::from_vec(
                height * width,
                config.in_channels,
                (0..height * width * config.in_channels)
                    .map(|index| (index as f32 * 0.23 + phase).cos())
                    .collect(),
            )
        };
        let (latent, noise) = (wave(0.0), wave(1.7));
        let conditioning = conditioning(&config, height, width);
        let mut rng = rand::SeedableRng::seed_from_u64(7);
        let mut lora = DitLora::new(&dit, 4, 4.0, &mut rng);
        lora.move_to_cuda(&context).unwrap();
        let device = DeviceDit::upload(&context, &dit).unwrap();
        let optimizer = crate::optimizers::Optimizer::adam(1e-2);
        let losses: Vec<f32> = (1..=40)
            .map(|step| {
                device
                    .flow_step(
                        &dit,
                        &mut lora,
                        &latent,
                        &noise,
                        0.5,
                        &conditioning,
                        &optimizer,
                        step,
                    )
                    .unwrap()
            })
            .collect();
        println!("flow losses {:?} .. {:?}", &losses[..3], &losses[37..]);
        // One sample at one noise level: the adapters can fit it, so a step
        // that pushes the wrong way or does not reach them shows as a flat loss.
        assert!(losses[39] < 0.5 * losses[0], "{losses:?}");
    }

    #[test]
    fn device_adapter_gradients_match_the_host_or_skip_without_device() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = config();
        let path = std::env::temp_dir().join("rusting_brain_cuda_mmdit_grad.safetensors");
        random_checkpoint(&config, &path);
        let dit = Dit::load(&path, "", config.clone()).unwrap();
        std::fs::remove_file(&path).ok();
        let (height, width) = (3, 5);
        let patches = Matrix::from_vec(
            height * width,
            config.in_channels,
            (0..height * width * config.in_channels)
                .map(|index| (index as f32 * 0.23).cos())
                .collect(),
        );
        let conditioning = conditioning(&config, height, width);
        let mut host = adapters(&dit);
        let mut device_lora = host.clone();
        let (output, tape) = host.forward(&dit, &patches, 0.4, &conditioning).unwrap();
        let grad = Matrix::from_vec(
            output.rows,
            output.cols,
            (0..output.data.len())
                .map(|index| (index as f32 * 0.41).sin())
                .collect(),
        );
        host.backward(&dit, &tape, &grad).unwrap();

        let device = DeviceDit::upload(&context, &dit).unwrap();
        device_lora.move_to_cuda(&context).unwrap();
        let (_, tape) = device
            .forward_train(&dit, &device_lora, &patches, 0.4, &conditioning)
            .unwrap();
        device
            .backward(&dit, &mut device_lora, &tape, &grad)
            .unwrap();
        device_lora.move_to_cpu().unwrap();

        // Errors are BF16 noise of 1-5% a tensor here, worst on query and key
        // `down` gradients, which come back through BF16 attention scores; a
        // missing or misplaced term puts its tensor near 100%.
        let (mut total_norm, mut total_error) = (0.0f32, 0.0f32);
        let mut worst = (0.0f32, String::new());
        for (key, list) in &host.adapters {
            for (index, (expected, actual)) in
                list.iter().zip(&device_lora.adapters[key]).enumerate()
            {
                let pairs = [
                    ("down", &expected.lora.down.grad, &actual.lora.down.grad),
                    ("up", &expected.lora.up.grad, &actual.lora.up.grad),
                ];
                for (name, expected, actual) in pairs {
                    let norm = expected.data.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let error = expected
                        .data
                        .iter()
                        .zip(&actual.data)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    assert!(norm > 0.0, "{key} {index} {name} has no gradient");
                    total_norm += norm * norm;
                    total_error += error * error;
                    if error / norm > worst.0 {
                        worst = (error / norm, format!("{key} {index} {name}"));
                    }
                }
            }
        }
        let overall = (total_error / total_norm).sqrt();
        eprintln!(
            "relative gradient error {overall} overall, worst {} at {}",
            worst.0, worst.1
        );
        assert!(overall < 2e-2, "gradients off by {overall} overall");
        assert!(worst.0 < 1e-1, "{} off by {}", worst.1, worst.0);
    }

    #[test]
    fn the_mmdit_kernels_match_the_host() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = config();
        let path = std::env::temp_dir().join("rusting_brain_cuda_mmdit_kernels.safetensors");
        checkpoint(&config, &path);
        let dit = Dit::load_at(&path, "", config.clone(), Precision::F32).unwrap();
        std::fs::remove_file(path).ok();
        let device = DeviceDit::upload(&context, &dit).unwrap();
        let gpu = Gpu { context: &context };
        let (width, heads) = (config.d_model, config.num_heads);
        let (text_rows, rows) = (2, 7);
        let values = |count: usize, step: f32| -> Vec<f32> {
            (0..count)
                .map(|index| (index as f32 * step).sin() * 1.7)
                .collect()
        };
        let close = |name: &str, actual: &[f32], expected: &[f32]| {
            let worst = actual
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-4, "{name}: worst difference {worst}");
        };

        // The head norm and rotation on the query columns of a qkv buffer,
        // text rows and image rows launched apart as the forward does.
        let qkv = values(rows * 3 * width, 0.37);
        let positions = Positions::new(&config, text_rows, 1, rows - text_rows);
        let mut on_device = gpu.upload(&qkv).unwrap();
        let (cos, sin) = (
            gpu.upload(&positions.cos).unwrap(),
            gpu.upload(&positions.sin).unwrap(),
        );
        let block = &device.single[0];
        let head_dim = width / heads;
        for (first, count) in [(0, text_rows), (text_rows, rows - text_rows)] {
            device
                .head_norm_rope(
                    &mut on_device,
                    &block.key_norm,
                    &cos,
                    &sin,
                    first,
                    count,
                    heads,
                    head_dim,
                    width,
                )
                .unwrap();
        }
        let actual = gpu.download(&on_device).unwrap();
        let mut keys = Matrix::new(rows, width);
        for row in 0..rows {
            keys.row_mut(row)
                .copy_from_slice(&qkv[row * 3 * width + width..row * 3 * width + 2 * width]);
        }
        dit.single[0].key_norm.apply(&mut keys, heads);
        positions.apply(&mut keys, heads);
        let actual: Vec<f32> = (0..rows)
            .flat_map(|row| actual[row * 3 * width + width..row * 3 * width + 2 * width].to_vec())
            .collect();
        close("head norm and rotation", &actual, &keys.data);

        // The modulated layer norm, shift then scale.
        let hidden = Matrix::from_vec(rows, width, values(rows * width, 0.11));
        let modulation = values(2 * width, 0.53);
        let mut normed = gpu.uninit(rows * width).unwrap();
        device
            .launch_norm(
                &gpu.upload(&hidden.data).unwrap().slice(..),
                &gpu.upload(&modulation).unwrap().slice(..),
                &mut normed,
                rows,
                width,
                config.eps,
            )
            .unwrap();
        let expected = crate::mmdit::modulate(
            &crate::mmdit::layer_norm(&hidden, config.eps),
            &modulation[..width],
            &modulation[width..],
        );
        close(
            "layer norm",
            &gpu.download(&normed).unwrap(),
            &expected.data,
        );

        // A column slice of a biased layer's output, with the GELU.
        let linear = &block.fused_in;
        let source = values(rows * linear.units, 0.07);
        let bias = gpu.download(linear.bias.as_ref().unwrap()).unwrap();
        let (offset, cols) = (3 * width, linear.units - 3 * width);
        let mut out = gpu.uninit(rows * cols).unwrap();
        device
            .bias(
                &gpu.upload(&source).unwrap(),
                linear,
                rows,
                offset,
                cols,
                true,
                &mut out.slice_mut(..),
            )
            .unwrap();
        let expected: Vec<f32> = (0..rows * cols)
            .map(|index| {
                let (row, col) = (index / cols, index % cols);
                crate::ffn::gelu(source[row * linear.units + offset + col] + bias[offset + col])
            })
            .collect();
        close("bias and GELU", &gpu.download(&out).unwrap(), &expected);

        // The backward kernels against the host reference's.
        let keys_of = |buffer: &[f32]| {
            let data = (0..rows)
                .flat_map(|row| {
                    buffer[row * 3 * width + width..row * 3 * width + 2 * width].to_vec()
                })
                .collect();
            Matrix::from_vec(rows, width, data)
        };
        let grad = values(rows * 3 * width, 0.29);
        let wide = gpu.upload(&grad).unwrap();
        let mut on_device = gpu.narrowed(&wide.slice(..), grad.len(), false).unwrap();
        let raw = gpu.upload(&qkv).unwrap();
        for (first, count) in [(0, text_rows), (text_rows, rows - text_rows)] {
            device
                .head_norm_rope_backward(
                    &mut on_device,
                    &raw,
                    &block.key_norm,
                    &cos,
                    &sin,
                    first,
                    count,
                    heads,
                    head_dim,
                    width,
                )
                .unwrap();
        }
        let mut expected = keys_of(&grad);
        positions.inverse().apply(&mut expected, heads);
        let expected = crate::dit_lora::head_norm_backward(
            &dit.single[0].key_norm,
            &keys_of(&qkv),
            heads,
            &expected,
        );
        close(
            "head norm and rotation backward",
            &keys_of(&gpu.download_act(&on_device).unwrap()).data,
            &expected.data,
        );

        let start = values(rows * width, 0.41);
        let grad_normed = values(rows * width, 0.19);
        let mut grad_hidden = gpu.upload(&start).unwrap();
        device
            .norm_backward(
                &mut grad_hidden.slice_mut(..),
                &gpu.upload(&grad_normed).unwrap().slice(..),
                &gpu.upload(&hidden.data).unwrap().slice(..),
                &gpu.upload(&modulation).unwrap().slice(..),
                rows,
                width,
                config.eps,
            )
            .unwrap();
        let scaled: Vec<f32> = grad_normed
            .iter()
            .enumerate()
            .map(|(index, grad)| grad * (1.0 + modulation[width + index % width]))
            .collect();
        let delta = crate::dit_lora::layer_norm_backward(
            &hidden,
            config.eps,
            &Matrix::from_vec(rows, width, scaled),
        );
        let expected: Vec<f32> = start.iter().zip(&delta.data).map(|(a, b)| a + b).collect();
        close(
            "layer norm backward",
            &gpu.download(&grad_hidden).unwrap(),
            &expected,
        );

        let (pre, grad) = (values(rows * cols, 0.07), values(rows * cols, 0.13));
        let mut on_device = gpu.upload(&grad).unwrap();
        device
            .gelu_backward(
                &mut on_device.slice_mut(..),
                &gpu.upload(&pre).unwrap().slice(..),
                pre.len(),
            )
            .unwrap();
        let expected: Vec<f32> = grad
            .iter()
            .zip(&pre)
            .map(|(grad, pre)| grad * crate::ffn::gelu_derivative(*pre))
            .collect();
        close(
            "GELU backward",
            &gpu.download(&on_device).unwrap(),
            &expected,
        );
    }

    /// A small FLUX-shaped denoiser over a 512x512 image (32x32 patches) and
    /// 256 text tokens, host against device. Run by hand: `cargo test
    /// --release --features cuda --lib time_a_dit -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn time_a_dit_over_a_512_image() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = DitConfig {
            in_channels: 64,
            out_channels: 64,
            d_model: 768,
            num_heads: 6,
            double_blocks: 2,
            single_blocks: 4,
            mlp_ratio: 4.0,
            text_dim: 512,
            pooled_dim: 256,
            axes_dim: vec![64, 64],
            ..tiny()
        };
        let path = std::env::temp_dir().join("rusting_brain_cuda_mmdit_time.safetensors");
        checkpoint(&config, &path);
        let models = [Precision::F32, Precision::Q8]
            .map(|precision| Dit::load_at(&path, "", config.clone(), precision).unwrap());
        std::fs::remove_file(path).ok();
        let (height, width) = (32, 32);
        let patches = Matrix::from_vec(
            height * width,
            config.in_channels,
            (0..height * width * config.in_channels)
                .map(|index| (index as f32 * 0.23).cos())
                .collect(),
        );
        let conditioning = Conditioning {
            text: Matrix::from_vec(
                256,
                config.text_dim,
                (0..256 * config.text_dim)
                    .map(|index| (index as f32 * 0.37).sin())
                    .collect(),
            ),
            ..conditioning(&config, height, width)
        };

        // BF16 weights against the byte path, each against its own host.
        for (dit, precision) in models.iter().zip([Precision::F32, Precision::Q8]) {
            let device = DeviceDit::upload(&context, dit).unwrap();
            let started = std::time::Instant::now();
            let expected = dit.forward_with(&patches, 0.4, &conditioning).unwrap();
            let host = started.elapsed();
            device
                .forward_with(dit, &patches, 0.4, &conditioning)
                .unwrap();
            let mut device_times = Vec::new();
            let mut actual = None;
            for _ in 0..5 {
                let started = std::time::Instant::now();
                actual = Some(
                    device
                        .forward_with(dit, &patches, 0.4, &conditioning)
                        .unwrap(),
                );
                device_times.push(started.elapsed());
            }
            let actual = actual.unwrap();
            let error: f32 = actual
                .data
                .iter()
                .zip(&expected.data)
                .map(|(actual, expected)| (actual - expected).powi(2))
                .sum::<f32>()
                .sqrt();
            let norm: f32 = expected.data.iter().map(|v| v * v).sum::<f32>().sqrt();
            println!(
                "{precision:?}: host {host:?}, device {device_times:?}, relative error {}",
                error / norm
            );
        }
    }

    /// One adapter training step - forward, backward - at 1024 image and 256
    /// text tokens on the shape above, adapters on every block projection.
    /// Run with `cargo test --release --features cuda --lib
    /// time_a_dit_adapter_step -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn time_a_dit_adapter_step() {
        let Some(context) = cuda_or_skip() else {
            return;
        };
        let config = DitConfig {
            in_channels: 64,
            out_channels: 64,
            d_model: 768,
            num_heads: 6,
            double_blocks: 2,
            single_blocks: 4,
            mlp_ratio: 4.0,
            text_dim: 512,
            pooled_dim: 256,
            axes_dim: vec![64, 64],
            ..tiny()
        };
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("target/rusting_brain_cuda_mmdit_step.safetensors");
        checkpoint(&config, &path);
        let dit = Dit::load(&path, "", config.clone()).unwrap();
        std::fs::remove_file(path).ok();
        let (height, width) = (32, 32);
        let patches = Matrix::from_vec(
            height * width,
            config.in_channels,
            (0..height * width * config.in_channels)
                .map(|index| (index as f32 * 0.23).cos())
                .collect(),
        );
        let conditioning = Conditioning {
            text: Matrix::from_vec(
                256,
                config.text_dim,
                (0..256 * config.text_dim)
                    .map(|index| (index as f32 * 0.37).sin())
                    .collect(),
            ),
            ..conditioning(&config, height, width)
        };
        let grad = Matrix::from_vec(
            patches.rows,
            config.out_channels,
            vec![1e-3; patches.rows * config.out_channels],
        );
        let device = DeviceDit::upload(&context, &dit).unwrap();
        let mut lora = adapters(&dit);
        lora.move_to_cuda(&context).unwrap();
        let mut times = Vec::new();
        for _ in 0..6 {
            let started = std::time::Instant::now();
            let (_, tape) = device
                .forward_train(&dit, &lora, &patches, 0.4, &conditioning)
                .unwrap();
            device.backward(&dit, &mut lora, &tape, &grad).unwrap();
            context.stream.synchronize().unwrap();
            times.push(started.elapsed());
        }
        let (free, total) = cudarc::driver::result::mem_get_info().unwrap();
        println!(
            "adapter step: {times:?}, device memory in use {} MiB",
            (total - free) >> 20
        );
    }
}
