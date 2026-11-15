//! SANA-Sprint on the device: Gemma-2, the Sana DiT and the DC-AE decoder
//! resident in BF16, with every activation, norm, attention and residual in
//! FP32 and every GEMM accumulating in FP32. The twin of [`crate::sana`].
//!
//! The SCM sampler, the timestep and caption embedders and the tokenizer stay
//! on the host: they are a few vectors a step.
//!
//! A 3x3 convolution is nine accumulating GEMMs over one zero-padded BF16
//! copy of its input: with the output laid out on the padded width, tap
//! `(dy, dx)` reads the input `dy * (width + 2) + dx` rows further on, so no
//! im2col buffer exists. The two padding columns of each output row are
//! dropped when the bias is added.

use crate::conv::{Dense, FeatureMap};
use crate::cuda_flash::compile_ptx;
use crate::cuda_training::{cfg, cuda_err};
use crate::dcae::{Block, Conv3, DcaeDecoder};
use crate::gpu_model::{Act, Gpu};
use crate::gpu_transformer::{GpuContext, act_plain, act_rhs_transposed};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::safetensors::ShardedSafeTensors;
use crate::sana::{
    CROSS_HEAD_DIM, CROSS_HEADS, DIT_DIM, DIT_HEAD_DIM, DIT_HEADS, DOWNSCALE, FF_HIDDEN, GEMMA_EPS,
    GEMMA_HEAD_DIM, GEMMA_HEADS, GEMMA_HIDDEN, GEMMA_KV_HEADS, GEMMA_LAYERS, GEMMA_SOFTCAP,
    GEMMA_THETA, GUIDANCE, GemmaEncoder, GemmaLayer, LATENT_CHANNELS, NORM_EPS, QK_EPS, SanaDit,
    check_size, gemma_norm, latent_map, sample, select_prompt, transpose,
};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaView, CudaViewMut, DevicePtrMut, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use half::bf16;
use std::path::Path;
use std::sync::{Arc, OnceLock};

type R<T> = Result<T, NetworkError>;

const KERNELS: &str = r#"
typedef unsigned short bf16_t;
typedef unsigned long long u64;
__device__ __forceinline__ bf16_t to_bf(float v){
  unsigned u=__float_as_uint(v);
  if((u&0x7f800000u)==0x7f800000u&&(u&0x007fffffu))return (bf16_t)0x7fc0u;
  unsigned r=((u>>16)&1u)+0x7fffu;return (bf16_t)((u+r)>>16);
}
__device__ __forceinline__ float silu(float v){return v/(1.f+expf(-v));}
__device__ __forceinline__ float gelu_tanh(float v){return 0.5f*v*(1.f+tanhf(0.7978845608f*(v+0.044715f*v*v*v)));}
__device__ __forceinline__ u64 gid(){return (u64)blockIdx.x*blockDim.x+threadIdx.x;}
__device__ float block_sum(float v){
  __shared__ float part[32];
  for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffffu,v,o);
  int lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=(blockDim.x+31)>>5;
  __syncthreads();if(lane==0)part[warp]=v;__syncthreads();
  v=lane<warps?part[lane]:0.f;
  for(int o=16;o>0;o>>=1)v+=__shfl_xor_sync(0xffffffffu,v,o);
  return v;
}
__device__ float block_max(float v){
  __shared__ float part[32];
  for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,o));
  int lane=threadIdx.x&31,warp=threadIdx.x>>5,warps=(blockDim.x+31)>>5;
  __syncthreads();if(lane==0)part[warp]=v;__syncthreads();
  v=lane<warps?part[lane]:__int_as_float(0xff800000);
  for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,o));
  return v;
}
// out = rms(x) * w + b, or out += that. One block a row.
extern "C" __global__ void rms(float*out,int out_stride,const float*x,int x_stride,const float*w,const float*b,int cols,float eps,int add){
  const float*xr=x+(u64)blockIdx.x*x_stride;float*o=out+(u64)blockIdx.x*out_stride;
  float s=0.f;for(int c=threadIdx.x;c<cols;c+=blockDim.x){float v=xr[c];s+=v*v;}
  float inv=rsqrtf(block_sum(s)/cols+eps);
  for(int c=threadIdx.x;c<cols;c+=blockDim.x){float v=xr[c]*inv*w[c]+b[c];if(add)o[c]+=v;else o[c]=v;}
}
extern "C" __global__ void rms_inplace(float*x,int stride,const float*w,int cols,float eps,int relu){
  float*xr=x+(u64)blockIdx.x*stride;
  float s=0.f;for(int c=threadIdx.x;c<cols;c+=blockDim.x){float v=xr[c];s+=v*v;}
  float inv=rsqrtf(block_sum(s)/cols+eps);
  for(int c=threadIdx.x;c<cols;c+=blockDim.x){float v=xr[c]*inv*w[c];xr[c]=relu?fmaxf(v,0.f):v;}
}
// Layer norm without a gain, then * (1 + scale) + shift.
extern "C" __global__ void ln_mod(float*out,const float*x,const float*shift,const float*scale,int cols,float eps){
  const float*xr=x+(u64)blockIdx.x*cols;float*o=out+(u64)blockIdx.x*cols;
  float s=0.f;for(int c=threadIdx.x;c<cols;c+=blockDim.x)s+=xr[c];
  float mean=block_sum(s)/cols;
  float q=0.f;for(int c=threadIdx.x;c<cols;c+=blockDim.x){float d=xr[c]-mean;q+=d*d;}
  float inv=rsqrtf(block_sum(q)/cols+eps);
  for(int c=threadIdx.x;c<cols;c+=blockDim.x)o[c]=(xr[c]-mean)*inv*(1.f+scale[c])+shift[c];
}
// x += b per column, then SiLU if act.
extern "C" __global__ void bias_act(float*x,const float*b,u64 n,int cols,int act){
  u64 i=gid();if(i>=n)return;float v=x[i]+b[i%cols];x[i]=act?silu(v):v;
}
// h += g * (u + b), or h += u + b without a gate.
extern "C" __global__ void gated_add(float*h,const float*u,const float*b,const float*g,u64 n,int cols,int gated){
  u64 i=gid();if(i>=n)return;int c=i%cols;float v=u[i]+b[c];h[i]+=gated?g[c]*v:v;
}
// Rows of [a | g] to gelu_tanh(a) * g (kind 0) or a * silu(g) (kind 1).
extern "C" __global__ void glu(float*out,const float*x,u64 n,int half,int kind){
  u64 i=gid();if(i>=n)return;u64 r=i/half;int c=i-r*half;
  float a=x[r*2*half+c],g=x[r*2*half+half+c];out[i]=kind?a*silu(g):gelu_tanh(a)*g;
}
// x as the sum of two BF16 operands, the second its rounding error.
extern "C" __global__ void split(bf16_t*hi,bf16_t*lo,const float*x,u64 n){
  u64 i=gid();if(i>=n)return;float v=x[i];bf16_t h=to_bf(v);hi[i]=h;lo[i]=to_bf(v-__uint_as_float(((unsigned)h)<<16));
}
// Rotary embedding, half-split pairs, in place.
extern "C" __global__ void rope(float*x,const float*cs,const float*sn,int rows,int heads,int hd,int stride,int off){
  int half=hd/2;u64 i=gid();if(i>=(u64)rows*heads*half)return;
  int p=i%half;u64 rh=i/half;int h=rh%heads;int r=rh/heads;
  float*v=x+(u64)r*stride+off+h*hd;float c=cs[(u64)r*half+p],s=sn[(u64)r*half+p];
  float a=v[p],b=v[p+half];v[p]=a*c-b*s;v[p+half]=b*c+a*s;
}
// Gemma's attention probabilities: scale, tanh cap, causal and padding mask.
// One block a (head, query) row of `len` scores, `stride` apart, in place.
extern "C" __global__ void softmax_cap(float*o_all,const float*s,int len,int stride,float scale,float cap,const int*mask){
  int row=blockIdx.x%len;const float*sr=s+(u64)blockIdx.x*stride;float*o=o_all+(u64)blockIdx.x*stride;
  float m=__int_as_float(0xff800000);
  for(int j=threadIdx.x;j<len;j+=blockDim.x)if(j<=row&&mask[j])m=fmaxf(m,tanhf(sr[j]*scale/cap)*cap);
  m=block_max(m);
  float sum=0.f;for(int j=threadIdx.x;j<len;j+=blockDim.x)if(j<=row&&mask[j])sum+=expf(tanhf(sr[j]*scale/cap)*cap-m);
  sum=block_sum(sum);
  for(int j=threadIdx.x;j<stride;j+=blockDim.x)o[j]=j<len&&j<=row&&mask[j]?expf(tanhf(sr[j]*scale/cap)*cap-m)/sum:0.f;
}
// Linear attention, pass 1: kv[h][i][j] = sum_t v_i k_j, and row d = sum_t k_j.
// A head's channel e sits at x[t * stride + h * head_stride + base + e].
extern "C" __global__ void lin_kv(float*kv,const float*x,int n,int d,int stride,int head_stride,int kb,int vb,int relu,int chunk){
  int h=blockIdx.x/(d+1),i=blockIdx.x%(d+1),j=threadIdx.x;if(j>=d)return;
  int t0=blockIdx.y*chunk,t1=min(n,t0+chunk);float s=0.f;
  for(int t=t0;t<t1;t++){const float*r=x+(u64)t*stride+h*head_stride;float k=r[kb+j];if(relu)k=fmaxf(k,0.f);s+=(i<d?r[vb+i]:1.f)*k;}
  atomicAdd(&kv[((u64)h*(d+1)+i)*d+j],s);
}
// Pass 2: out[t, h * d + i] = (kv[h][i] . q) / (kv[h][d] . q + 1e-15).
extern "C" __global__ void lin_out(float*out,const float*x,const float*kv,int n,int heads,int d,int stride,int head_stride,int qb,int relu){
  u64 idx=gid();if(idx>=(u64)n*heads*d)return;int i=idx%d;u64 th=idx/d;int h=th%heads;u64 t=th/heads;
  const float*q=x+t*stride+h*head_stride+qb;const float*m=kv+(u64)h*(d+1)*d;
  float num=0.f,den=0.f;
  for(int j=0;j<d;j++){float qj=q[j];if(relu)qj=fmaxf(qj,0.f);num+=m[i*d+j]*qj;den+=m[d*d+j]*qj;}
  out[idx]=num/(den+1e-15f);
}
// Softmax attention of each query over a few keys, one thread a (query, head).
extern "C" __global__ void cross_attn(float*out,const float*q,int q_stride,const float*k,const float*v,int kv_stride,int n,int m,int heads,int d,float scale){
  int idx=gid();if(idx>=n*heads)return;int h=idx%heads,t=idx/heads;
  const float*qq=q+(u64)t*q_stride+h*d;float mx=__int_as_float(0xff800000);
  for(int j=0;j<m;j++){const float*kk=k+(u64)j*kv_stride+h*d;float s=0.f;for(int e=0;e<d;e++)s+=qq[e]*kk[e];mx=fmaxf(mx,s*scale);}
  float acc[128];for(int e=0;e<d;e++)acc[e]=0.f;float sum=0.f;
  for(int j=0;j<m;j++){const float*kk=k+(u64)j*kv_stride+h*d;float s=0.f;for(int e=0;e<d;e++)s+=qq[e]*kk[e];
    float p=expf(s*scale-mx);sum+=p;const float*vv=v+(u64)j*kv_stride+h*d;for(int e=0;e<d;e++)acc[e]+=p*vv[e];}
  float*o=out+(u64)t*heads*d+h*d;for(int e=0;e<d;e++)o[e]=acc[e]/sum;
}
// k x k depthwise convolution, zero padded, token-major; w is [C, k, k].
extern "C" __global__ void dw(float*out,int out_stride,const float*x,int x_stride,const float*w,const float*b,int H,int W,int C,int k,int biased){
  u64 i=gid();if(i>=(u64)H*W*C)return;int c=i%C;u64 p=i/C;int y=p/W,xx=p%W,half=k/2;
  float s=biased?b[c]:0.f;
  for(int dy=0;dy<k;dy++){int sy=y+dy-half;if(sy<0||sy>=H)continue;
    for(int dx=0;dx<k;dx++){int sx=xx+dx-half;if(sx<0||sx>=W)continue;
      s+=w[c*k*k+dy*k+dx]*x[((u64)sy*W+sx)*x_stride+c];}}
  out[p*out_stride+c]=s;
}
// Grouped 1x1: out[t, o] = sum_e x[t, (o / g) g + e] w[o, e].
extern "C" __global__ void grouped(float*out,int out_stride,const float*x,const float*w,u64 n,int C,int g){
  u64 i=gid();if(i>=n*C)return;int o=i%C;u64 t=i/C;const float*src=x+t*C+(o/g)*g;const float*wo=w+(u64)o*g;
  float s=0.f;for(int e=0;e<g;e++)s+=src[e]*wo[e];out[t*out_stride+o]=s;
}
// A convolution's input on the padded grid: [(H + 2)(W + 2) + 2, C] BF16 with a
// zero border, read through a 2x nearest upsampling when `up`, after an
// activation (1 SiLU, 2 ReLU).
extern "C" __global__ void pad(bf16_t*out,const float*x,int H,int W,int C,int up,int act){
  u64 i=gid();if(i>=((u64)(H+2)*(W+2)+2)*C)return;int c=i%C;u64 p=i/C;int py=p/(W+2),px=p%(W+2);float v=0.f;
  if(py>=1&&py<=H&&px>=1&&px<=W){int y=py-1,xx=px-1;u64 s=up?(u64)(y/2)*(W/2)+xx/2:(u64)y*W+xx;v=x[s*C+c];
    if(act==1)v=silu(v);else if(act==2)v=fmaxf(v,0.f);}
  out[i]=to_bf(v);
}
// Drops the padded grid's two extra columns and adds the bias.
extern "C" __global__ void unpad(float*out,const float*tmp,const float*b,int H,int W,int C){
  u64 i=gid();if(i>=(u64)H*W*C)return;int c=i%C;u64 p=i/C;int y=p/W,x=p%W;
  out[i]=tmp[((u64)y*(W+2)+x)*C+c]+b[c];
}
// The DC-AE channel-repeat shortcuts: conv_in's (out[p, c] += src[p, c / r])
// and DCUpBlock2d's, through a pixel shuffle when `up`.
extern "C" __global__ void up_add(float*out,const float*src,int H,int W,int C,int Cs,int repeats,int up){
  u64 i=gid();if(i>=(u64)H*W*C)return;int c=i%C;u64 p=i/C;int y=p/W,x=p%W;u64 sp;int idx;
  if(up){int sub=(y%2)*2+x%2;sp=(u64)(y/2)*(W/2)+x/2;idx=(c*4+sub)/repeats;}else{sp=p;idx=c/repeats;}
  out[i]+=src[sp*Cs+idx];
}
"#;

fn ptx() -> R<&'static Ptx> {
    static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| compile_ptx(KERNELS).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| NetworkError::Cuda(format!("SANA kernel compilation failed: {error}")))
}

struct Kernels {
    rms: CudaFunction,
    rms_inplace: CudaFunction,
    ln_mod: CudaFunction,
    bias_act: CudaFunction,
    gated_add: CudaFunction,
    glu: CudaFunction,
    split: CudaFunction,
    rope: CudaFunction,
    softmax_cap: CudaFunction,
    lin_kv: CudaFunction,
    lin_out: CudaFunction,
    cross_attn: CudaFunction,
    dw: CudaFunction,
    grouped: CudaFunction,
    pad: CudaFunction,
    unpad: CudaFunction,
    up_add: CudaFunction,
}

/// One block a row, of 128 threads for a narrow row and 256 otherwise.
fn per_row(rows: usize, cols: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (if cols <= 128 { 128 } else { 256 }, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// A dense layer (or several stacked) as a BF16 `[units, inner]` weight and
/// an FP32 bias, zeros where the checkpoint has none.
struct Lin {
    w: Act,
    b: CudaSlice<f32>,
    units: usize,
    inner: usize,
}

struct Conv {
    /// `[out, (ky, kx, in)]`, as [`Conv3`] holds it.
    w: Act,
    b: CudaSlice<f32>,
    inputs: usize,
    outputs: usize,
}

struct GemmaBlock {
    input_norm: CudaSlice<f32>,
    post_attention_norm: CudaSlice<f32>,
    pre_feedforward_norm: CudaSlice<f32>,
    post_feedforward_norm: CudaSlice<f32>,
    /// q | k | v, `[2048 + 1024 + 1024, 2304]`.
    qkv: Lin,
    output: Lin,
    /// gate | up.
    gate_up: Lin,
    down: Lin,
}

struct DitBlock {
    table: Vec<f32>,
    qkv: Lin,
    out: Lin,
    query_norm: CudaSlice<f32>,
    key_norm: CudaSlice<f32>,
    cross_query: Lin,
    /// k | v over the prompt.
    cross_kv: Lin,
    cross_out: Lin,
    cross_query_norm: CudaSlice<f32>,
    cross_key_norm: CudaSlice<f32>,
    inverted: Lin,
    depth_weight: CudaSlice<f32>,
    depth_bias: CudaSlice<f32>,
    point: Lin,
}

struct Affine {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

struct Vit {
    qkv: Lin,
    multiscale_in: CudaSlice<f32>,
    multiscale_out: CudaSlice<f32>,
    out: Lin,
    attention_norm: Affine,
    inverted: Lin,
    depth_weight: CudaSlice<f32>,
    depth_bias: CudaSlice<f32>,
    point: Lin,
    norm: Affine,
}

#[allow(clippy::large_enum_variant)] // a few dozen blocks, built once
enum Stage {
    Up {
        conv: Conv,
        repeats: usize,
    },
    Res {
        conv1: Conv,
        conv2: Conv,
        norm: Affine,
    },
    Vit(Box<Vit>),
}

/// The whole SANA-Sprint pipeline resident on one device.
pub struct DeviceSana {
    context: Arc<GpuContext>,
    k: Kernels,
    /// Zeros as wide as any row, the bias of a bias-free norm.
    zeros: CudaSlice<f32>,
    encoder: GemmaEncoder,
    /// Gemma's embedding table stays on the host, in its own BF16: a lookup
    /// reads 506 of its 256000 rows.
    embed: Vec<bf16>,
    gemma: Vec<GemmaBlock>,
    gemma_norm: CudaSlice<f32>,
    /// The host DiT without its blocks: the time, guidance and caption
    /// embedders and the final modulation table.
    dit: SanaDit,
    patch: Lin,
    proj_out: Lin,
    blocks: Vec<DitBlock>,
    conv_in: Conv,
    stages: Vec<Vec<Stage>>,
    norm_out: Affine,
    conv_out: Conv,
}

impl std::fmt::Debug for DeviceSana {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceSana").finish_non_exhaustive()
    }
}

fn lin(gpu: &Gpu<'_>, parts: &[&Dense]) -> R<Lin> {
    let inner = parts[0].in_dim();
    let (mut w, mut b) = (Vec::new(), Vec::new());
    for part in parts {
        if part.in_dim() != inner {
            return Err(NetworkError::InvalidConfig(format!(
                "stacked layers take {inner} and {} inputs",
                part.in_dim()
            )));
        }
        w.extend_from_slice(&part.weight_f32().data);
        match &part.bias {
            Some(bias) => b.extend_from_slice(bias),
            None => b.extend(std::iter::repeat_n(0.0, part.out_dim())),
        }
    }
    let units = w.len() / inner;
    Ok(Lin {
        w: bf16_weight(gpu, &w)?,
        b: gpu.upload(&b)?,
        units,
        inner,
    })
}

fn bf16_weight(gpu: &Gpu<'_>, values: &[f32]) -> R<Act> {
    let wide = gpu.upload(values)?;
    gpu.narrowed(&wide.slice(..), values.len(), true)
}

fn conv(gpu: &Gpu<'_>, conv: &Conv3) -> R<Conv> {
    let outputs = conv.weight.rows;
    Ok(Conv {
        w: bf16_weight(gpu, &conv.weight.data)?,
        b: gpu.upload(&conv.bias.clone().unwrap_or_else(|| vec![0.0; outputs]))?,
        inputs: conv.inputs,
        outputs,
    })
}

fn affine(gpu: &Gpu<'_>, norm: &crate::dcae::Norm) -> R<Affine> {
    Ok(Affine {
        weight: gpu.upload(&norm.weight)?,
        bias: gpu.upload(&norm.bias)?,
    })
}

impl DeviceSana {
    /// Uploads a diffusers `Sana_Sprint_0.6B_1024px_diffusers` folder. Gemma
    /// goes up one layer at a time, so the host never holds it whole in FP32.
    pub fn load(context: &Arc<GpuContext>, model: impl AsRef<Path>) -> R<Self> {
        let model = model.as_ref();
        if context.half_accumulate {
            return Err(NetworkError::InvalidConfig(
                "SANA-Sprint needs a BF16 context, not an FP16 one".into(),
            ));
        }
        let gpu = Gpu { context };
        let module = context
            .stream
            .context()
            .load_module(ptx()?.clone())
            .map_err(cuda_err("SANA module loading"))?;
        let get = |name: &str| {
            module
                .load_function(name)
                .map_err(cuda_err("SANA kernel lookup"))
        };
        let k = Kernels {
            rms: get("rms")?,
            rms_inplace: get("rms_inplace")?,
            ln_mod: get("ln_mod")?,
            bias_act: get("bias_act")?,
            gated_add: get("gated_add")?,
            glu: get("glu")?,
            split: get("split")?,
            rope: get("rope")?,
            softmax_cap: get("softmax_cap")?,
            lin_kv: get("lin_kv")?,
            lin_out: get("lin_out")?,
            cross_attn: get("cross_attn")?,
            dw: get("dw")?,
            grouped: get("grouped")?,
            pad: get("pad")?,
            unpad: get("unpad")?,
            up_add: get("up_add")?,
        };

        let encoder = GemmaEncoder::open(model)?;
        let mut file = ShardedSafeTensors::open(&encoder.weights)?;
        let embed = file
            .tensor("embed_tokens.weight")?
            .0
            .into_iter()
            .map(bf16::from_f32)
            .collect();
        let mut gemma = Vec::with_capacity(GEMMA_LAYERS);
        for layer in 0..GEMMA_LAYERS {
            let l = GemmaLayer::read(&mut file, layer)?;
            gemma.push(GemmaBlock {
                input_norm: gpu.upload(&l.input_norm)?,
                post_attention_norm: gpu.upload(&l.post_attention_norm)?,
                pre_feedforward_norm: gpu.upload(&l.pre_feedforward_norm)?,
                post_feedforward_norm: gpu.upload(&l.post_feedforward_norm)?,
                qkv: lin(&gpu, &[&l.query, &l.key, &l.value])?,
                output: lin(&gpu, &[&l.output])?,
                gate_up: lin(&gpu, &[&l.gate, &l.up])?,
                down: lin(&gpu, &[&l.down])?,
            });
        }
        let gemma_norm = gpu.upload(&gemma_norm(&mut file, "norm.weight")?)?;
        drop(file);

        let mut dit = SanaDit::load(model)?;
        let blocks = dit
            .blocks
            .iter()
            .map(|b| {
                Ok(DitBlock {
                    table: b.table.clone(),
                    qkv: lin(&gpu, &[&b.query, &b.key, &b.value])?,
                    out: lin(&gpu, &[&b.out])?,
                    query_norm: gpu.upload(&b.query_norm)?,
                    key_norm: gpu.upload(&b.key_norm)?,
                    cross_query: lin(&gpu, &[&b.cross_query])?,
                    cross_kv: lin(&gpu, &[&b.cross_key, &b.cross_value])?,
                    cross_out: lin(&gpu, &[&b.cross_out])?,
                    cross_query_norm: gpu.upload(&b.cross_query_norm)?,
                    cross_key_norm: gpu.upload(&b.cross_key_norm)?,
                    inverted: lin(&gpu, &[&b.inverted])?,
                    depth_weight: gpu.upload(&b.depth_weight)?,
                    depth_bias: gpu.upload(&b.depth_bias)?,
                    point: lin(&gpu, &[&b.point])?,
                })
            })
            .collect::<R<Vec<_>>>()?;
        dit.blocks.clear();
        let patch = lin(&gpu, &[&dit.patch])?;
        let proj_out = lin(&gpu, &[&dit.proj_out])?;

        let decoder = DcaeDecoder::load(model)?;
        let stages = decoder
            .stages
            .iter()
            .map(|stage| {
                stage
                    .iter()
                    .map(|block| {
                        Ok(match block {
                            Block::Up { conv: c, repeats } => Stage::Up {
                                conv: conv(&gpu, c)?,
                                repeats: *repeats,
                            },
                            Block::Res { conv1, conv2, norm } => Stage::Res {
                                conv1: conv(&gpu, conv1)?,
                                conv2: conv(&gpu, conv2)?,
                                norm: affine(&gpu, norm)?,
                            },
                            Block::Vit(v) => Stage::Vit(Box::new(Vit {
                                qkv: lin(&gpu, &[&v.query, &v.key, &v.value])?,
                                multiscale_in: gpu.upload(&v.multiscale_in)?,
                                multiscale_out: gpu.upload(&v.multiscale_out)?,
                                out: lin(&gpu, &[&v.out])?,
                                attention_norm: affine(&gpu, &v.attention_norm)?,
                                inverted: lin(&gpu, &[&v.inverted])?,
                                depth_weight: gpu.upload(&v.depth_weight)?,
                                depth_bias: gpu.upload(&v.depth_bias)?,
                                point: lin(&gpu, &[&v.point])?,
                                norm: affine(&gpu, &v.norm)?,
                            })),
                        })
                    })
                    .collect::<R<Vec<_>>>()
            })
            .collect::<R<Vec<_>>>()?;
        Ok(Self {
            context: context.clone(),
            k,
            zeros: gpu.zeros(4 * GEMMA_HIDDEN)?,
            encoder,
            embed,
            gemma,
            gemma_norm,
            dit,
            patch,
            proj_out,
            blocks,
            conv_in: conv(&gpu, &decoder.conv_in)?,
            stages,
            norm_out: affine(&gpu, &decoder.norm_out)?,
            conv_out: conv(&gpu, &decoder.conv_out)?,
        })
    }

    fn gpu(&self) -> Gpu<'_> {
        Gpu {
            context: &self.context,
        }
    }

    /// `out[rows, units] = x . lin^T + beta * out`, rows `out_stride`
    /// apart, with `x` contiguous FP32. `x` goes in as two BF16 operands, its
    /// value and its rounding error, so it keeps about 16 bits and only the
    /// (already BF16) weights round: one BF16 operand puts Gemma at 2.2e-2
    /// and the DiT's step at 1.1e-2 against gates of 1e-2. The bias is the
    /// caller's.
    fn mm<T, O: DevicePtrMut<T>>(
        &self,
        lin: &Lin,
        x: &CudaView<'_, f32>,
        rows: usize,
        out: &mut O,
        out_stride: usize,
        beta: f32,
    ) -> R<()> {
        let gpu = self.gpu();
        let n = rows * lin.inner;
        let (mut hi, mut lo) = (gpu.act(n, true)?, gpu.act(n, true)?);
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.split)
                .arg(hi.destination())
                .arg(lo.destination())
                .arg(x)
                .arg(&(n as u64))
                .launch(cfg(n))
                .map_err(cuda_err("SANA operand split"))?;
        }
        for (part, beta) in [(&hi, beta), (&lo, 1.0)] {
            act_rhs_transposed(
                &self.context,
                &part.all(),
                lin.inner,
                &lin.w.all(),
                lin.inner,
                true,
                false,
                out,
                out_stride,
                rows,
                lin.units,
                lin.inner,
                1.0,
                beta,
            )?;
        }
        Ok(())
    }

    /// [`DeviceSana::mm`] into a fresh buffer.
    fn dense(&self, lin: &Lin, x: &CudaSlice<f32>, rows: usize) -> R<CudaSlice<f32>> {
        let mut out = self.gpu().uninit(rows * lin.units)?;
        self.mm(lin, &x.slice(..), rows, &mut out, lin.units, 0.0)?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn rms(
        &self,
        out: &mut CudaViewMut<'_, f32>,
        out_stride: usize,
        x: &CudaView<'_, f32>,
        x_stride: usize,
        weight: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        rows: usize,
        cols: usize,
        eps: f32,
        add: bool,
    ) -> R<()> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.rms)
                .arg(out)
                .arg(&(out_stride as i32))
                .arg(x)
                .arg(&(x_stride as i32))
                .arg(weight)
                .arg(bias.unwrap_or(&self.zeros))
                .arg(&(cols as i32))
                .arg(&eps)
                .arg(&i32::from(add))
                .launch(per_row(rows, cols))
                .map_err(cuda_err("SANA RMS norm"))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn rms_inplace(
        &self,
        x: &mut CudaViewMut<'_, f32>,
        stride: usize,
        weight: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        eps: f32,
        relu: bool,
    ) -> R<()> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.rms_inplace)
                .arg(x)
                .arg(&(stride as i32))
                .arg(weight)
                .arg(&(cols as i32))
                .arg(&eps)
                .arg(&i32::from(relu))
                .launch(per_row(rows, cols))
                .map_err(cuda_err("SANA in-place RMS norm"))?;
        }
        Ok(())
    }

    fn ln_mod(
        &self,
        x: &CudaSlice<f32>,
        shift: &CudaView<'_, f32>,
        scale: &CudaView<'_, f32>,
        rows: usize,
        cols: usize,
    ) -> R<CudaSlice<f32>> {
        let mut out = self.gpu().uninit(rows * cols)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.ln_mod)
                .arg(&mut out)
                .arg(x)
                .arg(shift)
                .arg(scale)
                .arg(&(cols as i32))
                .arg(&NORM_EPS)
                .launch(per_row(rows, cols))
                .map_err(cuda_err("SANA modulated layer norm"))?;
        }
        Ok(out)
    }

    fn bias_act(
        &self,
        x: &mut CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
        silu: bool,
    ) -> R<()> {
        let n = rows * cols;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.bias_act)
                .arg(x)
                .arg(bias)
                .arg(&(n as u64))
                .arg(&(cols as i32))
                .arg(&i32::from(silu))
                .launch(cfg(n))
                .map_err(cuda_err("SANA bias"))?;
        }
        Ok(())
    }

    fn gated_add(
        &self,
        hidden: &mut CudaSlice<f32>,
        update: &CudaSlice<f32>,
        bias: &CudaSlice<f32>,
        gate: Option<&CudaView<'_, f32>>,
        rows: usize,
        cols: usize,
    ) -> R<()> {
        let n = rows * cols;
        let ungated = bias.slice(..);
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.gated_add)
                .arg(hidden)
                .arg(update)
                .arg(bias)
                .arg(gate.unwrap_or(&ungated))
                .arg(&(n as u64))
                .arg(&(cols as i32))
                .arg(&i32::from(gate.is_some()))
                .launch(cfg(n))
                .map_err(cuda_err("SANA gated residual"))?;
        }
        Ok(())
    }

    fn glu(&self, x: &CudaSlice<f32>, rows: usize, half: usize, silu: bool) -> R<CudaSlice<f32>> {
        let n = rows * half;
        let mut out = self.gpu().uninit(n)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.glu)
                .arg(&mut out)
                .arg(x)
                .arg(&(n as u64))
                .arg(&(half as i32))
                .arg(&i32::from(silu))
                .launch(cfg(n))
                .map_err(cuda_err("SANA gated unit"))?;
        }
        Ok(out)
    }

    /// Linear attention over `heads` heads of `d`, whose query, key and value
    /// channel `e` sit at `x[t * stride + h * head_stride + {qb, kb, vb} + e]`.
    #[allow(clippy::too_many_arguments)]
    fn linear_attention(
        &self,
        x: &CudaSlice<f32>,
        n: usize,
        heads: usize,
        d: usize,
        stride: usize,
        head_stride: usize,
        [qb, kb, vb]: [usize; 3],
        relu: bool,
    ) -> R<CudaSlice<f32>> {
        let gpu = self.gpu();
        let mut kv = gpu.zeros(heads * (d + 1) * d)?;
        let chunk = 256;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.lin_kv)
                .arg(&mut kv)
                .arg(x)
                .arg(&(n as i32))
                .arg(&(d as i32))
                .arg(&(stride as i32))
                .arg(&(head_stride as i32))
                .arg(&(kb as i32))
                .arg(&(vb as i32))
                .arg(&i32::from(relu))
                .arg(&(chunk as i32))
                .launch(LaunchConfig {
                    grid_dim: ((heads * (d + 1)) as u32, n.div_ceil(chunk) as u32, 1),
                    block_dim: (d.next_multiple_of(32) as u32, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(cuda_err("SANA linear attention sums"))?;
        }
        let mut out = gpu.uninit(n * heads * d)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.lin_out)
                .arg(&mut out)
                .arg(x)
                .arg(&kv)
                .arg(&(n as i32))
                .arg(&(heads as i32))
                .arg(&(d as i32))
                .arg(&(stride as i32))
                .arg(&(head_stride as i32))
                .arg(&(qb as i32))
                .arg(&i32::from(relu))
                .launch(cfg(n * heads * d))
                .map_err(cuda_err("SANA linear attention"))?;
        }
        Ok(out)
    }

    /// A `k x k` depthwise convolution of `channels` channels.
    #[allow(clippy::too_many_arguments)]
    fn depthwise(
        &self,
        out: &mut CudaViewMut<'_, f32>,
        out_stride: usize,
        x: &CudaView<'_, f32>,
        x_stride: usize,
        weight: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        (height, width, channels): (usize, usize, usize),
        k: usize,
    ) -> R<()> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.dw)
                .arg(out)
                .arg(&(out_stride as i32))
                .arg(x)
                .arg(&(x_stride as i32))
                .arg(weight)
                .arg(bias.unwrap_or(weight))
                .arg(&(height as i32))
                .arg(&(width as i32))
                .arg(&(channels as i32))
                .arg(&(k as i32))
                .arg(&i32::from(bias.is_some()))
                .launch(cfg(height * width * channels))
                .map_err(cuda_err("SANA depthwise convolution"))?;
        }
        Ok(())
    }

    /// A 3x3 convolution, padding 1, of a `height x width` map (read through
    /// a 2x nearest upsampling of half that size when `up`), after `act` on
    /// its input (0 none, 1 SiLU, 2 ReLU). Bias included.
    fn conv(
        &self,
        conv: &Conv,
        x: &CudaSlice<f32>,
        height: usize,
        width: usize,
        up: bool,
        act: i32,
    ) -> R<CudaSlice<f32>> {
        let gpu = self.gpu();
        let (c, o) = (conv.inputs, conv.outputs);
        let padded = ((height + 2) * (width + 2) + 2) * c;
        let mut input = gpu.act(padded, true)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.pad)
                .arg(input.destination())
                .arg(x)
                .arg(&(height as i32))
                .arg(&(width as i32))
                .arg(&(c as i32))
                .arg(&i32::from(up))
                .arg(&act)
                .launch(cfg(padded))
                .map_err(cuda_err("SANA convolution padding"))?;
        }
        let rows = height * (width + 2);
        let mut wide = gpu.uninit(rows * o)?;
        for tap in 0..9 {
            let shift = (tap / 3) * (width + 2) + tap % 3;
            act_rhs_transposed(
                &self.context,
                &input.at(shift * c),
                c,
                &conv.w.at(tap * c),
                9 * c,
                true,
                false,
                &mut wide.slice_mut(..),
                o,
                rows,
                o,
                c,
                1.0,
                if tap == 0 { 0.0 } else { 1.0 },
            )?;
        }
        drop(input);
        let mut out = gpu.uninit(height * width * o)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.unpad)
                .arg(&mut out)
                .arg(&wide)
                .arg(&conv.b)
                .arg(&(height as i32))
                .arg(&(width as i32))
                .arg(&(o as i32))
                .launch(cfg(height * width * o))
                .map_err(cuda_err("SANA convolution bias"))?;
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn up_add(
        &self,
        out: &mut CudaSlice<f32>,
        source: &CudaSlice<f32>,
        (height, width, channels): (usize, usize, usize),
        source_channels: usize,
        repeats: usize,
        up: bool,
    ) -> R<()> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.up_add)
                .arg(out)
                .arg(source)
                .arg(&(height as i32))
                .arg(&(width as i32))
                .arg(&(channels as i32))
                .arg(&(source_channels as i32))
                .arg(&(repeats as i32))
                .arg(&i32::from(up))
                .launch(cfg(height * width * channels))
                .map_err(cuda_err("SANA shortcut"))?;
        }
        Ok(())
    }

    /// Gemma's final normed hidden states over `ids`, `[ids, 2304]`, as
    /// [`GemmaEncoder::hidden`] computes them.
    pub fn gemma_hidden(&self, ids: &[u32], mask: &[bool]) -> R<Matrix> {
        self.gemma_layers(ids, mask, self.gemma.len())
    }

    /// The residual stream after `layers` layers, or the final normed states
    /// after all of them.
    fn gemma_layers(&self, ids: &[u32], mask: &[bool], layers: usize) -> R<Matrix> {
        let gpu = self.gpu();
        let (len, d) = (ids.len(), GEMMA_HIDDEN);
        let (q_width, kv_width) = (
            GEMMA_HEADS * GEMMA_HEAD_DIM,
            GEMMA_KV_HEADS * GEMMA_HEAD_DIM,
        );
        let qkv_width = q_width + 2 * kv_width;
        let group = GEMMA_HEADS / GEMMA_KV_HEADS;
        let stride = len.next_multiple_of(8);
        let scale = (GEMMA_HEAD_DIM as f32).powf(-0.5);
        let mut rows = vec![0f32; len * d];
        let normalizer = (d as f32).sqrt();
        for (row, &id) in rows.chunks_mut(d).zip(ids) {
            let source = self
                .embed
                .get(id as usize * d..(id as usize + 1) * d)
                .ok_or_else(|| {
                    NetworkError::InvalidConfig(format!("token {id} is past Gemma's vocabulary"))
                })?;
            for (value, weight) in row.iter_mut().zip(source) {
                *value = weight.to_f32() * normalizer;
            }
        }
        let mut hidden = gpu.upload(&rows)?;
        let rotary = crate::text_encoder::Rotary::new(len, GEMMA_HEAD_DIM, GEMMA_THETA);
        let (cos, sin) = (gpu.upload(&rotary.cos)?, gpu.upload(&rotary.sin)?);
        let flags: Vec<i32> = mask.iter().map(|&real| i32::from(real)).collect();
        let flags = gpu.upload_flags(&flags)?;
        let mut x = gpu.uninit(len * d)?;
        let mut out = gpu.uninit(len * d)?;
        for layer in &self.gemma[..layers] {
            self.rms(
                &mut x.slice_mut(..),
                d,
                &hidden.slice(..),
                d,
                &layer.input_norm,
                None,
                len,
                d,
                GEMMA_EPS,
                false,
            )?;
            let mut qkv = gpu.act(len * qkv_width, false)?;
            self.mm(
                &layer.qkv,
                &x.slice(..),
                len,
                qkv.destination(),
                qkv_width,
                0.0,
            )?;
            for (heads, offset) in [(GEMMA_HEADS, 0), (GEMMA_KV_HEADS, q_width)] {
                unsafe {
                    self.context
                        .stream
                        .launch_builder(&self.k.rope)
                        .arg(qkv.destination())
                        .arg(&cos)
                        .arg(&sin)
                        .arg(&(len as i32))
                        .arg(&(heads as i32))
                        .arg(&(GEMMA_HEAD_DIM as i32))
                        .arg(&(qkv_width as i32))
                        .arg(&(offset as i32))
                        .launch(cfg(len * heads * GEMMA_HEAD_DIM / 2))
                        .map_err(cuda_err("SANA rotary embedding"))?;
                }
            }
            let mut scores = gpu.uninit(GEMMA_HEADS * len * stride)?;
            for head in 0..GEMMA_HEADS {
                act_rhs_transposed(
                    &self.context,
                    &qkv.at(head * GEMMA_HEAD_DIM),
                    qkv_width,
                    &qkv.at(q_width + head / group * GEMMA_HEAD_DIM),
                    qkv_width,
                    false,
                    false,
                    &mut scores.slice_mut(head * len * stride..),
                    stride,
                    len,
                    len,
                    GEMMA_HEAD_DIM,
                    1.0,
                    0.0,
                )?;
            }
            let mut probs = gpu.act(GEMMA_HEADS * len * stride, false)?;
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.k.softmax_cap)
                    .arg(probs.destination())
                    .arg(&scores)
                    .arg(&(len as i32))
                    .arg(&(stride as i32))
                    .arg(&scale)
                    .arg(&GEMMA_SOFTCAP)
                    .arg(&flags)
                    .launch(per_row(GEMMA_HEADS * len, len))
                    .map_err(cuda_err("SANA capped softmax"))?;
            }
            drop(scores);
            let mut merged = gpu.uninit(len * q_width)?;
            for head in 0..GEMMA_HEADS {
                act_plain(
                    &self.context,
                    &probs.at(head * len * stride),
                    stride,
                    &qkv.at(q_width + kv_width + head / group * GEMMA_HEAD_DIM),
                    qkv_width,
                    false,
                    false,
                    &mut merged.slice_mut(head * GEMMA_HEAD_DIM..),
                    q_width,
                    len,
                    GEMMA_HEAD_DIM,
                    len,
                    1.0,
                    0.0,
                )?;
            }
            self.mm(&layer.output, &merged.slice(..), len, &mut out, d, 0.0)?;
            self.rms(
                &mut hidden.slice_mut(..),
                d,
                &out.slice(..),
                d,
                &layer.post_attention_norm,
                None,
                len,
                d,
                GEMMA_EPS,
                true,
            )?;

            self.rms(
                &mut x.slice_mut(..),
                d,
                &hidden.slice(..),
                d,
                &layer.pre_feedforward_norm,
                None,
                len,
                d,
                GEMMA_EPS,
                false,
            )?;
            let gate_up = self.dense(&layer.gate_up, &x, len)?;
            let gated = self.glu(&gate_up, len, layer.down.inner, false)?;
            drop(gate_up);
            self.mm(&layer.down, &gated.slice(..), len, &mut out, d, 0.0)?;
            self.rms(
                &mut hidden.slice_mut(..),
                d,
                &out.slice(..),
                d,
                &layer.post_feedforward_norm,
                None,
                len,
                d,
                GEMMA_EPS,
                true,
            )?;
        }
        if layers < self.gemma.len() {
            return Ok(Matrix::from_vec(len, d, gpu.download(&hidden)?));
        }
        self.rms(
            &mut x.slice_mut(..),
            d,
            &hidden.slice(..),
            d,
            &self.gemma_norm,
            None,
            len,
            d,
            GEMMA_EPS,
            false,
        )?;
        Ok(Matrix::from_vec(len, d, gpu.download(&x)?))
    }

    /// The DiT condition for a prompt: Gemma's selected states through the
    /// caption projection, real tokens only, on the device.
    pub fn context(&self, prompt: &str) -> R<(Matrix, Vec<bool>, Matrix)> {
        let (ids, mask) = self.encoder.prompt_ids(prompt)?;
        let hidden = self.gemma_hidden(&ids, &mask)?;
        let (embeds, mask) = select_prompt(&hidden, &mask);
        let context = self.dit.context(&embeds, &mask)?;
        Ok((embeds, mask, context))
    }

    /// One denoiser call, as [`SanaDit::forward`]: `latent` and the result
    /// are `[32, height * width]`, channel-major; `context` is
    /// [`SanaDit::context`]'s output.
    pub fn dit_forward(
        &self,
        latent: &Matrix,
        height: usize,
        width: usize,
        context: &Matrix,
        t: f32,
    ) -> R<Matrix> {
        let gpu = self.gpu();
        let (n, d, m) = (height * width, DIT_DIM, context.rows);
        let tokens = transpose(latent);
        let tokens = gpu.upload(&tokens.data)?;
        let mut hidden = self.dense(&self.patch, &tokens, n)?;
        self.bias_act(&mut hidden, &self.patch.b, n, d, false)?;
        let (temb, embedded) = self.dit.conditioning(t, GUIDANCE)?;
        let modulation: Vec<f32> = self
            .blocks
            .iter()
            .flat_map(|block| block.table.iter().zip(&temb).map(|(a, b)| a + b))
            .collect();
        let modulation = gpu.upload(&modulation)?;
        let prompt = gpu.upload(&context.data)?;
        let mut update = gpu.uninit(n * d)?;
        for (index, block) in self.blocks.iter().enumerate() {
            let part = |i: usize| modulation.slice((index * 6 + i) * d..(index * 6 + i + 1) * d);

            let x = self.ln_mod(&hidden, &part(0), &part(1), n, d)?;
            let mut qkv = self.dense(&block.qkv, &x, n)?;
            self.rms_inplace(
                &mut qkv.slice_mut(..),
                3 * d,
                &block.query_norm,
                n,
                d,
                QK_EPS,
                true,
            )?;
            self.rms_inplace(
                &mut qkv.slice_mut(d..),
                3 * d,
                &block.key_norm,
                n,
                d,
                QK_EPS,
                true,
            )?;
            let attended = self.linear_attention(
                &qkv,
                n,
                DIT_HEADS,
                DIT_HEAD_DIM,
                3 * d,
                DIT_HEAD_DIM,
                [0, d, 2 * d],
                false,
            )?;
            self.mm(&block.out, &attended.slice(..), n, &mut update, d, 0.0)?;
            self.gated_add(&mut hidden, &update, &block.out.b, Some(&part(2)), n, d)?;

            let mut query = self.dense(&block.cross_query, &hidden, n)?;
            self.bias_act(&mut query, &block.cross_query.b, n, d, false)?;
            self.rms_inplace(
                &mut query.slice_mut(..),
                d,
                &block.cross_query_norm,
                n,
                d,
                QK_EPS,
                false,
            )?;
            let mut kv = self.dense(&block.cross_kv, &prompt, m)?;
            self.bias_act(&mut kv, &block.cross_kv.b, m, 2 * d, false)?;
            self.rms_inplace(
                &mut kv.slice_mut(..),
                2 * d,
                &block.cross_key_norm,
                m,
                d,
                QK_EPS,
                false,
            )?;
            let mut attended = gpu.uninit(n * d)?;
            let scale = (CROSS_HEAD_DIM as f32).powf(-0.5);
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.k.cross_attn)
                    .arg(&mut attended)
                    .arg(&query)
                    .arg(&(d as i32))
                    .arg(&kv.slice(..))
                    .arg(&kv.slice(d..))
                    .arg(&(2 * d as i32))
                    .arg(&(n as i32))
                    .arg(&(m as i32))
                    .arg(&(CROSS_HEADS as i32))
                    .arg(&(CROSS_HEAD_DIM as i32))
                    .arg(&scale)
                    .launch(cfg(n * CROSS_HEADS))
                    .map_err(cuda_err("SANA cross-attention"))?;
            }
            self.mm(
                &block.cross_out,
                &attended.slice(..),
                n,
                &mut update,
                d,
                0.0,
            )?;
            self.gated_add(&mut hidden, &update, &block.cross_out.b, None, n, d)?;

            let x = self.ln_mod(&hidden, &part(3), &part(4), n, d)?;
            let wide = 2 * FF_HIDDEN;
            let mut inverted = self.dense(&block.inverted, &x, n)?;
            self.bias_act(&mut inverted, &block.inverted.b, n, wide, true)?;
            let mut depth = gpu.uninit(n * wide)?;
            self.depthwise(
                &mut depth.slice_mut(..),
                wide,
                &inverted.slice(..),
                wide,
                &block.depth_weight,
                Some(&block.depth_bias),
                (height, width, wide),
                3,
            )?;
            drop(inverted);
            let gated = self.glu(&depth, n, FF_HIDDEN, true)?;
            self.mm(&block.point, &gated.slice(..), n, &mut update, d, 0.0)?;
            self.gated_add(&mut hidden, &update, &block.point.b, Some(&part(5)), n, d)?;
        }
        let table = &self.dit.table;
        let shift: Vec<f32> = (0..d).map(|i| table[i] + embedded[i]).collect();
        let scale: Vec<f32> = (0..d).map(|i| table[d + i] + embedded[i]).collect();
        let (shift, scale) = (gpu.upload(&shift)?, gpu.upload(&scale)?);
        let x = self.ln_mod(&hidden, &shift.slice(..), &scale.slice(..), n, d)?;
        let mut out = self.dense(&self.proj_out, &x, n)?;
        self.bias_act(&mut out, &self.proj_out.b, n, LATENT_CHANNELS, false)?;
        Ok(transpose(&Matrix::from_vec(
            n,
            LATENT_CHANNELS,
            gpu.download(&out)?,
        )))
    }

    fn vit(&self, v: &Vit, hidden: &mut CudaSlice<f32>, height: usize, width: usize) -> R<()> {
        let gpu = self.gpu();
        let (n, c) = (height * width, v.out.units);
        // [q | k | v | their multiscale aggregate], 6C channels a token.
        let mut cat = gpu.uninit(n * 6 * c)?;
        self.mm(&v.qkv, &hidden.slice(..), n, &mut cat, 6 * c, 0.0)?;
        let mut spread = gpu.uninit(n * 3 * c)?;
        self.depthwise(
            &mut spread.slice_mut(..),
            3 * c,
            &cat.slice(..),
            6 * c,
            &v.multiscale_in,
            None,
            (height, width, 3 * c),
            5,
        )?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.k.grouped)
                .arg(&mut cat.slice_mut(3 * c..))
                .arg(&(6 * c as i32))
                .arg(&spread)
                .arg(&v.multiscale_out)
                .arg(&(n as u64))
                .arg(&(3 * c as i32))
                .arg(&(crate::dcae::HEAD_DIM as i32))
                .launch(cfg(n * 3 * c))
                .map_err(cuda_err("SANA grouped convolution"))?;
        }
        drop(spread);
        let head = crate::dcae::HEAD_DIM;
        let heads = 6 * c / (3 * head);
        let attended = self.linear_attention(
            &cat,
            n,
            heads,
            head,
            6 * c,
            3 * head,
            [0, head, 2 * head],
            true,
        )?;
        drop(cat);
        let mut update = gpu.uninit(n * c)?;
        self.mm(&v.out, &attended.slice(..), n, &mut update, c, 0.0)?;
        self.rms(
            &mut hidden.slice_mut(..),
            c,
            &update.slice(..),
            c,
            &v.attention_norm.weight,
            Some(&v.attention_norm.bias),
            n,
            c,
            1e-5,
            true,
        )?;

        let wide = v.inverted.units;
        let mut inverted = self.dense(&v.inverted, hidden, n)?;
        self.bias_act(&mut inverted, &v.inverted.b, n, wide, true)?;
        let mut depth = gpu.uninit(n * wide)?;
        self.depthwise(
            &mut depth.slice_mut(..),
            wide,
            &inverted.slice(..),
            wide,
            &v.depth_weight,
            Some(&v.depth_bias),
            (height, width, wide),
            3,
        )?;
        drop(inverted);
        let gated = self.glu(&depth, n, wide / 2, true)?;
        drop(depth);
        self.mm(&v.point, &gated.slice(..), n, &mut update, c, 0.0)?;
        self.rms(
            &mut hidden.slice_mut(..),
            c,
            &update.slice(..),
            c,
            &v.norm.weight,
            Some(&v.norm.bias),
            n,
            c,
            1e-5,
            true,
        )
    }

    /// [`DcaeDecoder::decode`] on the device, handing each stage's output to
    /// `probe` (`up_blocks.5` first) when one is given.
    pub fn decode(
        &self,
        latent: &FeatureMap,
        mut probe: Option<&mut dyn FnMut(usize, FeatureMap)>,
    ) -> R<FeatureMap> {
        if latent.channels != LATENT_CHANNELS {
            return Err(NetworkError::InvalidConfig(format!(
                "the DC-AE decodes 32 latent channels, not {}",
                latent.channels
            )));
        }
        let gpu = self.gpu();
        let (mut height, mut width) = (latent.height, latent.width);
        let tokens = transpose(&Matrix::from_vec(
            LATENT_CHANNELS,
            height * width,
            latent.data.clone(),
        ));
        let source = gpu.upload(&tokens.data)?;
        let mut hidden = self.conv(&self.conv_in, &source, height, width, false, 0)?;
        let channels = self.conv_in.outputs;
        self.up_add(
            &mut hidden,
            &source,
            (height, width, channels),
            LATENT_CHANNELS,
            channels / LATENT_CHANNELS,
            false,
        )?;
        let mut channels = channels;
        for (index, stage) in self.stages.iter().enumerate() {
            for block in stage {
                match block {
                    Stage::Up { conv, repeats } => {
                        height *= 2;
                        width *= 2;
                        let mut up = self.conv(conv, &hidden, height, width, true, 0)?;
                        self.up_add(
                            &mut up,
                            &hidden,
                            (height, width, conv.outputs),
                            channels,
                            *repeats,
                            true,
                        )?;
                        hidden = up;
                        channels = conv.outputs;
                    }
                    Stage::Res { conv1, conv2, norm } => {
                        let x = self.conv(conv1, &hidden, height, width, false, 0)?;
                        let y = self.conv(conv2, &x, height, width, false, 1)?;
                        drop(x);
                        let n = height * width;
                        self.rms(
                            &mut hidden.slice_mut(..),
                            channels,
                            &y.slice(..),
                            channels,
                            &norm.weight,
                            Some(&norm.bias),
                            n,
                            channels,
                            1e-5,
                            true,
                        )?;
                    }
                    Stage::Vit(v) => self.vit(v, &mut hidden, height, width)?,
                }
            }
            if let Some(probe) = probe.as_mut() {
                let tokens = Matrix::from_vec(height * width, channels, gpu.download(&hidden)?);
                probe(index, crate::dcae::channel_major(&tokens, height, width));
            }
        }
        let n = height * width;
        let mut normed = gpu.uninit(n * channels)?;
        self.rms(
            &mut normed.slice_mut(..),
            channels,
            &hidden.slice(..),
            channels,
            &self.norm_out.weight,
            Some(&self.norm_out.bias),
            n,
            channels,
            1e-5,
            false,
        )?;
        drop(hidden);
        let rgb = self.conv(&self.conv_out, &normed, height, width, false, 2)?;
        let rgb = Matrix::from_vec(n, self.conv_out.outputs, gpu.download(&rgb)?);
        Ok(crate::dcae::channel_major(&rgb, height, width))
    }

    /// One image in `[-1, 1]`, `[3, height, width]`, as
    /// [`crate::sana::SanaSprint::generate`] makes it.
    pub fn generate(
        &self,
        prompt: &str,
        width: usize,
        height: usize,
        steps: usize,
        seed: Option<u64>,
    ) -> R<FeatureMap> {
        check_size(width, height, steps)?;
        let (_, _, context) = self.context(prompt)?;
        let (h, w) = (height / DOWNSCALE, width / DOWNSCALE);
        let start = crate::diffusion::noise(LATENT_CHANNELS, h * w, seed);
        let latent = sample(
            &start,
            steps,
            |x, t| self.dit_forward(x, h, w, &context, t),
            |i| {
                crate::diffusion::noise(
                    LATENT_CHANNELS,
                    h * w,
                    seed.map(|s| s.wrapping_add(1 + i as u64)),
                )
            },
        )?;
        self.decode(&latent_map(&latent, h, w), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sana::tests::{gate, reference, tensor};
    use crate::sana::{MAX_TIME, timesteps};

    fn device() -> Option<(DeviceSana, crate::safetensors::SafeTensors, String)> {
        if std::env::var("RB_SANA_CUDA").ok().as_deref() != Some("1") {
            return None;
        }
        let (reference, model) = reference()?;
        let dir = std::path::PathBuf::from(std::env::var("RB_SANA_REF").ok()?);
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("reference.json")).ok()?)
                .ok()?;
        let prompt = json["prompt"].as_str()?.to_string();
        let context = GpuContext::new(0).expect("a CUDA device");
        let start = std::time::Instant::now();
        let sana = DeviceSana::load(&context, &model).expect("the SANA-Sprint weights");
        println!("device load: {:.1} s", start.elapsed().as_secs_f32());
        Some((sana, reference, prompt))
    }

    #[test]
    #[ignore = "needs RB_SANA_REF, RB_SANA_CUDA=1 and a CUDA device"]
    fn sana_device_matches_diffusers() {
        let Some((sana, mut reference, prompt)) = device() else {
            panic!("set RB_SANA_REF and RB_SANA_CUDA=1");
        };
        // Gemma, from the prompt.
        let (ids, mask) = sana.encoder.prompt_ids(&prompt).unwrap();
        let first = sana.gemma_layers(&ids, &mask, 1).unwrap();
        gate(
            "device gemma layer 0",
            &first.data,
            &tensor(&mut reference, "gemma_layer0"),
            1e-2,
        );
        let hidden = sana.gemma_hidden(&ids, &mask).unwrap();
        gate(
            "device gemma hidden",
            &hidden.data,
            &tensor(&mut reference, "gemma_hidden"),
            1e-2,
        );
        let (embeds, own_mask) = select_prompt(&hidden, &mask);
        gate(
            "device prompt embeds",
            &embeds.data,
            &tensor(&mut reference, "prompt_embeds"),
            1e-2,
        );
        let own_context = sana.dit.context(&embeds, &own_mask).unwrap();

        // The DiT and the sampler, from the reference's embeds and noise.
        let want_embeds = tensor(&mut reference, "prompt_embeds");
        let want_mask: Vec<bool> = tensor(&mut reference, "prompt_mask")
            .iter()
            .map(|v| *v != 0.0)
            .collect();
        let context = sana
            .dit
            .context(
                &Matrix::from_vec(want_mask.len(), GEMMA_HIDDEN, want_embeds),
                &want_mask,
            )
            .unwrap();
        let grid = 32;
        let n = grid * grid;
        let input = Matrix::from_vec(LATENT_CHANNELS, n, tensor(&mut reference, "step0_input"));
        let s = MAX_TIME;
        let scm = s.sin() / (s.cos() + s.sin());
        let raw = sana.dit_forward(&input, grid, grid, &context, scm).unwrap();
        gate(
            "device dit step 0",
            &raw.data,
            &tensor(&mut reference, "step0_raw"),
            1e-2,
        );
        let start = Matrix::from_vec(LATENT_CHANNELS, n, tensor(&mut reference, "noise"));
        let draws: Vec<Matrix> = (0..timesteps(2).len() - 1)
            .map(|i| {
                Matrix::from_vec(
                    LATENT_CHANNELS,
                    n,
                    tensor(&mut reference, &format!("step{i}_noise")),
                )
            })
            .collect();
        let latent = sample(
            &start,
            2,
            |x, t| sana.dit_forward(x, grid, grid, &context, t),
            |i| draws[i].clone(),
        )
        .unwrap();
        gate(
            "device final latent",
            &latent.data,
            &tensor(&mut reference, "final_latents"),
            3e-2,
        );

        // The DC-AE, from the reference's latent.
        let vae_input = FeatureMap {
            channels: LATENT_CHANNELS,
            height: grid,
            width: grid,
            data: tensor(&mut reference, "vae_input"),
        };
        let mut stages = Vec::new();
        let mut keep = |index: usize, map: FeatureMap| {
            if index < 3 {
                stages.push(map)
            }
        };
        let image = sana.decode(&vae_input, Some(&mut keep)).unwrap();
        for (index, map) in stages.iter().enumerate() {
            let name = format!("vae_up{}", 5 - index);
            gate(
                &format!("device {name}"),
                &map.data,
                &tensor(&mut reference, &name),
                1e-2,
            );
        }
        let want_image = tensor(&mut reference, "image");
        gate("device image", &image.data, &want_image, 3e-2);

        // Chained: own embeds, own latent, own image, the reference's noise.
        let latent = sample(
            &start,
            2,
            |x, t| sana.dit_forward(x, grid, grid, &own_context, t),
            |i| draws[i].clone(),
        )
        .unwrap();
        gate(
            "device chained final latent",
            &latent.data,
            &tensor(&mut reference, "final_latents"),
            3e-2,
        );
        let image = sana.decode(&latent_map(&latent, grid, grid), None).unwrap();
        gate("device chained image", &image.data, &want_image, 3e-2);
    }

    #[cfg(feature = "images")]
    #[test]
    #[ignore = "needs RB_SANA_REF, RB_SANA_CUDA=1 and a CUDA device"]
    fn sana_device_timing() {
        let Some((sana, _, prompt)) = device() else {
            panic!("set RB_SANA_REF and RB_SANA_CUDA=1");
        };
        let run = |prompt: &str| {
            let start = std::time::Instant::now();
            let image = sana.generate(prompt, 1024, 1024, 2, Some(7)).unwrap();
            sana.context.synchronize().unwrap();
            (start.elapsed().as_secs_f32(), image)
        };
        let (warm, _) = run(&prompt);
        println!("first job: {warm:.3} s");
        let times: Vec<f32> = (0..3).map(|_| run(&prompt).0).collect();
        let mean = times.iter().sum::<f32>() / times.len() as f32;
        let spread = times.iter().cloned().fold(f32::MIN, f32::max)
            - times.iter().cloned().fold(f32::MAX, f32::min);
        println!("warm 1024^2 2-step jobs: {times:?} s, mean {mean:.3} s, spread {spread:.3} s");

        if let Ok(dir) = std::env::var("RB_SANA_SHOWCASE") {
            std::fs::create_dir_all(&dir).unwrap();
            let prompts = [
                (
                    "01_portrait",
                    "Close-up portrait of an elderly fisherman with a weathered face and grey beard, soft window light, 85mm photo",
                ),
                (
                    "02_city_night",
                    "A rainy neon-lit street in Tokyo at night, reflections on wet asphalt, people with umbrellas, cinematic",
                ),
                (
                    "03_food",
                    "A bowl of ramen with a soft boiled egg, chashu pork and spring onions on a wooden table, food photography",
                ),
                (
                    "04_text_sign",
                    "A vintage wooden shop sign that says OPEN DAILY in painted white letters, brick wall background",
                ),
                (
                    "05_fox_snow",
                    "A red fox walking through fresh snow in a pine forest at golden hour, wildlife photo, shallow depth of field",
                ),
                (
                    "06_product",
                    "A studio product photo of a matte black wireless headphone on a white background, soft shadows",
                ),
                (
                    "07_anime",
                    "Anime style illustration of a girl with a red scarf standing on a hill under a starry sky",
                ),
                (
                    "08_hands",
                    "Two hands holding a steaming cup of coffee, close-up, natural light, detailed fingers",
                ),
                (
                    "09_architecture",
                    "A modern glass house on a cliff above the ocean at sunset, architectural photography",
                ),
                (
                    "10_group",
                    "Three friends laughing at a cafe table outdoors, candid street photo, summer afternoon",
                ),
            ];
            for (name, prompt) in prompts {
                let (seconds, image) = run(prompt);
                crate::vae::save_png(&image, format!("{dir}/{name}.png")).unwrap();
                println!("{name}: {seconds:.3} s");
            }
        }
    }
}
