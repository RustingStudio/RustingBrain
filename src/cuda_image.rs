//! The device path for image generation: UNet, VAE decoder and CLIP tower.
//!
//! `gpu_model` is the same idea for a language model — activations
//! stay in device buffers for a whole pass and the weights are uploaded once —
//! but everything it fuses is transformer-shaped. An image model spends its
//! time somewhere else: convolutions, group norm over pixels, nearest-neighbour
//! upsampling and a *bidirectional* attention. So this is a second set of
//! kernels over the same [`GpuContext`], reusing its stream, its cuBLAS handle
//! and the `gemm_ex` wrappers that already know how to feed the tensor cores
//! narrow operands.
//!
//! Three decisions are worth stating, because they are what makes a 2.6B
//! parameter UNet fit in 12 GB:
//!
//! * **Everything on the device is FP16.** An `f32` SDXL UNet is 10.4 GB of
//!   weights alone, which does not fit; at two bytes it is 5.2 GB, which does.
//!   FP16 costs about the same relative error per weight (2^-11) as the
//!   [`Precision::Q8`](crate::transformer::Precision) path the CPU already
//!   runs. The GEMMs accumulate in FP16 too, because a consumer Ampere card
//!   halves its tensor-core rate for an FP32 accumulator and FP16's wider
//!   mantissa pays the accumulator's error back; the reduction kernels keep
//!   their FP32 running sums, where the accumulation is long and the operands
//!   are few.
//!
//!   NOTE: FP16 tops out at 65504, and the Stable Diffusion XL decoder is
//!   the one part of a published model known to reach that on some latents.
//!   Nothing seen here has. If one does, `DeviceVae::decode` returns an error
//!   instead of the black image the NaNs would become, and the upgrade path is
//!   a second
//!   module compiled from this same source with the two conversion helpers
//!   written for BF16, handed to the decoder alone.
//! * **`Precision::Q8` is dequantized on upload**, not in the kernel. Keeping
//!   the bytes and unpacking per GEMM would halve the weight footprint again,
//!   but it means a custom kernel for every matmul instead of cuBLAS, and the
//!   one byte per weight was only ever bought to make the model fit in *host*
//!   memory. On the device two bytes already fit.
//! * **im2col happens here**, one chunk of output pixels at a time. The column
//!   matrix for a 1024x1024 VAE layer is 2.4 GB if materialized whole, which is
//!   more than the weights; chunking bounds it at `COLUMN_BUDGET` and costs
//!   one extra GEMM launch per chunk. A stride-1 convolution expands only
//!   its horizontal taps (`im2row`) and runs one accumulating GEMM per kernel
//!   row, which writes a third of the scratch a 3x3 im2col would.
//!
//! The host keeps its own copy of every weight. That is what lets the parity
//! tests below run both paths against one another, and it is what a machine
//! with no device falls back to.
//!
//! NOTE: forward only, one stream, no graph capture, and the 1x1
//! convolutions are the only ones that skip im2col. Each of those is a
//! measurable-if-profiled improvement rather than a correctness gap.

use crate::clip::{ClipTextEncoder, Layer, Norm};
use crate::conv::{Conv2d, Dense, FeatureMap, GroupNorm};
use crate::cuda_flash::compile_ptx;
use crate::cuda_training::{cfg, cuda_alloc_err, cuda_err, device_context};
use crate::gpu_transformer::{GpuContext, act_lhs_transposed, act_plain, act_rhs_transposed};
use crate::matrix::Matrix;
use crate::network::NetworkError;
use crate::unet::{Projection, Unet};
use crate::vae::{VaeDecoder, VaeEncoder};
use crate::vit_encoder::VitEncoder;
use cudarc::cublas::{result as cublas, sys as cublas_sys, sys::cublasOperation_t};
use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaView, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use half::f16;
use lora::{AttentionRecord, BlockRecord, DeviceLora, Recorder};
use std::sync::{Arc, OnceLock};

/// The largest im2col scratch buffer, in FP16 elements: 32 MiB.
///
/// A convolution is split into as many chunks of output pixels as it takes to
/// stay under this. One chunk is one extra GEMM launch, and a GEMM at this size
/// is already far past the point where launch overhead matters.
const COLUMN_BUDGET: usize = 1 << 24;

/// The largest attention score scratch, in FP16 elements: 32 MiB.
///
/// The VAE's self-attention at 1024x1024 is 16384 queries over 16384 keys,
/// which is half a gigabyte of scores if they all exist at once. They do not
/// need to: the softmax is per query row, so the queries split into chunks.
const SCORE_BUDGET: usize = 1 << 24;

/// Threads per block for every reduction kernel below. Also the size of the
/// shared scratch they reduce through, so the two have to agree.
const THREADS: u32 = 256;

const IMAGE_KERNELS: &str = r#"
// FP16 helpers. NVRTC compiles this string with no header search path, so
// `cuda_fp16.h` is out of reach and the two conversions are written as the PTX
// instructions that header would have inlined anyway.
//
// The image path stores every activation and weight as FP16 rather than BF16
// because that is what lets the GEMMs accumulate in FP16 as well, and a
// consumer Ampere card runs its tensor cores at twice the rate when the
// accumulator is narrow. FP16's eleven mantissa bits buy back what the narrow
// accumulator loses: measured against an FP32 reference, an FP16 product with
// an FP16 accumulator is as accurate as a BF16 product with an FP32 one.
typedef unsigned short half_t;
__device__ __forceinline__ float h2f(half_t h){
  float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ half_t f2h(float v){
  half_t h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v)); return h;
}

#define THREADS 256
// How many elements one thread of a streaming kernel touches. At two bytes an
// element, a warp that loads once has sixty-four bytes in flight and the card
// sits at a sixth of its bandwidth waiting on latency; four independent loads
// per thread is what fills the pipe. The stride is the block, so every one of
// the four is still a coalesced access.
#define LANE 4
__device__ __forceinline__ float block_sum(float v,float*red){
  int t=threadIdx.x;red[t]=v;__syncthreads();
  for(int s=THREADS/2;s>0;s>>=1){if(t<s)red[t]+=red[t+s];__syncthreads();}
  float r=red[0];__syncthreads();return r;
}
__device__ __forceinline__ float block_max(float v,float*red){
  int t=threadIdx.x;red[t]=v;__syncthreads();
  for(int s=THREADS/2;s>0;s>>=1){if(t<s)red[t]=fmaxf(red[t],red[t+s]);__syncthreads();}
  float r=red[0];__syncthreads();return r;
}

// The activation a layer asks for by number: 0 SiLU, 1 GELU (the tanh
// approximation `ffn::gelu` uses), 2 the quick GELU the CLIP towers were
// trained with, 3 the exact GELU DINOv2 uses.
__device__ __forceinline__ float act_of(float x,int kind){
  if(kind==0)return x/(1.f+expf(-x));
  if(kind==2)return x/(1.f+expf(-1.702f*x));
  if(kind==3)return 0.5f*x*(1.f+erff(x*0.70710678f));
  float inner=0.7978846f*(x+0.044715f*x*x*x);
  return 0.5f*x*(1.f+tanhf(inner));
}

// One column of the lowered matrix per output pixel and one row per tap,
// which is the transpose of the layout `Conv2d::im2col` builds on the host.
// Two things fall out of the transpose. Consecutive threads now read
// consecutive input pixels instead of striding across kernel rows, so the
// gather is coalesced; and the GEMM that follows can compute
// `weight . columns` straight into the channel-major output, so the separate
// transposing epilogue this used to need is gone.
//
// One thread writes a whole patch column for one input channel — all nine
// taps of a 3x3 — rather than one tap. The stores are the same stores either
// way, but a thread that writes one FP16 and exits spends most of its life on
// the index arithmetic that found it, and nine independent stores in flight
// per thread is what keeps the memory pipe busy. It is also nine times fewer
// blocks, and the one division left, from the flat pixel index to a row and a
// column, is now paid once per nine values instead of once per value.
extern "C" __global__ void im2col(half_t*__restrict__ columns,const half_t*__restrict__ input,
    int in_h,int in_w,int kernel,int stride,int pad,int out_w,int row,int span,long long pitch){
  int s=blockIdx.x*blockDim.x+threadIdx.x;
  if(s>=span)return;
  int line=s/out_w,ox=s-line*out_w;
  int c=blockIdx.y;
  const half_t*source=input+(long long)c*in_h*in_w;
  half_t*target=columns+(long long)c*kernel*kernel*pitch+s;
  int y0=(row+line)*stride-pad,x0=ox*stride-pad;
  for(int ky=0;ky<kernel;ky++){
    int y=y0+ky;
    int inside=(y>=0&&y<in_h);
    for(int kx=0;kx<kernel;kx++){
      int x=x0+kx;
      float v=0.f;
      if(inside&&x>=0&&x<in_w)v=h2f(source[(long long)y*in_w+x]);
      target[(long long)(ky*kernel+kx)*pitch]=f2h(v);
    }
  }
}

// The stride-one gather, which expands only the horizontal taps. Line `l` of
// the buffer is input row `row - pad + l`, so the vertical tap `ky` of output
// line `l` is line `l + ky`: a whole-row offset into the same buffer. The
// convolution is then one accumulating GEMM per vertical tap, and the buffer
// holds `kernel` copies of the input instead of `kernel * kernel`.
extern "C" __global__ void im2row(half_t*__restrict__ columns,const half_t*__restrict__ input,
    int in_h,int in_w,int kernel,int pad,int out_w,int row,int span,long long pitch){
  int s=blockIdx.x*blockDim.x+threadIdx.x;
  if(s>=span)return;
  int line=s/out_w,ox=s-line*out_w;
  int c=blockIdx.y;
  int y=row-pad+line,x0=ox-pad;
  const half_t*source=input+((long long)c*in_h+y)*in_w;
  half_t*target=columns+(long long)c*kernel*pitch+s;
  int inside=(y>=0&&y<in_h);
  for(int kx=0;kx<kernel;kx++){
    int x=x0+kx;
    float v=0.f;
    if(inside&&x>=0&&x<in_w)v=h2f(source[x]);
    target[(long long)kx*pitch]=f2h(v);
  }
}

// The row rides in the grid, so the column index is a thread index rather
// than a 64-bit remainder per element.
extern "C" __global__ void add_bias_rows(half_t*__restrict__ x,const float*__restrict__ b,int cols){
  int c=blockIdx.x*blockDim.x+threadIdx.x;
  if(c>=cols)return;
  long long i=(long long)blockIdx.y*cols+c;
  x[i]=f2h(h2f(x[i])+b[c]);
}
// One value per channel, spread over that channel's plane: a convolution's
// bias, or the noise level a residual block carries in. The channel is the
// grid's second dimension, for the reason `im2col` takes its row that way.
extern "C" __global__ void add_channel(half_t*__restrict__ x,const half_t*__restrict__ v,
    long long pixels){
  half_t*row=x+(long long)blockIdx.y*pixels;
  float offset=h2f(v[blockIdx.y]);
  long long base=(long long)blockIdx.x*blockDim.x*LANE+threadIdx.x;
  for(int k=0;k<LANE;k++){
    long long p=base+(long long)k*blockDim.x;
    if(p<pixels)row[p]=f2h(h2f(row[p])+offset);
  }
}
// A convolution's bias, which stays f32.
extern "C" __global__ void add_plane_bias(half_t*__restrict__ x,const float*__restrict__ b,
    long long pixels){
  half_t*row=x+(long long)blockIdx.y*pixels;
  float bias=b[blockIdx.y];
  long long base=(long long)blockIdx.x*blockDim.x*LANE+threadIdx.x;
  for(int k=0;k<LANE;k++){
    long long p=base+(long long)k*blockDim.x;
    if(p<pixels)row[p]=f2h(h2f(row[p])+bias);
  }
}
extern "C" __global__ void add_inplace(half_t*__restrict__ x,const half_t*__restrict__ y,
    long long n){
  long long base=(long long)blockIdx.x*blockDim.x*LANE+threadIdx.x;
  for(int k=0;k<LANE;k++){
    long long i=base+(long long)k*blockDim.x;
    if(i<n)x[i]=f2h(h2f(x[i])+h2f(y[i]));
  }
}
extern "C" __global__ void activate(half_t*__restrict__ x,long long n,int kind){
  long long base=(long long)blockIdx.x*blockDim.x*LANE+threadIdx.x;
  for(int k=0;k<LANE;k++){
    long long i=base+(long long)k*blockDim.x;
    if(i<n)x[i]=f2h(act_of(h2f(x[i]),kind));
  }
}

// Group normalization in three launches instead of one.
//
// The obvious kernel is one block per group, which is what this was: the
// group's channels are contiguous and so are their pixels, so a group is one
// slice and a block can walk it twice. It is also thirty two blocks on a card
// with twenty eight multiprocessors, which reaches about a tenth of the
// available bandwidth and, at 1024x1024, cost more than the convolutions it
// sits between. So the group is split along its own span, every slice gets a
// block, and a second launch folds the slices together.
extern "C" __global__ void group_partials(const half_t*__restrict__ x,float*__restrict__ out,
    long long span,int splits){
  __shared__ float red[THREADS];
  long long g=blockIdx.y,s=blockIdx.x;
  const half_t*v=x+g*span;
  long long begin=span*s/splits,end=span*(s+1)/splits;
  float sum=0.f,square=0.f;
  for(long long i=begin+threadIdx.x;i<end;i+=THREADS){float t=h2f(v[i]);sum+=t;square+=t*t;}
  float a=block_sum(sum,red),b=block_sum(square,red);
  if(threadIdx.x==0){out[g*splits+s]=a;out[((long long)gridDim.y+g)*splits+s]=b;}
}

// One thread per group. The variance comes out of the sum and the sum of
// squares, which cancel in FP32 whenever the mean dominates the spread, so the
// fold runs in double; a few hundred double operations on a card that runs
// double at a thirty-second rate is still nothing.
extern "C" __global__ void group_fold(const float*__restrict__ partials,float*__restrict__ stats,
    int groups,int splits,double span,float eps){
  int g=blockIdx.x*blockDim.x+threadIdx.x;
  if(g>=groups)return;
  double sum=0.0,square=0.0;
  for(int s=0;s<splits;s++){
    sum+=(double)partials[g*splits+s];
    square+=(double)partials[(groups+g)*splits+s];
  }
  double mean=sum/span,var=square/span-mean*mean;
  if(var<0.0)var=0.0;
  stats[2*g]=(float)mean;
  stats[2*g+1]=(float)(1.0/sqrt(var+(double)eps));
}

// The scale, the offset and — because a group norm in this model is followed
// by an activation nearly every time — the activation, in one pass. `act` is
// negative when the caller wants the normalization on its own. The channel
// rides in the grid, so there is no division per element.
extern "C" __global__ void group_apply(half_t*__restrict__ x,const float*__restrict__ stats,
    const float*__restrict__ w,const float*__restrict__ b,
    long long pixels,int per_group,int act){
  int c=blockIdx.y,g=c/per_group;
  half_t*row=x+(long long)c*pixels;
  float mean=stats[2*g],inv=stats[2*g+1]*w[c],shift=b[c];
  long long base=(long long)blockIdx.x*blockDim.x*LANE+threadIdx.x;
  for(int k=0;k<LANE;k++){
    long long p=base+(long long)k*blockDim.x;
    if(p>=pixels)continue;
    float v=(h2f(row[p])-mean)*inv+shift;
    row[p]=f2h(act>=0?act_of(v,act):v);
  }
}

// One block per token. The learned scale and offset are the ones CLIP trains
// and the UNet's transformer blocks reuse.
extern "C" __global__ void layer_norm(half_t*out,const half_t*x,const float*w,const float*b,
    int cols,float eps){
  __shared__ float red[THREADS];
  const half_t*src=x+(long long)blockIdx.x*cols;
  half_t*dst=out+(long long)blockIdx.x*cols;
  float total=0.f;
  for(int i=threadIdx.x;i<cols;i+=THREADS)total+=h2f(src[i]);
  float mean=block_sum(total,red)/(float)cols;
  float square=0.f;
  for(int i=threadIdx.x;i<cols;i+=THREADS){float d=h2f(src[i])-mean;square+=d*d;}
  float inv=rsqrtf(block_sum(square,red)/(float)cols+eps);
  for(int i=threadIdx.x;i<cols;i+=THREADS)dst[i]=f2h((h2f(src[i])-mean)*inv*w[i]+b[i]);
}

// One block per query row of the batched score matrix, which is laid out
// [head][query][key]. `causal` is for the CLIP tower; the UNet's own attention
// and the VAE's are bidirectional, so every key is visible. Masked entries are
// written as zero, which is what the host path leaves them at.
// One pass to find the row's largest score and its exponent sum together, a
// second to turn the scores into probabilities where they lie. The old kernel
// made three passes over an FP32 score matrix and wrote the exponentials back
// between them, which is eighteen bytes of traffic per score against six.
// Every score is touched by exactly one thread at one index, so the rewrite is
// in place; nothing is read after it has been written.
//
// Holding the scores in FP16 costs about 0.05% on each one, which becomes the
// same on each probability, and the merge that follows averages a few hundred
// of them against values that are themselves FP16. It is the same trade the
// rest of this module already makes.
// One warp to a score row. A block-wide softmax over a thousand-wide row
// gives each of its two hundred and fifty-six threads four values and then
// spends sixteen barriers reducing them; a warp reduces in registers with no
// barrier at all, and eight rows share the block. Rows must be whole quads
// and nothing may be masked, which is every attention an image model runs.
#define WARPS (THREADS/32)
__device__ __forceinline__ float warp_max(float v){
  for(int s=16;s;s>>=1)v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,s));
  return v;
}
__device__ __forceinline__ float warp_sum(float v){
  for(int s=16;s;s>>=1)v+=__shfl_xor_sync(0xffffffff,v,s);
  return v;
}
extern "C" __global__ void softmax_warp(half_t*__restrict__ scores,
    int context,int rows){
  int row=blockIdx.x*WARPS+(threadIdx.x>>5);
  if(row>=rows)return;
  int lane=threadIdx.x&31,quads=context>>2;
  half_t*s=scores+(long long)row*context;
  float largest=__int_as_float(0xff800000),total=0.f;
  if(context&3){
    // A cross-attention row is as short as the prompt and never a whole
    // number of quads. It still wants a warp rather than a block.
    for(int i=lane;i<context;i+=32){
      float x=h2f(s[i]);
      if(x>largest){total*=__expf(largest-x);largest=x;}
      total+=__expf(x-largest);
    }
    float top=warp_max(largest);
    float scale=1.f/warp_sum(total*__expf(largest-top));
    for(int i=lane;i<context;i+=32)s[i]=f2h(__expf(h2f(s[i])-top)*scale);
    return;
  }
  ushort4*v=(ushort4*)s;
  for(int i=lane;i<quads;i+=32){
    ushort4 q=v[i];
    float a=h2f(q.x),b=h2f(q.y),c=h2f(q.z),d=h2f(q.w);
    float top=fmaxf(fmaxf(a,b),fmaxf(c,d));
    if(top>largest){total*=__expf(largest-top);largest=top;}
    total+=__expf(a-largest)+__expf(b-largest)+__expf(c-largest)+__expf(d-largest);
  }
  float peak=warp_max(largest);
  float inv=1.f/warp_sum(total*__expf(largest-peak));
  for(int i=lane;i<quads;i+=32){
    ushort4 q=v[i];
    q.x=f2h(__expf(h2f(q.x)-peak)*inv);
    q.y=f2h(__expf(h2f(q.y)-peak)*inv);
    q.z=f2h(__expf(h2f(q.z)-peak)*inv);
    q.w=f2h(__expf(h2f(q.w)-peak)*inv);
    v[i]=q;
  }
}

// The wide case, and the general one. A row long enough that a warp would
// walk it serially gets a whole block instead; `packed` says the row is a
// whole number of quads with nothing masked, so four scores move per load.
extern "C" __global__ void softmax_rows(half_t*__restrict__ scores,
    int context,int rows,int causal,int first,int packed){
  __shared__ float red[THREADS];
  half_t*s=scores+(long long)blockIdx.x*context;
  int query=(int)(blockIdx.x%rows)+first;
  int visible=causal?(query+1<context?query+1:context):context;
  float largest=__int_as_float(0xff800000),total=0.f;
  if(packed){
    ushort4*v=(ushort4*)s;
    int quads=context>>2;
    for(int i=threadIdx.x;i<quads;i+=THREADS){
      ushort4 q=v[i];
      float a=h2f(q.x),b=h2f(q.y),c=h2f(q.z),d=h2f(q.w);
      float top=fmaxf(fmaxf(a,b),fmaxf(c,d));
      if(top>largest){total*=__expf(largest-top);largest=top;}
      total+=__expf(a-largest)+__expf(b-largest)+__expf(c-largest)+__expf(d-largest);
    }
    float peak=block_max(largest,red);
    total*=__expf(largest-peak);
    float inv=1.f/block_sum(total,red);
    for(int i=threadIdx.x;i<quads;i+=THREADS){
      ushort4 q=v[i];
      q.x=f2h(__expf(h2f(q.x)-peak)*inv);
      q.y=f2h(__expf(h2f(q.y)-peak)*inv);
      q.z=f2h(__expf(h2f(q.z)-peak)*inv);
      q.w=f2h(__expf(h2f(q.w)-peak)*inv);
      v[i]=q;
    }
    return;
  }
  for(int i=threadIdx.x;i<visible;i+=THREADS){
    float v=h2f(s[i]);
    if(v>largest){total*=__expf(largest-v);largest=v;}
    total+=__expf(v-largest);
  }
  // Every thread's partial sum was taken against its own largest value, so it
  // is rescaled to the row's before the sums are added.
  float peak=block_max(largest,red);
  total*=__expf(largest-peak);
  float inv=1.f/block_sum(total,red);
  for(int i=threadIdx.x;i<context;i+=THREADS)
    s[i]=f2h(i<visible?__expf(h2f(s[i])-peak)*inv:0.f);
}

// `softmax_rows` for scores too large for FP16, which the FLUX.1 autoencoder's
// middle attention produces: f32 scores in, FP16 probabilities out.
extern "C" __global__ void softmax_wide(const float*__restrict__ scores,
    half_t*__restrict__ out,int context){
  __shared__ float red[THREADS];
  const float*s=scores+(long long)blockIdx.x*context;
  half_t*o=out+(long long)blockIdx.x*context;
  float largest=__int_as_float(0xff800000),total=0.f;
  for(int i=threadIdx.x;i<context;i+=THREADS){
    float v=s[i];
    if(v>largest){total*=__expf(largest-v);largest=v;}
    total+=__expf(v-largest);
  }
  float peak=block_max(largest,red);
  total*=__expf(largest-peak);
  float inv=1.f/block_sum(total,red);
  for(int i=threadIdx.x;i<context;i+=THREADS)o[i]=f2h(__expf(s[i]-peak)*inv);
}

// Half the projection is the value and half is its gate, which is the gated
// feed-forward every one of these transformer blocks ends in.
// The projection that feeds this is the widest tensor a transformer block
// writes, so its bias rides along here rather than in a pass of its own: a
// read and a write of `[tokens, 2 * inner]` that no longer happen. A null
// `bias` is a projection that has none.
extern "C" __global__ void gelu_gate(half_t*__restrict__ out,const half_t*__restrict__ x,
    const float*__restrict__ bias,int inner){
  const half_t*row=x+(long long)blockIdx.y*2*inner;
  half_t*target=out+(long long)blockIdx.y*inner;
  int base=blockIdx.x*blockDim.x*LANE+threadIdx.x;
  for(int k=0;k<LANE;k++){
    int c=base+k*blockDim.x;
    if(c<inner){
      float value=h2f(row[c]), gate=h2f(row[inner+c]);
      if(bias){ value+=bias[c]; gate+=bias[inner+c]; }
      target[c]=f2h(value*act_of(gate,1));
    }
  }
}

extern "C" __global__ void upsample(half_t*out,const half_t*in,
    long long n,int h,int w,int factor){
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  long long ow=(long long)w*factor,plane=ow*(long long)h*factor;
  long long p=i%plane;
  out[i]=in[((i/plane)*h+(p/ow)/factor)*w+(p%ow)/factor];
}

// Channel-major to one row per pixel and back. Naive, so one of the two
// directions is always uncoalesced.
// NOTE: a tiled shared-memory transpose is the standard fix; this runs
// twice per attention block, not per convolution, so it has not been worth it.
extern "C" __global__ void transpose(half_t*out,const half_t*in,long long rows,long long cols){
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<rows*cols)out[(i%cols)*rows+i/cols]=in[i];
}

// The backward pass, for training adapters on a frozen UNet. Gradients are
// FP16 like the activations they mirror; every reduction runs in FP32.

// The reverse of `im2col`, or with `tap` set of one vertical tap of `im2row`:
// every column entry is added back to the input value it was read from.
// Several taps land on one input value, so the sums are FP32 atomics into
// `out`, which `narrow` turns back into FP16 once every chunk is in.
extern "C" __global__ void col2im(float*__restrict__ out,const half_t*__restrict__ columns,
    int in_h,int in_w,int kernel,int stride,int pad,int out_w,int row,int span,long long pitch,
    int tap){
  int s=blockIdx.x*blockDim.x+threadIdx.x;
  if(s>=span)return;
  int line=s/out_w,ox=s-line*out_w;
  int c=blockIdx.y,taps=tap<0?kernel:1;
  float*target=out+(long long)c*in_h*in_w;
  const half_t*source=columns+(long long)c*taps*kernel*pitch+s;
  int y0=(row+line)*stride-pad,x0=ox*stride-pad;
  for(int t=0;t<taps;t++){
    int y=y0+(tap<0?t:tap);
    if(y<0||y>=in_h)continue;
    for(int kx=0;kx<kernel;kx++){
      int x=x0+kx;
      if(x>=0&&x<in_w)
        atomicAdd(target+(long long)y*in_w+x,h2f(source[(long long)(t*kernel+kx)*pitch]));
    }
  }
}
extern "C" __global__ void narrow(half_t*__restrict__ out,const float*__restrict__ in,long long n){
  long long base=(long long)blockIdx.x*blockDim.x*LANE+threadIdx.x;
  for(int k=0;k<LANE;k++){
    long long i=base+(long long)k*blockDim.x;
    if(i<n)out[i]=f2h(in[i]);
  }
}

// `dL/dx` of a group norm, in place over the incoming gradient `g`:
// `inv * (g*w - mean(g*w) - xhat * mean(xhat*g*w))` over each group. The two
// group means are split along the span as `group_partials` splits its sums,
// and `stats` are the forward pass's, from `group_fold`.
extern "C" __global__ void group_bwd_partials(const half_t*__restrict__ x,
    const half_t*__restrict__ g,const float*__restrict__ w,const float*__restrict__ stats,
    float*__restrict__ out,long long pixels,int per_group,int splits){
  __shared__ float red[THREADS];
  long long n=blockIdx.y,s=blockIdx.x,span=pixels*per_group;
  const half_t*xv=x+n*span;
  const half_t*gv=g+n*span;
  const float*wv=w+n*per_group;
  float mean=stats[2*n],inv=stats[2*n+1];
  long long begin=span*s/splits,end=span*(s+1)/splits;
  float plain=0.f,projected=0.f;
  for(long long i=begin+threadIdx.x;i<end;i+=THREADS){
    float d=h2f(gv[i])*wv[i/pixels];
    plain+=d;projected+=(h2f(xv[i])-mean)*inv*d;
  }
  float a=block_sum(plain,red),b=block_sum(projected,red);
  if(threadIdx.x==0){out[n*splits+s]=a;out[((long long)gridDim.y+n)*splits+s]=b;}
}
extern "C" __global__ void group_bwd_apply(half_t*__restrict__ g,const half_t*__restrict__ x,
    const float*__restrict__ w,const float*__restrict__ stats,const float*__restrict__ partials,
    long long pixels,int per_group,int groups,int splits,float span){
  __shared__ float means[2];
  int c=blockIdx.y,n=c/per_group;
  if(threadIdx.x==0){
    float a=0.f,b=0.f;
    for(int s=0;s<splits;s++){a+=partials[n*splits+s];b+=partials[(groups+n)*splits+s];}
    means[0]=a/span;means[1]=b/span;
  }
  __syncthreads();
  half_t*row=g+(long long)c*pixels;
  const half_t*xr=x+(long long)c*pixels;
  float mean=stats[2*n],inv=stats[2*n+1],scale=w[c],plain=means[0],projected=means[1];
  long long base=(long long)blockIdx.x*blockDim.x*LANE+threadIdx.x;
  for(int k=0;k<LANE;k++){
    long long p=base+(long long)k*blockDim.x;
    if(p>=pixels)continue;
    float xhat=(h2f(xr[p])-mean)*inv;
    row[p]=f2h(inv*(h2f(row[p])*scale-plain-xhat*projected));
  }
}

// `dL/dx` of `layer_norm`, in place over `g`, one block per token. Every
// value of the row is read into both sums before the first is written.
extern "C" __global__ void layer_norm_bwd(half_t*g,const half_t*x,const float*w,int cols,float eps){
  __shared__ float red[THREADS];
  const half_t*src=x+(long long)blockIdx.x*cols;
  half_t*grad=g+(long long)blockIdx.x*cols;
  float total=0.f;
  for(int i=threadIdx.x;i<cols;i+=THREADS)total+=h2f(src[i]);
  float mean=block_sum(total,red)/(float)cols;
  float square=0.f;
  for(int i=threadIdx.x;i<cols;i+=THREADS){float d=h2f(src[i])-mean;square+=d*d;}
  float inv=rsqrtf(block_sum(square,red)/(float)cols+eps);
  float plain=0.f,projected=0.f;
  for(int i=threadIdx.x;i<cols;i+=THREADS){
    float d=h2f(grad[i])*w[i];
    plain+=d;projected+=(h2f(src[i])-mean)*inv*d;
  }
  plain=block_sum(plain,red)/(float)cols;
  projected=block_sum(projected,red)/(float)cols;
  for(int i=threadIdx.x;i<cols;i+=THREADS){
    float xhat=(h2f(src[i])-mean)*inv;
    grad[i]=f2h(inv*(h2f(grad[i])*w[i]-plain-xhat*projected));
  }
}

// The derivative of the tanh GELU `act_of` computes as kind 1.
__device__ __forceinline__ float gelu_grad(float x){
  float t=tanhf(0.7978846f*(x+0.044715f*x*x*x));
  return 0.5f*(1.f+t)+0.5f*x*(1.f-t*t)*0.7978846f*(1.f+3.f*0.044715f*x*x);
}
// `dL/dx` of `gelu_gate` for `[tokens, 2 * inner]` of `x` and `[tokens,
// inner]` of `g`, with the same optional bias.
extern "C" __global__ void gelu_gate_bwd(half_t*__restrict__ out,const half_t*__restrict__ x,
    const float*__restrict__ bias,const half_t*__restrict__ g,int inner){
  const half_t*row=x+(long long)blockIdx.y*2*inner;
  const half_t*grad=g+(long long)blockIdx.y*inner;
  half_t*target=out+(long long)blockIdx.y*2*inner;
  int base=blockIdx.x*blockDim.x*LANE+threadIdx.x;
  for(int k=0;k<LANE;k++){
    int c=base+k*blockDim.x;
    if(c<inner){
      float value=h2f(row[c]),gate=h2f(row[inner+c]),d=h2f(grad[c]);
      if(bias){value+=bias[c];gate+=bias[inner+c];}
      target[c]=f2h(d*act_of(gate,1));
      target[inner+c]=f2h(d*value*gelu_grad(gate));
    }
  }
}

// `g *= silu'(pre)`.
extern "C" __global__ void silu_bwd(half_t*__restrict__ g,const half_t*__restrict__ pre,
    long long n){
  long long base=(long long)blockIdx.x*blockDim.x*LANE+threadIdx.x;
  for(int k=0;k<LANE;k++){
    long long i=base+(long long)k*blockDim.x;
    if(i>=n)continue;
    float x=h2f(pre[i]),s=1.f/(1.f+expf(-x));
    g[i]=f2h(h2f(g[i])*s*(1.f+x*(1.f-s)));
  }
}

// `dL/dx` of `upsample`: each of the `n` values of the small map sums the
// `factor * factor` copies made of it.
extern "C" __global__ void upsample_bwd(half_t*out,const half_t*in,
    long long n,int h,int w,int factor){
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=n)return;
  long long plane=(long long)h*w,c=i/plane,p=i%plane,y=p/w,x=p%w,wide=(long long)w*factor;
  const half_t*source=in+(c*h*factor+y*factor)*wide+x*factor;
  float sum=0.f;
  for(int dy=0;dy<factor;dy++)
    for(int dx=0;dx<factor;dx++)sum+=h2f(source[dy*wide+dx]);
  out[i]=f2h(sum);
}

// `dL/ds` of each softmax row times `scale`, in place over `dL/dp` in `g`:
// `scale * p * (g - p . g)`. A warp a row.
extern "C" __global__ void softmax_bwd(half_t*__restrict__ g,const half_t*__restrict__ p,
    int context,int rows,float scale){
  int row=blockIdx.x*WARPS+(threadIdx.x>>5);
  if(row>=rows)return;
  int lane=threadIdx.x&31;
  half_t*gr=g+(long long)row*context;
  const half_t*pr=p+(long long)row*context;
  float dot=0.f;
  for(int i=lane;i<context;i+=32)dot+=h2f(pr[i])*h2f(gr[i]);
  dot=warp_sum(dot);
  for(int i=lane;i<context;i+=32)gr[i]=f2h(scale*h2f(pr[i])*(h2f(gr[i])-dot));
}
"#;

mod lora;
pub(crate) mod wan_vae;
pub use lora::{DeviceAdapters, DeviceTape};

/// NVRTC compilation costs a few hundred milliseconds and the source never
/// varies, so a process that loads several pipelines compiles once.
fn image_ptx() -> Result<&'static Ptx, NetworkError> {
    static PTX: OnceLock<Result<Ptx, String>> = OnceLock::new();
    PTX.get_or_init(|| compile_ptx(IMAGE_KERNELS).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| {
            NetworkError::Cuda(format!("CUDA image kernel compilation failed: {error}"))
        })
}

/// One device, the [`GpuContext`] it shares with the transformer path, and the
/// image kernels resolved once.
pub struct ImageGpu {
    context: Arc<GpuContext>,
    im2col: CudaFunction,
    im2row: CudaFunction,
    add_bias_rows: CudaFunction,
    add_channel: CudaFunction,
    add_plane_bias: CudaFunction,
    add_inplace: CudaFunction,
    activate: CudaFunction,
    group_partials: CudaFunction,
    group_fold: CudaFunction,
    group_apply: CudaFunction,
    layer_norm: CudaFunction,
    softmax_rows: CudaFunction,
    softmax_warp: CudaFunction,
    softmax_wide: CudaFunction,
    gelu_gate: CudaFunction,
    upsample: CudaFunction,
    transpose: CudaFunction,
    col2im: CudaFunction,
    narrow: CudaFunction,
    group_bwd_partials: CudaFunction,
    group_bwd_apply: CudaFunction,
    layer_norm_bwd: CudaFunction,
    gelu_gate_bwd: CudaFunction,
    silu_bwd: CudaFunction,
    upsample_bwd: CudaFunction,
    softmax_bwd: CudaFunction,
    /// Keep attention scores in f32. Off by default, which keeps every
    /// existing result bit for bit. A decoder turns it on for good the first
    /// time it overflows FP16, which the FLUX.1 autoencoder (Z-Image) does in
    /// its middle attention: its scores reach 2e7.
    wide_scores: std::sync::atomic::AtomicBool,
}

impl std::fmt::Debug for ImageGpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageGpu").finish_non_exhaustive()
    }
}

impl ImageGpu {
    /// Fails closed: no device, no cuBLAS or no kernels is an error, never a
    /// quiet fallback. The caller decides whether to stay on the CPU.
    pub fn new(device: usize) -> Result<Arc<Self>, NetworkError> {
        let (context, _) = device_context(device)?;
        let module: Arc<CudaModule> = context
            .load_module(image_ptx()?.clone())
            .map_err(cuda_err("CUDA image module loading"))?;
        let get = |name: &str| {
            module
                .load_function(name)
                .map_err(cuda_err("CUDA kernel lookup"))
        };
        Ok(Arc::new(Self {
            context: GpuContext::half_accumulate(device)?,
            im2col: get("im2col")?,
            im2row: get("im2row")?,
            add_bias_rows: get("add_bias_rows")?,
            add_channel: get("add_channel")?,
            add_plane_bias: get("add_plane_bias")?,
            add_inplace: get("add_inplace")?,
            activate: get("activate")?,
            group_partials: get("group_partials")?,
            group_fold: get("group_fold")?,
            group_apply: get("group_apply")?,
            layer_norm: get("layer_norm")?,
            softmax_rows: get("softmax_rows")?,
            softmax_warp: get("softmax_warp")?,
            softmax_wide: get("softmax_wide")?,
            gelu_gate: get("gelu_gate")?,
            upsample: get("upsample")?,
            transpose: get("transpose")?,
            col2im: get("col2im")?,
            narrow: get("narrow")?,
            group_bwd_partials: get("group_bwd_partials")?,
            group_bwd_apply: get("group_bwd_apply")?,
            layer_norm_bwd: get("layer_norm_bwd")?,
            gelu_gate_bwd: get("gelu_gate_bwd")?,
            silu_bwd: get("silu_bwd")?,
            upsample_bwd: get("upsample_bwd")?,
            softmax_bwd: get("softmax_bwd")?,
            wide_scores: false.into(),
        }))
    }

    /// Blocks until every launch has retired, which is what a timing harness
    /// and the end of a forward pass need.
    pub fn synchronize(&self) -> Result<(), NetworkError> {
        self.context.synchronize()
    }

    /// An uninitialized buffer, for the many places whose every element is
    /// written by the kernel that follows.
    fn uninit(&self, rows: usize, cols: usize) -> Result<Tensor, NetworkError> {
        Ok(Tensor {
            data: unsafe { self.context.stream.alloc::<f16>((rows * cols).max(1)) }.map_err(
                cuda_alloc_err("device allocation", (rows * cols).max(1) * 2),
            )?,
            rows,
            cols,
        })
    }

    /// `[rows, cols]` of `f32` narrowed on the host and copied up once.
    ///
    /// Narrowing a whole checkpoint is billions of conversions, so a large
    /// weight is split over the cores. A small one is not worth a spawn.
    fn upload(&self, rows: usize, cols: usize, values: &[f32]) -> Result<Tensor, NetworkError> {
        let lanes = match values.len() >= 1 << 20 {
            true => std::thread::available_parallelism().map_or(1, |count| count.get()),
            false => 1,
        };
        let narrowed: Vec<f16> = match lanes {
            1 => values.iter().copied().map(f16::from_f32).collect(),
            _ => {
                let mut narrowed = vec![f16::ZERO; values.len()];
                let stride = values.len().div_ceil(lanes);
                std::thread::scope(|scope| {
                    for (slot, source) in narrowed.chunks_mut(stride).zip(values.chunks(stride)) {
                        scope.spawn(move || {
                            for (target, value) in slot.iter_mut().zip(source) {
                                *target = f16::from_f32(*value);
                            }
                        });
                    }
                });
                narrowed
            }
        };
        Ok(Tensor {
            data: self
                .context
                .stream
                .clone_htod(&narrowed)
                .map_err(cuda_alloc_err(
                    "host to device copy",
                    std::mem::size_of_val(&narrowed[..]),
                ))?,
            rows,
            cols,
        })
    }

    fn upload_matrix(&self, matrix: &Matrix) -> Result<Tensor, NetworkError> {
        self.upload(matrix.rows, matrix.cols, &matrix.data)
    }

    /// The per-channel vectors — norms, biases — stay `f32`. They are a
    /// rounding error of the model's size and they are what the reductions
    /// scale by, so there is nothing to buy by narrowing them.
    fn upload_f32(&self, values: &[f32]) -> Result<CudaSlice<f32>, NetworkError> {
        self.context
            .stream
            .clone_htod(if values.is_empty() {
                &[0.0][..]
            } else {
                values
            })
            .map_err(cuda_err("host to device copy"))
    }

    fn download(&self, tensor: &Tensor) -> Result<Matrix, NetworkError> {
        let narrow = self
            .context
            .stream
            .clone_dtoh(&tensor.data)
            .map_err(cuda_err("device to host copy"))?;
        Ok(Matrix::from_vec(
            tensor.rows,
            tensor.cols,
            narrow.iter().copied().map(f16::to_f32).collect(),
        ))
    }
}

/// A `[rows, cols]` row-major FP16 buffer.
///
/// A feature map is one of these too, held channel-major: `rows` is the channel
/// count and `cols` the pixel count, which is the layout every convolution and
/// every checkpoint reads.
struct Tensor {
    data: CudaSlice<f16>,
    rows: usize,
    cols: usize,
}

impl Tensor {
    fn len(&self) -> usize {
        self.rows * self.cols
    }

    /// The buffer as untyped bytes, which is how cuBLAS is told the element
    /// type separately from the pointer.
    fn bytes(&self) -> CudaView<'_, u8> {
        unsafe { self.data.transmute::<u8>(self.data.len() * 2) }
            .expect("a device allocation is byte-aligned")
    }

    fn view(&self) -> View<'_> {
        View {
            data: &self.data,
            base: 0,
            lead: self.cols,
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// `rows` whole rows starting at row `first`: one sequence out of several
    /// stacked into this buffer.
    fn window(&self, first: usize, rows: usize) -> View<'_> {
        View {
            base: first * self.cols,
            rows,
            ..self.view()
        }
    }

    /// The `index`th of several equally wide projections interleaved across
    /// this buffer's columns.
    fn part(&self, index: usize, cols: usize) -> View<'_> {
        View {
            data: &self.data,
            base: index * cols,
            lead: self.cols,
            rows: self.rows,
            cols,
        }
    }
}

/// A window on a tensor: `rows` rows of `cols` columns, starting `base`
/// elements in and `lead` elements apart. A fused projection writes its
/// queries, keys and values into one buffer and hands attention three of
/// these.
#[derive(Clone, Copy)]
struct View<'a> {
    data: &'a CudaSlice<f16>,
    base: usize,
    lead: usize,
    rows: usize,
    cols: usize,
}

/// A feature map: a [`Tensor`] plus the two dimensions its pixel axis folds
/// into.
struct Map {
    tensor: Tensor,
    height: usize,
    width: usize,
}

impl Map {
    fn channels(&self) -> usize {
        self.tensor.rows
    }

    fn pixels(&self) -> usize {
        self.tensor.cols
    }
}

/// The longest score row the softmax hands to a single warp. Above it a warp
/// walks too many values in sequence and a whole block is quicker.
const WARP_SCORES: usize = 2048;

/// How many elements one thread of a streaming kernel walks. Has to agree with
/// `LANE` in the kernel source.
const LANE: u32 = 4;

/// A grid of `rows` lines, each `width` elements wide: the launch shape for
/// every kernel that wants its row index without a division per element.
fn plane(width: usize, rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((width as u32).div_ceil(THREADS), rows as u32, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// The same, for the streaming kernels whose threads walk [`LANE`] elements.
fn lanes(width: usize, rows: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((width as u32).div_ceil(THREADS * LANE), rows as u32, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn grid(count: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (count as u32, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// The layers, as they sit on the device.
// ---------------------------------------------------------------------------

/// Whether a convolution runs as one GEMM per vertical tap over `im2row`
/// rather than one GEMM over `im2col`.
fn rowwise(kernel: usize, stride: usize) -> bool {
    kernel > 1 && stride == 1
}

struct DeviceConv {
    /// `[out_channels, in_channels * kernel * kernel]`, the same flattening the
    /// host layer holds, or for a [`rowwise`] convolution the vertical taps
    /// pulled out front: `[kernel][out_channels][in_channels * kernel]`.
    weight: Tensor,
    /// Always present. A convolution without one gets zeros, which keeps the
    /// epilogue kernel branchless.
    bias: CudaSlice<f32>,
    /// Whether `bias` holds anything but zeros, so the epilogue can be skipped.
    biased: bool,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    /// Extra zeros past the right and bottom edges only, which is how the VAE
    /// encoder's halving convolution pads. im2col already reads zeros past
    /// the edge, so this only changes how many output pixels there are.
    pad_end: usize,
}

struct DeviceDense {
    weight: Tensor,
    bias: Option<CudaSlice<f32>>,
}

/// Several projections of one width stacked into a single weight, so one
/// batched GEMM runs all of them.
struct DeviceStack {
    dense: DeviceDense,
    blocks: usize,
}

impl DeviceStack {
    fn units(&self) -> usize {
        self.dense.weight.rows / self.blocks
    }
}

struct DeviceGroupNorm {
    groups: usize,
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
    eps: f32,
}

struct DeviceNorm {
    weight: CudaSlice<f32>,
    bias: CudaSlice<f32>,
}

enum DeviceProjection {
    Convolution(DeviceConv),
    Linear(DeviceDense),
}

fn upload_conv(gpu: &ImageGpu, conv: &Conv2d) -> Result<DeviceConv, NetworkError> {
    let (k, out) = (conv.kernel, conv.weight.rows);
    let weight = match rowwise(k, conv.stride) {
        // `[ky][out][in * k + kx]`, one `[out, in * k]` matrix per vertical
        // tap, which is the layout the `im2row` path multiplies by.
        true => {
            let patch = conv.weight.cols;
            let mut data = Vec::with_capacity(conv.weight.data.len());
            for ky in 0..k {
                for o in 0..out {
                    for c in 0..conv.in_channels {
                        let base = o * patch + (c * k + ky) * k;
                        data.extend_from_slice(&conv.weight.data[base..base + k]);
                    }
                }
            }
            gpu.upload(k * out, patch / k, &data)?
        }
        false => gpu.upload_matrix(&conv.weight)?,
    };
    Ok(DeviceConv {
        weight,
        bias: gpu.upload_f32(&match &conv.bias {
            Some(bias) => bias.clone(),
            None => vec![0.0; conv.weight.rows],
        })?,
        biased: conv.bias.is_some(),
        in_channels: conv.in_channels,
        out_channels: conv.out_channels(),
        kernel: conv.kernel,
        stride: conv.stride,
        padding: conv.padding,
        pad_end: 0,
    })
}

fn upload_dense(gpu: &ImageGpu, dense: &Dense) -> Result<DeviceDense, NetworkError> {
    Ok(DeviceDense {
        weight: gpu.upload_matrix(dense.weight_f32().as_ref())?,
        bias: match &dense.bias {
            Some(bias) => Some(gpu.upload_f32(bias)?),
            None => None,
        },
    })
}

/// Several projections of the same input as one matrix, so one GEMM can do
/// all of them. A layer that leaves one of the biases off gets zeros there.
fn upload_stacked(gpu: &ImageGpu, parts: &[&Dense]) -> Result<DeviceStack, NetworkError> {
    let inner = parts[0].weight.cols;
    let units = parts[0].weight.rows;
    let mut weight = Vec::with_capacity(parts.len() * units * inner);
    let mut bias = Vec::with_capacity(parts.len() * units);
    let mut biased = false;
    for part in parts {
        if part.weight.cols != inner || part.weight.rows != units {
            return Err(NetworkError::InvalidConfig(format!(
                "stacked projections must match: {}x{} beside {}x{}",
                units, inner, part.weight.rows, part.weight.cols
            )));
        }
        weight.extend_from_slice(&part.weight_f32().as_ref().data);
        match &part.bias {
            Some(values) => {
                biased = true;
                bias.extend_from_slice(values);
            }
            None => bias.extend(std::iter::repeat_n(0.0, units)),
        }
    }
    Ok(DeviceStack {
        dense: DeviceDense {
            weight: gpu.upload(parts.len() * units, inner, &weight)?,
            bias: match biased {
                true => Some(gpu.upload_f32(&bias)?),
                false => None,
            },
        },
        blocks: parts.len(),
    })
}

fn upload_group_norm(gpu: &ImageGpu, norm: &GroupNorm) -> Result<DeviceGroupNorm, NetworkError> {
    Ok(DeviceGroupNorm {
        groups: norm.groups,
        weight: gpu.upload_f32(&norm.weight)?,
        bias: gpu.upload_f32(&norm.bias)?,
        eps: norm.eps,
    })
}

fn upload_norm(gpu: &ImageGpu, norm: &Norm) -> Result<DeviceNorm, NetworkError> {
    Ok(DeviceNorm {
        weight: gpu.upload_f32(&norm.weight)?,
        bias: gpu.upload_f32(&norm.bias)?,
    })
}

// ---------------------------------------------------------------------------
// The operations.
// ---------------------------------------------------------------------------

impl ImageGpu {
    /// `out[channels, pixels]`, the convolution lowered to im2col plus one GEMM
    /// per chunk of output pixels.
    fn conv(&self, conv: &DeviceConv, input: &Map) -> Result<Map, NetworkError> {
        self.conv_with_budget(conv, input, COLUMN_BUDGET)
    }

    /// The convolution added on top of what `target` already holds.
    ///
    /// Same trick as [`ImageGpu::dense_add`]: a residual shortcut is the
    /// convolution's GEMM accumulating into its output rather than a second
    /// kernel reading both maps back.
    fn conv_add(
        &self,
        conv: &DeviceConv,
        input: &Map,
        target: &mut Map,
    ) -> Result<(), NetworkError> {
        let produced = self.conv_into(conv, input, COLUMN_BUDGET, Some(target))?;
        debug_assert!(produced.is_none());
        Ok(())
    }

    /// The convolution with its im2col scratch bounded by `budget` FP16
    /// elements, which is a parameter only so a test can force the chunk loop
    /// without shrinking a constant the rest of the crate sizes itself by.
    fn conv_with_budget(
        &self,
        conv: &DeviceConv,
        input: &Map,
        budget: usize,
    ) -> Result<Map, NetworkError> {
        Ok(self
            .conv_into(conv, input, budget, None)?
            .expect("a convolution with no target allocates one"))
    }

    /// With a `target` the result is accumulated there and nothing is
    /// returned; without one a fresh map comes back.
    fn conv_into(
        &self,
        conv: &DeviceConv,
        input: &Map,
        budget: usize,
        target: Option<&mut Map>,
    ) -> Result<Option<Map>, NetworkError> {
        if input.channels() != conv.in_channels {
            return Err(NetworkError::InvalidConfig(format!(
                "this convolution reads {} channels and was handed {}",
                conv.in_channels,
                input.channels()
            )));
        }
        let size = |length: usize| match (length + 2 * conv.padding + conv.pad_end)
            .checked_sub(conv.kernel)
        {
            Some(span) => span / conv.stride + 1,
            None => 0,
        };
        let (height, width) = (size(input.height), size(input.width));
        if height == 0 || width == 0 {
            return Err(NetworkError::InvalidConfig(format!(
                "a {}x{} kernel over a {}x{} input leaves nothing",
                conv.kernel, conv.kernel, input.height, input.width
            )));
        }
        let pixels = height * width;
        let mut fresh = None;
        let (output, beta): (&mut Tensor, f32) = match target {
            Some(target)
                if target.channels() == conv.out_channels
                    && target.height == height
                    && target.width == width =>
            {
                (&mut target.tensor, 1.0)
            }
            Some(target) => {
                return Err(NetworkError::InvalidTarget {
                    expected: conv.out_channels * pixels,
                    actual: target.tensor.len(),
                });
            }
            None => (fresh.insert(self.uninit(conv.out_channels, pixels)?), 0.0),
        };

        // A one-by-one convolution reads each pixel where it already sits, so
        // there is nothing to gather: the whole layer is one GEMM straight into
        // channel-major. That covers every projection into and out of a
        // transformer, every residual shortcut, and the VAE's `post_quant_conv`.
        if conv.kernel == 1 && conv.stride == 1 && conv.padding == 0 {
            act_plain::<f16, _>(
                &self.context,
                &conv.weight.bytes(),
                conv.in_channels,
                &input.tensor.bytes(),
                pixels,
                true,
                true,
                &mut output.data,
                pixels,
                conv.out_channels,
                pixels,
                conv.in_channels,
                1.0,
                beta,
            )?;
        } else if rowwise(conv.kernel, conv.stride) {
            let k = conv.kernel;
            let patch = conv.in_channels * k;
            let group = if width % 8 == 0 { 1 } else { 8 };
            let stack = (budget / (patch * width).max(1))
                .saturating_sub(k - 1)
                .max(group)
                .next_multiple_of(group)
                .min(height);
            // The `k - 1` lines past the chunk are the halo its lower taps read.
            let pitch = (stack + k - 1) * width;
            let mut columns = self.uninit(patch, pitch)?;
            let taps = conv.weight.bytes();
            let tap = conv.out_channels * patch * 2;
            for row in (0..height).step_by(stack) {
                let rows = stack.min(height - row);
                let span = rows * width;
                let lines = (rows + k - 1) * width;
                unsafe {
                    self.context
                        .stream
                        .launch_builder(&self.im2row)
                        .arg(&mut columns.data)
                        .arg(&input.tensor.data)
                        .arg(&(input.height as i32))
                        .arg(&(input.width as i32))
                        .arg(&(k as i32))
                        .arg(&(conv.padding as i32))
                        .arg(&(width as i32))
                        .arg(&(row as i32))
                        .arg(&(lines as i32))
                        .arg(&(pitch as i64))
                        .launch(plane(lines, conv.in_channels))
                        .map_err(cuda_err("im2row kernel"))?;
                }
                let bytes = columns.bytes();
                let mut target = output.data.slice_mut(row * width..);
                for ky in 0..k {
                    act_plain::<f16, _>(
                        &self.context,
                        &taps.slice(ky * tap..(ky + 1) * tap),
                        patch,
                        &bytes.slice(ky * width * 2..),
                        pitch,
                        true,
                        true,
                        &mut target,
                        pixels,
                        conv.out_channels,
                        span,
                        patch,
                        1.0,
                        if ky == 0 { beta } else { 1.0 },
                    )?;
                }
            }
        } else {
            let patch = conv.in_channels * conv.kernel * conv.kernel;
            // Chunks are whole output rows, so the grid can address a pixel by
            // its row and column and the kernel needs no division. A chunk is
            // rounded to eight rows when the row length is odd enough to leave
            // the GEMM's write offset unaligned, and left at one row otherwise,
            // because eight rows of a 1024-wide VAE layer is already 75 MB of
            // scratch and there is no reason to round that up.
            let group = if width % 8 == 0 { 1 } else { 8 };
            let stack = (budget / (patch * width).max(1))
                .max(group)
                .next_multiple_of(group)
                .min(height);
            let pitch = stack * width;
            let mut columns = self.uninit(patch, pitch)?;
            for row in (0..height).step_by(stack) {
                let rows = stack.min(height - row);
                let span = rows * width;
                let (row32, span32, pitch64) = (row as i32, span as i32, pitch as i64);
                unsafe {
                    self.context
                        .stream
                        .launch_builder(&self.im2col)
                        .arg(&mut columns.data)
                        .arg(&input.tensor.data)
                        .arg(&(input.height as i32))
                        .arg(&(input.width as i32))
                        .arg(&(conv.kernel as i32))
                        .arg(&(conv.stride as i32))
                        .arg(&(conv.padding as i32))
                        .arg(&(width as i32))
                        .arg(&row32)
                        .arg(&span32)
                        .arg(&pitch64)
                        .launch(plane(span, conv.in_channels))
                        .map_err(cuda_err("im2col kernel"))?;
                }
                // `out[out_channels, span] = weight . columns`, written at
                // column `row * width` of a buffer whose rows are `pixels`
                // apart.
                let mut target = output.data.slice_mut(row * width..);
                act_plain::<f16, _>(
                    &self.context,
                    &conv.weight.bytes(),
                    patch,
                    &columns.bytes(),
                    pitch,
                    true,
                    true,
                    &mut target,
                    pixels,
                    conv.out_channels,
                    span,
                    patch,
                    1.0,
                    beta,
                )?;
            }
        }
        if conv.biased {
            self.add_channel_f32(output, &conv.bias, pixels)?;
        }
        Ok(fresh.map(|tensor| Map {
            tensor,
            height,
            width,
        }))
    }

    /// Several [`rowwise`] convolutions of one input, each added onto
    /// (`beta` of one) or written over (zero) its own same-sized output. The input is
    /// gathered once for all of them, which is what a causal convolution's
    /// taps want: every frame meets each tap, and the gather is the part
    /// that does not depend on the weights.
    fn conv_fan(
        &self,
        input: &Map,
        jobs: &mut [(&DeviceConv, &mut Map, f32)],
        budget: usize,
    ) -> Result<(), NetworkError> {
        let Some(first) = jobs.first().map(|job| job.0) else {
            return Ok(());
        };
        let (k, padding) = (first.kernel, first.padding);
        let pixels = input.pixels();
        for (conv, target, _) in jobs.iter() {
            if !rowwise(conv.kernel, conv.stride)
                || conv.kernel != k
                || conv.padding != padding
                || conv.pad_end != 0
                || k != 2 * padding + 1
                || conv.in_channels != input.channels()
                || target.channels() != conv.out_channels
                || target.height != input.height
                || target.width != input.width
            {
                return Err(NetworkError::InvalidConfig(format!(
                    "a fanned {k}x{k} convolution over {} channels does not fit a {}x{} input",
                    conv.in_channels, input.height, input.width
                )));
            }
        }
        let (height, width) = (input.height, input.width);
        let patch = input.channels() * k;
        let group = if width % 8 == 0 { 1 } else { 8 };
        let stack = (budget / (patch * width).max(1))
            .saturating_sub(k - 1)
            .max(group)
            .next_multiple_of(group)
            .min(height);
        let pitch = (stack + k - 1) * width;
        let mut columns = self.uninit(patch, pitch)?;
        for row in (0..height).step_by(stack) {
            let rows = stack.min(height - row);
            let span = rows * width;
            let lines = (rows + k - 1) * width;
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.im2row)
                    .arg(&mut columns.data)
                    .arg(&input.tensor.data)
                    .arg(&(input.height as i32))
                    .arg(&(input.width as i32))
                    .arg(&(k as i32))
                    .arg(&(padding as i32))
                    .arg(&(width as i32))
                    .arg(&(row as i32))
                    .arg(&(lines as i32))
                    .arg(&(pitch as i64))
                    .launch(plane(lines, input.channels()))
                    .map_err(cuda_err("im2row kernel"))?;
            }
            let bytes = columns.bytes();
            for (conv, output, beta) in jobs.iter_mut() {
                let taps = conv.weight.bytes();
                let tap = conv.out_channels * patch * 2;
                let mut target = output.tensor.data.slice_mut(row * width..);
                for ky in 0..k {
                    act_plain::<f16, _>(
                        &self.context,
                        &taps.slice(ky * tap..(ky + 1) * tap),
                        patch,
                        &bytes.slice(ky * width * 2..),
                        pitch,
                        true,
                        true,
                        &mut target,
                        pixels,
                        conv.out_channels,
                        span,
                        patch,
                        1.0,
                        if ky == 0 { *beta } else { 1.0 },
                    )?;
                }
            }
        }
        for (conv, output, _) in jobs.iter_mut() {
            if conv.biased {
                self.add_channel_f32(&mut output.tensor, &conv.bias, pixels)?;
            }
        }
        Ok(())
    }

    /// `out[rows, out_dim] = tokens . weight^T + bias`.
    fn dense(&self, dense: &DeviceDense, tokens: &Tensor) -> Result<Tensor, NetworkError> {
        let mut output = self.uninit(tokens.rows, dense.weight.rows)?;
        self.dense_into(dense, tokens, &mut output, 0.0)?;
        Ok(output)
    }

    /// The same product added on top of what `target` already holds.
    ///
    /// A residual connection is a projection followed by a sum, and the sum is
    /// free: cuBLAS accumulates into its output when told to scale it by one,
    /// which spares a whole read and write of the projection.
    fn dense_add(
        &self,
        dense: &DeviceDense,
        tokens: &Tensor,
        target: &mut Tensor,
    ) -> Result<(), NetworkError> {
        self.dense_into(dense, tokens, target, 1.0)
    }

    /// The product alone, for a caller that applies the bias itself.
    fn dense_unbiased(&self, dense: &DeviceDense, tokens: &Tensor) -> Result<Tensor, NetworkError> {
        let mut output = self.uninit(tokens.rows, dense.weight.rows)?;
        self.dense_into_with(dense, tokens, &mut output, 0.0, false)?;
        Ok(output)
    }

    fn dense_into(
        &self,
        dense: &DeviceDense,
        tokens: &Tensor,
        output: &mut Tensor,
        beta: f32,
    ) -> Result<(), NetworkError> {
        self.dense_into_with(dense, tokens, output, beta, true)
    }

    fn dense_into_with(
        &self,
        dense: &DeviceDense,
        tokens: &Tensor,
        output: &mut Tensor,
        beta: f32,
        apply_bias: bool,
    ) -> Result<(), NetworkError> {
        if tokens.cols != dense.weight.cols {
            return Err(NetworkError::InvalidTarget {
                expected: dense.weight.cols,
                actual: tokens.cols,
            });
        }
        let units = dense.weight.rows;
        if output.rows != tokens.rows || output.cols != units {
            return Err(NetworkError::InvalidTarget {
                expected: tokens.rows * units,
                actual: output.len(),
            });
        }
        act_rhs_transposed::<f16, _>(
            &self.context,
            &tokens.bytes(),
            tokens.cols,
            &dense.weight.bytes(),
            dense.weight.cols,
            true,
            true,
            &mut output.data,
            units,
            tokens.rows,
            units,
            tokens.cols,
            1.0,
            beta,
        )?;
        if let Some(bias) = dense.bias.as_ref().filter(|_| apply_bias) {
            let width = units as i32;
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.add_bias_rows)
                    .arg(&mut output.data)
                    .arg(bias)
                    .arg(&width)
                    .launch(plane(units, tokens.rows))
                    .map_err(cuda_err("bias kernel"))?;
            }
        }
        Ok(())
    }

    /// Group normalization, with the activation that follows it folded in:
    /// `act` is 0 for SiLU, or negative for the normalization alone.
    fn group_norm(
        &self,
        norm: &DeviceGroupNorm,
        map: &mut Map,
        act: i32,
    ) -> Result<(), NetworkError> {
        let (stats, per_group) = self.group_stats(norm, map)?;
        unsafe {
            let shape = lanes(map.pixels(), map.channels());
            let (pixels, per_group) = (map.pixels() as i64, per_group as i32);
            self.context
                .stream
                .launch_builder(&self.group_apply)
                .arg(&mut map.tensor.data)
                .arg(&stats)
                .arg(&norm.weight)
                .arg(&norm.bias)
                .arg(&pixels)
                .arg(&per_group)
                .arg(&act)
                .launch(shape)
                .map_err(cuda_err("group norm kernel"))?;
        }
        Ok(())
    }

    /// Each group's mean and inverse deviation, interleaved, and the channels
    /// per group.
    fn group_stats(
        &self,
        norm: &DeviceGroupNorm,
        map: &Map,
    ) -> Result<(CudaSlice<f32>, usize), NetworkError> {
        let per_group = map.channels() / norm.groups;
        if per_group * norm.groups != map.channels() {
            return Err(NetworkError::InvalidTarget {
                expected: norm.groups,
                actual: map.channels(),
            });
        }
        let span = per_group * map.pixels();
        // Enough slices to fill the card, capped so the fold stays a short
        // serial loop. One block per group is what this used to be, and it ran
        // at a tenth of the available bandwidth.
        let splits = (span / (THREADS as usize * 16)).clamp(1, 128);
        let mut partials = self
            .context
            .stream
            .alloc_zeros::<f32>(2 * norm.groups * splits)
            .map_err(cuda_err("device allocation"))?;
        let mut stats = self
            .context
            .stream
            .alloc_zeros::<f32>(2 * norm.groups)
            .map_err(cuda_err("device allocation"))?;
        let (span64, splits32) = (span as i64, splits as i32);
        unsafe {
            self.context
                .stream
                .launch_builder(&self.group_partials)
                .arg(&map.tensor.data)
                .arg(&mut partials)
                .arg(&span64)
                .arg(&splits32)
                .launch(LaunchConfig {
                    grid_dim: (splits as u32, norm.groups as u32, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(cuda_err("group statistics kernel"))?;
            let (groups, span) = (norm.groups as i32, span as f64);
            self.context
                .stream
                .launch_builder(&self.group_fold)
                .arg(&partials)
                .arg(&mut stats)
                .arg(&groups)
                .arg(&splits32)
                .arg(&span)
                .arg(&norm.eps)
                .launch(cfg(norm.groups))
                .map_err(cuda_err("group fold kernel"))?;
        }
        Ok((stats, per_group))
    }

    fn layer_norm(
        &self,
        norm: &DeviceNorm,
        tokens: &Tensor,
        eps: f32,
    ) -> Result<Tensor, NetworkError> {
        let mut output = self.uninit(tokens.rows, tokens.cols)?;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.layer_norm)
                .arg(&mut output.data)
                .arg(&tokens.data)
                .arg(&norm.weight)
                .arg(&norm.bias)
                .arg(&(tokens.cols as i32))
                .arg(&eps)
                .launch(grid(tokens.rows))
                .map_err(cuda_err("layer norm kernel"))?;
        }
        Ok(output)
    }

    /// `kind` is 0 for SiLU, 1 for GELU, 2 for the quick GELU.
    fn activate(&self, tensor: &mut Tensor, kind: i32) -> Result<(), NetworkError> {
        let len = tensor.len();
        unsafe {
            self.context
                .stream
                .launch_builder(&self.activate)
                .arg(&mut tensor.data)
                .arg(&(len as i64))
                .arg(&kind)
                .launch(lanes(len, 1))
                .map_err(cuda_err("activation kernel"))?;
        }
        Ok(())
    }

    fn add(&self, target: &mut Tensor, source: &Tensor) -> Result<(), NetworkError> {
        let len = target.len();
        if len != source.len() {
            return Err(NetworkError::InvalidTarget {
                expected: len,
                actual: source.len(),
            });
        }
        unsafe {
            self.context
                .stream
                .launch_builder(&self.add_inplace)
                .arg(&mut target.data)
                .arg(&source.data)
                .arg(&(len as i64))
                .launch(lanes(len, 1))
                .map_err(cuda_err("add kernel"))?;
        }
        Ok(())
    }

    /// One value per channel, added over that channel's whole plane.
    fn add_channel(&self, map: &mut Map, offsets: &Tensor) -> Result<(), NetworkError> {
        let shape = lanes(map.pixels(), map.channels());
        let pixels = map.pixels() as i64;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.add_channel)
                .arg(&mut map.tensor.data)
                .arg(&offsets.data)
                .arg(&pixels)
                .launch(shape)
                .map_err(cuda_err("channel offset kernel"))?;
        }
        Ok(())
    }

    /// The same for a bias that stayed `f32`, which is the convolution's.
    fn add_channel_f32(
        &self,
        tensor: &mut Tensor,
        bias: &CudaSlice<f32>,
        pixels: usize,
    ) -> Result<(), NetworkError> {
        let shape = lanes(pixels, tensor.len() / pixels.max(1));
        let pixels = pixels as i64;
        unsafe {
            self.context
                .stream
                .launch_builder(&self.add_plane_bias)
                .arg(&mut tensor.data)
                .arg(bias)
                .arg(&pixels)
                .launch(shape)
                .map_err(cuda_err("channel bias kernel"))?;
        }
        Ok(())
    }

    fn to_tokens(&self, map: &Map) -> Result<Tensor, NetworkError> {
        let mut output = self.uninit(map.pixels(), map.channels())?;
        self.transpose(&map.tensor, &mut output)?;
        Ok(output)
    }

    fn to_map(&self, tokens: &Tensor, height: usize, width: usize) -> Result<Map, NetworkError> {
        let mut output = self.uninit(tokens.cols, tokens.rows)?;
        self.transpose(tokens, &mut output)?;
        Ok(Map {
            tensor: output,
            height,
            width,
        })
    }

    fn transpose(&self, input: &Tensor, output: &mut Tensor) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.transpose)
                .arg(&mut output.data)
                .arg(&input.data)
                .arg(&(input.rows as i64))
                .arg(&(input.cols as i64))
                .launch(cfg(input.len()))
                .map_err(cuda_err("transpose kernel"))?;
        }
        Ok(())
    }

    fn upsample_nearest(&self, map: &Map, factor: usize) -> Result<Map, NetworkError> {
        let (height, width) = (map.height * factor, map.width * factor);
        let mut output = self.uninit(map.channels(), height * width)?;
        let len = output.len();
        unsafe {
            self.context
                .stream
                .launch_builder(&self.upsample)
                .arg(&mut output.data)
                .arg(&map.tensor.data)
                .arg(&(len as i64))
                .arg(&(map.height as i32))
                .arg(&(map.width as i32))
                .arg(&(factor as i32))
                .launch(cfg(len))
                .map_err(cuda_err("upsample kernel"))?;
        }
        Ok(Map {
            tensor: output,
            height,
            width,
        })
    }

    /// Two maps of the same size, stacked along their channels.
    fn concatenate(&self, first: &Map, second: &Map) -> Result<Map, NetworkError> {
        let mut output = self.uninit(first.channels() + second.channels(), first.pixels())?;
        let split = first.tensor.len();
        self.context
            .stream
            .memcpy_dtod(&first.tensor.data, &mut output.data.slice_mut(..split))
            .map_err(cuda_err("device to device copy"))?;
        self.context
            .stream
            .memcpy_dtod(
                &second.tensor.data,
                &mut output.data.slice_mut(split..split + second.tensor.len()),
            )
            .map_err(cuda_err("device to device copy"))?;
        Ok(Map {
            tensor: output,
            height: first.height,
            width: first.width,
        })
    }

    /// The gated feed-forward, taking the projection's bias rather than a
    /// tensor that already has it: see the kernel.
    fn gelu_gate(
        &self,
        projected: &Tensor,
        bias: Option<&CudaSlice<f32>>,
    ) -> Result<Tensor, NetworkError> {
        let inner = projected.cols / 2;
        let mut output = self.uninit(projected.rows, inner)?;
        let shape = lanes(inner, projected.rows);
        let null = 0u64;
        unsafe {
            let mut launch = self.context.stream.launch_builder(&self.gelu_gate);
            launch.arg(&mut output.data).arg(&projected.data);
            match bias {
                Some(bias) => launch.arg(bias),
                None => launch.arg(&null),
            };
            launch
                .arg(&(inner as i32))
                .launch(shape)
                .map_err(cuda_err("gated feed-forward kernel"))?;
        }
        Ok(output)
    }

    /// Multi-head attention over `[tokens, heads * head_dim]` operands.
    ///
    /// No head is ever gathered into its own buffer: a head is a slice of every
    /// row, so the batch stride of a strided-batched GEMM steps from one head
    /// to the next while the leading dimension steps from one token to the
    /// next. The queries are chunked so the score matrix stays bounded, which
    /// is what makes a 16384-pixel VAE attention finish in megabytes.
    fn attention(
        &self,
        queries: View<'_>,
        keys: View<'_>,
        values: View<'_>,
        heads: usize,
        causal: bool,
    ) -> Result<Tensor, NetworkError> {
        let (tokens, context) = (queries.rows, keys.rows);
        let head_dim = queries.cols / heads;
        if head_dim * heads != queries.cols || keys.cols != queries.cols {
            return Err(NetworkError::InvalidConfig(format!(
                "{} query columns and {} key columns do not divide into {heads} heads",
                queries.cols, keys.cols
            )));
        }
        let scale = (head_dim as f32).sqrt().recip();
        // The fused kernel keeps a tile of scores in registers and never
        // writes the `[tokens, context]` matrix at all, which is three passes
        // over the largest buffer in the pass. It knows one head width, it has
        // nothing to say about a masked row, and it reaches the values through
        // the key pointer, so the two have to be windows on one buffer -- which
        // is what a fused QKV projection hands it.
        if let Some(flash) = self.context.flash.as_ref().filter(|_| {
            head_dim == crate::cuda_flash::TILE
                && !causal
                && std::ptr::eq(keys.data, values.data)
                && keys.lead == values.lead
                && keys.lead % 8 == 0
                && !crate::cuda_flash::disabled()
        }) {
            // The kernel exponentiates in base two, so the scale carries the
            // change of base and no score pays for it.
            let log2_scale = scale * std::f32::consts::LOG2_E;
            let mut output = self.uninit(tokens, queries.cols)?;
            let config = cudarc::driver::LaunchConfig {
                grid_dim: (
                    tokens.div_ceil(crate::cuda_flash::IMAGE_ROWS) as u32,
                    heads as u32,
                    1,
                ),
                block_dim: (crate::cuda_flash::IMAGE_THREADS, 1, 1),
                shared_mem_bytes: crate::cuda_flash::SHARED_BYTES,
            };
            unsafe {
                self.context
                    .stream
                    .launch_builder(&flash.image)
                    .arg(queries.data)
                    .arg(keys.data)
                    .arg(&mut output.data)
                    .arg(&(tokens as i32))
                    .arg(&(context as i32))
                    .arg(&(queries.lead as i32))
                    .arg(&(keys.lead as i32))
                    .arg(&(output.cols as i32))
                    .arg(&(queries.base as i32))
                    .arg(&(keys.base as i32))
                    .arg(&(values.base as i32))
                    .arg(&log2_scale)
                    .launch(config)
                    .map_err(cuda_err("fused image attention kernel"))?;
            }
            return Ok(output);
        }
        let chunk = (SCORE_BUDGET / (heads * context).max(1)).clamp(1, tokens);
        let mut scores = self.uninit(heads * chunk, context)?;
        let mut output = self.uninit(tokens, queries.cols)?;
        let wide = !causal && self.wide_scores.load(std::sync::atomic::Ordering::Relaxed);
        let mut wide_scores = match wide {
            true => Some(
                unsafe { self.context.stream.alloc::<f32>(heads * chunk * context) }.map_err(
                    cuda_alloc_err("device allocation", heads * chunk * context * 4),
                )?,
            ),
            false => None,
        };

        for first in (0..tokens).step_by(chunk) {
            let rows = chunk.min(tokens - first);
            if let Some(wide) = wide_scores.as_mut() {
                let shape = (heads, head_dim, first, rows, context);
                self.wide_score_gemm(queries, keys, wide, shape, scale)?;
                let lines = heads * rows;
                unsafe {
                    self.context
                        .stream
                        .launch_builder(&self.softmax_wide)
                        .arg(&*wide)
                        .arg(&mut scores.data)
                        .arg(&(context as i32))
                        .launch(grid(lines))
                }
                .map_err(cuda_err("attention softmax kernel"))?;
                self.merge(
                    &scores,
                    values,
                    &mut output,
                    heads,
                    head_dim,
                    first,
                    rows,
                    context,
                )?;
                continue;
            }
            self.scores(
                queries,
                keys,
                &mut scores,
                heads,
                head_dim,
                first,
                rows,
                context,
                scale,
            )?;
            self.softmax(&mut scores, heads, rows, first, context, causal)?;
            self.merge(
                &scores,
                values,
                &mut output,
                heads,
                head_dim,
                first,
                rows,
                context,
            )?;
        }
        Ok(output)
    }

    /// Each of the `heads * rows` score rows of a chunk starting at query
    /// `first`, in place, to its softmax.
    fn softmax(
        &self,
        scores: &mut Tensor,
        heads: usize,
        rows: usize,
        first: usize,
        context: usize,
        causal: bool,
    ) -> Result<(), NetworkError> {
        // Nothing is masked and the row packs into whole quads, so the
        // softmax can move four scores per load. A short row goes to a warp,
        // a long one to a whole block.
        let packed = !causal && context % 4 == 0;
        let by_warp = !causal && context <= WARP_SCORES;
        let lines = heads * rows;
        unsafe {
            match by_warp {
                true => self
                    .context
                    .stream
                    .launch_builder(&self.softmax_warp)
                    .arg(&mut scores.data)
                    .arg(&(context as i32))
                    .arg(&(lines as i32))
                    .launch(grid(lines.div_ceil(THREADS as usize / 32))),
                false => self
                    .context
                    .stream
                    .launch_builder(&self.softmax_rows)
                    .arg(&mut scores.data)
                    .arg(&(context as i32))
                    .arg(&(rows as i32))
                    .arg(&i32::from(causal))
                    .arg(&(first as i32))
                    .arg(&i32::from(packed))
                    .launch(grid(lines)),
            }
            .map_err(cuda_err("attention softmax kernel"))?;
        }
        Ok(())
    }

    /// `scores[h, i, j] = scale * q[first + i, h] . k[j, h]`.
    #[allow(clippy::too_many_arguments)]
    fn scores(
        &self,
        queries: View<'_>,
        keys: View<'_>,
        scores: &mut Tensor,
        heads: usize,
        head_dim: usize,
        first: usize,
        rows: usize,
        context: usize,
        scale: f32,
    ) -> Result<(), NetworkError> {
        let (out_pointer, _out) = scores.data.device_ptr_mut(&self.context.stream);
        let shape = (heads, head_dim, first, rows, context);
        self.score_gemm(queries, keys, out_pointer, false, shape, scale)
    }

    /// [`ImageGpu::scores`] into `f32`, summed in `f32`.
    fn wide_score_gemm(
        &self,
        queries: View<'_>,
        keys: View<'_>,
        scores: &mut CudaSlice<f32>,
        shape: (usize, usize, usize, usize, usize),
        scale: f32,
    ) -> Result<(), NetworkError> {
        let (out_pointer, _out) = scores.device_ptr_mut(&self.context.stream);
        self.score_gemm(queries, keys, out_pointer, true, shape, scale)
    }

    fn score_gemm(
        &self,
        queries: View<'_>,
        keys: View<'_>,
        out_pointer: u64,
        wide: bool,
        (heads, head_dim, first, rows, context): (usize, usize, usize, usize, usize),
        scale: f32,
    ) -> Result<(), NetworkError> {
        let query = queries.data.slice(queries.base + first * queries.lead..);
        let key = keys.data.slice(keys.base..);
        let (key_pointer, _key) = key.device_ptr(&self.context.stream);
        let (query_pointer, _query) = query.device_ptr(&self.context.stream);
        let (scale, zero) = (f16::from_f32(scale), f16::ZERO);
        let (wide_scale, wide_zero) = (scale.to_f32(), 0f32);
        let (alpha, beta, out_type, compute) = match wide {
            true => (
                &wide_scale as *const f32 as *const std::ffi::c_void,
                &wide_zero as *const f32 as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            ),
            false => (
                &scale as *const f16 as *const std::ffi::c_void,
                &zero as *const f16 as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
            ),
        };
        unsafe {
            cublas::gemm_strided_batched_ex(
                *self.context.blas.handle(),
                cublasOperation_t::CUBLAS_OP_T,
                cublasOperation_t::CUBLAS_OP_N,
                context as i32,
                rows as i32,
                head_dim as i32,
                alpha,
                key_pointer as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                keys.lead as i32,
                head_dim as i64,
                query_pointer as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                queries.lead as i32,
                head_dim as i64,
                beta,
                out_pointer as *mut std::ffi::c_void,
                out_type,
                context as i32,
                (rows * context) as i64,
                heads as i32,
                compute,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(cuda_err("attention score GEMM"))
    }

    /// `out[first + i, h] = p[h, i, ..] . v[.., h]`, written straight into the
    /// merged output's slice for that head.
    #[allow(clippy::too_many_arguments)]
    fn merge(
        &self,
        probabilities: &Tensor,
        values: View<'_>,
        output: &mut Tensor,
        heads: usize,
        head_dim: usize,
        first: usize,
        rows: usize,
        context: usize,
    ) -> Result<(), NetworkError> {
        let width = output.cols;
        let mut target = output.data.slice_mut(first * width..);
        let value = values.data.slice(values.base..);
        let (value_pointer, _value) = value.device_ptr(&self.context.stream);
        let (probability_pointer, _probability) =
            probabilities.data.device_ptr(&self.context.stream);
        let (out_pointer, _out) = target.device_ptr_mut(&self.context.stream);
        let (one, zero) = (f16::ONE, f16::ZERO);
        unsafe {
            cublas::gemm_strided_batched_ex(
                *self.context.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                head_dim as i32,
                rows as i32,
                context as i32,
                &one as *const f16 as *const std::ffi::c_void,
                value_pointer as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                values.lead as i32,
                head_dim as i64,
                probability_pointer as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                context as i32,
                (rows * context) as i64,
                &zero as *const f16 as *const std::ffi::c_void,
                out_pointer as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                width as i32,
                head_dim as i64,
                heads as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(cuda_err("attention merge GEMM"))
    }
}

// ---------------------------------------------------------------------------
// The UNet.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The backward operations, for adapters trained on a frozen UNet.
// ---------------------------------------------------------------------------

impl ImageGpu {
    /// `dL/d(input)` of [`ImageGpu::conv`] for an input of `height` by
    /// `width`: `weight^T . grad` lowered to columns, a chunk of output rows at
    /// a time within `budget` FP16 elements, and each column added back where
    /// `im2col` read it from.
    fn conv_backward(
        &self,
        conv: &DeviceConv,
        grad: &Map,
        height: usize,
        width: usize,
        budget: usize,
    ) -> Result<Map, NetworkError> {
        if grad.channels() != conv.out_channels {
            return Err(NetworkError::InvalidTarget {
                expected: conv.out_channels,
                actual: grad.channels(),
            });
        }
        let (k, inputs, pixels) = (conv.kernel, conv.in_channels, grad.pixels());
        // `columns[patch, span] = weight[out, patch]^T . grad[out, span]`.
        let lower = |weight: &CudaView<'_, u8>,
                     patch: usize,
                     first: usize,
                     span: usize,
                     columns: &mut CudaSlice<f16>,
                     pitch: usize| {
            act_lhs_transposed::<f16, _>(
                &self.context,
                weight,
                patch,
                &grad.tensor.bytes().slice(first * 2..),
                pixels,
                true,
                true,
                columns,
                pitch,
                conv.out_channels,
                patch,
                span,
                1.0,
                0.0,
            )
        };
        if k == 1 && conv.stride == 1 && conv.padding == 0 {
            let mut output = self.uninit(inputs, pixels)?;
            lower(
                &conv.weight.bytes(),
                inputs,
                0,
                pixels,
                &mut output.data,
                pixels,
            )?;
            return Ok(Map {
                tensor: output,
                height,
                width,
            });
        }
        let mut sums = self
            .context
            .stream
            .alloc_zeros::<f32>(inputs * height * width)
            .map_err(cuda_err("device allocation"))?;
        let rows_of = |patch: usize| {
            let group = if grad.width % 8 == 0 { 1 } else { 8 };
            (budget / (patch * grad.width).max(1))
                .max(group)
                .next_multiple_of(group)
                .min(grad.height)
        };
        // Row-wise weights are one `[out, in * k]` matrix per vertical tap, so
        // each tap lowers and scatters on its own; `im2col` weights do all
        // taps at once.
        let (patch, taps): (usize, Vec<i32>) = match rowwise(k, conv.stride) {
            true => (inputs * k, (0..k as i32).collect()),
            false => (inputs * k * k, vec![-1]),
        };
        let stack = rows_of(patch);
        let pitch = stack * grad.width;
        let mut columns = self.uninit(patch, pitch)?;
        let weights = conv.weight.bytes();
        let tap_bytes = conv.out_channels * patch * 2;
        for row in (0..grad.height).step_by(stack) {
            let span = stack.min(grad.height - row) * grad.width;
            for &tap in &taps {
                let weight = match tap {
                    -1 => weights.slice(..),
                    _ => weights.slice(tap as usize * tap_bytes..(tap as usize + 1) * tap_bytes),
                };
                lower(
                    &weight,
                    patch,
                    row * grad.width,
                    span,
                    &mut columns.data,
                    pitch,
                )?;
                unsafe {
                    self.context
                        .stream
                        .launch_builder(&self.col2im)
                        .arg(&mut sums)
                        .arg(&columns.data)
                        .arg(&(height as i32))
                        .arg(&(width as i32))
                        .arg(&(k as i32))
                        .arg(&(conv.stride as i32))
                        .arg(&(conv.padding as i32))
                        .arg(&(grad.width as i32))
                        .arg(&(row as i32))
                        .arg(&(span as i32))
                        .arg(&(pitch as i64))
                        .arg(&tap)
                        .launch(plane(span, inputs))
                        .map_err(cuda_err("col2im kernel"))?;
                }
            }
        }
        Ok(Map {
            tensor: self.narrowed(&sums, inputs, height * width)?,
            height,
            width,
        })
    }

    /// FP32 `sums` as a `[rows, cols]` FP16 tensor.
    fn narrowed(
        &self,
        sums: &CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<Tensor, NetworkError> {
        let mut output = self.uninit(rows, cols)?;
        let len = output.len();
        unsafe {
            self.context
                .stream
                .launch_builder(&self.narrow)
                .arg(&mut output.data)
                .arg(sums)
                .arg(&(len as i64))
                .launch(lanes(len, 1))
                .map_err(cuda_err("narrowing kernel"))?;
        }
        Ok(output)
    }

    /// `dL/dx` of [`ImageGpu::group_norm`] with no activation, given its
    /// input, in place over `grad`.
    fn group_norm_backward(
        &self,
        norm: &DeviceGroupNorm,
        input: &Map,
        grad: &mut Map,
    ) -> Result<(), NetworkError> {
        let (stats, per_group) = self.group_stats(norm, input)?;
        let span = per_group * input.pixels();
        let splits = (span / (THREADS as usize * 16)).clamp(1, 128);
        let mut partials = self
            .context
            .stream
            .alloc_zeros::<f32>(2 * norm.groups * splits)
            .map_err(cuda_err("device allocation"))?;
        let (pixels, per_group, splits) = (input.pixels() as i64, per_group as i32, splits as i32);
        unsafe {
            self.context
                .stream
                .launch_builder(&self.group_bwd_partials)
                .arg(&input.tensor.data)
                .arg(&grad.tensor.data)
                .arg(&norm.weight)
                .arg(&stats)
                .arg(&mut partials)
                .arg(&pixels)
                .arg(&per_group)
                .arg(&splits)
                .launch(LaunchConfig {
                    grid_dim: (splits as u32, norm.groups as u32, 1),
                    block_dim: (THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(cuda_err("group norm backward kernel"))?;
            self.context
                .stream
                .launch_builder(&self.group_bwd_apply)
                .arg(&mut grad.tensor.data)
                .arg(&input.tensor.data)
                .arg(&norm.weight)
                .arg(&stats)
                .arg(&partials)
                .arg(&pixels)
                .arg(&per_group)
                .arg(&(norm.groups as i32))
                .arg(&splits)
                .arg(&(span as f32))
                .launch(lanes(input.pixels(), input.channels()))
                .map_err(cuda_err("group norm backward kernel"))?;
        }
        Ok(())
    }

    /// `dL/dx` of [`ImageGpu::layer_norm`], given its input, in place over
    /// `grad`.
    fn layer_norm_backward(
        &self,
        norm: &DeviceNorm,
        tokens: &Tensor,
        eps: f32,
        grad: &mut Tensor,
    ) -> Result<(), NetworkError> {
        unsafe {
            self.context
                .stream
                .launch_builder(&self.layer_norm_bwd)
                .arg(&mut grad.data)
                .arg(&tokens.data)
                .arg(&norm.weight)
                .arg(&(tokens.cols as i32))
                .arg(&eps)
                .launch(grid(tokens.rows))
                .map_err(cuda_err("layer norm backward kernel"))?;
        }
        Ok(())
    }

    /// `dL/d(projected)` of [`ImageGpu::gelu_gate`].
    fn gelu_gate_backward(
        &self,
        projected: &Tensor,
        bias: Option<&CudaSlice<f32>>,
        grad: &Tensor,
    ) -> Result<Tensor, NetworkError> {
        let inner = grad.cols;
        let mut output = self.uninit(projected.rows, projected.cols)?;
        let null = 0u64;
        unsafe {
            let mut launch = self.context.stream.launch_builder(&self.gelu_gate_bwd);
            launch.arg(&mut output.data).arg(&projected.data);
            match bias {
                Some(bias) => launch.arg(bias),
                None => launch.arg(&null),
            };
            launch
                .arg(&grad.data)
                .arg(&(inner as i32))
                .launch(lanes(inner, projected.rows))
                .map_err(cuda_err("gated GELU backward kernel"))?;
        }
        Ok(output)
    }

    /// `grad *= silu'(pre)`.
    fn silu_backward(&self, grad: &mut Tensor, pre: &Tensor) -> Result<(), NetworkError> {
        let len = grad.len();
        unsafe {
            self.context
                .stream
                .launch_builder(&self.silu_bwd)
                .arg(&mut grad.data)
                .arg(&pre.data)
                .arg(&(len as i64))
                .launch(lanes(len, 1))
                .map_err(cuda_err("SiLU backward kernel"))?;
        }
        Ok(())
    }

    /// `dL/d(map)` of [`ImageGpu::upsample_nearest`].
    fn upsample_backward(&self, grad: &Map, factor: usize) -> Result<Map, NetworkError> {
        let (height, width) = (grad.height / factor, grad.width / factor);
        let mut output = self.uninit(grad.channels(), height * width)?;
        let len = output.len();
        unsafe {
            self.context
                .stream
                .launch_builder(&self.upsample_bwd)
                .arg(&mut output.data)
                .arg(&grad.tensor.data)
                .arg(&(len as i64))
                .arg(&(height as i32))
                .arg(&(width as i32))
                .arg(&(factor as i32))
                .launch(cfg(len))
                .map_err(cuda_err("upsample backward kernel"))?;
        }
        Ok(Map {
            tensor: output,
            height,
            width,
        })
    }

    /// `dL/dq`, `dL/dk` and `dL/dv` of unmasked [`ImageGpu::attention`],
    /// given `dL/d(output)`. The probabilities are worked out again, a chunk
    /// of query rows at a time within `budget` scores; the key and value
    /// gradients sum over every chunk, so they stay FP32 until the end.
    fn attention_backward(
        &self,
        queries: View<'_>,
        keys: View<'_>,
        values: View<'_>,
        heads: usize,
        grad: &Tensor,
        budget: usize,
    ) -> Result<[Tensor; 3], NetworkError> {
        let (tokens, context, units) = (queries.rows, keys.rows, queries.cols);
        let head_dim = units / heads;
        let scale = (head_dim as f32).sqrt().recip();
        let chunk = (budget / (heads * context).max(1)).clamp(1, tokens);
        let mut probabilities = self.uninit(heads * chunk, context)?;
        let mut grad_scores = self.uninit(heads * chunk, context)?;
        let mut grad_queries = self.uninit(tokens, units)?;
        let zeros = || {
            let stream = &self.context.stream;
            stream
                .alloc_zeros::<f32>(context * units)
                .map_err(cuda_err("device allocation"))
        };
        let (mut grad_keys, mut grad_values) = (zeros()?, zeros()?);
        for first in (0..tokens).step_by(chunk) {
            let rows = chunk.min(tokens - first);
            let shape = (heads, head_dim, first, rows, context);
            self.scores(
                queries,
                keys,
                &mut probabilities,
                heads,
                head_dim,
                first,
                rows,
                context,
                scale,
            )?;
            self.softmax(&mut probabilities, heads, rows, first, context, false)?;
            self.scores(
                grad.view(),
                values,
                &mut grad_scores,
                heads,
                head_dim,
                first,
                rows,
                context,
                1.0,
            )?;
            self.sum_transposed(&probabilities, grad.view(), &mut grad_values, units, shape)?;
            let lines = (heads * rows) as i32;
            unsafe {
                self.context
                    .stream
                    .launch_builder(&self.softmax_bwd)
                    .arg(&mut grad_scores.data)
                    .arg(&probabilities.data)
                    .arg(&(context as i32))
                    .arg(&lines)
                    .arg(&scale)
                    .launch(grid((lines as usize).div_ceil(THREADS as usize / 32)))
                    .map_err(cuda_err("softmax backward kernel"))?;
            }
            self.merge(
                &grad_scores,
                keys,
                &mut grad_queries,
                heads,
                head_dim,
                first,
                rows,
                context,
            )?;
            self.sum_transposed(&grad_scores, queries, &mut grad_keys, units, shape)?;
        }
        Ok([
            grad_queries,
            self.narrowed(&grad_keys, context, units)?,
            self.narrowed(&grad_values, context, units)?,
        ])
    }

    /// `out[j, h] += sum_i p[h, i, j] * x[first + i, h]` over a chunk's
    /// `rows`, in FP32: the per-head product that sums a chunk of attention
    /// back onto the keys and values.
    fn sum_transposed(
        &self,
        probabilities: &Tensor,
        rows_of: View<'_>,
        out: &mut CudaSlice<f32>,
        units: usize,
        (heads, head_dim, first, rows, context): (usize, usize, usize, usize, usize),
    ) -> Result<(), NetworkError> {
        let source = rows_of.data.slice(rows_of.base + first * rows_of.lead..);
        self.sum_lhs_transposed(
            (&probabilities.data.slice(..), context, rows * context),
            (&source, rows_of.lead, head_dim),
            (out, units, head_dim),
            (rows, context, head_dim, heads),
        )
    }

    /// `out[n, m] += a[k, n]^T . b[k, m]` in FP32, `batch` times. Each
    /// operand is its buffer, its row pitch and its step from one batch entry
    /// to the next, in elements.
    fn sum_lhs_transposed(
        &self,
        (a, a_lead, a_step): (&CudaView<'_, f16>, usize, usize),
        (b, b_lead, b_step): (&CudaView<'_, f16>, usize, usize),
        (out, out_lead, out_step): (&mut CudaSlice<f32>, usize, usize),
        (k, n, m, batch): (usize, usize, usize, usize),
    ) -> Result<(), NetworkError> {
        let (a_pointer, _a) = a.device_ptr(&self.context.stream);
        let (b_pointer, _b) = b.device_ptr(&self.context.stream);
        let (out_pointer, _out) = out.device_ptr_mut(&self.context.stream);
        let one = 1.0f32;
        unsafe {
            cublas::gemm_strided_batched_ex(
                *self.context.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_T,
                m as i32,
                n as i32,
                k as i32,
                &one as *const f32 as *const std::ffi::c_void,
                b_pointer as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                b_lead as i32,
                b_step as i64,
                a_pointer as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                a_lead as i32,
                a_step as i64,
                &one as *const f32 as *const std::ffi::c_void,
                out_pointer as *mut std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                out_lead as i32,
                out_step as i64,
                batch as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
        }
        .map_err(cuda_err("FP32 accumulating GEMM"))
    }
}

struct DeviceResnet {
    norm1: DeviceGroupNorm,
    conv1: DeviceConv,
    time: DeviceDense,
    norm2: DeviceGroupNorm,
    conv2: DeviceConv,
    shortcut: Option<DeviceConv>,
}

struct DeviceAttention {
    /// The key and value weights, with the query weight joining them when it
    /// reads the same width — which is every self-attention.
    stacked: DeviceStack,
    /// Present only when the queries read the image and the keys read the
    /// prompt, so the query weight could not join the stack.
    query: Option<DeviceDense>,
    output: DeviceDense,
}

struct DeviceTransformerBlock {
    norm1: DeviceNorm,
    attention: DeviceAttention,
    norm2: DeviceNorm,
    cross: DeviceAttention,
    norm3: DeviceNorm,
    gate: DeviceDense,
    output: DeviceDense,
}

struct DeviceTransformer {
    norm: DeviceGroupNorm,
    input: DeviceProjection,
    blocks: Vec<DeviceTransformerBlock>,
    output: DeviceProjection,
    heads: usize,
}

struct DeviceBlock {
    resnets: Vec<DeviceResnet>,
    attentions: Vec<DeviceTransformer>,
    resampler: Option<DeviceConv>,
}

/// A UNet, resident on the device. A ControlNet's copy of the way down is
/// one with no way up.
pub struct DeviceUnet {
    gpu: Arc<ImageGpu>,
    conv_in: DeviceConv,
    down: Vec<DeviceBlock>,
    middle: (DeviceResnet, DeviceTransformer, DeviceResnet),
    up: Vec<DeviceBlock>,
    out: Option<(DeviceGroupNorm, DeviceConv)>,
    eps: f32,
}

impl std::fmt::Debug for DeviceUnet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceUnet").finish_non_exhaustive()
    }
}

/// A ControlNet on the device, with its strength folded into the
/// convolutions its corrections leave through.
pub struct DeviceControlNet {
    body: DeviceUnet,
    hint: Vec<DeviceConv>,
    zero: Vec<DeviceConv>,
    zero_middle: DeviceConv,
}

impl std::fmt::Debug for DeviceControlNet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceControlNet").finish_non_exhaustive()
    }
}

impl DeviceControlNet {
    pub fn upload(
        gpu: &Arc<ImageGpu>,
        control: &crate::unet::Control,
    ) -> Result<Self, NetworkError> {
        let net = &control.net;
        let scaled = |conv: &Conv2d| {
            let mut conv = conv.clone();
            conv.weight
                .data
                .iter_mut()
                .for_each(|value| *value *= control.scale);
            if let Some(bias) = &mut conv.bias {
                bias.iter_mut().for_each(|value| *value *= control.scale);
            }
            upload_conv(gpu, &conv)
        };
        Ok(Self {
            body: DeviceUnet::upload_encoder(gpu, &net.encoder, net.config.eps)?,
            hint: net
                .hint
                .iter()
                .map(|conv| upload_conv(gpu, conv))
                .collect::<Result<_, _>>()?,
            zero: net.zero.iter().map(scaled).collect::<Result<_, _>>()?,
            zero_middle: scaled(&net.zero_middle)?,
        })
    }
}

impl DeviceUnet {
    /// Uploads every weight once. A failure part way through drops whatever was
    /// already on the device, so the caller is left on the CPU path intact.
    pub fn upload(gpu: &Arc<ImageGpu>, model: &Unet) -> Result<Self, NetworkError> {
        Ok(Self {
            up: model
                .up
                .iter()
                .map(|block| upload_block(gpu, block))
                .collect::<Result<_, _>>()?,
            out: Some((
                upload_group_norm(gpu, &model.norm_out)?,
                upload_conv(gpu, &model.conv_out)?,
            )),
            ..Self::upload_encoder(gpu, &model.encoder, model.config().eps)?
        })
    }

    pub(crate) fn gpu(&self) -> &Arc<ImageGpu> {
        &self.gpu
    }

    fn upload_encoder(
        gpu: &Arc<ImageGpu>,
        encoder: &crate::unet::Encoder,
        eps: f32,
    ) -> Result<Self, NetworkError> {
        Ok(Self {
            conv_in: upload_conv(gpu, &encoder.conv_in)?,
            down: encoder
                .down
                .iter()
                .map(|block| upload_block(gpu, block))
                .collect::<Result<_, _>>()?,
            middle: (
                upload_resnet(gpu, &encoder.middle.0)?,
                upload_transformer(gpu, &encoder.middle.1)?,
                upload_resnet(gpu, &encoder.middle.2)?,
            ),
            up: Vec::new(),
            out: None,
            eps,
            gpu: gpu.clone(),
        })
    }

    /// The ladder, from the stem to the output convolution.
    ///
    /// `latent` is already scaled for the noise level and `time` is already the
    /// embedded, conditioned vector: both are a handful of flops that the host
    /// does once per pass, and keeping them there means the device path and the
    /// CPU path share that code rather than reimplementing it. Each control
    /// comes with its own embedded time and its picture, and reads `noisy`,
    /// the latent without an inpainting checkpoint's extra channels.
    pub fn forward(
        &self,
        latent: &FeatureMap,
        time: &[f32],
        text: &Matrix,
        noisy: &FeatureMap,
        controls: &[(&DeviceControlNet, Vec<f32>, &FeatureMap)],
    ) -> Result<FeatureMap, NetworkError> {
        let gpu = &self.gpu;
        let time = self.time(time)?;
        let text = gpu.upload_matrix(text)?;
        let (mut sample, mut skips) = self.down(latent, &time, &text, None, None)?;

        for (net, control_time, image) in controls {
            let mut hint = upload_map(gpu, image)?;
            let (last, stack) = net.hint.split_last().expect("a hint stack has an output");
            for conv in stack {
                hint = gpu.conv(conv, &hint)?;
                gpu.activate(&mut hint.tensor, 0)?;
            }
            let (middle, corrections) = net.body.down(
                noisy,
                &net.body.time(control_time)?,
                &text,
                Some((last, &hint)),
                None,
            )?;
            if corrections.len() != skips.len() || corrections.len() != net.zero.len() {
                return Err(NetworkError::InvalidConfig(
                    "the ControlNet does not have the UNet's rungs".into(),
                ));
            }
            for ((skip, correction), zero) in skips.iter_mut().zip(&corrections).zip(&net.zero) {
                gpu.conv_add(zero, correction, skip)?;
            }
            gpu.conv_add(&net.zero_middle, &middle, &mut sample)?;
        }

        let mut sample = self.up(sample, skips, &time, &text, None)?;
        let (norm_out, conv_out) = self.out.as_ref().expect("a UNet has an output");
        gpu.group_norm(norm_out, &mut sample, 0)?;
        let output = gpu.conv(conv_out, &sample)?;
        download_map(gpu, &output)
    }

    /// The way up, which reads the way down's rungs from the top.
    fn up(
        &self,
        mut sample: Map,
        mut skips: Vec<Map>,
        time: &Tensor,
        text: &Tensor,
        mut record: Option<&mut Recorder>,
    ) -> Result<Map, NetworkError> {
        let gpu = &self.gpu;
        for block in &self.up {
            for (index, resnet) in block.resnets.iter().enumerate() {
                let skip = skips.pop().ok_or_else(|| {
                    NetworkError::InvalidConfig(
                        "the way up has more rungs than the way down".into(),
                    )
                })?;
                if let Some(record) = record.as_deref_mut() {
                    record.records.joined.push(sample.channels());
                }
                let joined = gpu.concatenate(&sample, &skip)?;
                drop(skip);
                drop(sample);
                sample = self.resnet(resnet, joined, time, record.as_deref_mut())?;
                if let Some(transformer) = block.attentions.get(index) {
                    sample = self.transformer(transformer, &sample, text, record.as_deref_mut())?;
                }
            }
            if let Some(resampler) = &block.resampler {
                let large = gpu.upsample_nearest(&sample, 2)?;
                drop(sample);
                if let Some(record) = record.as_deref_mut() {
                    record.records.resampled.push((large.height, large.width));
                }
                sample = gpu.conv(resampler, &large)?;
            }
        }
        Ok(sample)
    }

    /// The embedded time on the device, through the SiLU every resnet reads
    /// it after.
    fn time(&self, time: &[f32]) -> Result<Tensor, NetworkError> {
        let mut time = self.gpu.upload(1, time.len(), time)?;
        self.gpu.activate(&mut time, 0)?;
        Ok(time)
    }

    /// The stem, the way down and the middle, keeping every rung's output for
    /// the way up. A ControlNet adds its picture after the stem, through the
    /// last convolution of its hint stack.
    fn down(
        &self,
        latent: &FeatureMap,
        time: &Tensor,
        text: &Tensor,
        hint: Option<(&DeviceConv, &Map)>,
        mut record: Option<&mut Recorder>,
    ) -> Result<(Map, Vec<Map>), NetworkError> {
        let gpu = &self.gpu;
        let mut sample = gpu.conv(&self.conv_in, &upload_map(gpu, latent)?)?;
        if let Some((conv, hint)) = hint {
            if (conv.out_channels, hint.height, hint.width)
                != (sample.channels(), sample.height, sample.width)
            {
                return Err(NetworkError::InvalidConfig(format!(
                    "a ControlNet picture that becomes {}x{} does not match a {}x{} latent; it is \
                     the size of the image being made",
                    hint.height, hint.width, sample.height, sample.width
                )));
            }
            gpu.conv_add(conv, hint, &mut sample)?;
        }
        let mut skips = vec![clone_map(gpu, &sample)?];
        for block in &self.down {
            for (index, resnet) in block.resnets.iter().enumerate() {
                sample = self.resnet(resnet, sample, time, record.as_deref_mut())?;
                if let Some(transformer) = block.attentions.get(index) {
                    sample = self.transformer(transformer, &sample, text, record.as_deref_mut())?;
                }
                skips.push(clone_map(gpu, &sample)?);
            }
            if let Some(resampler) = &block.resampler {
                if let Some(record) = record.as_deref_mut() {
                    record.records.resampled.push((sample.height, sample.width));
                }
                sample = gpu.conv(resampler, &sample)?;
                skips.push(clone_map(gpu, &sample)?);
            }
        }
        sample = self.resnet(&self.middle.0, sample, time, record.as_deref_mut())?;
        sample = self.transformer(&self.middle.1, &sample, text, record.as_deref_mut())?;
        sample = self.resnet(&self.middle.2, sample, time, record)?;
        Ok((sample, skips))
    }

    /// Takes the input by value, for the reason
    /// [`DeviceVae::resnet`](crate::cuda_image::DeviceVae) does: the skip is
    /// built first so the normalization can run in place.
    fn resnet(
        &self,
        resnet: &DeviceResnet,
        input: Map,
        time: &Tensor,
        record: Option<&mut Recorder>,
    ) -> Result<Map, NetworkError> {
        let gpu = &self.gpu;
        let kept = match record {
            Some(_) => Some(clone_map(gpu, &input)?),
            None => None,
        };
        let (mut output, mut hidden) = match &resnet.shortcut {
            Some(shortcut) => (gpu.conv(shortcut, &input)?, input),
            None => {
                let copy = clone_map(gpu, &input)?;
                (input, copy)
            }
        };
        gpu.group_norm(&resnet.norm1, &mut hidden, 0)?;
        let mut next = gpu.conv(&resnet.conv1, &hidden)?;
        drop(hidden);

        // The noise level arrives as one number per channel.
        let offsets = gpu.dense(&resnet.time, time)?;
        gpu.add_channel(&mut next, &offsets)?;
        if let (Some(record), Some(input)) = (record, kept) {
            let hidden = clone_map(gpu, &next)?;
            record.records.resnets.push((input, hidden));
        }

        gpu.group_norm(&resnet.norm2, &mut next, 0)?;
        gpu.conv_add(&resnet.conv2, &next, &mut output)?;
        Ok(output)
    }

    fn transformer(
        &self,
        transformer: &DeviceTransformer,
        input: &Map,
        text: &Tensor,
        mut record: Option<&mut Recorder>,
    ) -> Result<Map, NetworkError> {
        let gpu = &self.gpu;
        let adapters = match record.as_deref_mut() {
            Some(record) => Some(record.next()?),
            None => None,
        };
        let mut blocks = Vec::new();
        let (height, width) = (input.height, input.width);
        let mut normed = clone_map(gpu, input)?;
        gpu.group_norm(&transformer.norm, &mut normed, -1)?;
        let mut tokens = match &transformer.input {
            DeviceProjection::Convolution(conv) => gpu.to_tokens(&gpu.conv(conv, &normed)?)?,
            DeviceProjection::Linear(dense) => gpu.dense(dense, &gpu.to_tokens(&normed)?)?,
        };

        for (index, block) in transformer.blocks.iter().enumerate() {
            let lora = adapters.map(|adapters| &adapters[index]);
            let keep = |tokens: &Tensor| match lora {
                Some(_) => Ok(Some(clone_tensor(gpu, tokens)?)),
                None => Ok::<_, NetworkError>(None),
            };
            let before_self = keep(&tokens)?;
            let normed = gpu.layer_norm(&block.norm1, &tokens, self.eps)?;
            let own = self.attend(
                &block.attention,
                &normed,
                None,
                transformer.heads,
                &mut tokens,
                lora.map(|lora| &lora[0]),
            )?;

            let before_cross = keep(&tokens)?;
            let normed = gpu.layer_norm(&block.norm2, &tokens, self.eps)?;
            let cross = self.attend(
                &block.cross,
                &normed,
                Some(text),
                transformer.heads,
                &mut tokens,
                lora.map(|lora| &lora[1]),
            )?;

            let before_feed = keep(&tokens)?;
            let normed = gpu.layer_norm(&block.norm3, &tokens, self.eps)?;
            let projected = gpu.dense_unbiased(&block.gate, &normed)?;
            let hidden = gpu.gelu_gate(&projected, block.gate.bias.as_ref())?;
            gpu.dense_add(&block.output, &hidden, &mut tokens)?;
            if let (Some(before_self), Some(before_cross), Some(before_feed)) =
                (before_self, before_cross, before_feed)
            {
                blocks.push(BlockRecord {
                    before: [before_self, before_cross, before_feed],
                    own: own.expect("an adapted attention keeps its record"),
                    cross: cross.expect("an adapted attention keeps its record"),
                    projected,
                });
            }
        }

        let mut output = match &transformer.output {
            DeviceProjection::Convolution(conv) => {
                gpu.conv(conv, &gpu.to_map(&tokens, height, width)?)?
            }
            DeviceProjection::Linear(dense) => {
                gpu.to_map(&gpu.dense(dense, &tokens)?, height, width)?
            }
        };
        gpu.add(&mut output.tensor, &input.tensor)?;
        if let Some(record) = record {
            let input = clone_map(gpu, input)?;
            record.records.transformers.push((input, blocks));
        }
        Ok(output)
    }

    /// With `lora` (the adapters on `to_q`, `to_k`, `to_v` and `to_out.0`,
    /// any of which may be absent) their products are added and what the
    /// backward pass reads comes back.
    #[allow(clippy::too_many_arguments)]
    fn attend(
        &self,
        attention: &DeviceAttention,
        tokens: &Tensor,
        context: Option<&Tensor>,
        heads: usize,
        residual: &mut Tensor,
        lora: Option<&[Option<DeviceLora>; 4]>,
    ) -> Result<Option<AttentionRecord>, NetworkError> {
        let gpu = &self.gpu;
        let stack = &attention.stacked;
        let units = stack.units();
        let adapter = |slot: usize| lora.and_then(|lora| lora[slot].as_ref());
        // One GEMM for every projection that reads the same tensor. Three
        // narrow products leave the card's last wave of tiles half empty; one
        // wide product does not, and it is also two fewer launches.
        let (attended, queries, projected) = match &attention.query {
            None => {
                let mut qkv = gpu.dense(&stack.dense, tokens)?;
                for slot in 0..3 {
                    if let Some(adapter) = adapter(slot) {
                        let target = &mut qkv.data.slice_mut(slot * units..);
                        gpu.lora_add(adapter, tokens, target, 3 * units)?;
                    }
                }
                let attended = gpu.attention(
                    qkv.part(0, units),
                    qkv.part(1, units),
                    qkv.part(2, units),
                    heads,
                    false,
                )?;
                (attended, None, qkv)
            }
            Some(query) => {
                let mut queries = gpu.dense(query, tokens)?;
                if let Some(adapter) = adapter(0) {
                    gpu.lora_add(adapter, tokens, &mut queries.data, units)?;
                }
                let source = context.unwrap_or(tokens);
                let mut kv = gpu.dense(&stack.dense, source)?;
                for slot in 1..3 {
                    if let Some(adapter) = adapter(slot) {
                        let target = &mut kv.data.slice_mut((slot - 1) * units..);
                        gpu.lora_add(adapter, source, target, 2 * units)?;
                    }
                }
                let attended = gpu.attention(
                    queries.view(),
                    kv.part(0, units),
                    kv.part(1, units),
                    heads,
                    false,
                )?;
                (attended, Some(queries), kv)
            }
        };
        gpu.dense_add(&attention.output, &attended, residual)?;
        if let Some(adapter) = adapter(3) {
            let width = residual.cols;
            gpu.lora_add(adapter, &attended, &mut residual.data, width)?;
        }
        Ok(lora.map(|_| AttentionRecord {
            queries,
            projected,
            attended,
        }))
    }
}

fn clone_map(gpu: &ImageGpu, map: &Map) -> Result<Map, NetworkError> {
    Ok(Map {
        tensor: clone_tensor(gpu, &map.tensor)?,
        height: map.height,
        width: map.width,
    })
}

fn clone_tensor(gpu: &ImageGpu, tensor: &Tensor) -> Result<Tensor, NetworkError> {
    let mut copy = gpu.uninit(tensor.rows, tensor.cols)?;
    gpu.context
        .stream
        .memcpy_dtod(&tensor.data, &mut copy.data)
        .map_err(cuda_err("device to device copy"))?;
    Ok(copy)
}

fn upload_map(gpu: &ImageGpu, map: &FeatureMap) -> Result<Map, NetworkError> {
    Ok(Map {
        tensor: gpu.upload(map.channels, map.pixels(), &map.data)?,
        height: map.height,
        width: map.width,
    })
}

fn download_map(gpu: &ImageGpu, map: &Map) -> Result<FeatureMap, NetworkError> {
    let values = gpu.download(&map.tensor)?;
    FeatureMap::from_vec(map.channels(), map.height, map.width, values.data)
}

fn upload_block(
    gpu: &Arc<ImageGpu>,
    block: &crate::unet::Block,
) -> Result<DeviceBlock, NetworkError> {
    Ok(DeviceBlock {
        resnets: block
            .resnets
            .iter()
            .map(|resnet| upload_resnet(gpu, resnet))
            .collect::<Result<_, _>>()?,
        attentions: block
            .attentions
            .iter()
            .map(|transformer| upload_transformer(gpu, transformer))
            .collect::<Result<_, _>>()?,
        resampler: match &block.resampler {
            Some(conv) => Some(upload_conv(gpu, conv)?),
            None => None,
        },
    })
}

fn upload_resnet(
    gpu: &Arc<ImageGpu>,
    resnet: &crate::unet::Resnet,
) -> Result<DeviceResnet, NetworkError> {
    Ok(DeviceResnet {
        norm1: upload_group_norm(gpu, &resnet.norm1)?,
        conv1: upload_conv(gpu, &resnet.conv1)?,
        time: upload_dense(gpu, &resnet.time)?,
        norm2: upload_group_norm(gpu, &resnet.norm2)?,
        conv2: upload_conv(gpu, &resnet.conv2)?,
        shortcut: match &resnet.shortcut {
            Some(conv) => Some(upload_conv(gpu, conv)?),
            None => None,
        },
    })
}

fn upload_transformer(
    gpu: &Arc<ImageGpu>,
    transformer: &crate::unet::Transformer,
) -> Result<DeviceTransformer, NetworkError> {
    Ok(DeviceTransformer {
        norm: upload_group_norm(gpu, &transformer.norm)?,
        input: upload_projection(gpu, &transformer.input)?,
        output: upload_projection(gpu, &transformer.output)?,
        heads: transformer.heads,
        blocks: transformer
            .blocks
            .iter()
            .map(|block| {
                Ok(DeviceTransformerBlock {
                    norm1: upload_norm(gpu, &block.norm1)?,
                    attention: upload_attention(gpu, &block.attention, true)?,
                    norm2: upload_norm(gpu, &block.norm2)?,
                    cross: upload_attention(gpu, &block.cross, false)?,
                    norm3: upload_norm(gpu, &block.norm3)?,
                    gate: upload_dense(gpu, &block.gate)?,
                    output: upload_dense(gpu, &block.output)?,
                })
            })
            .collect::<Result<_, NetworkError>>()?,
    })
}

fn upload_projection(
    gpu: &Arc<ImageGpu>,
    projection: &Projection,
) -> Result<DeviceProjection, NetworkError> {
    Ok(match projection {
        Projection::Convolution(conv) => DeviceProjection::Convolution(upload_conv(gpu, conv)?),
        Projection::Linear(dense) => DeviceProjection::Linear(upload_dense(gpu, dense)?),
    })
}

fn upload_attention(
    gpu: &Arc<ImageGpu>,
    attention: &crate::unet::Attention,
    // Self-attention reads one tensor, so its three projections are one GEMM.
    // Cross-attention reads the prompt for two of them and the image for the
    // third, so only those two stack.
    itself: bool,
) -> Result<DeviceAttention, NetworkError> {
    Ok(DeviceAttention {
        stacked: match itself {
            true => upload_stacked(gpu, &[&attention.query, &attention.key, &attention.value])?,
            false => upload_stacked(gpu, &[&attention.key, &attention.value])?,
        },
        query: match itself {
            true => None,
            false => Some(upload_dense(gpu, &attention.query)?),
        },
        output: upload_dense(gpu, &attention.output)?,
    })
}

// ---------------------------------------------------------------------------
// The VAE decoder.
// ---------------------------------------------------------------------------

struct DeviceVaeResnet {
    norm1: DeviceGroupNorm,
    conv1: DeviceConv,
    norm2: DeviceGroupNorm,
    conv2: DeviceConv,
    shortcut: Option<DeviceConv>,
}

struct DeviceVaeAttention {
    norm: DeviceGroupNorm,
    query: DeviceDense,
    key: DeviceDense,
    value: DeviceDense,
    output: DeviceDense,
}

struct DeviceUpBlock {
    resnets: Vec<DeviceVaeResnet>,
    upsampler: Option<DeviceConv>,
}

/// A VAE decoder, resident on the device.
pub struct DeviceVae {
    gpu: Arc<ImageGpu>,
    post_quant: Option<DeviceConv>,
    conv_in: DeviceConv,
    mid_first: DeviceVaeResnet,
    mid_attention: DeviceVaeAttention,
    mid_last: DeviceVaeResnet,
    up_blocks: Vec<DeviceUpBlock>,
    norm_out: DeviceGroupNorm,
    conv_out: DeviceConv,
}

impl std::fmt::Debug for DeviceVae {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceVae").finish_non_exhaustive()
    }
}

impl DeviceVae {
    pub fn upload(gpu: &Arc<ImageGpu>, decoder: &VaeDecoder) -> Result<Self, NetworkError> {
        Ok(Self {
            post_quant: match &decoder.post_quant {
                Some(conv) => Some(upload_conv(gpu, conv)?),
                None => None,
            },
            conv_in: upload_conv(gpu, &decoder.conv_in)?,
            mid_first: upload_vae_resnet(gpu, &decoder.mid_first)?,
            mid_attention: upload_vae_attention(gpu, &decoder.mid_attention)?,
            mid_last: upload_vae_resnet(gpu, &decoder.mid_last)?,
            up_blocks: decoder
                .up_blocks
                .iter()
                .map(|block| {
                    Ok(DeviceUpBlock {
                        resnets: block
                            .resnets
                            .iter()
                            .map(|resnet| upload_vae_resnet(gpu, resnet))
                            .collect::<Result<_, NetworkError>>()?,
                        upsampler: match &block.upsampler {
                            Some(conv) => Some(upload_conv(gpu, conv)?),
                            None => None,
                        },
                    })
                })
                .collect::<Result<_, NetworkError>>()?,
            norm_out: upload_group_norm(gpu, &decoder.norm_out)?,
            conv_out: upload_conv(gpu, &decoder.conv_out)?,
            gpu: gpu.clone(),
        })
    }

    /// `latent` has already had the model's scale and shift undone by the host,
    /// which is one multiply per value and keeps that arithmetic in one place.
    pub fn decode(&self, latent: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        use std::sync::atomic::Ordering::Relaxed;
        match self.decode_once(latent) {
            // Retry once with f32 scores, and keep them for every later decode.
            Err(NetworkError::Cuda(message))
                if message.contains("overflowed") && !self.gpu.wide_scores.swap(true, Relaxed) =>
            {
                self.decode_once(latent)
            }
            result => result,
        }
    }

    fn decode_once(&self, latent: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        let gpu = &self.gpu;
        let mut hidden = Map {
            tensor: gpu.upload(latent.channels, latent.pixels(), &latent.data)?,
            height: latent.height,
            width: latent.width,
        };
        // The convolution the decoder's stem sits behind. Skipping it is the
        // bug `a_decoder_runs_the_convolution_that_sits_in_front_of_its_stem`
        // guards on the host, and it is just as invisible to a shape check
        // here.
        if let Some(post_quant) = &self.post_quant {
            hidden = gpu.conv(post_quant, &hidden)?;
        }
        hidden = gpu.conv(&self.conv_in, &hidden)?;
        hidden = vae_resnet(gpu, &self.mid_first, hidden)?;
        hidden = vae_attention(gpu, &self.mid_attention, &hidden)?;
        hidden = vae_resnet(gpu, &self.mid_last, hidden)?;

        for block in &self.up_blocks {
            for resnet in &block.resnets {
                hidden = vae_resnet(gpu, resnet, hidden)?;
            }
            if let Some(upsampler) = &block.upsampler {
                let large = gpu.upsample_nearest(&hidden, 2)?;
                // The small map is dead the moment the large one exists, and
                // at this resolution holding it costs a quarter of a gigabyte.
                drop(hidden);
                hidden = gpu.conv(upsampler, &large)?;
            }
        }

        gpu.group_norm(&self.norm_out, &mut hidden, 0)?;
        let output = gpu.conv(&self.conv_out, &hidden)?;
        let image = download_map(gpu, &output)?;
        // An FP16 overflow anywhere above turns into infinities, and the first
        // group norm after it into NaN, which an image writer clamps to black.
        // Saying so is the whole guard: the host path decodes the same latent
        // in f32.
        if image.data.iter().any(|value| !value.is_finite()) {
            return Err(NetworkError::Cuda(
                "VAE decode overflowed FP16 on the device; decode this latent on the host".into(),
            ));
        }
        Ok(image)
    }
}

/// A VAE encoder, resident on the device. The same blocks as [`DeviceVae`],
/// run the other way.
pub struct DeviceVaeEncoder {
    gpu: Arc<ImageGpu>,
    conv_in: DeviceConv,
    down_blocks: Vec<(Vec<DeviceVaeResnet>, Option<DeviceConv>)>,
    mid_first: DeviceVaeResnet,
    mid_attention: DeviceVaeAttention,
    mid_last: DeviceVaeResnet,
    norm_out: DeviceGroupNorm,
    conv_out: DeviceConv,
    quant: Option<DeviceConv>,
}

impl std::fmt::Debug for DeviceVaeEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceVaeEncoder").finish_non_exhaustive()
    }
}

impl DeviceVaeEncoder {
    pub fn upload(gpu: &Arc<ImageGpu>, encoder: &VaeEncoder) -> Result<Self, NetworkError> {
        Ok(Self {
            conv_in: upload_conv(gpu, &encoder.conv_in)?,
            down_blocks: encoder
                .down_blocks
                .iter()
                .map(|block| {
                    let resnets = block
                        .resnets
                        .iter()
                        .map(|resnet| upload_vae_resnet(gpu, resnet))
                        .collect::<Result<_, NetworkError>>()?;
                    let downsampler = match &block.downsampler {
                        Some(conv) => Some(DeviceConv {
                            pad_end: 1,
                            ..upload_conv(gpu, conv)?
                        }),
                        None => None,
                    };
                    Ok((resnets, downsampler))
                })
                .collect::<Result<_, NetworkError>>()?,
            mid_first: upload_vae_resnet(gpu, &encoder.mid_first)?,
            mid_attention: upload_vae_attention(gpu, &encoder.mid_attention)?,
            mid_last: upload_vae_resnet(gpu, &encoder.mid_last)?,
            norm_out: upload_group_norm(gpu, &encoder.norm_out)?,
            conv_out: upload_conv(gpu, &encoder.conv_out)?,
            quant: match &encoder.quant {
                Some(conv) => Some(upload_conv(gpu, conv)?),
                None => None,
            },
            gpu: gpu.clone(),
        })
    }

    /// The moments, mean and log-variance, before the host picks the mean
    /// and applies the scale and shift.
    pub fn moments(&self, image: &FeatureMap) -> Result<FeatureMap, NetworkError> {
        let gpu = &self.gpu;
        let mut hidden = Map {
            tensor: gpu.upload(image.channels, image.pixels(), &image.data)?,
            height: image.height,
            width: image.width,
        };
        hidden = gpu.conv(&self.conv_in, &hidden)?;
        for (resnets, downsampler) in &self.down_blocks {
            for resnet in resnets {
                hidden = vae_resnet(gpu, resnet, hidden)?;
            }
            if let Some(downsampler) = downsampler {
                hidden = gpu.conv(downsampler, &hidden)?;
            }
        }
        hidden = vae_resnet(gpu, &self.mid_first, hidden)?;
        hidden = vae_attention(gpu, &self.mid_attention, &hidden)?;
        hidden = vae_resnet(gpu, &self.mid_last, hidden)?;
        gpu.group_norm(&self.norm_out, &mut hidden, 0)?;
        hidden = gpu.conv(&self.conv_out, &hidden)?;
        if let Some(quant) = &self.quant {
            hidden = gpu.conv(quant, &hidden)?;
        }
        let moments = download_map(gpu, &hidden)?;
        if moments.data.iter().any(|value| !value.is_finite()) {
            return Err(NetworkError::Cuda(
                "VAE encode overflowed FP16 on the device; encode this image on the host".into(),
            ));
        }
        Ok(moments)
    }
}

/// Takes the input by value so the normalization can run in place.
///
/// At the decoder's last rung one feature map is half a gigabyte, and the
/// naive order — clone, normalize the clone, convolve, then build the skip
/// — holds three of them at once. Building the skip first instead lets the
/// input be normalized where it lies and freed as soon as the first
/// convolution has read it.
fn vae_resnet(gpu: &ImageGpu, resnet: &DeviceVaeResnet, input: Map) -> Result<Map, NetworkError> {
    let (skip, mut hidden) = match &resnet.shortcut {
        Some(shortcut) => (gpu.conv(shortcut, &input)?, input),
        // Without one the input is the skip, so the normalization is what
        // needs the copy.
        None => {
            let copy = clone_map(gpu, &input)?;
            (input, copy)
        }
    };
    gpu.group_norm(&resnet.norm1, &mut hidden, 0)?;
    let mut next = gpu.conv(&resnet.conv1, &hidden)?;
    drop(hidden);
    gpu.group_norm(&resnet.norm2, &mut next, 0)?;
    let mut output = skip;
    gpu.conv_add(&resnet.conv2, &next, &mut output)?;
    Ok(output)
}

/// One head over the full channel width, which is what the decoder's own
/// attention is.
fn vae_attention(
    gpu: &ImageGpu,
    attention: &DeviceVaeAttention,
    input: &Map,
) -> Result<Map, NetworkError> {
    let mut normed = clone_map(gpu, input)?;
    gpu.group_norm(&attention.norm, &mut normed, -1)?;
    let tokens = gpu.to_tokens(&normed)?;

    let queries = gpu.dense(&attention.query, &tokens)?;
    let keys = gpu.dense(&attention.key, &tokens)?;
    let values = gpu.dense(&attention.value, &tokens)?;
    let attended = gpu.attention(queries.view(), keys.view(), values.view(), 1, false)?;
    let projected = gpu.dense(&attention.output, &attended)?;

    let mut output = gpu.to_map(&projected, input.height, input.width)?;
    gpu.add(&mut output.tensor, &input.tensor)?;
    Ok(output)
}

fn upload_vae_attention(
    gpu: &Arc<ImageGpu>,
    attention: &crate::vae::AttentionBlock,
) -> Result<DeviceVaeAttention, NetworkError> {
    Ok(DeviceVaeAttention {
        norm: upload_group_norm(gpu, &attention.norm)?,
        query: upload_dense(gpu, &attention.query)?,
        key: upload_dense(gpu, &attention.key)?,
        value: upload_dense(gpu, &attention.value)?,
        output: upload_dense(gpu, &attention.output)?,
    })
}

fn upload_vae_resnet(
    gpu: &Arc<ImageGpu>,
    resnet: &crate::vae::ResnetBlock,
) -> Result<DeviceVaeResnet, NetworkError> {
    Ok(DeviceVaeResnet {
        norm1: upload_group_norm(gpu, &resnet.norm1)?,
        conv1: upload_conv(gpu, &resnet.conv1)?,
        norm2: upload_group_norm(gpu, &resnet.norm2)?,
        conv2: upload_conv(gpu, &resnet.conv2)?,
        shortcut: match &resnet.shortcut {
            Some(conv) => Some(upload_conv(gpu, conv)?),
            None => None,
        },
    })
}

// ---------------------------------------------------------------------------
// The CLIP text tower.
// ---------------------------------------------------------------------------

struct DeviceClipLayer {
    attention_norm: DeviceNorm,
    query: DeviceDense,
    key: DeviceDense,
    value: DeviceDense,
    output: DeviceDense,
    mlp_norm: DeviceNorm,
    mlp_in: DeviceDense,
    mlp_out: DeviceDense,
}

/// A CLIP text tower, resident on the device.
pub struct DeviceClip {
    gpu: Arc<ImageGpu>,
    layers: Vec<DeviceClipLayer>,
    final_norm: DeviceNorm,
    heads: usize,
    eps: f32,
    /// 2 for the quick GELU the published towers were trained with, 1 for the
    /// tanh approximation.
    activation: i32,
}

impl std::fmt::Debug for DeviceClip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceClip").finish_non_exhaustive()
    }
}

impl DeviceClip {
    pub fn upload(gpu: &Arc<ImageGpu>, encoder: &ClipTextEncoder) -> Result<Self, NetworkError> {
        Ok(Self {
            layers: upload_clip_layers(gpu, &encoder.layers)?,
            final_norm: upload_norm(gpu, &encoder.final_norm)?,
            heads: encoder.config().num_heads,
            eps: encoder.config().eps,
            activation: i32::from(encoder.config().quick_gelu) + 1,
            gpu: gpu.clone(),
        })
    }

    /// The tower, from the embedded tokens to the final norm.
    ///
    /// The embedding gather and the position table stay on the host: they are a
    /// table lookup per token, and the result is the one buffer this uploads.
    pub fn forward(
        &self,
        embedded: &Matrix,
        skip: usize,
    ) -> Result<(Matrix, Matrix), NetworkError> {
        let gpu = &self.gpu;
        let mut hidden = gpu.upload_matrix(embedded)?;
        let stop = self.layers.len().saturating_sub(skip);
        let mut skipped = None;

        for (index, layer) in self.layers.iter().enumerate() {
            if index == stop {
                skipped = Some(gpu.download(&hidden)?);
            }
            // A text tower reads left to right, unlike everything else here.
            let sequence = hidden.rows;
            run_clip_layer(
                gpu,
                layer,
                &mut hidden,
                self.heads,
                self.eps,
                self.activation,
                true,
                sequence,
            )?;
        }

        let normalized = gpu.layer_norm(&self.final_norm, &hidden, self.eps)?;
        let normalized = gpu.download(&normalized)?;
        Ok((skipped.unwrap_or_else(|| normalized.clone()), normalized))
    }
}

fn upload_clip_layers(
    gpu: &ImageGpu,
    layers: &[Layer],
) -> Result<Vec<DeviceClipLayer>, NetworkError> {
    layers
        .iter()
        .map(|layer| {
            Ok(DeviceClipLayer {
                attention_norm: upload_norm(gpu, &layer.attention_norm)?,
                query: upload_dense(gpu, &layer.query)?,
                key: upload_dense(gpu, &layer.key)?,
                value: upload_dense(gpu, &layer.value)?,
                output: upload_dense(gpu, &layer.output)?,
                mlp_norm: upload_norm(gpu, &layer.mlp_norm)?,
                mlp_in: upload_dense(gpu, &layer.mlp_in)?,
                mlp_out: upload_dense(gpu, &layer.mlp_out)?,
            })
        })
        .collect()
}

/// One CLIP layer in place, the device twin of [`crate::clip::run_layer`].
///
/// `hidden` holds whole sequences of `sequence` rows stacked one after
/// another. Everything but attention reads a row at a time, so it runs over the
/// whole stack at once; attention runs over each sequence's window on its own,
/// so no sequence sees another's tokens.
#[allow(clippy::too_many_arguments)]
fn run_clip_layer(
    gpu: &ImageGpu,
    layer: &DeviceClipLayer,
    hidden: &mut Tensor,
    heads: usize,
    eps: f32,
    activation: i32,
    causal: bool,
    sequence: usize,
) -> Result<(), NetworkError> {
    if sequence == 0 || hidden.rows % sequence != 0 {
        return Err(NetworkError::InvalidConfig(format!(
            "{} rows do not split into sequences of {sequence}",
            hidden.rows
        )));
    }
    let normed = gpu.layer_norm(&layer.attention_norm, hidden, eps)?;
    let queries = gpu.dense(&layer.query, &normed)?;
    let keys = gpu.dense(&layer.key, &normed)?;
    let values = gpu.dense(&layer.value, &normed)?;
    let attended = match hidden.rows == sequence {
        true => gpu.attention(queries.view(), keys.view(), values.view(), heads, causal)?,
        false => {
            let mut attended = gpu.uninit(hidden.rows, queries.cols)?;
            let width = sequence * queries.cols;
            for first in (0..hidden.rows).step_by(sequence) {
                let one = gpu.attention(
                    queries.window(first, sequence),
                    keys.window(first, sequence),
                    values.window(first, sequence),
                    heads,
                    causal,
                )?;
                let start = first * queries.cols;
                gpu.context
                    .stream
                    .memcpy_dtod(
                        &one.data,
                        &mut attended.data.slice_mut(start..start + width),
                    )
                    .map_err(cuda_err("device to device copy"))?;
            }
            attended
        }
    };
    gpu.dense_add(&layer.output, &attended, hidden)?;

    let normed = gpu.layer_norm(&layer.mlp_norm, hidden, eps)?;
    let mut wide = gpu.dense(&layer.mlp_in, &normed)?;
    gpu.activate(&mut wide, activation)?;
    gpu.dense_add(&layer.mlp_out, &wide, hidden)
}

// ---------------------------------------------------------------------------
// The CLIP image tower.
// ---------------------------------------------------------------------------

/// A CLIP image tower, resident on the device.
pub struct DeviceVitEncoder {
    gpu: Arc<ImageGpu>,
    patch: DeviceDense,
    pre_norm: Option<DeviceNorm>,
    layers: Vec<DeviceClipLayer>,
    post_norm: DeviceNorm,
    heads: usize,
    eps: f32,
    activation: i32,
}

impl std::fmt::Debug for DeviceVitEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceVitEncoder").finish_non_exhaustive()
    }
}

impl DeviceVitEncoder {
    pub fn upload(gpu: &Arc<ImageGpu>, encoder: &VitEncoder) -> Result<Self, NetworkError> {
        let config = encoder.config();
        Ok(Self {
            patch: upload_dense(gpu, &encoder.patch)?,
            pre_norm: encoder
                .pre_norm
                .as_ref()
                .map(|norm| upload_norm(gpu, norm))
                .transpose()?,
            layers: upload_clip_layers(gpu, &encoder.layers)?,
            post_norm: upload_norm(gpu, &encoder.post_norm)?,
            heads: config.num_heads,
            eps: config.eps,
            activation: match encoder.exact_gelu {
                true => 3,
                false => i32::from(config.quick_gelu) + 1,
            },
            gpu: gpu.clone(),
        })
    }

    /// The tower over a stack of images, `tokens` rows each.
    ///
    /// `patches` is `[images * tokens, channels * patch * patch]`, every
    /// image's flattened patches behind a zero row where its class token goes.
    /// `embedding` is `[images * tokens, d_model]`: the class token and the
    /// position table, which the patch projection lands on top of. Both are
    /// built on the host, the same way [`DeviceClip`] leaves the embedding
    /// gather there.
    pub fn forward(
        &self,
        patches: &Matrix,
        embedding: &Matrix,
        tokens: usize,
    ) -> Result<Matrix, NetworkError> {
        let gpu = &self.gpu;
        let mut hidden = gpu.upload_matrix(embedding)?;
        gpu.dense_add(&self.patch, &gpu.upload_matrix(patches)?, &mut hidden)?;
        if let Some(norm) = &self.pre_norm {
            hidden = gpu.layer_norm(norm, &hidden, self.eps)?;
        }
        for layer in &self.layers {
            run_clip_layer(
                gpu,
                layer,
                &mut hidden,
                self.heads,
                self.eps,
                self.activation,
                false,
                tokens,
            )?;
        }
        gpu.download(&gpu.layer_norm(&self.post_norm, &hidden, self.eps)?)
    }
}

/// Free and total device memory, in bytes, for a benchmark that wants to say
/// what a run cost.
pub fn memory_info() -> Option<(usize, usize)> {
    cudarc::runtime::result::get_mem_info().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv::{silu, upsample_nearest};
    use crate::mmdit::attention as host_attention;
    use crate::transformer::Precision;

    /// Same contract as the other CUDA tests: no device means the parity tests
    /// report success without running, a broken device is a failure.
    pub(super) fn device() -> Option<Arc<ImageGpu>> {
        match crate::cuda_training::cuda_doctor(0, 512) {
            Ok(_) => Some(ImageGpu::new(0).expect("a device that the doctor cleared")),
            Err(NetworkError::Cuda(message)) if message.contains("initialization") => {
                crate::cuda_training::skip_without_cuda();
                None
            }
            Err(error) => panic!("CUDA device present but unusable: {error}"),
        }
    }

    fn ramp(count: usize, seed: f32) -> Vec<f32> {
        (0..count)
            .map(|index| (((index % 23) as f32 - 11.0) * 0.07 + seed).sin())
            .collect()
    }

    fn map(channels: usize, height: usize, width: usize, seed: f32) -> FeatureMap {
        FeatureMap::from_vec(
            channels,
            height,
            width,
            ramp(channels * height * width, seed),
        )
        .expect("a well-shaped map")
    }

    /// FP16 carries eleven mantissa bits, so a value is good to about 0.05%
    /// before any accumulation. The bars below are relative to the largest
    /// value in the reference, which is what makes them comparable across
    /// layers of very different scale.
    pub(super) fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        let scale = expected
            .iter()
            .fold(0.0f32, |largest, value| largest.max(value.abs()))
            .max(1e-6);
        let (worst, at) = actual.iter().zip(expected).enumerate().fold(
            (0.0f32, 0usize),
            |(worst, at), (index, (actual, expected))| {
                let error = (actual - expected).abs() / scale;
                if error > worst {
                    (error, index)
                } else {
                    (worst, at)
                }
            },
        );
        assert!(
            worst <= tolerance,
            "{label}: {worst} relative error at {at} ({} against {}), over {tolerance}",
            actual[at],
            expected[at]
        );
    }

    fn upload_map(gpu: &ImageGpu, source: &FeatureMap) -> Map {
        Map {
            tensor: gpu
                .upload(source.channels, source.pixels(), &source.data)
                .expect("an upload"),
            height: source.height,
            width: source.width,
        }
    }

    #[test]
    fn a_device_convolution_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        // A 3x3 with padding, a 1x1 (the GEMM-only path) and a stride of two;
        // then the row-wise path unpadded and with a wider kernel and padding.
        for (kernel, stride, padding, in_channels, out_channels) in [
            (3, 1, 1, 3, 5),
            (1, 1, 0, 4, 6),
            (3, 2, 1, 2, 4),
            (3, 1, 0, 3, 5),
            (5, 1, 2, 2, 3),
        ] {
            let weight = Matrix::from_vec(
                out_channels,
                in_channels * kernel * kernel,
                ramp(out_channels * in_channels * kernel * kernel, 0.3),
            );
            let conv = Conv2d::new(
                weight,
                Some(ramp(out_channels, 1.1)),
                in_channels,
                kernel,
                stride,
                padding,
            )
            .expect("a convolution");
            let input = map(in_channels, 7, 9, 0.2);

            let expected = conv.forward(&input).expect("a host convolution");
            let device = upload_conv(&gpu, &conv).expect("an upload");
            let actual = gpu
                .conv(&device, &upload_map(&gpu, &input))
                .expect("a device convolution");
            let actual = download_map(&gpu, &actual).expect("a download");

            assert_eq!(
                (actual.channels, actual.height, actual.width),
                (expected.channels, expected.height, expected.width)
            );
            assert_close(
                &format!("{kernel}x{kernel}/{stride}"),
                &actual.data,
                &expected.data,
                0.01,
            );
            // Adding into a map with fewer channels than the convolution
            // makes would write past it.
            let mut short = upload_map(
                &gpu,
                &map(out_channels - 1, actual.height, actual.width, 0.0),
            );
            assert!(
                gpu.conv_add(&device, &upload_map(&gpu, &input), &mut short)
                    .is_err()
            );
        }
    }

    /// Each backward kernel against the host function `unet_lora` computes
    /// the same gradient with.
    #[test]
    fn backward_kernels_match_the_host_or_skip_without_a_device() {
        use crate::unet_lora as host;
        let Some(gpu) = device() else { return };
        // The forward test's convolutions, which cover the 1x1, `im2col` and
        // row-wise paths, each in one chunk and in one row per chunk.
        for (kernel, stride, padding, in_channels, out_channels) in [
            (3, 1, 1, 3, 5),
            (1, 1, 0, 4, 6),
            (3, 2, 1, 2, 4),
            (3, 1, 0, 3, 5),
            (5, 1, 2, 2, 3),
        ] {
            let weight = Matrix::from_vec(
                out_channels,
                in_channels * kernel * kernel,
                ramp(out_channels * in_channels * kernel * kernel, 0.3),
            );
            let conv = Conv2d::new(weight, None, in_channels, kernel, stride, padding)
                .expect("a convolution");
            let input = map(in_channels, 7, 9, 0.2);
            let output = conv.forward(&input).expect("a host convolution");
            let grad = map(out_channels, output.height, output.width, 0.7);
            let expected = host::conv_backward(&conv, &grad, &input);
            let device = upload_conv(&gpu, &conv).expect("an upload");
            for budget in [COLUMN_BUDGET, 1] {
                let actual = gpu
                    .conv_backward(&device, &upload_map(&gpu, &grad), 7, 9, budget)
                    .expect("a device convolution backward");
                assert_close(
                    &format!("{kernel}x{kernel}/{stride} backward, budget {budget}"),
                    &download_map(&gpu, &actual).expect("a download").data,
                    &expected.data,
                    0.01,
                );
            }
        }

        let norm = GroupNorm::new(4, ramp(8, 0.9), ramp(8, 0.1), 1e-5).expect("a group norm");
        let (input, grad) = (map(8, 6, 7, 0.4), map(8, 6, 7, 1.3));
        let expected = host::group_norm_backward(&norm, &input, &grad);
        let mut actual = upload_map(&gpu, &grad);
        gpu.group_norm_backward(
            &upload_group_norm(&gpu, &norm).expect("an upload"),
            &upload_map(&gpu, &input),
            &mut actual,
        )
        .expect("a group norm backward");
        let actual = download_map(&gpu, &actual).expect("a download");
        assert_close("group norm backward", &actual.data, &expected.data, 0.01);

        let norm = Norm {
            weight: ramp(24, 0.6),
            bias: ramp(24, 0.2),
        };
        let tokens = Matrix::from_vec(5, 24, ramp(5 * 24, 0.5));
        let grad = Matrix::from_vec(5, 24, ramp(5 * 24, 1.7));
        let expected = host::norm_backward(&norm, &tokens, 1e-5, &grad);
        let mut actual = gpu.upload_matrix(&grad).expect("an upload");
        gpu.layer_norm_backward(
            &upload_norm(&gpu, &norm).expect("an upload"),
            &gpu.upload_matrix(&tokens).expect("an upload"),
            1e-5,
            &mut actual,
        )
        .expect("a layer norm backward");
        let actual = gpu.download(&actual).expect("a download");
        assert_close("layer norm backward", &actual.data, &expected.data, 0.01);

        let projected = Matrix::from_vec(5, 2 * 12, ramp(5 * 24, 0.8));
        let grad = Matrix::from_vec(5, 12, ramp(5 * 12, 2.1));
        let expected = host::geglu_backward(&projected, &grad);
        let actual = gpu
            .gelu_gate_backward(
                &gpu.upload_matrix(&projected).expect("an upload"),
                None,
                &gpu.upload_matrix(&grad).expect("an upload"),
            )
            .expect("a gated GELU backward");
        let actual = gpu.download(&actual).expect("a download");
        assert_close("GEGLU backward", &actual.data, &expected.data, 0.01);

        let (pre, grad) = (map(3, 5, 6, 0.3), map(3, 5, 6, 1.9));
        let expected = host::silu_backward(&grad, &pre);
        let mut actual = upload_map(&gpu, &grad);
        gpu.silu_backward(&mut actual.tensor, &upload_map(&gpu, &pre).tensor)
            .expect("a SiLU backward");
        let actual = download_map(&gpu, &actual).expect("a download");
        assert_close("SiLU backward", &actual.data, &expected.data, 0.01);

        let grad = map(3, 8, 10, 0.6);
        let expected = host::upsample_backward(&grad);
        let actual = gpu
            .upsample_backward(&upload_map(&gpu, &grad), 2)
            .expect("an upsample backward");
        let actual = download_map(&gpu, &actual).expect("a download");
        assert_eq!((actual.height, actual.width), (4, 5));
        assert_close("upsample backward", &actual.data, &expected.data, 0.01);
    }

    #[test]
    fn a_convolution_that_needs_more_than_one_chunk_matches_the_host_or_skips() {
        let Some(gpu) = device() else { return };
        let conv = Conv2d::new(
            Matrix::from_vec(4, 2 * 9, ramp(4 * 18, 0.5)),
            None,
            2,
            3,
            1,
            1,
        )
        .expect("a convolution");
        let input = map(2, 16, 16, 0.4);
        let expected = conv.forward(&input).expect("a host convolution");

        let device = upload_conv(&gpu, &conv).expect("an upload");
        // A budget of one element is what `COLUMN_BUDGET / patch` would give
        // for a patch larger than the whole budget, so the real convolution
        // runs its chunk loop at the smallest chunk it is ever asked for.
        let output = gpu
            .conv_with_budget(&device, &upload_map(&gpu, &input), 1)
            .expect("a device convolution");
        let actual = download_map(&gpu, &output).expect("a download");
        assert_close("chunked convolution", &actual.data, &expected.data, 0.01);
    }

    #[test]
    fn device_group_norm_and_silu_match_the_host_or_skip_without_a_device() {
        let Some(gpu) = device() else { return };
        let norm = GroupNorm::new(4, ramp(8, 0.9), ramp(8, 0.1), 1e-5).expect("a norm");
        let source = map(8, 5, 6, 0.8);

        let mut expected = source.clone();
        norm.forward(&mut expected).expect("a host norm");
        silu(&mut expected);

        let device = upload_group_norm(&gpu, &norm).expect("an upload");
        let mut actual = upload_map(&gpu, &source);
        gpu.group_norm(&device, &mut actual, 0)
            .expect("a device norm and SiLU");
        let actual = download_map(&gpu, &actual).expect("a download");

        assert_close("group norm and SiLU", &actual.data, &expected.data, 0.01);
    }

    #[test]
    fn a_device_dense_matches_the_host_including_when_quantized_or_skips() {
        let Some(gpu) = device() else { return };
        let weight = Matrix::from_vec(7, 5, ramp(35, 0.6));
        let tokens = Matrix::from_vec(4, 5, ramp(20, 0.2));
        let mut dense = Dense::new(weight, Some(ramp(7, 0.4))).expect("a layer");
        let expected = dense.forward(&tokens).expect("a host layer");

        let device = upload_dense(&gpu, &dense).expect("an upload");
        let actual = gpu
            .dense(&device, &gpu.upload_matrix(&tokens).expect("an upload"))
            .expect("a device layer");
        assert_close(
            "dense",
            &gpu.download(&actual).expect("a download").data,
            &expected.data,
            0.01,
        );

        // A quantized layer is dequantized on the way up, so it lands within
        // the sum of the two roundings rather than either one.
        dense.quantize();
        let device = upload_dense(&gpu, &dense).expect("an upload");
        let actual = gpu
            .dense(&device, &gpu.upload_matrix(&tokens).expect("an upload"))
            .expect("a device layer");
        assert_close(
            "quantized dense",
            &gpu.download(&actual).expect("a download").data,
            &expected.data,
            0.03,
        );
    }

    #[test]
    fn device_attention_matches_the_host_both_ways_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        for (heads, causal, context) in [(2usize, false, 6usize), (2, true, 6), (1, false, 9)] {
            let width = heads * 4;
            let queries = Matrix::from_vec(6, width, ramp(6 * width, 0.15));
            let keys = Matrix::from_vec(context, width, ramp(context * width, 0.35));
            let values = Matrix::from_vec(context, width, ramp(context * width, 0.55));

            let expected = host_attention(&queries, &keys, &values, heads, heads, causal);
            let (queries, keys, values) = (
                gpu.upload_matrix(&queries).expect("an upload"),
                gpu.upload_matrix(&keys).expect("an upload"),
                gpu.upload_matrix(&values).expect("an upload"),
            );
            let actual = gpu
                .attention(queries.view(), keys.view(), values.view(), heads, causal)
                .expect("a device attention");

            assert_close(
                &format!("attention heads={heads} causal={causal}"),
                &gpu.download(&actual).expect("a download").data,
                &expected.data,
                0.02,
            );
        }
    }

    #[test]
    fn device_attention_backward_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        // SD 1.5's 40-wide heads on stacked self-attention, and narrow heads
        // across a five-token prompt; each in one chunk and in chunks of five
        // query rows with a short last one.
        for (heads, head_dim, context) in [(2usize, 40usize, None), (3, 8, Some(5usize))] {
            let (tokens, units) = (37, heads * head_dim);
            let context_rows = context.unwrap_or(tokens);
            let queries = Matrix::from_vec(tokens, units, ramp(tokens * units, 0.15));
            let keys = Matrix::from_vec(context_rows, units, ramp(context_rows * units, 0.35));
            let values = Matrix::from_vec(context_rows, units, ramp(context_rows * units, 0.55));
            let grad = Matrix::from_vec(tokens, units, ramp(tokens * units, 0.75));
            let expected =
                crate::dit_lora::attention_backward(&queries, &keys, &values, heads, &grad);
            let stacked = |parts: &[&Matrix]| {
                let rows = parts[0].rows;
                let data = (0..rows)
                    .flat_map(|row| parts.iter().flat_map(move |part| part.row(row).to_vec()))
                    .collect();
                gpu.upload_matrix(&Matrix::from_vec(rows, parts.len() * units, data))
                    .expect("an upload")
            };
            let (own, query, kv) = match context {
                None => (Some(stacked(&[&queries, &keys, &values])), None, None),
                Some(_) => (
                    None,
                    Some(gpu.upload_matrix(&queries).expect("an upload")),
                    Some(stacked(&[&keys, &values])),
                ),
            };
            let (q, k, v) = match (&own, &query, &kv) {
                (Some(own), _, _) => (own.part(0, units), own.part(1, units), own.part(2, units)),
                (_, Some(query), Some(kv)) => (query.view(), kv.part(0, units), kv.part(1, units)),
                _ => unreachable!(),
            };
            let grad_device = gpu.upload_matrix(&grad).expect("an upload");
            for budget in [SCORE_BUDGET, 5 * heads * context_rows] {
                let actual = gpu
                    .attention_backward(q, k, v, heads, &grad_device, budget)
                    .expect("a device attention backward");
                let (dq, dk, dv) = &expected;
                for ((name, actual), expected) in
                    ["dq", "dk", "dv"].iter().zip(&actual).zip([dq, dk, dv])
                {
                    assert_close(
                        &format!("{name}, {heads}x{head_dim}, budget {budget}"),
                        &gpu.download(actual).expect("a download").data,
                        &expected.data,
                        0.02,
                    );
                }
            }
        }
    }

    /// The other parity test runs four columns to a head, which is not a
    /// width the fused kernel knows, so it never reaches the branch. This one
    /// does: sixty-four columns to a head, keys and values two windows on one
    /// buffer, and a token count that leaves the last tile of each axis part
    /// empty, which is the only place the kernel masks.
    #[test]
    fn the_fused_attention_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        if gpu.context.flash.is_none() {
            eprintln!("skipped: no fused attention module on this device");
            return;
        }
        let (heads, head_dim, tokens) = (2usize, crate::cuda_flash::TILE, 70usize);
        let units = heads * head_dim;
        let fused = Matrix::from_vec(tokens, 3 * units, ramp(tokens * 3 * units, 0.2));
        let part = |index: usize| {
            Matrix::from_vec(
                tokens,
                units,
                (0..tokens)
                    .flat_map(|row| {
                        let start = row * fused.cols + index * units;
                        fused.data[start..start + units].to_vec()
                    })
                    .collect(),
            )
        };
        let (queries, keys, values) = (part(0), part(1), part(2));
        let expected = host_attention(&queries, &keys, &values, heads, heads, false);

        let fused = gpu.upload_matrix(&fused).expect("an upload");
        let actual = gpu
            .attention(
                fused.part(0, units),
                fused.part(1, units),
                fused.part(2, units),
                heads,
                false,
            )
            .expect("a device attention");
        assert_close(
            "fused attention",
            &gpu.download(&actual).expect("a download").data,
            &expected.data,
            0.02,
        );
    }

    #[test]
    fn device_reshapes_match_the_host_or_skip_without_a_device() {
        let Some(gpu) = device() else { return };
        let source = map(3, 4, 5, 0.25);
        let device = upload_map(&gpu, &source);
        // These three move values without arithmetic, so the only error is the
        // narrowing the upload did.
        let narrowing = 0.005;

        let tokens = gpu.to_tokens(&device).expect("a transpose");
        assert_close(
            "to_tokens",
            &gpu.download(&tokens).expect("a download").data,
            &source.to_tokens().data,
            narrowing,
        );
        let back = gpu.to_map(&tokens, 4, 5).expect("a transpose");
        assert_close(
            "to_map",
            &download_map(&gpu, &back).expect("a download").data,
            &source.data,
            narrowing,
        );

        let large = gpu.upsample_nearest(&device, 2).expect("an upsample");
        assert_close(
            "upsample",
            &download_map(&gpu, &large).expect("a download").data,
            &upsample_nearest(&source, 2).data,
            narrowing,
        );
    }

    #[test]
    fn the_gated_feed_forward_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        let projected = Matrix::from_vec(3, 8, ramp(24, 0.45));
        let inner = 4;
        // The bias the projection did not apply, which this kernel owes the
        // gate and the value alike.
        let bias = ramp(8, 0.1);
        let mut expected = Vec::with_capacity(12);
        for row in projected.data.chunks(8) {
            for index in 0..inner {
                expected.push(
                    (row[index] + bias[index])
                        * crate::ffn::gelu(row[inner + index] + bias[inner + index]),
                );
            }
        }
        let bias = gpu.context.stream.clone_htod(&bias).expect("a bias upload");
        let actual = gpu
            .gelu_gate(
                &gpu.upload_matrix(&projected).expect("an upload"),
                Some(&bias),
            )
            .expect("a gate");
        assert_close(
            "gelu gate",
            &gpu.download(&actual).expect("a download").data,
            &expected,
            0.01,
        );
    }

    #[test]
    fn a_whole_unet_pass_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        let config = crate::unet::tests::tiny();
        let path = std::env::temp_dir().join("rusting_brain_cuda_unet.safetensors");
        crate::unet::tests::checkpoint(&config, &path);
        let mut model = Unet::load_at(&path, config.clone(), Precision::F32).expect("a model");
        std::fs::remove_file(&path).ok();

        let latent = map(config.in_channels, 8, 8, 0.31);
        let mut text = Matrix::new(5, config.cross_dim);
        text.data.copy_from_slice(&ramp(5 * config.cross_dim, 0.62));
        let conditioning = crate::mmdit::Conditioning {
            text,
            pooled: Vec::new(),
            height: 8,
            width: 8,
            guidance: 0.0,
        };

        let expected = model
            .forward(&latent, 1.0, &conditioning)
            .expect("a host pass");
        model.attach_device(gpu).expect("an upload");
        let actual = model
            .forward(&latent, 1.0, &conditioning)
            .expect("a device pass");

        assert_eq!(
            (actual.channels, actual.height, actual.width),
            (expected.channels, expected.height, expected.width)
        );
        assert_close("unet", &actual.data, &expected.data, 0.05);
    }

    #[test]
    fn a_controlnet_on_the_device_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        let config = crate::unet::tests::tiny();
        let path = std::env::temp_dir().join("rusting_brain_cuda_controlled_unet.safetensors");
        crate::unet::tests::checkpoint(&config, &path);
        let mut model = Unet::load_at(&path, config.clone(), Precision::F32).expect("a model");
        std::fs::remove_file(&path).ok();

        let latent = map(config.in_channels, 8, 8, 0.31);
        let mut text = Matrix::new(5, config.cross_dim);
        text.data.copy_from_slice(&ramp(5 * config.cross_dim, 0.62));
        let conditioning = crate::mmdit::Conditioning {
            text,
            pooled: Vec::new(),
            height: 8,
            width: 8,
            guidance: 0.0,
        };
        let plain = model
            .forward(&latent, 1.0, &conditioning)
            .expect("a host pass");
        // Two controls, so their corrections have to add up.
        let net = crate::unet::tests::controlnet(&config);
        model.set_controls(
            [(0.43, 0.8), (0.17, 0.5)]
                .map(|(seed, scale)| crate::unet::Control {
                    net: net.clone(),
                    image: map(3, 64, 64, seed),
                    scale,
                })
                .to_vec(),
        );
        let expected = model
            .forward(&latent, 1.0, &conditioning)
            .expect("a host pass");
        model.attach_device(gpu).expect("an upload");
        let actual = model
            .forward(&latent, 1.0, &conditioning)
            .expect("a device pass");

        let farthest = |from: &[f32]| {
            from.iter()
                .zip(&expected.data)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        };
        // A device pass that dropped the controls would land on `plain`.
        let (moved, error) = (farthest(&plain.data), farthest(&actual.data));
        assert!(
            error < moved / 10.0,
            "the device is {error} off, and the controls move the output {moved}"
        );
        assert_close("controlled unet", &actual.data, &expected.data, 0.05);
    }

    #[test]
    fn a_whole_vae_decode_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        let config = crate::vae::tests::tiny();
        let path = std::env::temp_dir().join("rusting_brain_cuda_vae.safetensors");
        crate::vae::tests::checkpoint(&config, &path);
        let mut decoder = VaeDecoder::load(&path, "decoder.", config.clone()).expect("a decoder");
        std::fs::remove_file(&path).ok();

        let latent = map(config.latent_channels, 4, 4, 0.18);
        let expected = decoder.decode(&latent).expect("a host decode");
        decoder.attach_device(gpu).expect("an upload");
        let actual = decoder.decode(&latent).expect("a device decode");

        assert_eq!(
            (actual.channels, actual.height, actual.width),
            (expected.channels, expected.height, expected.width)
        );
        assert_close("vae", &actual.data, &expected.data, 0.05);
    }

    #[test]
    fn a_whole_vae_encode_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        let config = crate::vae::tests::tiny();
        let path = std::env::temp_dir().join("rusting_brain_cuda_vae_encoder.safetensors");
        crate::vae::tests::checkpoint(&config, &path);
        let mut encoder = VaeEncoder::load(&path, "encoder.", config).expect("an encoder");
        std::fs::remove_file(&path).ok();

        // Every halving convolution pads one extra row and column at the end.
        let image = map(3, 8, 8, 0.5);
        let expected = encoder.encode(&image).expect("a host encode");
        encoder.attach_device(gpu).expect("an upload");
        let actual = encoder.encode(&image).expect("a device encode");

        assert_eq!(
            (actual.channels, actual.height, actual.width),
            (expected.channels, expected.height, expected.width)
        );
        assert_close("vae encoder", &actual.data, &expected.data, 0.05);
    }

    #[test]
    fn a_vae_decode_that_overflows_fp16_fails_instead_of_returning_nan() {
        let Some(gpu) = device() else { return };
        let config = crate::vae::tests::tiny();
        let path = std::env::temp_dir().join("rusting_brain_cuda_vae_overflow.safetensors");
        crate::vae::tests::checkpoint(&config, &path);
        let mut decoder = VaeDecoder::load(&path, "decoder.", config.clone()).expect("a decoder");
        std::fs::remove_file(&path).ok();

        // Past FP16's 65504 once scaled, so the upload alone is infinite, while
        // the host's group norms keep its f32 decode finite.
        let mut latent = map(config.latent_channels, 4, 4, 0.18);
        latent.data.iter_mut().for_each(|value| *value *= 1.0e6);
        let host = decoder.decode(&latent).expect("a host decode");
        assert!(host.data.iter().all(|value| value.is_finite()));
        decoder.attach_device(gpu).expect("an upload");
        let error = decoder.decode(&latent).expect_err("an overflow error");
        assert!(error.to_string().contains("overflowed FP16"), "{error}");
    }

    #[test]
    fn a_whole_clip_tower_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        let config = crate::clip::tests::tiny();
        let path = std::env::temp_dir().join("rusting_brain_cuda_clip.safetensors");
        crate::clip::tests::checkpoint(&config, &path, false);
        let mut encoder =
            ClipTextEncoder::load_at(&path, "", config.clone(), Precision::F32).expect("a tower");
        std::fs::remove_file(&path).ok();

        let ids: Vec<u32> = (0..6)
            .map(|index| index % config.vocab_size as u32)
            .collect();
        let (expected_skipped, expected) = encoder.forward_skipping(&ids, 1).expect("a host pass");
        encoder.attach_device(gpu).expect("an upload");
        let (actual_skipped, actual) = encoder.forward_skipping(&ids, 1).expect("a device pass");

        assert_close("clip", &actual.data, &expected.data, 0.05);
        assert_close(
            "clip skipped",
            &actual_skipped.data,
            &expected_skipped.data,
            0.05,
        );
    }

    #[test]
    fn a_whole_vit_tower_matches_the_host_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        let path = std::env::temp_dir().join("rusting_brain_cuda_vit.safetensors");
        crate::vit_encoder::tests::checkpoint(&path, false);
        // CLIP's layout, then DINOv2's: no pre-norm and the exact GELU.
        for dinov2 in [false, true] {
            let mut tower =
                VitEncoder::load(&path, "", crate::vit_encoder::tests::tiny(), Precision::F32)
                    .expect("a tower");
            if dinov2 {
                tower.pre_norm = None;
                tower.exact_gelu = true;
            }
            let image = map(
                tower.channels(),
                tower.image_size(),
                tower.image_size(),
                0.27,
            );
            let expected = tower.encode(&image).expect("a host pass");
            tower.attach_device(gpu.clone()).expect("an upload");
            let actual = tower.encode(&image).expect("a device pass");

            assert_eq!((actual.rows, actual.cols), (expected.rows, expected.cols));
            assert_close("vit", &actual.data, &expected.data, 0.05);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_vit_batch_keeps_each_image_to_itself_or_skips_without_a_device() {
        let Some(gpu) = device() else { return };
        let path = std::env::temp_dir().join("rusting_brain_cuda_vit_batch.safetensors");
        crate::vit_encoder::tests::checkpoint(&path, false);
        let mut tower =
            VitEncoder::load(&path, "", crate::vit_encoder::tests::tiny(), Precision::F32)
                .expect("a tower");
        std::fs::remove_file(&path).ok();

        // Three different images. The host comparison has to allow for FP16,
        // and this tiny tower is dominated by its norms, so a token that
        // attended to another image moves by less than that allowance. The
        // single-image device passes run the same kernels, so the batch has to
        // match them far more tightly, and that is what catches the mixing.
        let size = tower.image_size();
        let images: Vec<_> = [0.0, 1.5, 3.0]
            .into_iter()
            .map(|seed| map(tower.channels(), size, size, seed))
            .collect();
        let expected: Vec<_> = images
            .iter()
            .map(|image| tower.encode(image).expect("a host pass"))
            .collect();
        tower.attach_device(gpu).expect("an upload");
        let actual = tower.encode_batch(&images).expect("a device batch");

        assert_eq!(actual.len(), expected.len());
        for ((actual, expected), image) in actual.iter().zip(&expected).zip(&images) {
            assert_eq!((actual.rows, actual.cols), (expected.rows, expected.cols));
            assert_close("vit batch", &actual.data, &expected.data, 0.05);
            let single = tower.encode(image).expect("a single device pass");
            assert_close("vit batch beside one", &actual.data, &single.data, 0.002);
        }
    }
}
