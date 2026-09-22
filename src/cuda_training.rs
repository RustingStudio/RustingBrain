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
    time::Instant,
};

/// Backwards-compatible name for the shared session measurements.
pub use crate::accelerator::AcceleratorStats as CudaTrainingStats;
pub(crate) use crate::accelerator::MIB;
/// Re-exported so `cuda_training::estimate_tensor_memory_mib` keeps resolving;
/// the estimate itself is backend-independent and lives in `accelerator`.
pub use crate::accelerator::estimate_tensor_memory_mib;

const KERNELS: &str = r#"
// bfloat16 helpers, written against the raw bit pattern rather than
// `cuda_bf16.h`: NVRTC compiles this string without a header search path, and
// bf16 is just FP32 with the low 16 mantissa bits dropped. Rounding is
// round-to-nearest-even, the same rule the tensor cores use, so a value that
// makes a round trip through bf16 and back matches what cuBLAS saw.
typedef unsigned short bf16_t;
__device__ __forceinline__ float bf16_to_f32(bf16_t b){return __uint_as_float((unsigned int)b<<16);}
__device__ __forceinline__ bf16_t f32_to_bf16(float v){
  unsigned int u=__float_as_uint(v);
  if((u&0x7f800000u)==0x7f800000u&&(u&0x007fffffu))return (bf16_t)0x7fc0u;
  unsigned int r=((u>>16)&1u)+0x7fffu;
  return (bf16_t)((u+r)>>16);
}
// A GEMM operand is stored narrow whenever the context is in reduced
// precision, so the kernels that produce one take their destination as raw
// bytes plus a `narrow` flag rather than a typed pointer. The branch is
// uniform across the whole grid, and writing two bytes per element instead of
// four is what the caller is paying for.
__device__ __forceinline__ void store_act(void*p,size_t i,float v,int narrow){
  if(narrow)((bf16_t*)p)[i]=f32_to_bf16(v); else ((float*)p)[i]=v;
}
__device__ __forceinline__ void store_act4(void*p,size_t i,float4 v,int narrow){
  if(narrow){ushort4 o;o.x=f32_to_bf16(v.x);o.y=f32_to_bf16(v.y);o.z=f32_to_bf16(v.z);o.w=f32_to_bf16(v.w);((ushort4*)p)[i]=o;}
  else ((float4*)p)[i]=v;
}
__device__ __forceinline__ float load_act(const void*p,size_t i,int narrow){
  return narrow ? __uint_as_float((unsigned)((const unsigned short*)p)[i]<<16) : ((const float*)p)[i];
}
// `dst = src`, narrowed or not, for an operand whose producer is a cuBLAS call
// rather than one of the kernels above.
extern "C" __global__ void cast_act(void*dst,const float*src,int n,int narrow){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)store_act(dst,i,src[i],narrow);
}

extern "C" __global__ void bias_act(float *x,const float*b,int rows,int cols,int act){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*cols){float v=x[i]+b[i%cols];if(act==1)v=v>0?v:0;else if(act==2)v=1.f/(1.f+expf(-v));else if(act==3)v=tanhf(v);x[i]=v;}}
extern "C" __global__ void output_delta(float*d,const float*y,const float*t,int n,int act){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=t[i]-y[i];float a=y[i];if(act==1)v*=a>0;else if(act==2)v*=a*(1-a);else if(act==3)v*=1-a*a;d[i]=v;}}
extern "C" __global__ void act_derivative(float*d,const float*a,int n,int act){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=a[i];if(act==1)d[i]*=v>0;else if(act==2)d[i]*=v*(1-v);else if(act==3)d[i]*=1-v*v;}}
extern "C" __global__ void grad_b(float*g,const float*d,int batch,int out){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<out){float s=0;for(int b=0;b<batch;b++)s+=d[b*out+i];g[i]=s/(float)batch;}}
extern "C" __global__ void sgd(float*x,const float*g,int n,float lr){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]+=lr*g[i];}
extern "C" __global__ void adam(float*x,const float*g,float*m,float*v,int n,float lr,float b1,float b2,float eps,float c1,float c2,float wd){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float q=g[i];float mi=b1*m[i]+(1-b1)*q;float vi=b2*v[i]+(1-b2)*q*q;m[i]=mi;v[i]=vi;x[i]-=lr*wd*x[i];x[i]+=lr*(mi/c1)/(sqrtf(vi/c2)+eps);}}
extern "C" __global__ void mse_sum(float *out,const float*y,const float*t,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=y[i]-t[i];atomicAdd(out,z*z/(float)n);}}
extern "C" __global__ void mse_epoch_sum(float *out,const float*y,const float*t,int n,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=y[i]-t[i];atomicAdd(out,z*z*scale);}}
extern "C" __global__ void gather_rows(float*out,const float*all,const unsigned int*order,int start,int rows,int width){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*width){int r=i/width,c=i%width;out[i]=all[order[start+r]*width+c];}}
extern "C" __global__ void scale_inplace(float*g,int n,float s){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)g[i]*=s;}

// im2col and its transpose for a trainable convolution, in FP32.
//
// `crate::cuda_image` has an FP16 im2col for inference; this pair is the
// training one, so it keeps the precision the optimizer needs and it has the
// scatter half that a backward pass needs. Layout: the input is
// `[batch * height * width, channels]` with the channel fastest, and a column
// row holds `(ky * kernel + kx) * channels + channel`, which is the ordering
// the weight `[out_channels, in_channels * kernel * kernel]` was trained in.
extern "C" __global__ void im2col_train(float*columns,const float*input,
    int batch,int channels,int height,int width,int kernel,int stride,int pad,int out_h,int out_w){
  long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  long long patch=(long long)channels*kernel*kernel;
  long long total=(long long)batch*out_h*out_w*patch;
  if(idx>=total)return;
  long long row=idx/patch,off=idx%patch;
  int c=(int)(off%channels);long long t=off/channels;
  int kx=(int)(t%kernel),ky=(int)(t/kernel);
  int plane=out_h*out_w;
  int b=(int)(row/plane),rem=(int)(row%plane);
  int oy=rem/out_w,ox=rem%out_w;
  int y=oy*stride+ky-pad,x=ox*stride+kx-pad;
  float v=0.f;
  if(y>=0&&y<height&&x>=0&&x<width)v=input[(((long long)b*height+y)*width+x)*channels+c];
  columns[idx]=v;
}
// One thread per input element, gathering every column entry that read it.
// A scatter would need atomics and would not be reproducible run to run; this
// costs kernel*kernel reads per element and is deterministic.
extern "C" __global__ void col2im_train(float*grad,const float*columns,
    int batch,int channels,int height,int width,int kernel,int stride,int pad,int out_h,int out_w){
  long long idx=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  long long total=(long long)batch*height*width*channels;
  if(idx>=total)return;
  int c=(int)(idx%channels);long long p=idx/channels;
  int x=(int)(p%width);long long q=p/width;
  int y=(int)(q%height);int b=(int)(q/height);
  long long patch=(long long)channels*kernel*kernel;
  float s=0.f;
  for(int ky=0;ky<kernel;ky++){
    int sy=y+pad-ky;if(sy<0||sy%stride)continue;int oy=sy/stride;if(oy>=out_h)continue;
    for(int kx=0;kx<kernel;kx++){
      int sx=x+pad-kx;if(sx<0||sx%stride)continue;int ox=sx/stride;if(ox>=out_w)continue;
      long long row=((long long)b*out_h+oy)*out_w+ox;
      s+=columns[row*patch+(long long)(ky*kernel+kx)*channels+c];
    }
  }
  grad[idx]=s;
}
extern "C" __global__ void scatter_rows_neg(float*grad,const float*upstream,const unsigned int*rows_of,int rows,int width){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows*width){int r=i/width,c=i%width;atomicAdd(&grad[rows_of[r]*width+c],-upstream[i]);}}

// The transformer path's fused kernels. Everything a decoder layer does
// between two matmuls lives here, so a layer runs device-side end to end: see
// `gpu_model` for how they are sequenced.
// One block of ROW_THREADS threads cooperates on one row, striding over the
// columns so neighbouring threads touch neighbouring addresses. The previous
// one-thread-per-row shape read a row serially, which made every access
// uncoalesced: adjacent threads were `cols` floats apart. Rows outnumber
// blocks, so each block walks a grid-stride slice of them.
#define ROW_THREADS 256
__device__ __forceinline__ float row_sum(float v,float*red){
  int tid=threadIdx.x;red[tid]=v;__syncthreads();
  for(int s=ROW_THREADS/2;s>0;s>>=1){if(tid<s)red[tid]+=red[tid+s];__syncthreads();}
  float r=red[0];__syncthreads();return r;
}
// `do_copy` asks for a plain duplicate of `x` in `copy`. A residual branch
// needs one: the projection below it accumulates with `beta = 1` over the
// block input. Writing it here is a store on a row this kernel has already
// read, which is cheaper than the separate device-to-device copy it replaces.
extern "C" __global__ void rmsnorm_fwd(void*out,float*inv,const float*x,const float*w,int rows,int cols,float eps,int narrow,float*copy,int do_copy){
  __shared__ float red[ROW_THREADS];int tid=threadIdx.x;
  for(int r=blockIdx.x;r<rows;r+=gridDim.x){
    const float*src=x+(size_t)r*cols;
    float*dup=copy+(size_t)r*cols;
    float s=0.f;for(int c=tid;c<cols;c+=ROW_THREADS){float v=src[c];if(do_copy)dup[c]=v;s+=v*v;}
    s=row_sum(s,red);
    float t=1.f/sqrtf(s/(float)cols+eps);
    if(tid==0)inv[r]=t;
    size_t base=(size_t)r*cols;
    for(int c=tid;c<cols;c+=ROW_THREADS)store_act(out,base+c,src[c]*t*w[c],narrow);
  }
}
// `use_smem` says the host sized the dynamic shared memory to hold `cols`
// floats, so the scale gradient accumulates per block and lands in `gw` with
// one atomic per column instead of one per element. A model wide enough to
// overflow shared memory falls back to the direct atomic.
extern "C" __global__ void rmsnorm_bwd(float*gx,float*gw,const float*x,const float*gy,const float*w,const float*inv,int rows,int cols,int use_smem,const float*res,int add_res,void*copy,int copy_mode){
  extern __shared__ float acc[];
  __shared__ float red[ROW_THREADS];int tid=threadIdx.x;
  if(use_smem)for(int c=tid;c<cols;c+=ROW_THREADS)acc[c]=0.f;
  __syncthreads();
  for(int r=blockIdx.x;r<rows;r+=gridDim.x){
    size_t base=(size_t)r*cols;
    const float*src=x+base;const float*up=gy+base;float t=inv[r];
    float proj=0.f;for(int c=tid;c<cols;c+=ROW_THREADS)proj+=up[c]*w[c]*src[c];
    proj=row_sum(proj,red);
    float shared=proj*t*t*t/(float)cols;
    float*dst=gx+base;
    for(int c=tid;c<cols;c+=ROW_THREADS){
      float g_in=up[c]*w[c]*t-src[c]*shared;
      // The residual carries the upstream gradient past the branch, so it is
      // one more addend here rather than a separate pass over the whole tensor.
      if(add_res)g_in+=res[base+c];
      dst[c]=g_in;
      // The GEMMs downstream want this gradient as an operand, so the copy
      // they read is written here instead of by a separate pass.
      if(copy_mode)store_act(copy,base+c,g_in,copy_mode-1);
      float g=up[c]*src[c]*t;
      if(use_smem)acc[c]+=g;else atomicAdd(&gw[c],g);
    }
  }
  if(use_smem){__syncthreads();for(int c=tid;c<cols;c+=ROW_THREADS)atomicAdd(&gw[c],acc[c]);}
}
// `width` and `off` locate the rotated block inside a wider row: queries and
// keys are two slices of one fused projection output, so they share a row
// stride and differ only in where they start.
extern "C" __global__ void rope_rotate(void*x,const float*cs,const float*sn,int rows,int heads,int head_dim,int seq_len,float dir,int width,int off,int narrow){int half=head_dim/2;int total=rows*heads*half;int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;int ch=i%half;int h=(i/half)%heads;int r=i/(half*heads);int t=(r%seq_len)*half+ch;float c=cs[t];float s=dir*sn[t];size_t base=(size_t)r*width+off+(size_t)h*head_dim+ch;float lo=load_act(x,base,narrow);float hi=load_act(x,base+half,narrow);store_act(x,base,lo*c-hi*s,narrow);store_act(x,base+half,hi*c+lo*s,narrow);}
// Attention rows are short (one per query position, and only the positions up
// to it are visible), so a whole block per row would leave most of its threads
// idle. One warp per row instead, reducing through shuffles with no shared
// memory and no block-wide barrier.
#define WARP 32
__device__ __forceinline__ float warp_max(float v){
  for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,o));
  return __shfl_sync(0xffffffff,v,0);
}
__device__ __forceinline__ float warp_sum(float v){
  for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffff,v,o);
  return __shfl_sync(0xffffffff,v,0);
}
// Row softmax in place, plus the one log-sum-exp per query the backward pass
// needs to rebuild these same probabilities. Writing that number here is what
// lets the forward cache hold a few kilobytes per layer instead of the whole
// probability matrix.
//
// `causal` masks each query to the keys at or before it, which is what a
// decoder wants. Cross-attention clears the flag: it has two independent
// lengths and every query sees every key, so `cols` is the key count rather
// than the query count and the row index carries no position at all.
extern "C" __global__ void attention_softmax_lse(float*s,float*lse,int rows,int cols,int causal){
  int lane=threadIdx.x%WARP,warps=blockDim.x/WARP;
  int stride=gridDim.x*warps;
  for(int i=blockIdx.x*warps+threadIdx.x/WARP;i<rows;i+=stride){
    int seq_len=cols;
    int vis=causal?i%cols+1:cols;float*row=s+(size_t)i*seq_len;
    float m=-3.0e38f;for(int j=lane;j<vis;j+=WARP)m=fmaxf(m,row[j]);
    m=warp_max(m);
    float part=0.f;for(int j=lane;j<vis;j+=WARP){float e=__expf(row[j]-m);row[j]=e;part+=e;}
    float sum=warp_sum(part);
    if(lane==0)lse[i]=m+__logf(sum);
    if(sum>0.f){float inv=1.f/sum;for(int j=lane;j<vis;j+=WARP)row[j]*=inv;}
    for(int j=vis+lane;j<seq_len;j+=WARP)row[j]=0.f;
  }
}
extern "C" __global__ void attention_softmax_bwd(float*g,const float*p,int rows,int cols,int causal){
  int lane=threadIdx.x%WARP,warps=blockDim.x/WARP;
  int stride=gridDim.x*warps;
  for(int i=blockIdx.x*warps+threadIdx.x/WARP;i<rows;i+=stride){
    int seq_len=cols;
    int vis=causal?i%cols+1:cols;float*gr=g+(size_t)i*seq_len;const float*pr=p+(size_t)i*seq_len;
    float part=0.f;for(int j=lane;j<vis;j+=WARP)part+=pr[j]*gr[j];
    float dot=warp_sum(part);
    for(int j=lane;j<vis;j+=WARP)gr[j]=pr[j]*(gr[j]-dot);
    for(int j=vis+lane;j<seq_len;j+=WARP)gr[j]=0.f;
  }
}
// Rebuild the attention probabilities from the scaled scores and the stored
// log-sum-exp. The forward pass kept only that one number per query, so the
// backward pass pays an exponential per element instead of a second reduction,
// and the probability matrix lives only for the length of one block's backward.
extern "C" __global__ void attention_probs_from_lse(
    float*s,const float*lse,int rows,int cols,int causal){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=rows*cols)return;
  int r=i/cols,c=i-r*cols;
  s[i]=(!causal||c<=r%cols)?__expf(s[i]-lse[r]):0.f;
}

// Gate and up share an input and a shape, so they are one GEMM at twice the
// width: `gu` holds both halves of every row, gate first. cuBLAS is close to
// twice as fast on the wider shape, which is why the two projections are not
// kept apart here.
__device__ inline float silu_mul(float x,float u){return x*u/(1.f+expf(-x));}
__device__ inline void silu_mul_grad(float x,float u,float p,float&dg,float&du){float s=1.f/(1.f+expf(-x));dg=p*u*s*(1.f+x*(1.f-s));du=p*x*s;}
// These two are memory bound, so an even width is read and written four floats
// at a time. The host picks the thread count to match; the scalar tail below
// serves an odd width, which no preset uses but a caller may configure.
extern "C" __global__ void swiglu_fwd(void*h,const float*gu,int rows,int width,int narrow){int i=blockIdx.x*blockDim.x+threadIdx.x;
 if((width&3)==0){int q=width>>2;if(i>=rows*q)return;int r=i/q;size_t b=(size_t)r*2*q+(i-r*q);const float4*s=(const float4*)gu;float4 g=s[b],u=s[b+q];float4 o;o.x=silu_mul(g.x,u.x);o.y=silu_mul(g.y,u.y);o.z=silu_mul(g.z,u.z);o.w=silu_mul(g.w,u.w);store_act4(h,i,o,narrow);return;}
 if(i>=rows*width)return;int r=i/width;size_t b=(size_t)r*2*width+(i-r*width);store_act(h,i,silu_mul(gu[b],gu[b+width]),narrow);}
extern "C" __global__ void swiglu_bwd(void*ggu,const float*gu,const float*gh,int rows,int width,int narrow){int i=blockIdx.x*blockDim.x+threadIdx.x;
 if((width&3)==0){int q=width>>2;if(i>=rows*q)return;int r=i/q;size_t b=(size_t)r*2*q+(i-r*q);const float4*s=(const float4*)gu;float4 g=s[b],u=s[b+q],p=((const float4*)gh)[i];float4 dg,du;silu_mul_grad(g.x,u.x,p.x,dg.x,du.x);silu_mul_grad(g.y,u.y,p.y,dg.y,du.y);silu_mul_grad(g.z,u.z,p.z,dg.z,du.z);silu_mul_grad(g.w,u.w,p.w,dg.w,du.w);store_act4(ggu,b,dg,narrow);store_act4(ggu,b+q,du,narrow);return;}
 if(i>=rows*width)return;int r=i/width;size_t b=(size_t)r*2*width+(i-r*width);float dg,du;silu_mul_grad(gu[b],gu[b+width],gh[i],dg,du);store_act(ggu,b,dg,narrow);store_act(ggu,b+width,du,narrow);}
// AdaLN, in the shape `src/adaln.rs` describes: the conditioning is one row
// per sequence and every token of that sequence reads it. A row of `triple` is
// `[shift, scale, gate]`, each `cols` wide, so the gate the residual below
// needs is the same buffer at offset `2 * cols` rather than a tensor of its
// own.
extern "C" __global__ void adaln_modulate(float*out,const float*x,const float*triple,int rows,int cols,int seq_len){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;int r=i/cols,c=i-r*cols;
 const float*t=triple+(size_t)(r/seq_len)*3*cols;out[i]=x[i]*(1.f+t[cols+c])+t[c];}
// The shift and scale gradients are sums over a sequence's tokens, so they land
// with atomics: the whole conditioning gradient is one zeroed buffer that every
// sub-layer of every block accumulates into, and it is downloaded once.
extern "C" __global__ void adaln_modulate_bwd(float*gx,float*gt,const float*gy,const float*x,const float*triple,int rows,int cols,int seq_len){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;int r=i/cols,c=i-r*cols;
 size_t base=(size_t)(r/seq_len)*3*cols;float u=gy[i];
 gx[i]=u*(1.f+triple[base+cols+c]);
 atomicAdd(&gt[base+c],u);atomicAdd(&gt[base+cols+c],u*x[i]);}
// `out = residual + gate * branch`, the other half of an AdaLN sub-layer.
extern "C" __global__ void gate_residual_fwd(float*out,const float*residual,const float*branch,const float*triple,int rows,int cols,int seq_len){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;int r=i/cols,c=i-r*cols;
 out[i]=residual[i]+triple[(size_t)(r/seq_len)*3*cols+2*cols+c]*branch[i];}
// `dL/dresidual` is the upstream gradient unchanged, so it is not written: the
// caller already holds that buffer.
extern "C" __global__ void gate_residual_bwd(float*gb,float*gt,const float*branch,const float*triple,const float*gout,int rows,int cols,int seq_len){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*cols)return;int r=i/cols,c=i-r*cols;
 size_t gate=(size_t)(r/seq_len)*3*cols+2*cols+c;float u=gout[i];
 gb[i]=u*triple[gate];atomicAdd(&gt[gate],u*branch[i]);}
extern "C" __global__ void softmax_lse(float*p,float*lse,const float*x,int rows,int cols){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;const float*src=x+(size_t)r*cols;float*dst=p+(size_t)r*cols;float m=-3.0e38f;for(int c=0;c<cols;c++)m=fmaxf(m,src[c]);float sum=0;for(int c=0;c<cols;c++){float e=expf(src[c]-m);dst[c]=e;sum+=e;}lse[r]=m+logf(sum);if(sum>0)for(int c=0;c<cols;c++)dst[c]/=sum;}
extern "C" __global__ void topk_gate(int*expert_of,float*gate_of,const float*p,const int*valid,int rows,int experts,int top_k){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;int base=r*top_k;if(valid!=0&&valid[r]==0){for(int k=0;k<top_k;k++){expert_of[base+k]=-1;gate_of[base+k]=0;}return;}const float*row=p+(size_t)r*experts;float total=0;for(int k=0;k<top_k;k++){int best=-1;for(int e=0;e<experts;e++){int taken=0;for(int j=0;j<k;j++)if(expert_of[base+j]==e)taken=1;if(taken)continue;if(best<0||row[e]>row[best])best=e;}expert_of[base+k]=best;gate_of[base+k]=row[best];total+=row[best];}if(total>0)for(int k=0;k<top_k;k++)gate_of[base+k]/=total;}
extern "C" __global__ void gather_scale_rows(float*out,const float*src,const unsigned int*rows_of,const float*scale,int rows,int width){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*width)return;int r=i/width;int c=i%width;out[i]=scale[r]*src[(size_t)rows_of[r]*width+c];}
extern "C" __global__ void scatter_add_scaled(float*out,const float*src,const unsigned int*rows_of,const float*scale,int rows,int width){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*width)return;int r=i/width;int c=i%width;atomicAdd(&out[(size_t)rows_of[r]*width+c],scale[r]*src[i]);}
extern "C" __global__ void scatter_add_rows(float*out,const float*src,const unsigned int*rows_of,int rows,int width){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*width)return;int r=i/width;int c=i%width;atomicAdd(&out[(size_t)rows_of[r]*width+c],src[i]);}
// One warp per routed row. A thread per row reads `width` consecutive floats
// alone, which is the worst access pattern the memory system has; a warp
// striding the row reads it in coalesced 128-byte lines instead and was about
// nine times faster on the default MoE configuration.
extern "C" __global__ void row_dot_scatter(float*out,const unsigned int*out_of,const float*a,const unsigned int*rows_of,const float*b,int rows,int width){
  int lane=threadIdx.x&(WARP-1);
  int warp=(blockIdx.x*blockDim.x+threadIdx.x)/WARP;
  int warps=(gridDim.x*blockDim.x)/WARP;
  for(int r=warp;r<rows;r+=warps){
    const float*x=a+(size_t)rows_of[r]*width;
    const float*y=b+(size_t)r*width;
    float s=0;
    for(int c=lane;c<width;c+=WARP)s+=x[c]*y[c];
    s=warp_sum(s);
    if(lane==0)out[out_of[r]]=s;
  }
}
extern "C" __global__ void moe_stats(float*sums,const float*p,const float*lse,const int*valid,int rows,int experts){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;if(valid!=0&&valid[r]==0)return;const float*row=p+(size_t)r*experts;for(int e=0;e<experts;e++)atomicAdd(&sums[e],row[e]);atomicAdd(&sums[experts],lse[r]*lse[r]);}
extern "C" __global__ void moe_grad_probs(float*gp,const float*gg,const float*p,const int*expert_of,int rows,int experts,int top_k){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;int base=r*top_k;if(expert_of[base]<0)return;const float*row=p+(size_t)r*experts;float total=0;float weighted=0;for(int k=0;k<top_k;k++){int e=expert_of[base+k];total+=row[e];weighted+=gg[base+k]*row[e];}if(total<=0)return;float*dst=gp+(size_t)r*experts;for(int k=0;k<top_k;k++){int e=expert_of[base+k];dst[e]+=gg[base+k]/total-weighted/(total*total);}}
extern "C" __global__ void moe_aux_grad(float*gp,const float*loads,const int*expert_of,int rows,int experts,int top_k,float scale){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*experts)return;int r=i/experts;int e=i%experts;if(expert_of[r*top_k]<0)return;gp[i]+=scale*loads[e];}
extern "C" __global__ void router_grad_logits(float*gl,const float*gp,const float*p,const float*lse,const int*expert_of,int rows,int experts,int top_k,float zfactor){int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=rows)return;float*dst=gl+(size_t)r*experts;if(expert_of[r*top_k]<0){for(int e=0;e<experts;e++)dst[e]=0;return;}const float*prob=p+(size_t)r*experts;const float*up=gp+(size_t)r*experts;float dot=0;for(int e=0;e<experts;e++)dot+=prob[e]*up[e];float f=zfactor*lse[r];for(int e=0;e<experts;e++)dst[e]=prob[e]*(up[e]-dot)+f*prob[e];}
// `keep` says `a` already holds a value worth adding to. A gradient buffer the
// optimizer has not refilled since the last step holds a stale one instead, and
// the first writer of the step overwrites it rather than paying for a zeroing
// pass over every parameter.
extern "C" __global__ void add_inplace(float*a,const float*b,int off,int n,int keep){int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(((n|off)&3)==0){int q=n>>2;if(i>=q)return;float4*d=(float4*)a;float4 y=((const float4*)(b+off))[i];if(keep){float4 x=d[i];y.x+=x.x;y.y+=x.y;y.z+=x.z;y.w+=x.w;}d[i]=y;return;}
 if(i<n)a[i]=keep?a[i]+b[off+i]:b[off+i];}

// Cross-entropy over one chunk of logit rows, in place: the row comes in as
// logits and leaves as dL/dlogits, so the largest tensor in a training step is
// never held twice. `target` is the next token, or -1 for a row that predicts
// nothing (padding, and the last position of every sequence), whose gradient
// is zero. One block of CE_THREADS threads per row; `loss` accumulates the
// unscaled negative log-likelihood, which the host divides by the predicted
// count.
extern "C" __global__ void to_bf16(bf16_t*out,const float*in,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=f32_to_bf16(in[i]);
}
extern "C" __global__ void from_bf16(float*out,const bf16_t*in,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=bf16_to_f32(in[i]);
}
extern "C" __global__ void accumulate_bf16(float*acc,const bf16_t*in,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)acc[i]+=bf16_to_f32(in[i]);
}

#define CE_THREADS 256
extern "C" __global__ void ce_loss_grad(float*x,float*loss,const int*target,int rows,int vocab,float inv_predicted){
  int r=blockIdx.x;if(r>=rows)return;
  float*row=x+(size_t)r*vocab;
  int tid=threadIdx.x;
  int t=target[r];
  if(t<0){for(int c=tid;c<vocab;c+=CE_THREADS)row[c]=0.f;return;}
  // The running (max, sum) pair of the online softmax: one read pass over the
  // row instead of a max pass followed by a sum pass. At this vocabulary the
  // row is far too big for any cache, so the second pass cost a full trip to
  // memory. Rescaling the accumulator only when a new maximum arrives keeps the
  // common element down to a single exponential, and the fast intrinsic is
  // ample for a result that is stored as bfloat16.
  __shared__ float redm[CE_THREADS];
  __shared__ float reds[CE_THREADS];
  __shared__ float tgt;
  float m=-3.0e38f,acc=0.f;
  for(int c=tid;c<vocab;c+=CE_THREADS){float v=row[c];if(v>m){acc=acc*__expf(m-v)+1.f;m=v;}else acc+=__expf(v-m);}
  redm[tid]=m;reds[tid]=acc;__syncthreads();
  for(int s=CE_THREADS/2;s>0;s>>=1){
    if(tid<s){float a=redm[tid],b=redm[tid+s];float nm=fmaxf(a,b);
      reds[tid]=reds[tid]*expf(a-nm)+reds[tid+s]*expf(b-nm);redm[tid]=nm;}
    __syncthreads();
  }
  m=redm[0];
  float sum=reds[0];
  if(tid==0)tgt=row[t];
  __syncthreads();
  float scale=inv_predicted/sum;
  if(tid==0)atomicAdd(loss,(m+logf(sum))-tgt);
  for(int c=tid;c<vocab;c+=CE_THREADS)row[c]=__expf(row[c]-m)*scale;
  __syncthreads();
  if(tid==0)row[t]-=inv_predicted;
}

// The bf16 mirror of `ce_loss_grad`. Identical arithmetic in FP32 registers;
// only the row's storage is half as wide, which halves the traffic over the
// largest tensor a training step touches.
extern "C" __global__ void ce_loss_grad_bf16(bf16_t*x,float*loss,const int*target,int rows,int vocab,float inv_predicted){
  int r=blockIdx.x;if(r>=rows)return;
  bf16_t*row=x+(size_t)r*vocab;
  int tid=threadIdx.x;
  int t=target[r];
  if(t<0){for(int c=tid;c<vocab;c+=CE_THREADS)row[c]=(bf16_t)0;return;}
  // The running (max, sum) pair of the online softmax: one read pass over the
  // row instead of a max pass followed by a sum pass. At this vocabulary the
  // row is far too big for any cache, so the second pass cost a full trip to
  // memory. Rescaling the accumulator only when a new maximum arrives keeps the
  // common element down to a single exponential, and the fast intrinsic is
  // ample for a result that is stored as bfloat16.
  __shared__ float redm[CE_THREADS];
  __shared__ float reds[CE_THREADS];
  __shared__ float tgt;
  // Both passes over the row move two values per access, because a warp of
  // single bfloat16 loads only asks for 64 bytes at a time and this row is the
  // largest tensor in the step. An odd vocabulary has no pairs and falls to the
  // scalar tail, which is also where an even vocabulary's last value never
  // lands.
  int half=(vocab&1)==0?vocab>>1:0;
  float m=-3.0e38f,acc=0.f;
  {
    const unsigned int*pairs=(const unsigned int*)row;
    for(int c=tid;c<half;c+=CE_THREADS){
      unsigned int p=pairs[c];
      float v=bf16_to_f32((bf16_t)(p&0xffffu));
      if(v>m){acc=acc*__expf(m-v)+1.f;m=v;}else acc+=__expf(v-m);
      v=bf16_to_f32((bf16_t)(p>>16));
      if(v>m){acc=acc*__expf(m-v)+1.f;m=v;}else acc+=__expf(v-m);
    }
  }
  for(int c=half*2+tid;c<vocab;c+=CE_THREADS){float v=bf16_to_f32(row[c]);if(v>m){acc=acc*__expf(m-v)+1.f;m=v;}else acc+=__expf(v-m);}
  redm[tid]=m;reds[tid]=acc;__syncthreads();
  for(int s=CE_THREADS/2;s>0;s>>=1){
    if(tid<s){float a=redm[tid],b=redm[tid+s];float nm=fmaxf(a,b);
      reds[tid]=reds[tid]*expf(a-nm)+reds[tid+s]*expf(b-nm);redm[tid]=nm;}
    __syncthreads();
  }
  m=redm[0];
  float sum=reds[0];
  if(tid==0)tgt=bf16_to_f32(row[t]);
  __syncthreads();
  float scale=inv_predicted/sum;
  if(tid==0)atomicAdd(loss,(m+logf(sum))-tgt);
  {
    unsigned int*pairs=(unsigned int*)row;
    for(int c=tid;c<half;c+=CE_THREADS){
      unsigned int p=pairs[c];
      unsigned int lo=f32_to_bf16(__expf(bf16_to_f32((bf16_t)(p&0xffffu))-m)*scale);
      unsigned int hi=f32_to_bf16(__expf(bf16_to_f32((bf16_t)(p>>16))-m)*scale);
      pairs[c]=(hi<<16)|lo;
    }
  }
  for(int c=half*2+tid;c<vocab;c+=CE_THREADS)row[c]=f32_to_bf16(__expf(bf16_to_f32(row[c])-m)*scale);
  __syncthreads();
  if(tid==0)row[t]=f32_to_bf16(bf16_to_f32(row[t])-inv_predicted);
}
"#;

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

pub(crate) fn cuda_err<E: std::fmt::Display>(
    stage: &'static str,
) -> impl FnOnce(E) -> NetworkError {
    move |e| NetworkError::Cuda(format!("{stage} failed: {e}"))
}

/// As [`cuda_err`], for a call that asks the driver for `bytes` of device
/// memory.
///
/// A driver out-of-memory becomes [`NetworkError::CudaOutOfMemory`], which a
/// caller can match on to retry with a smaller batch. Every other failure keeps
/// its [`NetworkError::Cuda`], because nothing about it says a smaller
/// allocation would have worked.
pub(crate) fn cuda_alloc_err<E: std::fmt::Display>(
    stage: &'static str,
    bytes: usize,
) -> impl FnOnce(E) -> NetworkError {
    move |e| {
        let message = e.to_string();
        if !is_out_of_memory(&message) {
            return NetworkError::Cuda(format!("{stage} failed: {e}"));
        }
        NetworkError::CudaOutOfMemory {
            requested_mib: bytes.div_ceil(MIB),
            free_mib: cudarc::runtime::result::get_mem_info()
                .map(|(free, _)| free / MIB)
                .unwrap_or(0),
        }
    }
}

/// Whether a driver error is the one that says the allocation did not fit.
///
/// Matched on the text because cudarc's error types differ between the driver
/// and the runtime API and both reach this crate; the spelling is the driver's
/// own `CUDA_ERROR_OUT_OF_MEMORY` and the runtime's `cudaErrorMemoryAllocation`.
fn is_out_of_memory(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    lowered.contains("out_of_memory")
        || lowered.contains("out of memory")
        || lowered.contains("memoryallocation")
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
type LoadedDevice = (Arc<CudaContext>, Arc<CudaModule>);

pub(crate) fn device_context(
    device: usize,
) -> Result<(Arc<CudaContext>, Arc<CudaModule>), NetworkError> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, LoadedDevice>>> = OnceLock::new();
    let mut contexts = CONTEXTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map_err(|_| NetworkError::Cuda("CUDA context cache lock was poisoned".into()))?;
    if let Some(entry) = contexts.get(&device) {
        return Ok(entry.clone());
    }
    let ctx = CudaContext::new(device).map_err(cuda_err("CUDA driver/device initialization"))?;
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
pub(crate) fn cfg(n: usize) -> LaunchConfig {
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
    /// Validation set, uploaded once by `prepare_validation` (P09: device-side
    /// validation avoids a full model host round-trip every epoch). Identity
    /// order, since validation never shuffles.
    eval_input: Option<CudaSlice<f32>>,
    eval_target: Option<CudaSlice<f32>>,
    eval_order: Option<CudaSlice<u32>>,
    eval_loss: Option<CudaSlice<f32>>,
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
    /// Row count of the device-resident validation set, or 0 when
    /// `prepare_validation` has not been called (or declined for budget).
    eval_rows: usize,
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
            host: (!resident).then_some(HostDataset {
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
            eval_rows: 0,
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

    /// Uploads `dataset` to the device once so `validate_epoch` can score it
    /// without a host round-trip. Returns `Ok(false)` (not an error) when it
    /// would not fit under `budget_mib` alongside what is already resident;
    /// the caller should fall back to host-side validation in that case.
    pub fn prepare_validation(
        &mut self,
        dataset: &Dataset,
        budget_mib: usize,
    ) -> Result<bool, NetworkError> {
        if dataset.is_empty() {
            return Err(NetworkError::EmptyDataset);
        }
        for (x, y) in dataset.inputs.iter().zip(&dataset.targets) {
            self.model.validate_input(x)?;
            self.model.validate_target(y)?;
        }
        let xs: Vec<f32> = dataset.inputs.iter().flatten().copied().collect();
        let ys: Vec<f32> = dataset.targets.iter().flatten().copied().collect();
        let rows = dataset.len();
        let bytes = (xs.len() + ys.len()) * size_of::<f32>() + rows * size_of::<u32>();
        if self.state.allocated_bytes() + bytes > budget_mib.saturating_mul(MIB) {
            return Ok(false);
        }
        let stream = self.state.stream.clone();
        let eval_input = stream
            .clone_htod(&xs)
            .map_err(cuda_err("validation dataset upload"))?;
        let eval_target = stream
            .clone_htod(&ys)
            .map_err(cuda_err("validation dataset upload"))?;
        // Identity order: validation is never shuffled, but `gather_rows`
        // already does exactly the contiguous-batch copy this needs.
        let identity: Vec<u32> = (0..rows as u32).collect();
        let eval_order = stream
            .clone_htod(&identity)
            .map_err(cuda_err("validation order upload"))?;
        let eval_loss = stream
            .alloc_zeros::<f32>(1)
            .map_err(cuda_err("device allocation"))?;
        self.stats.host_to_device_bytes += bytes;
        self.state.eval_input = Some(eval_input);
        self.state.eval_target = Some(eval_target);
        self.state.eval_order = Some(eval_order);
        self.state.eval_loss = Some(eval_loss);
        self.eval_rows = rows;
        Ok(true)
    }

    /// Device-side validation loss over the set uploaded by
    /// `prepare_validation`, using the weights currently resident on the
    /// device (no `copy_back`/host round-trip of the full model). Forward
    /// pass only: no gradients, no optimizer step, no mutation of training
    /// state. Numerically the same computation as `Network::evaluate_loss`
    /// (MSE averaged over every output element), just done on-device.
    pub fn validate_epoch(&mut self) -> Result<f32, NetworkError> {
        if self.eval_rows == 0 {
            return Err(NetworkError::Accelerator(
                "validate_epoch called before prepare_validation succeeded".into(),
            ));
        }
        let rows = self.eval_rows;
        let output_size = self.target_width;
        {
            let loss = self.state.eval_loss.as_mut().expect("prepared");
            self.state
                .stream
                .memset_zeros(loss)
                .map_err(cuda_err("validation loss reset"))?;
        }
        let scale = 1.0 / (rows * output_size) as f32;
        for start in (0..rows).step_by(self.batch_size) {
            let b = self.batch_size.min(rows - start);
            let f = &self.state.kernels.gather_rows;
            let all_input = self.state.eval_input.as_ref().expect("prepared");
            let all_target = self.state.eval_target.as_ref().expect("prepared");
            let order = self.state.eval_order.as_ref().expect("prepared");
            unsafe {
                self.state
                    .stream
                    .launch_builder(f)
                    .arg(&mut self.state.input)
                    .arg(all_input)
                    .arg(order)
                    .arg(&(start as i32))
                    .arg(&(b as i32))
                    .arg(&(self.input_width as i32))
                    .launch(cfg(b * self.input_width))
                    .map_err(cuda_err("validation input gather kernel"))?;
                self.state
                    .stream
                    .launch_builder(f)
                    .arg(&mut self.state.target)
                    .arg(all_target)
                    .arg(order)
                    .arg(&(start as i32))
                    .arg(&(b as i32))
                    .arg(&(self.target_width as i32))
                    .launch(cfg(b * self.target_width))
                    .map_err(cuda_err("validation target gather kernel"))?;
            }
            forward_pass(
                &self.state.stream,
                &self.state.blas,
                &self.state.kernels,
                &mut self.state.layers,
                &self.model,
                &self.state.input,
                b,
            )?;
            let last = self.state.layers.len() - 1;
            let la = &self.state.layers[last].a;
            let n = b * output_size;
            let fl = &self.state.kernels.mse_epoch_sum;
            let loss = self.state.eval_loss.as_mut().expect("prepared");
            unsafe {
                self.state
                    .stream
                    .launch_builder(fl)
                    .arg(loss)
                    .arg(la)
                    .arg(&self.state.target)
                    .arg(&(n as i32))
                    .arg(&scale)
                    .launch(cfg(n))
                    .map_err(cuda_err("validation loss kernel"))?;
            }
        }
        self.state
            .stream
            .synchronize()
            .map_err(cuda_err("validation synchronization"))?;
        let loss = self
            .state
            .stream
            .clone_dtoh(self.state.eval_loss.as_ref().expect("prepared"))
            .map_err(cuda_err("validation loss download"))?[0];
        self.stats.device_to_host_bytes += size_of::<f32>();
        if !loss.is_finite() {
            return Err(NetworkError::Cuda(
                "numerical preflight failed: non-finite validation loss".into(),
            ));
        }
        Ok(loss)
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
            eval_input: None,
            eval_target: None,
            eval_order: None,
            eval_loss: None,
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
                + self.all_target.as_ref().map_or(0, CudaSlice::len)
                + self.eval_input.as_ref().map_or(0, CudaSlice::len)
                + self.eval_target.as_ref().map_or(0, CudaSlice::len)
                + self.eval_loss.as_ref().map_or(0, CudaSlice::len))
                * size_of::<f32>()
            + (self.order.as_ref().map_or(0, CudaSlice::len)
                + self.eval_order.as_ref().map_or(0, CudaSlice::len))
                * size_of::<u32>()
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
            // GW[out, inp] = (1/b) * D^T . X, through cuBLAS rather than a
            // hand-written kernel: this is a real GEMM, and a kernel giving
            // each of the out*inp threads a serial strided loop over the batch
            // reaches a fraction of the same bandwidth.
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
            Optimizer::Lion { .. } => {
                return Err(NetworkError::UnsupportedCuda(
                    "the Lion optimizer, which has no device kernel".into(),
                ));
            }
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
/// Forward pass only: fills every `layers[i].a` from `input`, leaving
/// `layers.last().a` holding the network's output. No gradients, no
/// parameter update. Used by `CudaTrainingSession::validate_epoch`; kept
/// separate from `State::train_batch`'s own (duplicated) forward section so
/// this addition cannot change what the tested training path does.
fn forward_pass(
    stream: &Arc<CudaStream>,
    blas: &CudaBlas,
    kernels: &Kernels,
    layers: &mut [DevLayer],
    net: &Network,
    input: &CudaSlice<f32>,
    b: usize,
) -> Result<(), NetworkError> {
    for i in 0..layers.len() {
        let units = if i == 0 {
            net.input_size
        } else {
            net.layers[i - 1].weights.rows
        };
        let out_units = net.layers[i].weights.rows;
        if i == 0 {
            let l = &mut layers[0];
            gemm(blas, input, &l.w, &mut l.a, b, units, out_units)?;
        } else {
            let (left, right) = layers.split_at_mut(i);
            let previous = &left[i - 1].a;
            let l = &mut right[0];
            gemm(blas, previous, &l.w, &mut l.a, b, units, out_units)?;
        }
        let f = &kernels.bias_act;
        let l = &mut layers[i];
        unsafe {
            stream
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
    Ok(())
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
        // compute Z^T = W * X^T. OP_N would also typecheck here, and is
        // wrong: it multiplies that column-major reinterpretation of W, which
        // only coincides with W itself when the layer is square.
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

    /// P09: `validate_epoch`'s device-side forward pass must agree with the
    /// CPU `Network::evaluate_loss` path it replaces, or early stopping
    /// would silently start deciding on wrong numbers. Trains a couple of
    /// epochs first so weights are not at their (numerically forgiving)
    /// random init, then checks both paths against the same, unsynced,
    /// device-resident weights.
    #[test]
    fn cuda_validate_epoch_matches_cpu_evaluate_loss_or_skips_without_device() {
        if !cuda_or_skip() {
            return;
        }
        let train = Dataset::new(
            (0..23)
                .map(|i| vec![(i as f32 - 11.0) / 9.0, ((i * 3) % 7) as f32 / 7.0])
                .collect(),
            (0..23)
                .map(|i| vec![((i * 5) % 11) as f32 / 11.0 - 0.5])
                .collect(),
        );
        let validation = Dataset::new(
            (0..13)
                .map(|i| vec![(i as f32 - 6.0) / 5.0, ((i * 2) % 5) as f32 / 5.0])
                .collect(),
            (0..13)
                .map(|i| vec![((i * 3) % 7) as f32 / 7.0 - 0.3])
                .collect(),
        );
        let base = Network::builder()
            .input_size(2)
            .dense(6, Activation::Tanh)
            .dense(1, Activation::Linear)
            .loss(Loss::Mse)
            .optimizer(Optimizer::adam(0.01))
            .seed(0xC0FFEE)
            .build();
        let config = TrainConfig {
            epochs: 1,
            batch_size: 5,
            shuffle: true,
            seed: Some(42),
        };
        let mut session = CudaTrainingSession::new(&base, &train, config, 0, 8192).unwrap();
        assert!(session.prepare_validation(&validation, 8192).unwrap());
        session.train_epoch().unwrap();
        session.train_epoch().unwrap();

        let device_loss = session.validate_epoch().unwrap();

        let mut host_network = base;
        session.synchronize_network(&mut host_network).unwrap();
        let cpu_loss = host_network.evaluate_loss(&validation).unwrap();

        assert!(
            (device_loss - cpu_loss).abs() <= 1e-5,
            "device validation loss {device_loss} vs CPU {cpu_loss}"
        );

        // validate_epoch must not have perturbed training: another epoch and
        // checkpoint should still look like ordinary continued training.
        let loss_after = session.train_epoch().unwrap();
        assert!(loss_after.is_finite());
    }
}
