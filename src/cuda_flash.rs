//! Fused causal attention for the Ampere tensor cores.
//!
//! The three-kernel attention path in [`crate::gpu_model`] writes a whole
//! `[seq_len, seq_len]` score matrix to global memory, reads it back for the
//! softmax, and reads it a third time for the value matmul. Profiling a
//! 101.7M-parameter model at `seq_len` 1024 put `causal_softmax_lse`,
//! `causal_softmax_bwd` and `causal_probs_from_lse` together at 21.3% of all
//! GPU time, every one of them running at 302-384 GB/s against a card whose
//! ceiling is 360 GB/s. There is nothing left to win inside those kernels: the
//! round trip itself is the cost, and the only way to remove it is not to
//! materialize the matrix at all.
//!
//! This module holds the flash-attention forward kernel that does that. It
//! keeps one tile of scores in registers, runs the softmax there with a
//! running maximum, and accumulates the output in place, so the score matrix
//! never reaches global memory. The matmuls run on the tensor cores through
//! inline `mma.sync` PTX, which is why the source here is compiled separately
//! from [`crate::cuda_training::KERNELS`]: it needs
//! `--gpu-architecture=compute_80` and would not build for an older device.
//!
//! The kernel is loaded only where it can run, and [`flash_kernels`] returns
//! `None` otherwise, leaving the three-kernel path in place.

use crate::cuda_training::cuda_err;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Query rows, key columns and head dimension of one tile. `head_dim` is 64 in
/// every configuration this crate validates, and fixing it is what lets the
/// fragment layout below be written out rather than computed.
pub(crate) const TILE: usize = 64;
/// Threads per block: four warps, sixteen query rows each.
pub(crate) const THREADS: u32 = 128;
/// Two `[64][64]` BF16 tiles, each row padded by eight elements so that the
/// eight lanes of an `mma` fragment land in eight different shared-memory
/// banks instead of all in one.
pub(crate) const SHARED_BYTES: u32 = 2 * (TILE * (TILE + 8) * 2) as u32;
/// Keys twice - key-major for the scores, dimension-major for the query
/// gradient - and values once.
pub(crate) const DQ_SHARED_BYTES: u32 = 3 * (TILE * (TILE + 8) * 2) as u32;
/// Queries and output gradients, each stored both ways round, plus the
/// log-sum-exp and delta of the query tile.
pub(crate) const DKV_SHARED_BYTES: u32 = 4 * (TILE * (TILE + 8) * 2) as u32 + 2 * TILE as u32 * 4;

const KERNELS: &str = r#"
#define FA_D    64
#define FA_ROW  72
#define NEG_INF (-3.0e38f)

// BF16 is the top 16 bits of an FP32 under round-to-nearest-even. Doing the
// rounding by hand keeps NVRTC away from <cuda_bf16.h> and the include path it
// would need, and two of them pack into the 32-bit register an mma operand is.
__device__ __forceinline__ unsigned short bf16_bits(float f){
  unsigned u=__float_as_uint(f);
  u+=0x7fffu+((u>>16)&1u);
  return (unsigned short)(u>>16);
}
// Every activation these kernels touch is BF16: `eligible` runs them only in
// mixed precision, so the type is known at compile time rather than passed in.
__device__ __forceinline__ float bf16_load(const unsigned short*p,size_t i){
  return __uint_as_float((unsigned)p[i]<<16);
}
// Two adjacent BF16 elements as the 32-bit register an mma operand already is,
// in one load and no conversion at all. `i` must be even.
__device__ __forceinline__ unsigned bf16_pair(const unsigned short*p,size_t i){
  return *(const unsigned*)(p+i);
}
__device__ __forceinline__ unsigned pack2(float lo,float hi){
  return (unsigned)bf16_bits(lo)|((unsigned)bf16_bits(hi)<<16);
}

// D[16x8] += A[16x16] * B[16x8], A row-major, B k-major, accumulating in FP32.
// The fragment layout is the one the PTX ISA fixes for m16n8k16: with
// g = lane/4 and t = lane%4, a thread holds A rows {g, g+8} at columns
// {2t, 2t+1} and {2t+8, 2t+9}, B column g at the same four k, and D rows
// {g, g+8} at columns {2t, 2t+1}.
#define MMA(d,a,b) asm volatile( \
  "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
  "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
  : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
  : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]))

// FP16 conversions, for the image path, whose activations are FP16 rather
// than BF16. `cvt` is the instruction <cuda_fp16.h> would have inlined, and
// writing it out keeps NVRTC away from the include path it would need.
__device__ __forceinline__ float h2f(unsigned short h){
  float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h)); return f;
}
__device__ __forceinline__ unsigned short f2h(float v){
  unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v)); return h;
}
__device__ __forceinline__ unsigned packh2(float lo,float hi){
  return (unsigned)f2h(lo)|((unsigned)f2h(hi)<<16);
}
#define MMAH(d,a,b) asm volatile( \
  "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " \
  "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n" \
  : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]) \
  : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]))

// The same fused attention for the image path: FP16 operands, no mask, and
// queries that may come from one tensor while keys and values come from
// another, which is what a cross-attention is. `q_base`, `k_base` and
// `v_base` are the column each projection starts at, so a fused QKV buffer
// needs no split.
//
// One block per (query tile, head). Nothing here writes a log-sum-exp: there
// is no backward pass over an image model in this crate.
extern "C" __global__ __launch_bounds__(128) void image_attention_fwd(
    const unsigned short* __restrict__ queries,
    const unsigned short* __restrict__ keys,
    unsigned short* __restrict__ out,
    int rows, int context,
    int q_lead, int kv_lead, int out_lead,
    int q_base, int k_base, int v_base,
    float scale)
{
  extern __shared__ unsigned short smem[];
  unsigned short* ks = smem;                      // [key][dim]
  unsigned short* vt = smem + 64*FA_ROW;          // [dim][key], transposed

  const int h = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int q_row0 = blockIdx.x*64 + warp*16;
  const int row_a = q_row0 + g, row_b = q_row0 + g + 8;
  const int head = h*FA_D;

  unsigned qf[4][4];
  {
    const size_t base = (size_t)q_base + head;
    #pragma unroll
    for (int kk=0; kk<4; ++kk) {
      int c = kk*16 + t*2;
      float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f;
      if (row_a < rows) {
        size_t p = base + (size_t)row_a*q_lead;
        a0=h2f(queries[p+c])*scale;   a1=h2f(queries[p+c+1])*scale;
        a4=h2f(queries[p+c+8])*scale; a5=h2f(queries[p+c+9])*scale;
      }
      if (row_b < rows) {
        size_t p = base + (size_t)row_b*q_lead;
        a2=h2f(queries[p+c])*scale;   a3=h2f(queries[p+c+1])*scale;
        a6=h2f(queries[p+c+8])*scale; a7=h2f(queries[p+c+9])*scale;
      }
      qf[kk][0]=packh2(a0,a1); qf[kk][1]=packh2(a2,a3);
      qf[kk][2]=packh2(a4,a5); qf[kk][3]=packh2(a6,a7);
    }
  }

  float acc[8][4];
  #pragma unroll
  for (int n=0;n<8;++n)
    #pragma unroll
    for (int i=0;i<4;++i) acc[n][i]=0.f;
  float m_a=NEG_INF, m_b=NEG_INF, l_a=0.f, l_b=0.f;

  for (int kt=0; kt*64 < context; ++kt) {
    __syncthreads();
    for (int i = threadIdx.x; i < 64*FA_D; i += 128) {
      int r = i >> 6, c = i & 63;
      int key = kt*64 + r;
      unsigned short kb = 0, vb = 0;
      if (key < context) {
        size_t p = (size_t)key*kv_lead + head;
        kb = keys[p + k_base + c];
        vb = keys[p + v_base + c];
      }
      ks[r*FA_ROW + c] = kb;
      vt[c*FA_ROW + r] = vb;
    }
    __syncthreads();

    float s[8][4];
    #pragma unroll
    for (int n=0;n<8;++n)
      #pragma unroll
      for (int i=0;i<4;++i) s[n][i]=0.f;
    #pragma unroll
    for (int n=0;n<8;++n) {
      #pragma unroll
      for (int kk=0; kk<4; ++kk) {
        const unsigned short* p = ks + (n*8+g)*FA_ROW + kk*16 + t*2;
        unsigned bf[2];
        bf[0] = *(const unsigned*)p;
        bf[1] = *(const unsigned*)(p+8);
        MMAH(s[n], qf[kk], bf);
      }
    }

    // A key past the end read as zero, and a zero score is a probability of
    // one. Only the last tile can hold one, so only it pays for the check.
    if ((kt+1)*64 > context) {
      #pragma unroll
      for (int n=0;n<8;++n) {
        int key = kt*64 + n*8 + t*2;
        if (key   >= context) { s[n][0]=NEG_INF; s[n][2]=NEG_INF; }
        if (key+1 >= context) { s[n][1]=NEG_INF; s[n][3]=NEG_INF; }
      }
    }

    float max_a=NEG_INF, max_b=NEG_INF;
    #pragma unroll
    for (int n=0;n<8;++n) {
      max_a=fmaxf(max_a,fmaxf(s[n][0],s[n][1]));
      max_b=fmaxf(max_b,fmaxf(s[n][2],s[n][3]));
    }
    max_a=fmaxf(max_a,__shfl_xor_sync(0xffffffff,max_a,1));
    max_a=fmaxf(max_a,__shfl_xor_sync(0xffffffff,max_a,2));
    max_b=fmaxf(max_b,__shfl_xor_sync(0xffffffff,max_b,1));
    max_b=fmaxf(max_b,__shfl_xor_sync(0xffffffff,max_b,2));

    float new_a=fmaxf(m_a,max_a), new_b=fmaxf(m_b,max_b);
    float corr_a=__expf(m_a-new_a), corr_b=__expf(m_b-new_b);
    float sum_a=0.f, sum_b=0.f;
    #pragma unroll
    for (int n=0;n<8;++n) {
      s[n][0]=__expf(s[n][0]-new_a); s[n][1]=__expf(s[n][1]-new_a);
      s[n][2]=__expf(s[n][2]-new_b); s[n][3]=__expf(s[n][3]-new_b);
      sum_a+=s[n][0]+s[n][1]; sum_b+=s[n][2]+s[n][3];
    }
    sum_a+=__shfl_xor_sync(0xffffffff,sum_a,1); sum_a+=__shfl_xor_sync(0xffffffff,sum_a,2);
    sum_b+=__shfl_xor_sync(0xffffffff,sum_b,1); sum_b+=__shfl_xor_sync(0xffffffff,sum_b,2);
    l_a=l_a*corr_a+sum_a; l_b=l_b*corr_b+sum_b; m_a=new_a; m_b=new_b;
    #pragma unroll
    for (int n=0;n<8;++n) {
      acc[n][0]*=corr_a; acc[n][1]*=corr_a; acc[n][2]*=corr_b; acc[n][3]*=corr_b;
    }

    #pragma unroll
    for (int kk=0; kk<4; ++kk) {
      unsigned pf[4];
      pf[0]=packh2(s[2*kk][0],  s[2*kk][1]);
      pf[1]=packh2(s[2*kk][2],  s[2*kk][3]);
      pf[2]=packh2(s[2*kk+1][0],s[2*kk+1][1]);
      pf[3]=packh2(s[2*kk+1][2],s[2*kk+1][3]);
      #pragma unroll
      for (int n=0;n<8;++n) {
        const unsigned short* p = vt + (n*8+g)*FA_ROW + kk*16 + t*2;
        unsigned bf[2];
        bf[0] = *(const unsigned*)p;
        bf[1] = *(const unsigned*)(p+8);
        MMAH(acc[n], pf, bf);
      }
    }
  }

  float inv_a = l_a>0.f ? 1.f/l_a : 0.f;
  float inv_b = l_b>0.f ? 1.f/l_b : 0.f;
  #pragma unroll
  for (int n=0;n<8;++n) {
    int c = n*8 + t*2;
    if (row_a < rows) {
      size_t p = (size_t)row_a*out_lead + head + c;
      *(unsigned*)(out+p) = packh2(acc[n][0]*inv_a,acc[n][1]*inv_a);
    }
    if (row_b < rows) {
      size_t p = (size_t)row_b*out_lead + head + c;
      *(unsigned*)(out+p) = packh2(acc[n][2]*inv_b,acc[n][3]*inv_b);
    }
  }
}

// One block per (query tile, head, sequence). Queries live in registers for
// the whole block; keys and values stream through shared memory one tile at a
// time. `lse` is written in the same layout the three-kernel path used, so the
// backward pass rebuilds the probabilities from it unchanged.
extern "C" __global__ __launch_bounds__(128) void flash_attention_fwd(
    const unsigned short* __restrict__ qkv,
    unsigned short* __restrict__ out,
    float* __restrict__ lse,
    int seq_len, int qkv_width, int query_width,
    int key_base, int value_base, int group,
    int lse_head_stride, float scale)
{
  extern __shared__ unsigned short smem[];
  unsigned short* ks = smem;                      // [key][dim]
  unsigned short* vt = smem + 64*FA_ROW;          // [dim][key], transposed

  const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int kv = (h / group) * FA_D;
  const size_t seq_base = (size_t)b * seq_len;
  const int q_row0 = qt*64 + warp*16;
  const int row_a = q_row0 + g, row_b = q_row0 + g + 8;

  // Queries, scaled once here rather than by the matmul's alpha as the cuBLAS
  // path did, and rounded to BF16 exactly as that path's tensor cores did.
  unsigned qf[4][4];
  {
    const size_t base = seq_base*(size_t)qkv_width + h*FA_D;
    #pragma unroll
    for (int kk=0; kk<4; ++kk) {
      int c = kk*16 + t*2;
      float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f;
      if (row_a < seq_len) {
        size_t p = base + (size_t)row_a*qkv_width;
        a0=bf16_load(qkv,p+c)*scale;   a1=bf16_load(qkv,p+c+1)*scale;
        a4=bf16_load(qkv,p+c+8)*scale; a5=bf16_load(qkv,p+c+9)*scale;
      }
      if (row_b < seq_len) {
        size_t p = base + (size_t)row_b*qkv_width;
        a2=bf16_load(qkv,p+c)*scale;   a3=bf16_load(qkv,p+c+1)*scale;
        a6=bf16_load(qkv,p+c+8)*scale; a7=bf16_load(qkv,p+c+9)*scale;
      }
      qf[kk][0]=pack2(a0,a1); qf[kk][1]=pack2(a2,a3);
      qf[kk][2]=pack2(a4,a5); qf[kk][3]=pack2(a6,a7);
    }
  }

  float acc[8][4];
  #pragma unroll
  for (int n=0;n<8;++n)
    #pragma unroll
    for (int i=0;i<4;++i) acc[n][i]=0.f;
  // Running maximum and sum of the two query rows this thread holds.
  float m_a=NEG_INF, m_b=NEG_INF, l_a=0.f, l_b=0.f;

  // Causal: query tile `qt` sees key tiles 0..qt and nothing beyond.
  for (int kt=0; kt<=qt; ++kt) {
    __syncthreads();
    // One element a thread, not two: the value tile is stored transposed, and
    // measured, the wider load does not pay for the extra shared-memory bank
    // conflicts that come with it.
    for (int i = threadIdx.x; i < 64*FA_D; i += 128) {
      int r = i >> 6, c = i & 63;
      int key = kt*64 + r;
      unsigned short kb = 0, vb = 0;
      if (key < seq_len) {
        size_t p = (seq_base + key)*(size_t)qkv_width;
        kb = qkv[p + key_base + kv + c];
        vb = qkv[p + value_base + kv + c];
      }
      ks[r*FA_ROW + c] = kb;
      vt[c*FA_ROW + r] = vb;
    }
    __syncthreads();

    float s[8][4];
    #pragma unroll
    for (int n=0;n<8;++n)
      #pragma unroll
      for (int i=0;i<4;++i) s[n][i]=0.f;
    // scores = queries * keys^T, eight key columns and sixteen k per mma.
    #pragma unroll
    for (int n=0;n<8;++n) {
      #pragma unroll
      for (int kk=0; kk<4; ++kk) {
        const unsigned short* p = ks + (n*8+g)*FA_ROW + kk*16 + t*2;
        unsigned bf[2];
        bf[0] = *(const unsigned*)p;
        bf[1] = *(const unsigned*)(p+8);
        MMA(s[n], qf[kk], bf);
      }
    }

    // Only the tile on the diagonal is partly masked, and only it can hold
    // keys past the end of a sequence whose length is not a multiple of 64.
    if (kt == qt) {
      #pragma unroll
      for (int n=0;n<8;++n) {
        int key = kt*64 + n*8 + t*2;
        if (key   > row_a || key   >= seq_len) s[n][0]=NEG_INF;
        if (key+1 > row_a || key+1 >= seq_len) s[n][1]=NEG_INF;
        if (key   > row_b || key   >= seq_len) s[n][2]=NEG_INF;
        if (key+1 > row_b || key+1 >= seq_len) s[n][3]=NEG_INF;
      }
    }

    // Online softmax. A row of scores is spread over the four lanes that share
    // `g`, which are adjacent, so the reduction is two shuffles.
    float max_a=NEG_INF, max_b=NEG_INF;
    #pragma unroll
    for (int n=0;n<8;++n) {
      max_a=fmaxf(max_a,fmaxf(s[n][0],s[n][1]));
      max_b=fmaxf(max_b,fmaxf(s[n][2],s[n][3]));
    }
    max_a=fmaxf(max_a,__shfl_xor_sync(0xffffffff,max_a,1));
    max_a=fmaxf(max_a,__shfl_xor_sync(0xffffffff,max_a,2));
    max_b=fmaxf(max_b,__shfl_xor_sync(0xffffffff,max_b,1));
    max_b=fmaxf(max_b,__shfl_xor_sync(0xffffffff,max_b,2));

    float new_a=fmaxf(m_a,max_a), new_b=fmaxf(m_b,max_b);
    float corr_a=__expf(m_a-new_a), corr_b=__expf(m_b-new_b);
    float sum_a=0.f, sum_b=0.f;
    #pragma unroll
    for (int n=0;n<8;++n) {
      s[n][0]=__expf(s[n][0]-new_a); s[n][1]=__expf(s[n][1]-new_a);
      s[n][2]=__expf(s[n][2]-new_b); s[n][3]=__expf(s[n][3]-new_b);
      sum_a+=s[n][0]+s[n][1]; sum_b+=s[n][2]+s[n][3];
    }
    sum_a+=__shfl_xor_sync(0xffffffff,sum_a,1); sum_a+=__shfl_xor_sync(0xffffffff,sum_a,2);
    sum_b+=__shfl_xor_sync(0xffffffff,sum_b,1); sum_b+=__shfl_xor_sync(0xffffffff,sum_b,2);
    l_a=l_a*corr_a+sum_a; l_b=l_b*corr_b+sum_b; m_a=new_a; m_b=new_b;
    #pragma unroll
    for (int n=0;n<8;++n) {
      acc[n][0]*=corr_a; acc[n][1]*=corr_a; acc[n][2]*=corr_b; acc[n][3]*=corr_b;
    }

    // output += probabilities * values. The accumulator fragment of the score
    // matmul is bit-for-bit the operand layout the value matmul wants, so the
    // probabilities go straight from registers into the next mma.
    #pragma unroll
    for (int kk=0; kk<4; ++kk) {
      unsigned pf[4];
      pf[0]=pack2(s[2*kk][0],  s[2*kk][1]);
      pf[1]=pack2(s[2*kk][2],  s[2*kk][3]);
      pf[2]=pack2(s[2*kk+1][0],s[2*kk+1][1]);
      pf[3]=pack2(s[2*kk+1][2],s[2*kk+1][3]);
      #pragma unroll
      for (int n=0;n<8;++n) {
        const unsigned short* p = vt + (n*8+g)*FA_ROW + kk*16 + t*2;
        unsigned bf[2];
        bf[0] = *(const unsigned*)p;
        bf[1] = *(const unsigned*)(p+8);
        MMA(acc[n], pf, bf);
      }
    }
  }

  float inv_a = l_a>0.f ? 1.f/l_a : 0.f;
  float inv_b = l_b>0.f ? 1.f/l_b : 0.f;
  // `out` is read by the output projection and by its weight-gradient GEMM and
  // by nothing else, so it is stored in whatever width those operands want.
  const size_t obase = seq_base*(size_t)query_width + h*FA_D;
  #pragma unroll
  for (int n=0;n<8;++n) {
    int c = n*8 + t*2;
    if (row_a < seq_len) {
      size_t p = obase + (size_t)row_a*query_width + c;
      *(unsigned*)(out+p) = pack2(acc[n][0]*inv_a,acc[n][1]*inv_a);
    }
    if (row_b < seq_len) {
      size_t p = obase + (size_t)row_b*query_width + c;
      *(unsigned*)(out+p) = pack2(acc[n][2]*inv_b,acc[n][3]*inv_b);
    }
  }
  if (t == 0) {
    float* p = lse + (size_t)h*lse_head_stride + seq_base;
    if (row_a < seq_len) p[row_a]=m_a+__logf(l_a);
    if (row_b < seq_len) p[row_b]=m_b+__logf(l_b);
  }
}
// Row dot product of the attention output with its gradient, which is the
// `delta` term the softmax backward needs. One warp per row, two dimensions a
// lane, so both loads are contiguous.
extern "C" __global__ void flash_attention_delta(
    const unsigned short* __restrict__ out,
    const unsigned short* __restrict__ grad_out,
    float* __restrict__ delta,
    int rows, int heads, int query_width, int delta_head_stride)
{
  int lane = threadIdx.x & 31;
  int total = rows*heads;
  int stride = (gridDim.x*blockDim.x) >> 5;
  for (int warp = (blockIdx.x*blockDim.x + threadIdx.x) >> 5; warp < total; warp += stride) {
    int h = warp / rows, r = warp % rows;
    size_t o = (size_t)r*query_width + h*FA_D;
    size_t g = (size_t)r*query_width + h*FA_D;
    float sum = bf16_load(out,o+lane)*bf16_load(grad_out,g+lane)
              + bf16_load(out,o+lane+32)*bf16_load(grad_out,g+lane+32);
    #pragma unroll
    for (int step=16; step; step>>=1) sum += __shfl_xor_sync(0xffffffff,sum,step);
    if (lane == 0) delta[(size_t)h*delta_head_stride + r] = sum;
  }
}

// Gradient of the queries. The same tiling as the forward pass, one block per
// (query tile, head, sequence), except that the probabilities are rebuilt from
// the log-sum-exp instead of being carried forward. Three key tiles live in
// shared memory: keys twice, once each way round, because the score matmul
// wants them key-major and the query matmul wants them dimension-major.
extern "C" __global__ __launch_bounds__(128) void flash_attention_dq(
    const unsigned short* __restrict__ qkv,
    const unsigned short* __restrict__ grad_out,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    unsigned short* __restrict__ grad_qkv,
    int seq_len, int qkv_width, int query_width,
    int key_base, int value_base, int group,
    int lse_head_stride, float scale)
{
  extern __shared__ unsigned short smem[];
  unsigned short* ks = smem;                        // [key][dim]
  unsigned short* vs = smem + 64*FA_ROW;            // [key][dim]
  unsigned short* kt = smem + 2*64*FA_ROW;          // [dim][key]

  const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int kv = (h / group) * FA_D;
  const size_t seq_base = (size_t)b * seq_len;
  const int q_row0 = qt*64 + warp*16;
  const int row_a = q_row0 + g, row_b = q_row0 + g + 8;

  unsigned qf[4][4], df[4][4];
  float lse_a=0.f, lse_b=0.f, del_a=0.f, del_b=0.f;
  {
    const size_t qbase = seq_base*(size_t)qkv_width + h*FA_D;
    const size_t gbase = seq_base*(size_t)query_width + h*FA_D;
    #pragma unroll
    for (int kk=0; kk<4; ++kk) {
      int c = kk*16 + t*2;
      float q0=0.f,q1=0.f,q2=0.f,q3=0.f,q4=0.f,q5=0.f,q6=0.f,q7=0.f;
      df[kk][0]=0; df[kk][1]=0; df[kk][2]=0; df[kk][3]=0;
      if (row_a < seq_len) {
        size_t p = qbase + (size_t)row_a*qkv_width;
        q0=bf16_load(qkv,p+c)*scale;   q1=bf16_load(qkv,p+c+1)*scale;
        q4=bf16_load(qkv,p+c+8)*scale; q5=bf16_load(qkv,p+c+9)*scale;
        size_t e = gbase + (size_t)row_a*query_width;
        df[kk][0]=bf16_pair(grad_out,e+c); df[kk][2]=bf16_pair(grad_out,e+c+8);
      }
      if (row_b < seq_len) {
        size_t p = qbase + (size_t)row_b*qkv_width;
        q2=bf16_load(qkv,p+c)*scale;   q3=bf16_load(qkv,p+c+1)*scale;
        q6=bf16_load(qkv,p+c+8)*scale; q7=bf16_load(qkv,p+c+9)*scale;
        size_t e = gbase + (size_t)row_b*query_width;
        df[kk][1]=bf16_pair(grad_out,e+c); df[kk][3]=bf16_pair(grad_out,e+c+8);
      }
      qf[kk][0]=pack2(q0,q1); qf[kk][1]=pack2(q2,q3);
      qf[kk][2]=pack2(q4,q5); qf[kk][3]=pack2(q6,q7);
    }
    const float* row = lse + (size_t)h*lse_head_stride + seq_base;
    const float* dd  = delta + (size_t)h*lse_head_stride + seq_base;
    if (row_a < seq_len) { lse_a=row[row_a]; del_a=dd[row_a]; }
    if (row_b < seq_len) { lse_b=row[row_b]; del_b=dd[row_b]; }
  }

  float acc[8][4];
  #pragma unroll
  for (int n=0;n<8;++n)
    #pragma unroll
    for (int i=0;i<4;++i) acc[n][i]=0.f;

  for (int ktile=0; ktile<=qt; ++ktile) {
    __syncthreads();
    for (int i = threadIdx.x*2; i < 64*FA_D; i += 256) {
      int r = i >> 6, c = i & 63;
      int key = ktile*64 + r;
      unsigned kw = 0, vw = 0;
      if (key < seq_len) {
        size_t p = (seq_base + key)*(size_t)qkv_width;
        kw = bf16_pair(qkv,p + key_base + kv + c);
        vw = bf16_pair(qkv,p + value_base + kv + c);
      }
      *(unsigned*)(ks + r*FA_ROW + c) = kw;
      kt[c*FA_ROW + r] = (unsigned short)kw;
      kt[(c+1)*FA_ROW + r] = (unsigned short)(kw >> 16);
      *(unsigned*)(vs + r*FA_ROW + c) = vw;
    }
    __syncthreads();

    float s[8][4], dp[8][4];
    #pragma unroll
    for (int n=0;n<8;++n)
      #pragma unroll
      for (int i=0;i<4;++i) { s[n][i]=0.f; dp[n][i]=0.f; }
    #pragma unroll
    for (int n=0;n<8;++n) {
      #pragma unroll
      for (int kk=0; kk<4; ++kk) {
        const unsigned short* pk = ks + (n*8+g)*FA_ROW + kk*16 + t*2;
        const unsigned short* pv = vs + (n*8+g)*FA_ROW + kk*16 + t*2;
        unsigned bk[2], bv[2];
        bk[0]=*(const unsigned*)pk; bk[1]=*(const unsigned*)(pk+8);
        bv[0]=*(const unsigned*)pv; bv[1]=*(const unsigned*)(pv+8);
        MMA(s[n], qf[kk], bk);
        MMA(dp[n], df[kk], bv);
      }
    }

    if (ktile == qt) {
      #pragma unroll
      for (int n=0;n<8;++n) {
        int key = ktile*64 + n*8 + t*2;
        if (key   > row_a || key   >= seq_len) s[n][0]=NEG_INF;
        if (key+1 > row_a || key+1 >= seq_len) s[n][1]=NEG_INF;
        if (key   > row_b || key   >= seq_len) s[n][2]=NEG_INF;
        if (key+1 > row_b || key+1 >= seq_len) s[n][3]=NEG_INF;
      }
    }

    // dS = P * (dP - delta), the softmax Jacobian with the row sum already
    // folded into `delta` by the kernel above.
    #pragma unroll
    for (int n=0;n<8;++n) {
      s[n][0]=__expf(s[n][0]-lse_a)*(dp[n][0]-del_a);
      s[n][1]=__expf(s[n][1]-lse_a)*(dp[n][1]-del_a);
      s[n][2]=__expf(s[n][2]-lse_b)*(dp[n][2]-del_b);
      s[n][3]=__expf(s[n][3]-lse_b)*(dp[n][3]-del_b);
    }

    #pragma unroll
    for (int kk=0; kk<4; ++kk) {
      unsigned pf[4];
      pf[0]=pack2(s[2*kk][0],  s[2*kk][1]);
      pf[1]=pack2(s[2*kk][2],  s[2*kk][3]);
      pf[2]=pack2(s[2*kk+1][0],s[2*kk+1][1]);
      pf[3]=pack2(s[2*kk+1][2],s[2*kk+1][3]);
      #pragma unroll
      for (int n=0;n<8;++n) {
        const unsigned short* p = kt + (n*8+g)*FA_ROW + kk*16 + t*2;
        unsigned bf[2];
        bf[0]=*(const unsigned*)p;
        bf[1]=*(const unsigned*)(p+8);
        MMA(acc[n], pf, bf);
      }
    }
  }

  const size_t obase = seq_base*(size_t)qkv_width + h*FA_D;
  #pragma unroll
  for (int n=0;n<8;++n) {
    int c = n*8 + t*2;
    if (row_a < seq_len) {
      size_t p = obase + (size_t)row_a*qkv_width;
      *(unsigned*)(grad_qkv+p+c) = pack2(acc[n][0]*scale,acc[n][1]*scale);
    }
    if (row_b < seq_len) {
      size_t p = obase + (size_t)row_b*qkv_width;
      *(unsigned*)(grad_qkv+p+c) = pack2(acc[n][2]*scale,acc[n][3]*scale);
    }
  }
}

// Gradient of the keys and values. This one is blocked the other way round,
// one block per (key tile, key/value head, sequence), and walks the query
// tiles that can see that key tile. Blocking on the key/value head rather than
// the query head is what keeps grouped-query attention exact without atomics:
// every query head that shares a key head is summed inside one block.
extern "C" __global__ __launch_bounds__(128) void flash_attention_dkv(
    const unsigned short* __restrict__ qkv,
    const unsigned short* __restrict__ grad_out,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    unsigned short* __restrict__ grad_qkv,
    int seq_len, int qkv_width, int query_width,
    int key_base, int value_base, int group,
    int lse_head_stride, float scale)
{
  extern __shared__ unsigned short smem[];
  unsigned short* qs  = smem;                       // [query][dim]
  unsigned short* qtr = smem + 64*FA_ROW;           // [dim][query]
  unsigned short* gs  = smem + 2*64*FA_ROW;         // [query][dim]
  unsigned short* gtr = smem + 3*64*FA_ROW;         // [dim][query]
  float* lse_s = (float*)(smem + 4*64*FA_ROW);
  float* del_s = lse_s + 64;

  const int jt = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const size_t seq_base = (size_t)b * seq_len;
  const int k_row0 = jt*64 + warp*16;
  const int key_a = k_row0 + g, key_b = k_row0 + g + 8;
  const int tiles = (seq_len + 63) >> 6;

  // Keys and values stay in registers for the whole block, as the operand of
  // every matmul below.
  unsigned kf[4][4], vf[4][4];
  {
    const size_t base = seq_base*(size_t)qkv_width + kvh*FA_D;
    #pragma unroll
    for (int kk=0; kk<4; ++kk) {
      int c = kk*16 + t*2;
      kf[kk][0]=0; kf[kk][1]=0; kf[kk][2]=0; kf[kk][3]=0;
      vf[kk][0]=0; vf[kk][1]=0; vf[kk][2]=0; vf[kk][3]=0;
      if (key_a < seq_len) {
        size_t p = base + (size_t)key_a*qkv_width;
        kf[kk][0]=bf16_pair(qkv,p+key_base+c);
        kf[kk][2]=bf16_pair(qkv,p+key_base+c+8);
        vf[kk][0]=bf16_pair(qkv,p+value_base+c);
        vf[kk][2]=bf16_pair(qkv,p+value_base+c+8);
      }
      if (key_b < seq_len) {
        size_t p = base + (size_t)key_b*qkv_width;
        kf[kk][1]=bf16_pair(qkv,p+key_base+c);
        kf[kk][3]=bf16_pair(qkv,p+key_base+c+8);
        vf[kk][1]=bf16_pair(qkv,p+value_base+c);
        vf[kk][3]=bf16_pair(qkv,p+value_base+c+8);
      }
    }
  }

  float dk[8][4], dv[8][4];
  #pragma unroll
  for (int n=0;n<8;++n)
    #pragma unroll
    for (int i=0;i<4;++i) { dk[n][i]=0.f; dv[n][i]=0.f; }

  for (int hq = kvh*group; hq < (kvh+1)*group; ++hq) {
    for (int it = jt; it < tiles; ++it) {
      __syncthreads();
      for (int i = threadIdx.x*2; i < 64*FA_D; i += 256) {
        int r = i >> 6, c = i & 63;
        int q = it*64 + r;
        unsigned qw = 0, gw = 0;
        if (q < seq_len) {
          qw = bf16_pair(qkv,(seq_base+q)*(size_t)qkv_width + hq*FA_D + c);
          gw = bf16_pair(grad_out,(seq_base+q)*(size_t)query_width + hq*FA_D + c);
        }
        // The queries are stored unscaled, so the scores carry the scale
        // instead: one multiply per score rather than one per element here.
        *(unsigned*)(qs + r*FA_ROW + c) = qw;
        qtr[c*FA_ROW + r] = (unsigned short)qw;
        qtr[(c+1)*FA_ROW + r] = (unsigned short)(qw >> 16);
        *(unsigned*)(gs + r*FA_ROW + c) = gw;
        gtr[c*FA_ROW + r] = (unsigned short)gw;
        gtr[(c+1)*FA_ROW + r] = (unsigned short)(gw >> 16);
      }
      if (threadIdx.x < 64) {
        int q = it*64 + threadIdx.x;
        bool ok = q < seq_len;
        lse_s[threadIdx.x] = ok ? lse[(size_t)hq*lse_head_stride + seq_base + q] : 0.f;
        del_s[threadIdx.x] = ok ? delta[(size_t)hq*lse_head_stride + seq_base + q] : 0.f;
      }
      __syncthreads();

      // Transposed scores and transposed dP: rows are keys, columns queries.
      float st[8][4], dpt[8][4];
      #pragma unroll
      for (int n=0;n<8;++n)
        #pragma unroll
        for (int i=0;i<4;++i) { st[n][i]=0.f; dpt[n][i]=0.f; }
      #pragma unroll
      for (int n=0;n<8;++n) {
        #pragma unroll
        for (int kk=0; kk<4; ++kk) {
          const unsigned short* pq = qs + (n*8+g)*FA_ROW + kk*16 + t*2;
          const unsigned short* pg = gs + (n*8+g)*FA_ROW + kk*16 + t*2;
          unsigned bq[2], bg[2];
          bq[0]=*(const unsigned*)pq; bq[1]=*(const unsigned*)(pq+8);
          bg[0]=*(const unsigned*)pg; bg[1]=*(const unsigned*)(pg+8);
          MMA(st[n],  kf[kk], bq);
          MMA(dpt[n], vf[kk], bg);
        }
      }

      #pragma unroll
      for (int n=0;n<8;++n) {
        int col = n*8 + t*2;
        int q0 = it*64 + col, q1 = q0 + 1;
        float l0 = lse_s[col], l1 = lse_s[col+1];
        float d0 = del_s[col], d1 = del_s[col+1];
        float p0 = (q0 >= key_a && q0 < seq_len) ? __expf(st[n][0]*scale-l0) : 0.f;
        float p1 = (q1 >= key_a && q1 < seq_len) ? __expf(st[n][1]*scale-l1) : 0.f;
        float p2 = (q0 >= key_b && q0 < seq_len) ? __expf(st[n][2]*scale-l0) : 0.f;
        float p3 = (q1 >= key_b && q1 < seq_len) ? __expf(st[n][3]*scale-l1) : 0.f;
        st[n][0]=p0; st[n][1]=p1; st[n][2]=p2; st[n][3]=p3;
        dpt[n][0]=p0*(dpt[n][0]-d0); dpt[n][1]=p1*(dpt[n][1]-d1);
        dpt[n][2]=p2*(dpt[n][2]-d0); dpt[n][3]=p3*(dpt[n][3]-d1);
      }

      #pragma unroll
      for (int kk=0; kk<4; ++kk) {
        unsigned pf[4], sf[4];
        pf[0]=pack2(st[2*kk][0],  st[2*kk][1]);
        pf[1]=pack2(st[2*kk][2],  st[2*kk][3]);
        pf[2]=pack2(st[2*kk+1][0],st[2*kk+1][1]);
        pf[3]=pack2(st[2*kk+1][2],st[2*kk+1][3]);
        sf[0]=pack2(dpt[2*kk][0],  dpt[2*kk][1]);
        sf[1]=pack2(dpt[2*kk][2],  dpt[2*kk][3]);
        sf[2]=pack2(dpt[2*kk+1][0],dpt[2*kk+1][1]);
        sf[3]=pack2(dpt[2*kk+1][2],dpt[2*kk+1][3]);
        #pragma unroll
        for (int n=0;n<8;++n) {
          const unsigned short* pg = gtr + (n*8+g)*FA_ROW + kk*16 + t*2;
          const unsigned short* pq = qtr + (n*8+g)*FA_ROW + kk*16 + t*2;
          unsigned bg[2], bq[2];
          bg[0]=*(const unsigned*)pg; bg[1]=*(const unsigned*)(pg+8);
          bq[0]=*(const unsigned*)pq; bq[1]=*(const unsigned*)(pq+8);
          MMA(dv[n], pf, bg);
          MMA(dk[n], sf, bq);
        }
      }
    }
  }

  const size_t base = seq_base*(size_t)qkv_width + kvh*FA_D;
  #pragma unroll
  for (int n=0;n<8;++n) {
    int c = n*8 + t*2;
    if (key_a < seq_len) {
      size_t p = base + (size_t)key_a*qkv_width;
      *(unsigned*)(grad_qkv+p+key_base+c) = pack2(dk[n][0]*scale,dk[n][1]*scale);
      *(unsigned*)(grad_qkv+p+value_base+c) = pack2(dv[n][0],dv[n][1]);
    }
    if (key_b < seq_len) {
      size_t p = base + (size_t)key_b*qkv_width;
      *(unsigned*)(grad_qkv+p+key_base+c) = pack2(dk[n][2]*scale,dk[n][3]*scale);
      *(unsigned*)(grad_qkv+p+value_base+c) = pack2(dv[n][2],dv[n][3]);
    }
  }
}

"#;

/// The fused kernels, resolved once per device.
pub(crate) struct FlashKernels {
    pub(crate) forward: CudaFunction,
    /// The image path's variant: FP16, unmasked, and able to read its keys
    /// from a different tensor than its queries.
    pub(crate) image: CudaFunction,
    pub(crate) delta: CudaFunction,
    pub(crate) grad_query: CudaFunction,
    pub(crate) grad_key_value: CudaFunction,
}

/// Compiles and loads the fused attention module for `context`, or returns
/// `None` when the device cannot run it.
///
/// `mma.sync` with BF16 operands is an Ampere instruction, so a device below
/// compute capability 8.0 gets `None` and keeps the three-kernel path. A
/// compilation or load failure is also `None` rather than an error: the fused
/// path is an optimization, and losing it should slow a run down, not stop it.
pub(crate) fn flash_kernels(device: usize, context: &Arc<CudaContext>) -> Option<FlashKernels> {
    static MODULES: OnceLock<Mutex<HashMap<usize, Option<Arc<CudaModule>>>>> = OnceLock::new();

    match context.compute_capability() {
        Ok((major, _)) if major >= 8 => {}
        _ => return None,
    }

    let mut modules = MODULES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .ok()?;
    let module = modules
        .entry(device)
        .or_insert_with(|| {
            let options = CompileOptions {
                arch: Some("compute_80"),
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(KERNELS, options).ok()?;
            context.load_module(ptx).ok()
        })
        .clone()?;

    let get = |name: &str| {
        module
            .load_function(name)
            .map_err(cuda_err("fused attention kernel lookup"))
            .ok()
    };
    Some(FlashKernels {
        forward: get("flash_attention_fwd")?,
        image: get("image_attention_fwd")?,
        delta: get("flash_attention_delta")?,
        grad_query: get("flash_attention_dq")?,
        grad_key_value: get("flash_attention_dkv")?,
    })
}

/// Whether the fused path can serve this shape.
///
/// The fragment layout is written for a 64-wide head, and the BF16 operands
/// are only an honest substitute where the cuBLAS path was already rounding
/// its own operands to BF16 - that is, under mixed precision.
pub(crate) fn eligible(mixed_precision: bool, head_dim: usize) -> bool {
    mixed_precision && head_dim == TILE && !disabled()
}

/// Set by `RUSTING_BRAIN_NO_FLASH`, and by the parity test that runs the same
/// model both ways. An escape hatch worth keeping: a fused attention kernel is
/// the kind of thing that is wrong on exactly one shape, and a run that can
/// fall back without a rebuild is a run that can be bisected.
pub(crate) static DISABLED: AtomicBool = AtomicBool::new(false);

pub(crate) fn disabled() -> bool {
    static FROM_ENV: OnceLock<()> = OnceLock::new();
    FROM_ENV.get_or_init(|| {
        if std::env::var_os("RUSTING_BRAIN_NO_FLASH").is_some() {
            DISABLED.store(true, Ordering::Relaxed);
        }
    });
    DISABLED.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    /// NVRTC only sees this source at runtime, so nothing in a `cargo build`
    /// catches a typo in the PTX. Without this test a kernel that fails to
    /// compile is silently `None` and every parity test still passes, having
    /// compared the three-kernel path against itself.
    #[test]
    fn the_fused_module_compiles_or_skips_without_device() {
        let Ok(context) = cudarc::driver::CudaContext::new(0) else {
            return;
        };
        match context.compute_capability() {
            Ok((major, _)) if major >= 8 => {}
            _ => return,
        }
        assert!(
            super::flash_kernels(0, &context).is_some(),
            "the device is an Ampere or later but the fused attention kernels did not load"
        );
    }
}
