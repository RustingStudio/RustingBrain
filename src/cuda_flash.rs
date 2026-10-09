//! Fused attention for the Ampere tensor cores, causal or bidirectional.
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
use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Query rows and key columns of one tile, and the head width of the image
/// kernel. The training kernels are also built for a head width of 128.
pub(crate) const TILE: usize = 64;
/// Threads per block: four warps, sixteen query rows each.
pub(crate) const THREADS: u32 = 128;

/// The image variant's block: eight warps over 128 query rows rather than
/// four over 64, which is twice the work out of every key tile it streams
/// through shared memory, for the same shared memory and the same occupancy.
pub(crate) const IMAGE_ROWS: usize = 128;
pub(crate) const IMAGE_THREADS: u32 = 256;
/// Two `[64][64]` BF16 tiles, each row padded by eight elements so that the
/// eight lanes of an `mma` fragment land in eight different shared-memory
/// banks instead of all in one.
pub(crate) const SHARED_BYTES: u32 = 2 * (TILE * (TILE + 8) * 2) as u32;

/// Bytes of one `[64][head_dim]` tile and of one `[head_dim][64]` tile, each
/// row padded by eight elements as above.
const fn tiles(head_dim: usize) -> (usize, usize) {
    (TILE * (head_dim + 8) * 2, head_dim * (TILE + 8) * 2)
}
/// Keys key-major and values dimension-major.
pub(crate) const fn forward_shared_bytes(head_dim: usize) -> u32 {
    let (rows, columns) = tiles(head_dim);
    (rows + columns) as u32
}
/// Keys and values, key-major; the query gradient reads the keys transposed.
pub(crate) const fn dq_shared_bytes(head_dim: usize) -> u32 {
    let (rows, _) = tiles(head_dim);
    (2 * rows) as u32
}
/// Queries and output gradients, query-major, plus the log-sum-exp and delta
/// of the query tile.
pub(crate) const fn dkv_shared_bytes(head_dim: usize) -> u32 {
    let (rows, _) = tiles(head_dim);
    (2 * rows + 2 * TILE * 4) as u32
}

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
// Four 8x8 BF16 matrices out of shared memory in one instruction, each lane
// naming one 16-byte row: lanes 8m..8m+7 give the rows of matrix m, which
// lands in r[m] laid out as an mma B operand half (row g, columns 2t, 2t+1).
__device__ __forceinline__ void ldsm_x4(unsigned r[4],const unsigned short*row){
  unsigned a=(unsigned)__cvta_generic_to_shared(row);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(a));
}
// The same with each matrix transposed on the way: a thread gets column g at
// rows 2t, 2t+1, so a k-major tile serves as the B operand of its transpose.
__device__ __forceinline__ void ldsm_x4_trans(unsigned r[4],const unsigned short*row){
  unsigned a=(unsigned)__cvta_generic_to_shared(row);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(a));
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

// The same product with an FP16 accumulator, which a consumer Ampere card
// issues at twice the rate. The two result registers hold the four elements
// the FP32 form spreads over four, packed in pairs. Only the scores use it:
// they are a dot product over one head width, where eleven mantissa bits are
// more than the softmax downstream can tell apart.
#define MMAH16(d,a,b) asm volatile( \
  "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 " \
  "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n" \
  : "+r"(d[0]),"+r"(d[1]) \
  : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]))
__device__ __forceinline__ float lowh(unsigned p){ return h2f((unsigned short)(p&0xffff)); }
__device__ __forceinline__ float highh(unsigned p){ return h2f((unsigned short)(p>>16)); }

// The same fused attention for the image path: FP16 operands, no mask, and
// queries that may come from one tensor while keys and values come from
// another, which is what a cross-attention is. `q_base`, `k_base` and
// `v_base` are the column each projection starts at, so a fused QKV buffer
// needs no split.
//
// One block per (query tile, head). Nothing here writes a log-sum-exp: there
// is no backward pass over an image model in this crate.
extern "C" __global__ __launch_bounds__(256) void image_attention_fwd(
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
  const int q_row0 = blockIdx.x*128 + warp*16;
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
    for (int i = threadIdx.x; i < 64*FA_D; i += 256) {
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

    unsigned packed[8][2];
    #pragma unroll
    for (int n=0;n<8;++n) { packed[n][0]=0; packed[n][1]=0; }
    #pragma unroll
    for (int n=0;n<8;++n) {
      #pragma unroll
      for (int kk=0; kk<4; ++kk) {
        const unsigned short* p = ks + (n*8+g)*FA_ROW + kk*16 + t*2;
        unsigned bf[2];
        bf[0] = *(const unsigned*)p;
        bf[1] = *(const unsigned*)(p+8);
        MMAH16(packed[n], qf[kk], bf);
      }
    }
    float s[8][4];
    #pragma unroll
    for (int n=0;n<8;++n) {
      s[n][0]=lowh(packed[n][0]); s[n][1]=highh(packed[n][0]);
      s[n][2]=lowh(packed[n][1]); s[n][3]=highh(packed[n][1]);
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
    // The caller folded log2(e) into the scale, so every score arrives in
    // base two and `exp2f` is the bare hardware instruction that `__expf`
    // wraps in a multiply. A softmax has no base of its own: the running
    // maximum, the correction and the sum all live in whichever one it is.
    float corr_a=exp2f(m_a-new_a), corr_b=exp2f(m_b-new_b);
    float sum_a=0.f, sum_b=0.f;
    #pragma unroll
    for (int n=0;n<8;++n) {
      s[n][0]=exp2f(s[n][0]-new_a); s[n][1]=exp2f(s[n][1]-new_a);
      s[n][2]=exp2f(s[n][2]-new_b); s[n][3]=exp2f(s[n][3]-new_b);
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

// The training kernels below are templates over the head width `D`, 64 or
// 128. A tile is always 64 queries or keys; only the head dimension changes.
// A `[row][dim]` tile is padded to `DROW` and a `[dim][row]` one to `FA_ROW`,
// both chosen so the eight rows of an mma fragment land in different banks.
#define DROW (D+8)

// One block per (query tile, head, sequence). Queries live in registers for
// the whole block; keys and values stream through shared memory one tile at a
// time. `lse` is written in the same layout the three-kernel path used, so the
// backward pass rebuilds the probabilities from it unchanged.
template<int D> __device__ __forceinline__ void flash_fwd(
    const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ kvp,
    unsigned short* __restrict__ out,
    float* __restrict__ lse,
    int q_len, int kv_len, int q_width, int kv_width, int query_width,
    int key_base, int value_base, int group,
    int lse_head_stride, float scale, int causal)
{
  extern __shared__ unsigned short smem[];
  unsigned short* ks = smem;                      // [key][dim]
  unsigned short* vt = smem + 64*DROW;            // [dim][key], transposed

  const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int kv = (h / group) * D;
  const size_t seq_base = (size_t)b * q_len, kv_seq = (size_t)b * kv_len;
  const int q_row0 = qt*64 + warp*16;
  const int row_a = q_row0 + g, row_b = q_row0 + g + 8;

  // Queries, scaled once here rather than by the matmul's alpha as the cuBLAS
  // path did, and rounded to BF16 exactly as that path's tensor cores did.
  unsigned qf[D/16][4];
  {
    const size_t base = seq_base*(size_t)q_width + h*D;
    #pragma unroll
    for (int kk=0; kk<D/16; ++kk) {
      int c = kk*16 + t*2;
      float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f;
      if (row_a < q_len) {
        size_t p = base + (size_t)row_a*q_width;
        a0=bf16_load(q,p+c)*scale;   a1=bf16_load(q,p+c+1)*scale;
        a4=bf16_load(q,p+c+8)*scale; a5=bf16_load(q,p+c+9)*scale;
      }
      if (row_b < q_len) {
        size_t p = base + (size_t)row_b*q_width;
        a2=bf16_load(q,p+c)*scale;   a3=bf16_load(q,p+c+1)*scale;
        a6=bf16_load(q,p+c+8)*scale; a7=bf16_load(q,p+c+9)*scale;
      }
      qf[kk][0]=pack2(a0,a1); qf[kk][1]=pack2(a2,a3);
      qf[kk][2]=pack2(a4,a5); qf[kk][3]=pack2(a6,a7);
    }
  }

  float acc[D/8][4];
  #pragma unroll
  for (int n=0;n<D/8;++n)
    #pragma unroll
    for (int i=0;i<4;++i) acc[n][i]=0.f;
  // Running maximum and sum of the two query rows this thread holds.
  float m_a=NEG_INF, m_b=NEG_INF, l_a=0.f, l_b=0.f;

  // Causal: `causal` is one more than the key position of query row 0, zero
  // in training, so query row r sees keys up to r + shift and query tile `qt`
  // sees key tiles up to the one holding its last row's. Bidirectional: every
  // query tile sees every key tile.
  const int shift = causal ? causal - 1 : 0;
  const int key_tiles = ((kv_len + 63) >> 6) - 1;
  const int last = causal ? min((qt*64 + 63 + shift) >> 6, key_tiles) : key_tiles;
  for (int kt=0; kt<=last; ++kt) {
    __syncthreads();
    // One element a thread, not two: the value tile is stored transposed, and
    // measured, the wider load does not pay for the extra shared-memory bank
    // conflicts that come with it.
    for (int i = threadIdx.x; i < 64*D; i += 128) {
      int r = (unsigned)i / D, c = (unsigned)i % D;
      int key = kt*64 + r;
      unsigned short kb = 0, vb = 0;
      if (key < kv_len) {
        size_t p = (kv_seq + key)*(size_t)kv_width;
        kb = kvp[p + key_base + kv + c];
        vb = kvp[p + value_base + kv + c];
      }
      ks[r*DROW + c] = kb;
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
      for (int kk=0; kk<D/16; ++kk) {
        const unsigned short* p = ks + (n*8+g)*DROW + kk*16 + t*2;
        unsigned bf[2];
        bf[0] = *(const unsigned*)p;
        bf[1] = *(const unsigned*)(p+8);
        MMA(s[n], qf[kk], bf);
      }
    }

    // Only the tiles the diagonal crosses are partly masked under a causal
    // mask - the last one in training, up to two with a shift - and only the
    // last can hold keys past the end of a sequence whose length is not a
    // multiple of 64.
    if (kt == last || (causal && kt*64 + 63 > qt*64 + shift)) {
      const int see_a = causal ? row_a + shift : kv_len, see_b = causal ? row_b + shift : kv_len;
      #pragma unroll
      for (int n=0;n<8;++n) {
        int key = kt*64 + n*8 + t*2;
        if (key   > see_a || key   >= kv_len) s[n][0]=NEG_INF;
        if (key+1 > see_a || key+1 >= kv_len) s[n][1]=NEG_INF;
        if (key   > see_b || key   >= kv_len) s[n][2]=NEG_INF;
        if (key+1 > see_b || key+1 >= kv_len) s[n][3]=NEG_INF;
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
    for (int n=0;n<D/8;++n) {
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
      for (int n=0;n<D/8;++n) {
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
  const size_t obase = seq_base*(size_t)query_width + h*D;
  #pragma unroll
  for (int n=0;n<D/8;++n) {
    int c = n*8 + t*2;
    if (row_a < q_len) {
      size_t p = obase + (size_t)row_a*query_width + c;
      *(unsigned*)(out+p) = pack2(acc[n][0]*inv_a,acc[n][1]*inv_a);
    }
    if (row_b < q_len) {
      size_t p = obase + (size_t)row_b*query_width + c;
      *(unsigned*)(out+p) = pack2(acc[n][2]*inv_b,acc[n][3]*inv_b);
    }
  }
  if (t == 0) {
    float* p = lse + (size_t)h*lse_head_stride + seq_base;
    if (row_a < q_len) p[row_a]=m_a+__logf(l_a);
    if (row_b < q_len) p[row_b]=m_b+__logf(l_b);
  }
}
// Row dot product of the attention output with its gradient, which is the
// `delta` term the softmax backward needs. One warp per row, `D / 32`
// dimensions a lane, so every load is contiguous.
template<int D> __device__ __forceinline__ void flash_delta(
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
    size_t o = (size_t)r*query_width + h*D;
    float sum = 0.f;
    #pragma unroll
    for (int c = lane; c < D; c += 32) sum += bf16_load(out,o+c)*bf16_load(grad_out,o+c);
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
template<int D> __device__ __forceinline__ void flash_dq(
    const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ kvp,
    const unsigned short* __restrict__ grad_out,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    unsigned short* __restrict__ grad_q,
    int q_len, int kv_len, int q_width, int kv_width, int query_width,
    int key_base, int value_base, int group,
    int lse_head_stride, float scale, int causal)
{
  extern __shared__ unsigned short smem[];
  unsigned short* ks = smem;                        // [key][dim]
  unsigned short* vs = smem + 64*DROW;              // [key][dim]

  const int qt = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int kv = (h / group) * D;
  const size_t seq_base = (size_t)b * q_len, kv_seq = (size_t)b * kv_len;
  const int q_row0 = qt*64 + warp*16;
  const int row_a = q_row0 + g, row_b = q_row0 + g + 8;

  unsigned qf[D/16][4], df[D/16][4];
  float lse_a=0.f, lse_b=0.f, del_a=0.f, del_b=0.f;
  {
    const size_t qbase = seq_base*(size_t)q_width + h*D;
    const size_t gbase = seq_base*(size_t)query_width + h*D;
    #pragma unroll
    for (int kk=0; kk<D/16; ++kk) {
      int c = kk*16 + t*2;
      float q0=0.f,q1=0.f,q2=0.f,q3=0.f,q4=0.f,q5=0.f,q6=0.f,q7=0.f;
      df[kk][0]=0; df[kk][1]=0; df[kk][2]=0; df[kk][3]=0;
      if (row_a < q_len) {
        size_t p = qbase + (size_t)row_a*q_width;
        q0=bf16_load(q,p+c)*scale;   q1=bf16_load(q,p+c+1)*scale;
        q4=bf16_load(q,p+c+8)*scale; q5=bf16_load(q,p+c+9)*scale;
        size_t e = gbase + (size_t)row_a*query_width;
        df[kk][0]=bf16_pair(grad_out,e+c); df[kk][2]=bf16_pair(grad_out,e+c+8);
      }
      if (row_b < q_len) {
        size_t p = qbase + (size_t)row_b*q_width;
        q2=bf16_load(q,p+c)*scale;   q3=bf16_load(q,p+c+1)*scale;
        q6=bf16_load(q,p+c+8)*scale; q7=bf16_load(q,p+c+9)*scale;
        size_t e = gbase + (size_t)row_b*query_width;
        df[kk][1]=bf16_pair(grad_out,e+c); df[kk][3]=bf16_pair(grad_out,e+c+8);
      }
      qf[kk][0]=pack2(q0,q1); qf[kk][1]=pack2(q2,q3);
      qf[kk][2]=pack2(q4,q5); qf[kk][3]=pack2(q6,q7);
    }
    const float* row = lse + (size_t)h*lse_head_stride + seq_base;
    const float* dd  = delta + (size_t)h*lse_head_stride + seq_base;
    if (row_a < q_len) { lse_a=row[row_a]; del_a=dd[row_a]; }
    if (row_b < q_len) { lse_b=row[row_b]; del_b=dd[row_b]; }
  }

  float acc[D/8][4];
  #pragma unroll
  for (int n=0;n<D/8;++n)
    #pragma unroll
    for (int i=0;i<4;++i) acc[n][i]=0.f;

  const int last = causal ? qt : ((kv_len + 63) >> 6) - 1;
  for (int ktile=0; ktile<=last; ++ktile) {
    __syncthreads();
    for (int i = threadIdx.x*2; i < 64*D; i += 256) {
      int r = (unsigned)i / D, c = (unsigned)i % D;
      int key = ktile*64 + r;
      unsigned kw = 0, vw = 0;
      if (key < kv_len) {
        size_t p = (kv_seq + key)*(size_t)kv_width;
        kw = bf16_pair(kvp,p + key_base + kv + c);
        vw = bf16_pair(kvp,p + value_base + kv + c);
      }
      *(unsigned*)(ks + r*DROW + c) = kw;
      *(unsigned*)(vs + r*DROW + c) = vw;
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
      for (int kk=0; kk<D/16; ++kk) {
        const unsigned short* pk = ks + (n*8+g)*DROW + kk*16 + t*2;
        const unsigned short* pv = vs + (n*8+g)*DROW + kk*16 + t*2;
        unsigned bk[2], bv[2];
        bk[0]=*(const unsigned*)pk; bk[1]=*(const unsigned*)(pk+8);
        bv[0]=*(const unsigned*)pv; bv[1]=*(const unsigned*)(pv+8);
        MMA(s[n], qf[kk], bk);
        MMA(dp[n], df[kk], bv);
      }
    }

    if (ktile == last) {
      #pragma unroll
      for (int n=0;n<8;++n) {
        int key = ktile*64 + n*8 + t*2;
        if ((causal && key   > row_a) || key   >= kv_len) s[n][0]=NEG_INF;
        if ((causal && key+1 > row_a) || key+1 >= kv_len) s[n][1]=NEG_INF;
        if ((causal && key   > row_b) || key   >= kv_len) s[n][2]=NEG_INF;
        if ((causal && key+1 > row_b) || key+1 >= kv_len) s[n][3]=NEG_INF;
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
      for (int n=0;n<D/8;n+=2) {
        // Matrix m is output columns (n + m/2)*8, k (key) half m%2.
        unsigned bf[4];
        ldsm_x4_trans(bf, ks + (kk*16 + ((lane>>3)&1)*8 + (lane&7))*DROW + (n + (lane>>4))*8);
        MMA(acc[n],   pf, bf);
        MMA(acc[n+1], pf, (bf+2));
      }
    }
  }

  const size_t obase = seq_base*(size_t)q_width + h*D;
  #pragma unroll
  for (int n=0;n<D/8;++n) {
    int c = n*8 + t*2;
    if (row_a < q_len) {
      size_t p = obase + (size_t)row_a*q_width;
      *(unsigned*)(grad_q+p+c) = pack2(acc[n][0]*scale,acc[n][1]*scale);
    }
    if (row_b < q_len) {
      size_t p = obase + (size_t)row_b*q_width;
      *(unsigned*)(grad_q+p+c) = pack2(acc[n][2]*scale,acc[n][3]*scale);
    }
  }
}

// Gradient of the keys and values. This one is blocked the other way round,
// one block per (key tile, key/value head, sequence), and walks the query
// tiles that can see that key tile. Blocking on the key/value head rather than
// the query head is what keeps grouped-query attention exact without atomics:
// every query head that shares a key head is summed inside one block.
template<int D> __device__ __forceinline__ void flash_dkv(
    const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ kvp,
    const unsigned short* __restrict__ grad_out,
    const float* __restrict__ lse,
    const float* __restrict__ delta,
    unsigned short* __restrict__ grad_kv,
    int q_len, int kv_len, int q_width, int kv_width, int query_width,
    int key_base, int value_base, int group,
    int lse_head_stride, float scale, int causal)
{
  extern __shared__ unsigned short smem[];
  // Both tiles query-major; the products that want them dimension-major read
  // them through `ldsm_x4_trans`.
  unsigned short* qs  = smem;                       // [query][dim]
  unsigned short* gs  = qs + 64*DROW;               // [query][dim]
  float* lse_s = (float*)(gs + 64*DROW);
  float* del_s = lse_s + 64;

  const int jt = blockIdx.x, kvh = blockIdx.y, b = blockIdx.z;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const size_t seq_base = (size_t)b * q_len, kv_seq = (size_t)b * kv_len;
  const int k_row0 = jt*64 + warp*16;
  const int key_a = k_row0 + g, key_b = k_row0 + g + 8;
  const int tiles = (q_len + 63) >> 6;

  // Keys and values stay in registers for the whole block, as the operand of
  // every matmul below.
  unsigned kf[D/16][4], vf[D/16][4];
  {
    const size_t base = kv_seq*(size_t)kv_width + kvh*D;
    #pragma unroll
    for (int kk=0; kk<D/16; ++kk) {
      int c = kk*16 + t*2;
      kf[kk][0]=0; kf[kk][1]=0; kf[kk][2]=0; kf[kk][3]=0;
      vf[kk][0]=0; vf[kk][1]=0; vf[kk][2]=0; vf[kk][3]=0;
      if (key_a < kv_len) {
        size_t p = base + (size_t)key_a*kv_width;
        kf[kk][0]=bf16_pair(kvp,p+key_base+c);
        kf[kk][2]=bf16_pair(kvp,p+key_base+c+8);
        vf[kk][0]=bf16_pair(kvp,p+value_base+c);
        vf[kk][2]=bf16_pair(kvp,p+value_base+c+8);
      }
      if (key_b < kv_len) {
        size_t p = base + (size_t)key_b*kv_width;
        kf[kk][1]=bf16_pair(kvp,p+key_base+c);
        kf[kk][3]=bf16_pair(kvp,p+key_base+c+8);
        vf[kk][1]=bf16_pair(kvp,p+value_base+c);
        vf[kk][3]=bf16_pair(kvp,p+value_base+c+8);
      }
    }
  }

  float dk[D/8][4], dv[D/8][4];
  #pragma unroll
  for (int n=0;n<D/8;++n)
    #pragma unroll
    for (int i=0;i<4;++i) { dk[n][i]=0.f; dv[n][i]=0.f; }

  for (int hq = kvh*group; hq < (kvh+1)*group; ++hq) {
    for (int it = causal ? jt : 0; it < tiles; ++it) {
      __syncthreads();
      for (int i = threadIdx.x*2; i < 64*D; i += 256) {
        int r = (unsigned)i / D, c = (unsigned)i % D;
        int qi = it*64 + r;
        unsigned qw = 0, gw = 0;
        if (qi < q_len) {
          qw = bf16_pair(q,(seq_base+qi)*(size_t)q_width + hq*D + c);
          gw = bf16_pair(grad_out,(seq_base+qi)*(size_t)query_width + hq*D + c);
        }
        // The queries are stored unscaled, so the scores carry the scale
        // instead: one multiply per score rather than one per element here.
        *(unsigned*)(qs + r*DROW + c) = qw;
        *(unsigned*)(gs + r*DROW + c) = gw;
      }
      if (threadIdx.x < 64) {
        int qi = it*64 + threadIdx.x;
        bool ok = qi < q_len;
        lse_s[threadIdx.x] = ok ? lse[(size_t)hq*lse_head_stride + seq_base + qi] : 0.f;
        del_s[threadIdx.x] = ok ? delta[(size_t)hq*lse_head_stride + seq_base + qi] : 0.f;
      }
      __syncthreads();

      // Transposed scores and transposed dP: rows are keys, columns queries.
      // At width 128 the query tile goes in two halves of 32, so only half
      // the scores are live beside the key, value and gradient fragments: the
      // whole tile's scores spilled. At 64 one pass is quicker.
      constexpr int HALVES = D > 64 ? 2 : 1, NQ = 8 / HALVES;
      #pragma unroll
      for (int half=0; half<HALVES; ++half) {
        float st[NQ][4], dpt[NQ][4];
        #pragma unroll
        for (int n=0;n<NQ;++n)
          #pragma unroll
          for (int i=0;i<4;++i) { st[n][i]=0.f; dpt[n][i]=0.f; }
        #pragma unroll
        for (int n=0;n<NQ;++n) {
          #pragma unroll
          for (int kk=0; kk<D/16; kk+=2) {
            // Matrix m is columns kk*16 + 8m: both halves of two k steps.
            int at = ((half*NQ + n)*8 + (lane&7))*DROW + kk*16 + (lane>>3)*8;
            unsigned bq[4], bg[4];
            ldsm_x4(bq, qs + at);
            ldsm_x4(bg, gs + at);
            MMA(st[n],  kf[kk],   bq);
            MMA(dpt[n], vf[kk],   bg);
            MMA(st[n],  kf[kk+1], (bq+2));
            MMA(dpt[n], vf[kk+1], (bg+2));
          }
        }

        #pragma unroll
        for (int n=0;n<NQ;++n) {
          int col = (half*NQ + n)*8 + t*2;
          int q0 = it*64 + col, q1 = q0 + 1;
          float l0 = lse_s[col], l1 = lse_s[col+1];
          float d0 = del_s[col], d1 = del_s[col+1];
          float p0 = ((!causal || q0 >= key_a) && q0 < q_len) ? __expf(st[n][0]*scale-l0) : 0.f;
          float p1 = ((!causal || q1 >= key_a) && q1 < q_len) ? __expf(st[n][1]*scale-l1) : 0.f;
          float p2 = ((!causal || q0 >= key_b) && q0 < q_len) ? __expf(st[n][2]*scale-l0) : 0.f;
          float p3 = ((!causal || q1 >= key_b) && q1 < q_len) ? __expf(st[n][3]*scale-l1) : 0.f;
          st[n][0]=p0; st[n][1]=p1; st[n][2]=p2; st[n][3]=p3;
          dpt[n][0]=p0*(dpt[n][0]-d0); dpt[n][1]=p1*(dpt[n][1]-d1);
          dpt[n][2]=p2*(dpt[n][2]-d0); dpt[n][3]=p3*(dpt[n][3]-d1);
        }

        #pragma unroll
        for (int kk=0; kk<NQ/2; ++kk) {
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
          for (int n=0;n<D/8;n+=2) {
            // Matrix m is output columns (n + m/2)*8, k (query) half m%2.
            int at = ((half*NQ/2 + kk)*16 + ((lane>>3)&1)*8 + (lane&7))*DROW + (n + (lane>>4))*8;
            unsigned bg[4], bq[4];
            ldsm_x4_trans(bg, gs + at);
            ldsm_x4_trans(bq, qs + at);
            MMA(dv[n],   pf, bg);
            MMA(dk[n],   sf, bq);
            MMA(dv[n+1], pf, (bg+2));
            MMA(dk[n+1], sf, (bq+2));
          }
        }
      }
    }
  }

  const size_t base = kv_seq*(size_t)kv_width + kvh*D;
  #pragma unroll
  for (int n=0;n<D/8;++n) {
    int c = n*8 + t*2;
    if (key_a < kv_len) {
      size_t p = base + (size_t)key_a*kv_width;
      *(unsigned*)(grad_kv+p+key_base+c) = pack2(dk[n][0]*scale,dk[n][1]*scale);
      *(unsigned*)(grad_kv+p+value_base+c) = pack2(dv[n][0],dv[n][1]);
    }
    if (key_b < kv_len) {
      size_t p = base + (size_t)key_b*kv_width;
      *(unsigned*)(grad_kv+p+key_base+c) = pack2(dk[n][2]*scale,dk[n][3]*scale);
      *(unsigned*)(grad_kv+p+value_base+c) = pack2(dv[n][2],dv[n][3]);
    }
  }
}

// One entry point per kernel and head width, named `<kernel>_<width>`.
// The key/value gradient at width 64 needs at most 170 registers to fit three
// blocks on an SM: separate query and key buffers took it from 168 to 173,
// two blocks, and 25% slower, so the bound holds it there.
// Queries come from `q` and keys and values from `kv`, `q_len` and `kv_len`
// per sequence: self-attention passes one fused buffer as both, with equal
// lengths, and cross-attention passes two.
#define FLASH_ARGS \
    int q_len, int kv_len, int q_width, int kv_width, int query_width, \
    int key_base, int value_base, int group, int lse_head_stride, float scale, int causal
#define FLASH_PASS \
    q_len, kv_len, q_width, kv_width, query_width, key_base, value_base, group, \
    lse_head_stride, scale, causal
#define FLASH_WIDTH(W) \
extern "C" __global__ __launch_bounds__(128) void flash_attention_fwd_##W( \
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ kv, \
    unsigned short* __restrict__ out, float* __restrict__ lse, FLASH_ARGS) \
{ flash_fwd<W>(q, kv, out, lse, FLASH_PASS); } \
extern "C" __global__ void flash_attention_delta_##W( \
    const unsigned short* __restrict__ out, const unsigned short* __restrict__ grad_out, \
    float* __restrict__ delta, int rows, int heads, int query_width, int delta_head_stride) \
{ flash_delta<W>(out, grad_out, delta, rows, heads, query_width, delta_head_stride); } \
extern "C" __global__ __launch_bounds__(128) void flash_attention_dq_##W( \
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ kv, \
    const unsigned short* __restrict__ grad_out, \
    const float* __restrict__ lse, const float* __restrict__ delta, \
    unsigned short* __restrict__ grad_q, FLASH_ARGS) \
{ flash_dq<W>(q, kv, grad_out, lse, delta, grad_q, FLASH_PASS); } \
extern "C" __global__ __launch_bounds__(128, (W) == 64 ? 3 : 1) void flash_attention_dkv_##W( \
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ kv, \
    const unsigned short* __restrict__ grad_out, \
    const float* __restrict__ lse, const float* __restrict__ delta, \
    unsigned short* __restrict__ grad_kv, FLASH_ARGS) \
{ flash_dkv<W>(q, kv, grad_out, lse, delta, grad_kv, FLASH_PASS); }

FLASH_WIDTH(64)
FLASH_WIDTH(128)

// `y[r, n] = scales[n] * (x[r, ..] . q[n, ..]) + beta * y[r, n]` for up to 16
// BF16 rows of `x`, on the tensor cores: the int8 decode GEMV for a batch.
// A block owns 16 weight rows, the mma's A, and the rows of `x` are its B, so
// each weight byte is read once whatever the batch. Its eight warps split the
// width in steps of 32 and add their partial sums in shared memory: one warp
// per 16 rows alone leaves too few warps in flight to cover the loads. The k
// order inside an mma is free as long as A and B agree, so a thread reads 8
// consecutive weights of each of its two rows and the same 8 inputs of each
// of its sequences, one load apiece, and feeds them to two mmas: k slots
// {2t, 2t+1} take inputs {4s, 4s+1} of the 8 and slots {2t+8, 2t+9} take
// {4s+2, 4s+3}. An int8 value converts to BF16 exactly. `inner` is a
// multiple of 32.
#define Q8_WARPS 8
extern "C" __global__ __launch_bounds__(256) void gemv_q8_mma(
    const signed char* __restrict__ q, const float* __restrict__ scales,
    const unsigned short* __restrict__ x, float* __restrict__ y,
    int units, int inner, int rows, float beta)
{
  __shared__ float partial[Q8_WARPS-1][8][32];
  int warp=threadIdx.x/32,lane=threadIdx.x%32,g=lane/4,t=lane%4;
  int u0=blockIdx.x*16;
  int ua=min(u0+g,units-1),ub=min(u0+g+8,units-1);
  const signed char*wa=q+(size_t)ua*inner+t*8;
  const signed char*wb=q+(size_t)ub*inner+t*8;
  bool in0=g<rows,in1=g+8<rows,two=rows>8;
  const unsigned short*x0=x+(size_t)(in0?g:0)*inner+t*8;
  const unsigned short*x1=x+(size_t)(in1?g+8:0)*inner+t*8;
  float d[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
  for(int k=warp*32;k<inner;k+=Q8_WARPS*32){
    int2 pa=*(const int2*)(wa+k),pb=*(const int2*)(wb+k);
    uint4 xa=in0?*(const uint4*)(x0+k):make_uint4(0,0,0,0);
    uint4 xb=two&&in1?*(const uint4*)(x1+k):make_uint4(0,0,0,0);
    const signed char*ba=(const signed char*)&pa,*bb=(const signed char*)&pb;
    const unsigned*ia=(const unsigned*)&xa,*ib=(const unsigned*)&xb;
    #pragma unroll
    for(int s=0;s<2;s++){
      unsigned a[4]={pack2(ba[4*s],ba[4*s+1]),pack2(bb[4*s],bb[4*s+1]),
                     pack2(ba[4*s+2],ba[4*s+3]),pack2(bb[4*s+2],bb[4*s+3])};
      unsigned b0[2]={ia[2*s],ia[2*s+1]};
      MMA(d,a,b0);
      if(two){unsigned b1[2]={ib[2*s],ib[2*s+1]};MMA((d+4),a,b1);}
    }
  }
  if(warp>0){
    #pragma unroll
    for(int e=0;e<8;e++)partial[warp-1][e][lane]=d[e];
  }
  __syncthreads();
  if(warp>0)return;
  #pragma unroll
  for(int w=0;w<Q8_WARPS-1;w++){
    #pragma unroll
    for(int e=0;e<8;e++)d[e]+=partial[w][e][lane];
  }
  float sa=scales[ua],sb=scales[ub];
  #pragma unroll
  for(int e=0;e<8;e++){
    int unit=u0+g+((e&2)?8:0),n=2*t+(e&1)+(e>=4?8:0);
    if(unit>=units||n>=rows)continue;
    size_t i=(size_t)n*units+unit;
    float v=d[e]*((e&2)?sb:sa);
    y[i]=beta==0.f?v:v+beta*y[i];
  }
}

"#;

/// The training kernels for one head width.
pub(crate) struct FlashWidth {
    pub(crate) forward: CudaFunction,
    pub(crate) delta: CudaFunction,
    pub(crate) grad_query: CudaFunction,
    pub(crate) grad_key_value: CudaFunction,
}

/// The fused kernels, resolved once per device.
pub(crate) struct FlashKernels {
    /// The image path's variant: FP16, unmasked, and able to read its keys
    /// from a different tensor than its queries.
    pub(crate) image: CudaFunction,
    /// The int8 decode GEMV for a batch, on the tensor cores.
    pub(crate) gemv_q8: CudaFunction,
    narrow: FlashWidth,
    wide: FlashWidth,
}

impl FlashKernels {
    /// The kernels for a head width [`eligible`] accepted.
    pub(crate) fn width(&self, head_dim: usize) -> &FlashWidth {
        if head_dim == 128 {
            &self.wide
        } else {
            &self.narrow
        }
    }
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
    let width = |head_dim: usize| {
        let kernels = FlashWidth {
            forward: get(&format!("flash_attention_fwd_{head_dim}"))?,
            delta: get(&format!("flash_attention_delta_{head_dim}"))?,
            grad_query: get(&format!("flash_attention_dq_{head_dim}"))?,
            grad_key_value: get(&format!("flash_attention_dkv_{head_dim}"))?,
        };
        // At 128 the backward kernels want more than the 48 KiB of shared
        // memory a launch gets without asking.
        let opt_in = |function: &CudaFunction, bytes: u32| {
            function
                .set_attribute(
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    bytes as i32,
                )
                .ok()
        };
        opt_in(&kernels.forward, forward_shared_bytes(head_dim))?;
        opt_in(&kernels.grad_query, dq_shared_bytes(head_dim))?;
        opt_in(&kernels.grad_key_value, dkv_shared_bytes(head_dim))?;
        Some(kernels)
    };
    Some(FlashKernels {
        image: get("image_attention_fwd")?,
        gemv_q8: get("gemv_q8_mma")?,
        narrow: width(64)?,
        wide: width(128)?,
    })
}

/// Whether the fused path can serve this shape.
///
/// The kernels are built for heads 64 and 128 wide, and the BF16 operands
/// are only an honest substitute where the cuBLAS path was already rounding
/// its own operands to BF16 - that is, under mixed precision.
pub(crate) fn eligible(mixed_precision: bool, head_dim: usize) -> bool {
    mixed_precision && matches!(head_dim, 64 | 128) && !disabled()
}

/// Set by `RUSTING_BRAIN_NO_FLASH`. An escape hatch worth keeping: a fused
/// attention kernel is the kind of thing that is wrong on exactly one shape,
/// and a run that can fall back without a rebuild is a run that can be
/// bisected.
pub(crate) fn disabled() -> bool {
    static FROM_ENV: OnceLock<bool> = OnceLock::new();
    #[cfg(test)]
    if OFF_ON_THIS_THREAD.get() {
        return true;
    }
    *FROM_ENV.get_or_init(|| std::env::var_os("RUSTING_BRAIN_NO_FLASH").is_some())
}

#[cfg(test)]
thread_local! {
    /// The parity tests' switch, which runs the same model both ways. It is
    /// per thread because a process-wide one turned the fused path off under
    /// whichever other test was running alongside.
    pub(crate) static OFF_ON_THIS_THREAD: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
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
            crate::cuda_training::skip_without_cuda();
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
