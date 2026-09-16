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

// One block per (query tile, head, sequence). Queries live in registers for
// the whole block; keys and values stream through shared memory one tile at a
// time. `lse` is written in the same layout the three-kernel path used, so the
// backward pass rebuilds the probabilities from it unchanged.
extern "C" __global__ __launch_bounds__(128) void flash_attention_fwd(
    const float* __restrict__ qkv,
    float* __restrict__ out,
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
    const float* base = qkv + seq_base*(size_t)qkv_width + h*FA_D;
    #pragma unroll
    for (int kk=0; kk<4; ++kk) {
      int c = kk*16 + t*2;
      float a0=0.f,a1=0.f,a2=0.f,a3=0.f,a4=0.f,a5=0.f,a6=0.f,a7=0.f;
      if (row_a < seq_len) {
        const float* p = base + (size_t)row_a*qkv_width;
        a0=p[c]*scale; a1=p[c+1]*scale; a4=p[c+8]*scale; a5=p[c+9]*scale;
      }
      if (row_b < seq_len) {
        const float* p = base + (size_t)row_b*qkv_width;
        a2=p[c]*scale; a3=p[c+1]*scale; a6=p[c+8]*scale; a7=p[c+9]*scale;
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
    for (int i = threadIdx.x; i < 64*FA_D; i += 128) {
      int r = i >> 6, c = i & 63;
      int key = kt*64 + r;
      float kval = 0.f, vval = 0.f;
      if (key < seq_len) {
        const float* p = qkv + (seq_base + key)*(size_t)qkv_width;
        kval = p[key_base + kv + c];
        vval = p[value_base + kv + c];
      }
      ks[r*FA_ROW + c] = bf16_bits(kval);
      vt[c*FA_ROW + r] = bf16_bits(vval);
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
  float* obase = out + seq_base*(size_t)query_width + h*FA_D;
  #pragma unroll
  for (int n=0;n<8;++n) {
    int c = n*8 + t*2;
    if (row_a < seq_len) {
      float* p = obase + (size_t)row_a*query_width;
      p[c]=acc[n][0]*inv_a; p[c+1]=acc[n][1]*inv_a;
    }
    if (row_b < seq_len) {
      float* p = obase + (size_t)row_b*query_width;
      p[c]=acc[n][2]*inv_b; p[c+1]=acc[n][3]*inv_b;
    }
  }
  if (t == 0) {
    float* p = lse + (size_t)h*lse_head_stride + seq_base;
    if (row_a < seq_len) p[row_a]=m_a+__logf(l_a);
    if (row_b < seq_len) p[row_b]=m_b+__logf(l_b);
  }
}
"#;

/// The fused kernels, resolved once per device.
pub(crate) struct FlashKernels {
    pub(crate) forward: CudaFunction,
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

    Some(FlashKernels {
        forward: module
            .load_function("flash_attention_fwd")
            .map_err(cuda_err("fused attention kernel lookup"))
            .ok()?,
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

fn disabled() -> bool {
    static FROM_ENV: OnceLock<()> = OnceLock::new();
    FROM_ENV.get_or_init(|| {
        if std::env::var_os("RUSTING_BRAIN_NO_FLASH").is_some() {
            DISABLED.store(true, Ordering::Relaxed);
        }
    });
    DISABLED.load(Ordering::Relaxed)
}
