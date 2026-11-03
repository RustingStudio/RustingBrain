//! FP8 E4M3 GEMM on the tensor cores of sm_89 and later (RustingFrame X3a).
//!
//! `y = (x . W^T) + beta * y` for a weight in the X1 storage (E4M3 bytes, one
//! scale per row) and activations the `quantize_rows_e4m3` kernel quantizes to
//! E4M3 per row, through `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
//! (PTX ISA 8.4, sm_89+). It accumulates in FP32 and applies both scales in
//! the epilogue; the host math is `Quantized::matmul_fp8_activations`.
//!
//! UNVERIFIED: no sm_89 or sm_120 card has run this kernel. On this repo's
//! sm_86 card it is only compiled (to PTX and to sm_89 / sm_120 SASS, in the
//! tests below) and never loaded. It is off by default: only
//! `RUSTING_BRAIN_FP8_MMA` set, or a direct [`load`], turns it on, and every
//! other device keeps X1's widen-then-cuBLAS path.

use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::sync::Arc;

/// The first compute capability with FP8 `mma.sync`, as major * 10 + minor.
const FP8_MMA_ARCH: i32 = 89;

/// Block tile edge in rows of `x` and rows of `W`.
pub(crate) const TILE: usize = 64;

// A block of four warps owns a 64 x 64 tile of `y`, each warp 32 x 32: two
// m16 by four n8 mmas per step of 32 along k. A step stages 64 x 32 bytes of
// each operand in shared memory, rows 48 bytes apart so the fragment loads
// (row g, bytes 4t..4t+3) of a warp hit 32 distinct banks. Out-of-range rows
// and k read as zeros, so any shape works; whole 16-byte loads need `inner`
// to be a multiple of 16, bytes one by one otherwise.
//
// Fragments, from the PTX ISA's m16n8k32 8-bit layout (g = lane / 4,
// t = lane % 4): A regs 0..3 are row g / g+8 / g / g+8 at bytes 4t / 4t /
// 16+4t / 16+4t; B regs 0..1 are column g at k 4t / 16+4t, which is four
// consecutive bytes of row g of `W`; C regs are row g / g / g+8 / g+8 at
// columns 2t / 2t+1 / 2t / 2t+1.
pub(crate) const KERNELS: &str = r#"
#define FP8_LD 48
__device__ __forceinline__ void fp8_stage(unsigned char*s,const unsigned char*g,
    int count,int inner,int first,int k0){
  int r=threadIdx.x>>1,c=(threadIdx.x&1)*16,row=first+r,k=k0+c;
  uint4 v=make_uint4(0u,0u,0u,0u);
  if(row<count){
    const unsigned char*p=g+(size_t)row*inner+k;
    if((inner&15)==0&&k+16<=inner)v=*(const uint4*)p;
    else{
      unsigned w[4]={0u,0u,0u,0u};
      for(int i=0;i<16&&k+i<inner;i++)w[i>>2]|=(unsigned)p[i]<<(8*(i&3));
      v=make_uint4(w[0],w[1],w[2],w[3]);
    }
  }
  *(uint4*)(s+r*FP8_LD+c)=v;
}
extern "C" __global__ __launch_bounds__(128) void gemm_fp8_mma(
    const unsigned char* __restrict__ xq,const float* __restrict__ xs,
    const unsigned char* __restrict__ w,const float* __restrict__ ws,
    float* __restrict__ y,int rows,int units,int inner,float beta)
{
  __shared__ __align__(16) unsigned char sa[64*FP8_LD],sb[64*FP8_LD];
  int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  int m0=blockIdx.y*64,n0=blockIdx.x*64,wm=(warp>>1)*32,wn=(warp&1)*32;
  float d[2][4][4];
  #pragma unroll
  for(int i=0;i<2;i++)
    #pragma unroll
    for(int j=0;j<4;j++)
      #pragma unroll
      for(int e=0;e<4;e++)d[i][j][e]=0.f;
  for(int k0=0;k0<inner;k0+=32){
    fp8_stage(sa,xq,rows,inner,m0,k0);
    fp8_stage(sb,w,units,inner,n0,k0);
    __syncthreads();
    unsigned a[2][4],b[4][2];
    #pragma unroll
    for(int i=0;i<2;i++){
      const unsigned char*p=sa+(wm+i*16+g)*FP8_LD+t*4;
      a[i][0]=*(const unsigned*)p;
      a[i][1]=*(const unsigned*)(p+8*FP8_LD);
      a[i][2]=*(const unsigned*)(p+16);
      a[i][3]=*(const unsigned*)(p+8*FP8_LD+16);
    }
    #pragma unroll
    for(int j=0;j<4;j++){
      const unsigned char*p=sb+(wn+j*8+g)*FP8_LD+t*4;
      b[j][0]=*(const unsigned*)p;
      b[j][1]=*(const unsigned*)(p+16);
    }
    #pragma unroll
    for(int i=0;i<2;i++)
      #pragma unroll
      for(int j=0;j<4;j++)
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          :"+f"(d[i][j][0]),"+f"(d[i][j][1]),"+f"(d[i][j][2]),"+f"(d[i][j][3])
          :"r"(a[i][0]),"r"(a[i][1]),"r"(a[i][2]),"r"(a[i][3]),"r"(b[j][0]),"r"(b[j][1]));
    __syncthreads();
  }
  #pragma unroll
  for(int i=0;i<2;i++)
    #pragma unroll
    for(int j=0;j<4;j++)
      #pragma unroll
      for(int e=0;e<4;e++){
        int row=m0+wm+i*16+g+(e>=2?8:0),col=n0+wn+j*8+2*t+(e&1);
        if(row>=rows||col>=units)continue;
        size_t o=(size_t)row*units+col;
        float v=d[i][j][e]*xs[row]*ws[col];
        y[o]=beta==0.f?v:v+beta*y[o];
      }
}
"#;

/// Whether `RUSTING_BRAIN_FP8_MMA` asks for the FP8 GEMM on the devices that
/// have it. Off by default.
pub(crate) fn enabled() -> bool {
    std::env::var_os("RUSTING_BRAIN_FP8_MMA").is_some()
}

/// The GEMM for `context`, or `None` below compute capability 8.9 (nothing is
/// compiled or loaded there) or when compiling or loading fails.
///
/// UNVERIFIED on sm_89 and sm_120: no such card has run it.
pub(crate) fn load(context: &Arc<CudaContext>) -> Option<CudaFunction> {
    let (major, minor) = context.compute_capability().ok()?;
    if major * 10 + minor < FP8_MMA_ARCH {
        return None;
    }
    // The process target when it has the instruction, else compute_89, whose
    // PTX the driver JIT-compiles forward.
    let arch = match crate::cuda_flash::nvrtc_arch() {
        Some((number, name)) if number >= FP8_MMA_ARCH => name,
        _ => "compute_89",
    };
    let options = CompileOptions {
        arch: Some(arch),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNELS, options).ok()?;
    context
        .load_module(ptx)
        .ok()?
        .load_function("gemm_fp8_mma")
        .ok()
}

#[cfg(test)]
mod tests {
    use super::KERNELS;
    use cudarc::nvrtc::result as nvrtc;
    use std::ffi::CString;

    /// Compiles [`KERNELS`] with `--gpu-architecture=<arch>`; for an `sm_`
    /// target NVRTC runs ptxas and the result is a CUBIN. Returns the output's
    /// size and the log (ptxas's register report), or the log as the error.
    fn nvrtc_compile(arch: &str) -> Result<(usize, String), String> {
        let src = CString::new(KERNELS).unwrap();
        let program = nvrtc::create_program(&src, None).map_err(|e| e.to_string())?;
        let options = [
            format!("--gpu-architecture={arch}"),
            "--ptxas-options=-v".to_string(),
        ];
        // SAFETY: `program` is live until the destroy below, and the getters
        // write the sizes they report into buffers that long.
        unsafe {
            let compiled = nvrtc::compile_program(program, &options);
            let log = nvrtc::get_program_log(program)
                .map(|log| {
                    let bytes: Vec<u8> = log.iter().map(|&c| c as u8).collect();
                    String::from_utf8_lossy(&bytes)
                        .trim_end_matches('\0')
                        .to_string()
                })
                .unwrap_or_default();
            let result = match compiled {
                Err(error) => Err(format!("{error}: {log}")),
                Ok(()) if arch.starts_with("sm_") => {
                    use cudarc::nvrtc::sys::{nvrtcGetCUBINSize, nvrtcResult};
                    let mut size = 0usize;
                    match nvrtcGetCUBINSize(program, &mut size) {
                        nvrtcResult::NVRTC_SUCCESS => Ok((size, log)),
                        other => Err(format!("{other:?}: {log}")),
                    }
                }
                Ok(()) => nvrtc::get_ptx(program)
                    .map(|ptx| (ptx.len(), log))
                    .map_err(|e| e.to_string()),
            };
            let _ = nvrtc::destroy_program(program);
            result
        }
    }

    /// NVRTC sees the PTX only at runtime and the 3060 never loads it, so a
    /// typo or an operand the instruction rejects would surface first on an
    /// sm_89 card. This builds it for compute_89 and compute_120, and through
    /// ptxas to sm_89 and sm_120 SASS, on any machine with NVRTC; and checks
    /// that sm_86 rejects the instruction, which is what the runtime gate in
    /// [`super::load`] is for. Compiling is all it proves: running it on
    /// sm_89 / sm_120 is UNVERIFIED.
    #[test]
    fn the_fp8_gemm_compiles_for_sm_89_and_sm_120_but_not_sm_86() {
        if nvrtc::create_program(c"", None).is_err() {
            eprintln!("skipped: NVRTC not available");
            return;
        }
        for arch in ["compute_89", "compute_120", "sm_89", "sm_120"] {
            match nvrtc_compile(arch) {
                Ok((size, log)) => {
                    eprintln!("{arch}: {size} bytes\n{log}");
                    assert!(size > 0, "{arch}: empty output");
                }
                Err(log) => panic!("{arch} failed:\n{log}"),
            }
        }
        let rejected = nvrtc_compile("sm_86");
        eprintln!("sm_86: {:?}", rejected.as_ref().map(|(size, _)| size));
        assert!(rejected.is_err(), "sm_86 accepted FP8 mma");
    }

    /// Below 8.9 [`super::load`] compiles and loads nothing.
    #[test]
    fn the_fp8_gemm_is_not_loaded_below_sm_89_or_skips_without_device() {
        let Ok(context) = cudarc::driver::CudaContext::new(0) else {
            crate::cuda_training::skip_without_cuda();
            return;
        };
        let (major, minor) = context.compute_capability().unwrap();
        let loaded = super::load(&context);
        eprintln!(
            "device 0 is {major}.{minor}, FP8 GEMM loaded: {}",
            loaded.is_some()
        );
        if major * 10 + minor < super::FP8_MMA_ARCH {
            assert!(loaded.is_none());
        } else {
            // UNVERIFIED: no sm_89+ card has run this.
            assert!(
                loaded.is_some(),
                "an sm_89+ device failed to load the FP8 GEMM"
            );
        }
    }
}
