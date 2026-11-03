//! W8A8: int8 weights times int8 activations on the int8 tensor cores
//! (RustingFrame P5), for the device MMDiT's block linears.
//!
//! `y = x . W^T + beta * y` for a Q8 weight (int8, one scale per row or per
//! `group` columns) and BF16 rows `x`, which `quantize_rows_i8` first takes to
//! int8 with one absmax scale per token per 64 columns. The GEMM runs
//! `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` (sm_80+) with an exact
//! int32 sum per 64-wide slab of k, then adds `sum * activation scale *
//! weight scale` into an FP32 accumulator. Output is FP32, as the cuBLAS path
//! writes it. Host math: the reference in the tests below.
//!
//! One scale a whole token was the first cut, and the MLP down projections
//! lost the most to it (their SiLU-gated inputs carry a few huge channels):
//! klein's final latent went 3.3e-2 to 1.05e-1 against diffusers, 4.2e-2
//! with those layers left on cuBLAS. A scale per 64 columns keeps an outlier
//! from flattening the rest of its row.
//!
//! The default for FLUX.2 on compute capability 8.0+ (owner, 2026-11-03);
//! `RB_FLUX2_W8A8=0`, read once, turns it off, and then nothing here is
//! compiled or loaded and every linear keeps the dequantize-then-cuBLAS path.
//! It changes the numbers (activations lose to int8): klein's parity test
//! holds this path to its own gates, the off path to the strict ones.

use crate::cuda_training::cuda_err;
use crate::gpu_model::Gpu;
use crate::network::NetworkError;
use cudarc::driver::{CudaFunction, CudaSlice, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

// The GEMM: a block of eight warps owns a 128 x 128 tile of `y`, each warp
// 64 x 32 (four m16 by four n8 mmas per step of 32 along k). A 64-wide slab
// of k of both operands goes into shared memory through `cp.async` in a
// three-stage ring, 64-byte rows whose four 16-byte chunks are XOR-swizzled
// by `(row / 2) % 4`, so every eight-row `ldmatrix` phase hits 32 distinct
// banks. Out-of-range rows and k are zero-filled, so any `rows` and `units`
// work; `inner` must be a multiple of 16 (whole 16-byte chunks).
//
// Fragments (PTX ISA, m16n8k32 8-bit, g = lane / 4, t = lane % 4): A regs
// 0..3 are row g / g+8 / g / g+8 at bytes 4t / 4t / 16+4t / 16+4t, which is
// `ldmatrix.x4` over (rows 0-7, chunk 0), (rows 8-15, chunk 0), (rows 0-7,
// chunk 1), (rows 8-15, chunk 1); B regs 0..1 are W row g at bytes 4t /
// 16+4t, so one `ldmatrix.x4` over an n8 tile's four chunks gives the B
// fragments of both k32 steps. C regs are row g / g / g+8 / g+8 at columns
// 2t / 2t+1 / 2t / 2t+1.
//
// Each 64-wide slab's int32 sum is exact and goes into the FP32 accumulator
// times its activation scale (one a row a slab) and its group's weight scale
// (`kpg` slabs a group, `groups` scales a weight row; one scale a row is
// `groups = 1`, `kpg = all slabs`). The sum
// starts at the bits of 1.5 * 2^23, and |sum| <= 64 * 127 * 127 < 2^22, so
// the int32 result read as a float is exactly 1.5 * 2^23 + sum: one FADD
// instead of an I2F, which runs at a quarter of the rate and cost as much
// time as the mmas.
//
// Blocks walk the tile grid eight row tiles at a time, so the blocks in
// flight share a few A and B tiles in L2 rather than all reading new ones.
pub(crate) const KERNELS: &str = r#"
#define BM 128
#define BN 128
#define BK 64
#define STAGES 3
#define TILE_BYTES (BM*BK)
#define MAGIC 0x4B400000
__device__ __forceinline__ unsigned smem(const void*p){return (unsigned)__cvta_generic_to_shared(p);}
__device__ __forceinline__ int swz(int r,int c){return r*BK+((c^((r>>1)&3))<<4);}
__device__ __forceinline__ void stage(unsigned char*s,const signed char*g,int count,int inner,int first,int k0){
  #pragma unroll
  for(int j=0;j<2;j++){
    int i=threadIdx.x+j*256,r=i>>2,c=i&3,row=first+r,k=k0+c*16;
    bool ok=row<count&&k<inner;
    const signed char*src=ok?g+(size_t)row*inner+k:g;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
      ::"r"(smem(s+swz(r,c))),"l"(src),"r"(ok?16:0));
  }
}
extern "C" __global__ __launch_bounds__(256,1) void gemm_w8a8(
    const signed char* __restrict__ xq,const float* __restrict__ xs,
    const signed char* __restrict__ w,const float* __restrict__ ws,
    float* __restrict__ y,int rows,int units,int inner,int groups,int kpg,float beta)
{
  __shared__ __align__(128) unsigned char sa[STAGES][TILE_BYTES],sb[STAGES][TILE_BYTES];
  int warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,t=lane&3;
  // Eight row tiles at a time, column tiles inside them.
  int tm=(rows+BM-1)/BM,tn=(units+BN-1)/BN,id=blockIdx.x,band=id/(8*tn),first=band*8;
  int height=min(8,tm-first),in_band=id-band*8*tn;
  int m0=(first+in_band%height)*BM,n0=(in_band/height)*BN;
  int wm=(warp&1)*64,wn=(warp>>1)*32;
  int acc[4][4][4];float f[4][4][4];
  #pragma unroll
  for(int i=0;i<4;i++)
    #pragma unroll
    for(int j=0;j<4;j++)
      #pragma unroll
      for(int e=0;e<4;e++){acc[i][j][e]=MAGIC;f[i][j][e]=0.f;}
  int ktiles=(inner+BK-1)/BK;
  #pragma unroll
  for(int s=0;s<STAGES-1;s++){
    if(s<ktiles){stage(sa[s],xq,rows,inner,m0,s*BK);stage(sb[s],w,units,inner,n0,s*BK);}
    asm volatile("cp.async.commit_group;\n"::);
  }
  // This thread's eight weight rows (output columns) and the offsets of its
  // eight activation rows' scales, clamped in range.
  int col[4][2],xrow[4][2];
  #pragma unroll
  for(int j=0;j<4;j++)
    #pragma unroll
    for(int h=0;h<2;h++){col[j][h]=min(n0+wn+j*8+2*t+h,units-1);xrow[j][h]=min(m0+wm+j*16+g+h*8,rows-1)*ktiles;}
  for(int kt=0;kt<ktiles;kt++){
    asm volatile("cp.async.wait_group %0;\n"::"n"(STAGES-2));
    __syncthreads();
    int nk=kt+STAGES-1;
    if(nk<ktiles){stage(sa[nk%STAGES],xq,rows,inner,m0,nk*BK);stage(sb[nk%STAGES],w,units,inner,n0,nk*BK);}
    asm volatile("cp.async.commit_group;\n"::);
    const unsigned char*A=sa[kt%STAGES],*B=sb[kt%STAGES];
    unsigned b[4][4];
    #pragma unroll
    for(int j=0;j<4;j++){
      int r=wn+j*8+(lane&7);
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        :"=r"(b[j][0]),"=r"(b[j][1]),"=r"(b[j][2]),"=r"(b[j][3]):"r"(smem(B+swz(r,lane>>3))));
    }
    // Every A fragment of the slab first, so no mma waits on its ldmatrix.
    unsigned a[2][4][4];
    #pragma unroll
    for(int s=0;s<2;s++)
      #pragma unroll
      for(int i=0;i<4;i++){
        int r=wm+i*16+(lane&15);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
          :"=r"(a[s][i][0]),"=r"(a[s][i][1]),"=r"(a[s][i][2]),"=r"(a[s][i][3]):"r"(smem(A+swz(r,2*s+(lane>>4)))));
      }
    #pragma unroll
    for(int s=0;s<2;s++)
      #pragma unroll
      for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<4;j++)
          asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            :"+r"(acc[i][j][0]),"+r"(acc[i][j][1]),"+r"(acc[i][j][2]),"+r"(acc[i][j][3])
            :"r"(a[s][i][0]),"r"(a[s][i][1]),"r"(a[s][i][2]),"r"(a[s][i][3]),"r"(b[j][2*s]),"r"(b[j][2*s+1]));
    {
      int group=kt/kpg;float sc[4][2],sx[4][2];
      #pragma unroll
      for(int j=0;j<4;j++)
        #pragma unroll
        for(int h=0;h<2;h++){sc[j][h]=__ldg(ws+(size_t)col[j][h]*groups+group);sx[j][h]=__ldg(xs+xrow[j][h]+kt);}
      #pragma unroll
      for(int i=0;i<4;i++)
        #pragma unroll
        for(int j=0;j<4;j++)
          #pragma unroll
          for(int e=0;e<4;e++){f[i][j][e]=fmaf(__int_as_float(acc[i][j][e])-12582912.f,sx[i][e>>1]*sc[j][e&1],f[i][j][e]);acc[i][j][e]=MAGIC;}
    }
  }
  asm volatile("cp.async.wait_group 0;\n"::);
  bool pair=(units&1)==0;
  #pragma unroll
  for(int i=0;i<4;i++)
    #pragma unroll
    for(int h=0;h<2;h++){
      int row=m0+wm+i*16+g+h*8;
      if(row>=rows)continue;
      #pragma unroll
      for(int j=0;j<4;j++){
        int c=n0+wn+j*8+2*t;
        if(c>=units)continue;
        float*o=y+(size_t)row*units+c;
        float v0=f[i][j][2*h],v1=f[i][j][2*h+1];
        if(pair){
          float2 v=make_float2(v0,v1);
          if(beta!=0.f){float2 p=*(float2*)o;v.x=fmaf(beta,p.x,v0);v.y=fmaf(beta,p.y,v1);}
          *(float2*)o=v;
        }else{
          o[0]=beta==0.f?v0:fmaf(beta,o[0],v0);
          if(c+1<units)o[1]=beta==0.f?v1:fmaf(beta,o[1],v1);
        }
      }
    }
}
// Each 64-column group of each BF16 row of `x` to int8 with one absmax
// scale, `scales[row * slabs + group]`: `scale = max|v| / 127`, `q = rint(v
// / scale)` as `v * (1 / scale)`, the weight quantizer's rule. One warp a
// group, two columns a lane; `cols` is even.
extern "C" __global__ __launch_bounds__(256) void quantize_rows_i8(
    const unsigned short* __restrict__ x,signed char* __restrict__ q,float* __restrict__ scales,int rows,int cols)
{
  int slabs=(cols+63)/64,id=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;
  if(id>=rows*slabs)return;
  int r=id/slabs,k=(id-r*slabs)*64+2*lane;
  bool ok=k<cols;
  unsigned v=ok?*(const unsigned*)(x+(size_t)r*cols+k):0u;
  float a=__uint_as_float(v<<16),b=__uint_as_float(v&0xffff0000u),m=fmaxf(fabsf(a),fabsf(b));
  #pragma unroll
  for(int o=16;o>0;o>>=1)m=fmaxf(m,__shfl_xor_sync(0xffffffffu,m,o));
  float scale=m/127.f,inv=scale>0.f?1.f/scale:0.f;
  if(ok){
    int ca=(int)fminf(fmaxf(rintf(a*inv),-127.f),127.f),cb=(int)fminf(fmaxf(rintf(b*inv),-127.f),127.f);
    *(unsigned short*)(q+(size_t)r*cols+k)=(unsigned short)((ca&0xff)|((cb&0xff)<<8));
  }
  if(lane==0)scales[id]=scale;
}
"#;

/// Row tiles of the GEMM.
const BM: usize = 128;
/// Below this many rows the 128-row tiles are mostly empty, and the linear
/// keeps its old path (the modulation GEMV has one row).
const MIN_ROWS: usize = 64;

/// Whether the W8A8 path is wanted: yes unless `RB_FLUX2_W8A8=0`. Read once.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("RB_FLUX2_W8A8").as_deref() != Ok("0"))
}

/// Set once [`W8a8::load`] has loaded the kernels on some device, so a test
/// can tell which path actually ran.
static LOADED: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
pub(crate) fn loaded() -> bool {
    LOADED.load(Ordering::Relaxed)
}

/// The two kernels, loaded on one device.
pub(crate) struct W8a8 {
    gemm: CudaFunction,
    quantize: CudaFunction,
}

impl W8a8 {
    /// Compiles and loads the kernels for `gpu`'s device, or `None` below
    /// compute capability 8.0 (no int8 `mma.sync` m16n8k32 or `cp.async`).
    pub(crate) fn load(gpu: &Gpu<'_>) -> Result<Option<Self>, NetworkError> {
        let context = gpu.context.stream.context();
        match context.compute_capability() {
            Ok((major, _)) if major >= 8 => {}
            _ => return Ok(None),
        }
        let arch = match crate::cuda_flash::nvrtc_arch() {
            Some((number, name)) if number >= 80 => name,
            _ => "compute_80",
        };
        let options = CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNELS, options).map_err(|error| {
            NetworkError::Cuda(format!("W8A8 kernel compilation failed: {error}"))
        })?;
        let module = context
            .load_module(ptx)
            .map_err(cuda_err("W8A8 module loading"))?;
        let get = |name| {
            module
                .load_function(name)
                .map_err(cuda_err("W8A8 kernel lookup"))
        };
        let kernels = Self {
            gemm: get("gemm_w8a8")?,
            quantize: get("quantize_rows_i8")?,
        };
        LOADED.store(true, Ordering::Relaxed);
        Ok(Some(kernels))
    }

    /// `out = x . W^T + beta * out` for `rows` BF16 rows `x` and the int8
    /// weight `bytes` / `scales` (`group` columns a scale, 0 for one a row),
    /// `units x inner`. `false`, with nothing launched, when the shape is not
    /// one this path takes; the caller then keeps its own.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project(
        &self,
        gpu: &Gpu<'_>,
        (bytes, scales, group): (&CudaView<'_, u8>, &CudaView<'_, f32>, usize),
        units: usize,
        inner: usize,
        x: &CudaView<'_, u8>,
        out: &mut CudaViewMut<'_, f32>,
        rows: usize,
        beta: f32,
    ) -> Result<bool, NetworkError> {
        if rows < MIN_ROWS || !inner.is_multiple_of(16) || !group.is_multiple_of(64) {
            return Ok(false);
        }
        let (xq, xs) = self.quantized_rows(gpu, x, rows, inner)?;
        let slabs = inner.div_ceil(64);
        let (groups, kpg) = match group {
            0 => (1, slabs),
            group => (inner.div_ceil(group), group / 64),
        };
        let tiles = rows.div_ceil(BM) * units.div_ceil(BM);
        unsafe {
            gpu.context
                .stream
                .launch_builder(&self.gemm)
                .arg(&xq)
                .arg(&xs)
                .arg(bytes)
                .arg(scales)
                .arg(out)
                .arg(&(rows as i32))
                .arg(&(units as i32))
                .arg(&(inner as i32))
                .arg(&(groups as i32))
                .arg(&(kpg as i32))
                .arg(&beta)
                .launch(LaunchConfig {
                    grid_dim: (tiles as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(cuda_err("W8A8 GEMM kernel"))?;
        }
        Ok(true)
    }

    /// `rows x cols` BF16 rows as int8 and one scale a row a 64 columns.
    fn quantized_rows(
        &self,
        gpu: &Gpu<'_>,
        x: &CudaView<'_, u8>,
        rows: usize,
        cols: usize,
    ) -> Result<(CudaSlice<u8>, CudaSlice<f32>), NetworkError> {
        let stream = &gpu.context.stream;
        let mut q = unsafe { stream.alloc::<u8>(rows * cols) }
            .map_err(cuda_err("W8A8 activation allocation"))?;
        let groups = rows * cols.div_ceil(64);
        let mut scales = gpu.uninit(groups)?;
        unsafe {
            stream
                .launch_builder(&self.quantize)
                .arg(x)
                .arg(&mut q)
                .arg(&mut scales)
                .arg(&(rows as i32))
                .arg(&(cols as i32))
                .launch(LaunchConfig {
                    grid_dim: (groups.div_ceil(8) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
                .map_err(cuda_err("W8A8 activation quantize kernel"))?;
        }
        Ok((q, scales))
    }
}

#[cfg(test)]
mod tests {
    use super::W8a8;
    use crate::gpu_model::Gpu;
    use crate::gpu_transformer::GpuContext;

    /// The host twin of `quantize_rows_i8`.
    fn quantize_rows(x: &[f32], cols: usize) -> (Vec<i8>, Vec<f32>) {
        let mut q = Vec::with_capacity(x.len());
        let mut scales = Vec::new();
        for row in x.chunks(cols).flat_map(|row| row.chunks(64)) {
            let m = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            let scale = m / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            q.extend(
                row.iter()
                    .map(|v| (v * inv).round_ties_even().clamp(-127.0, 127.0) as i8),
            );
            scales.push(scale);
        }
        (q, scales)
    }

    /// The host twin of `gemm_w8a8`: an exact int32 sum per 64-wide slab,
    /// `fma` into FP32 times its activation and weight scales in k order.
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        xq: &[i8],
        xs: &[f32],
        w: &[i8],
        ws: &[f32],
        y: &mut [f32],
        (rows, units, inner): (usize, usize, usize),
        group: usize,
        beta: f32,
    ) {
        let span = if group == 0 { inner } else { group };
        let (groups, slabs) = (inner.div_ceil(span), inner.div_ceil(64));
        for m in 0..rows {
            for n in 0..units {
                let mut f = 0f32;
                for slab in (0..inner).step_by(64) {
                    let sum: i32 = (slab..(slab + 64).min(inner))
                        .map(|k| i32::from(xq[m * inner + k]) * i32::from(w[n * inner + k]))
                        .sum();
                    let scale = xs[m * slabs + slab / 64] * ws[n * groups + slab / span];
                    f = (sum as f32).mul_add(scale, f);
                }
                let o = &mut y[m * units + n];
                *o = if beta == 0.0 { f } else { beta.mul_add(*o, f) };
            }
        }
    }

    /// Odd shapes (row and column tails, a k tail short of a slab, a group
    /// spanning two slabs, one scale a row, `beta` 0 and 1) against the host
    /// twin. The quantizer must match bit for bit; the GEMM's int32 sums are
    /// exact, so only an FP32 rounding may differ.
    #[test]
    fn the_w8a8_gemm_matches_the_host_or_skips_without_device() {
        let Ok(context) = GpuContext::new(0) else {
            crate::cuda_training::skip_without_cuda();
            return;
        };
        let gpu = Gpu { context: &context };
        let Some(kernels) = W8a8::load(&gpu).unwrap() else {
            eprintln!("skipped: device below compute capability 8.0");
            return;
        };
        let stream = &context.stream;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut random = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        };
        for (rows, units, inner, group, beta) in [
            (77, 201, 336, 64, 0.0f32),
            (130, 130, 208, 0, 1.0),
            (200, 257, 512, 128, 0.0),
            (64, 96, 3072, 64, 1.0),
        ] {
            // Activations with a few large outliers, as a DiT's have.
            let x: Vec<f32> = (0..rows * inner)
                .map(|i| random() * if i % 97 == 0 { 20.0 } else { 1.0 })
                .collect();
            let x = gpu
                .narrowed(&gpu.upload(&x).unwrap().slice(..), rows * inner, true)
                .unwrap();
            let x_host = gpu.download_act(&x).unwrap();
            let w: Vec<f32> = (0..units * inner).map(|_| random() * 0.05).collect();
            let w = gpu.upload(&w).unwrap();
            let weight = match group {
                0 => gpu.quantized(&w.slice(..), units, inner),
                group => gpu.quantized_grouped(&w.slice(..), units, inner, group),
            }
            .unwrap();
            let (bytes, scales, stored_group) = weight.int8().unwrap();
            assert_eq!(stored_group, group);
            let w_host: Vec<i8> = stream
                .clone_dtoh(&bytes.slice(..units * inner))
                .unwrap()
                .into_iter()
                .map(|b| b as i8)
                .collect();
            let ws_host = stream.clone_dtoh(scales).unwrap();

            let (xq, xs) = kernels.quantized_rows(&gpu, &x.all(), rows, inner).unwrap();
            let (xq_host, xs_host) = quantize_rows(&x_host, inner);
            let xq_device: Vec<i8> = stream
                .clone_dtoh(&xq)
                .unwrap()
                .into_iter()
                .map(|b| b as i8)
                .collect();
            assert_eq!(xq_device, xq_host, "quantized rows differ");
            assert_eq!(
                stream.clone_dtoh(&xs).unwrap(),
                xs_host,
                "activation scales differ"
            );

            let start: Vec<f32> = (0..rows * units).map(|_| random()).collect();
            let mut y = gpu.upload(&start).unwrap();
            let parts = (&bytes.slice(..), &scales.slice(..), stored_group);
            let ran = kernels
                .project(
                    &gpu,
                    parts,
                    units,
                    inner,
                    &x.all(),
                    &mut y.slice_mut(..),
                    rows,
                    beta,
                )
                .unwrap();
            assert!(ran);
            let device = gpu.download(&y).unwrap();
            let mut host = start;
            let shape = (rows, units, inner);
            gemm(
                &xq_host, &xs_host, &w_host, &ws_host, &mut host, shape, group, beta,
            );
            let (mut worst, mut exact) = (0f32, 0usize);
            for (d, h) in device.iter().zip(&host) {
                worst = worst.max((d - h).abs() / h.abs().max(1e-3));
                exact += usize::from(d.to_bits() == h.to_bits());
            }
            eprintln!(
                "{rows}x{units}x{inner} group {group} beta {beta}: worst relative {worst:.2e}, \
                 {exact}/{} bit-exact",
                host.len()
            );
            assert!(
                worst < 1e-5,
                "{rows}x{units}x{inner}: worst relative {worst}"
            );
        }
    }

    /// Speed against the path it replaces (Q8 dequantize then cuBLAS BF16)
    /// and plain cuBLAS BF16, on klein's block linears at 1024² (4608 rows).
    /// `cargo test --release --features cuda --lib -- --ignored --nocapture
    /// time_the_w8a8_gemm`, under the GPU lock.
    #[test]
    #[ignore]
    fn time_the_w8a8_gemm() {
        let context = GpuContext::new(0).expect("a CUDA device");
        let gpu = Gpu { context: &context };
        let kernels = W8a8::load(&gpu).unwrap().expect("compute capability 8.0+");
        let rows = 4608;
        for (name, units, inner) in [
            ("qkv", 9216, 3072),
            ("mlp_in", 18432, 3072),
            ("mlp_out", 3072, 9216),
            ("fused_in", 27648, 3072),
        ] {
            let x = gpu.upload(&vec![0.5f32; rows * inner]).unwrap();
            let x = gpu.narrowed(&x.slice(..), rows * inner, true).unwrap();
            let w: Vec<f32> = (0..units * inner)
                .map(|i| ((i % 251) as f32 - 125.0) * 1e-3)
                .collect();
            let w = gpu.upload(&w).unwrap();
            let q8 = gpu
                .quantized_grouped(&w.slice(..), units, inner, 64)
                .unwrap();
            let bf16 = gpu.narrowed(&w.slice(..), units * inner, true).unwrap();
            drop(w);
            let mut y = gpu.uninit(rows * units).unwrap();
            let (bytes, scales, group) = q8.int8().unwrap();
            let time = |run: &mut dyn FnMut()| {
                (0..5).for_each(|_| run());
                context.stream.synchronize().unwrap();
                let started = std::time::Instant::now();
                for _ in 0..10 {
                    run();
                }
                context.stream.synchronize().unwrap();
                started.elapsed().as_secs_f64() / 10.0
            };
            let w8a8 = time(&mut || {
                let parts = (&bytes.slice(..), &scales.slice(..), group);
                let out = &mut y.slice_mut(..);
                assert!(
                    kernels
                        .project(&gpu, parts, units, inner, &x.all(), out, rows, 0.0)
                        .unwrap()
                );
            });
            let dequantized = time(&mut || {
                let out = &mut y.slice_mut(..);
                gpu.project(&q8, units, inner, &x.all(), true, out, rows, 0.0)
                    .unwrap();
            });
            let plain = time(&mut || {
                let out = &mut y.slice_mut(..);
                gpu.project(&bf16, units, inner, &x.all(), true, out, rows, 0.0)
                    .unwrap();
            });
            let tops = |seconds: f64| 2.0 * (rows * units * inner) as f64 / seconds / 1e12;
            eprintln!(
                "{name} {rows}x{units}x{inner}: W8A8 {:.2} ms ({:.1} TOPS), Q8 dequantize + cuBLAS \
                 {:.2} ms ({:.1}), cuBLAS BF16 {:.2} ms ({:.1})",
                w8a8 * 1e3,
                tops(w8a8),
                dequantized * 1e3,
                tops(dequantized),
                plain * 1e3,
                tops(plain)
            );
        }
    }
}
