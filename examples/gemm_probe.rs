//! Which cuBLAS kernel the projection GEMMs land on, and what it costs.
//!
//! The fused-attention profile left three `cutlass_80_tensorop_s1688bf16gemm`
//! variants at 51% of GPU time. `s1688` is the m16n8k8 tensor-core shape and
//! `align4` says the operands are four-byte aligned, which is what FP32
//! operands with a `32F_FAST_16BF` compute type get. The language-model head,
//! whose operands are already BF16, lands on `s16816 ... align8` instead:
//! twice the k per instruction. This measures the difference on the shapes the
//! model actually runs.

use cudarc::cublas::{CudaBlas, result as cublas, sys as cublas_sys, sys::cublasOperation_t};
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use half::bf16;
use std::ffi::c_void;

/// The same round-to-nearest-even narrowing `Gpu::cast_to_bf16` runs.
const CAST: &str = r#"
extern "C" __global__ void to_bf16(unsigned short* dst, const float* src, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned u = __float_as_uint(src[i]);
    u += 0x7fff + ((u >> 16) & 1);
    dst[i] = (unsigned short)(u >> 16);
}
"#;

fn main() {
    let context = CudaContext::new(0).expect("no device");
    let stream = context.default_stream();
    let blas = CudaBlas::new(stream.clone()).unwrap();

    // out[rows, units] = x[rows, inner] . w[units, inner]^T, the shape every
    // projection in the block has, at batch 4 and sequence 1024.
    let rows = 4096;
    let shapes = [
        ("qkv projection", 1280usize, 768usize),
        ("attention output", 768, 768),
        ("feed forward in", 2816, 768),
        ("feed forward out", 768, 1408),
    ];

    let module = context.load_module(compile_ptx(CAST).unwrap()).unwrap();
    let cast = module.load_function("to_bf16").unwrap();

    println!(
        "{:<20} {:>12} {:>10} {:>10} {:>10} {:>8}",
        "shape", "units/inner", "fp32 in", "bf16 in", "+cast", "net"
    );
    for (name, units, inner) in shapes {
        let a = stream.alloc_zeros::<f32>(units * inner).unwrap();
        let b = stream.alloc_zeros::<f32>(rows * inner).unwrap();
        let mut c = stream.alloc_zeros::<f32>(rows * units).unwrap();
        let a16 = stream.alloc_zeros::<bf16>(units * inner).unwrap();
        let b16 = stream.alloc_zeros::<bf16>(rows * inner).unwrap();

        let flops = 2.0 * rows as f64 * units as f64 * inner as f64;
        let mut run = |wide: bool, iterations: usize| {
            let (ap, _ag) = a.device_ptr(&stream);
            let (bp, _bg) = b.device_ptr(&stream);
            let (a16p, _a16g) = a16.device_ptr(&stream);
            let (b16p, _b16g) = b16.device_ptr(&stream);
            let (cp, _cg) = c.device_ptr_mut(&stream);
            let (alpha, beta) = (1.0f32, 0.0f32);
            let (operand, compute) = if wide {
                (
                    cublas_sys::cudaDataType_t::CUDA_R_32F,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF,
                )
            } else {
                (
                    cublas_sys::cudaDataType_t::CUDA_R_16BF,
                    cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                )
            };
            let (left, right) = if wide { (ap, bp) } else { (a16p, b16p) };
            for _ in 0..iterations {
                unsafe {
                    cublas::gemm_ex(
                        *blas.handle(),
                        cublasOperation_t::CUBLAS_OP_T,
                        cublasOperation_t::CUBLAS_OP_N,
                        units as i32,
                        rows as i32,
                        inner as i32,
                        &alpha as *const f32 as *const c_void,
                        left as *const c_void,
                        operand,
                        inner as i32,
                        right as *const c_void,
                        operand,
                        inner as i32,
                        &beta as *const f32 as *const c_void,
                        cp as *mut c_void,
                        cublas_sys::cudaDataType_t::CUDA_R_32F,
                        units as i32,
                        compute,
                        cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                }
                .unwrap();
            }
            stream.synchronize().unwrap();
        };

        // What a `gemm_dispatch` that narrowed its own operands would pay:
        // both operands cast into scratch on every call.
        let cast_both = |iterations: usize| {
            for _ in 0..iterations {
                for (dst, src, n) in [(&a16, &a, units * inner), (&b16, &b, rows * inner)] {
                    let mut dst = dst.clone();
                    let config = LaunchConfig::for_num_elems(n as u32);
                    let n = n as i32;
                    unsafe {
                        stream
                            .launch_builder(&cast)
                            .arg(&mut dst)
                            .arg(src)
                            .arg(&n)
                            .launch(config)
                    }
                    .unwrap();
                }
            }
            stream.synchronize().unwrap();
        };

        let mut seconds = |wide: bool, iterations: usize| {
            run(wide, 5);
            let started = std::time::Instant::now();
            run(wide, iterations);
            started.elapsed().as_secs_f64() / iterations as f64
        };
        let wide = seconds(true, 50);
        let narrow = seconds(false, 50);
        cast_both(5);
        let started = std::time::Instant::now();
        cast_both(50);
        let cast_seconds = started.elapsed().as_secs_f64() / 50.0;
        println!(
            "{name:<20} {:>12} {:>7.2} TF {:>7.2} TF {:>7.2} TF {:>7.2}x",
            format!("{units}/{inner}"),
            flops / wide / 1e12,
            flops / narrow / 1e12,
            flops / (narrow + cast_seconds) / 1e12,
            wide / (narrow + cast_seconds)
        );
    }
}
