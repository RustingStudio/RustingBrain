//! What each fused attention kernel costs on the shape the benchmark runs.
//!
//! The three kernels are 20% of a training step and the profile only gives
//! their total. This runs them on their own, interleaved and best of five, so
//! a change to one of them can be measured in a second rather than in a full
//! benchmark run whose noise is about as large as the effect.

use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

/// The kernel source out of `src/cuda_flash.rs`, which is a `const` string
/// this crate keeps private.
fn kernels() -> String {
    let source = include_str!("../src/cuda_flash.rs");
    let body = source.split("r#\"").nth(1).expect("the kernel string");
    body.rsplit_once("\"#").expect("the kernel string ends").0.to_string()
}

fn main() {
    let context = CudaContext::new(0).expect("no device");
    let stream = context.default_stream();
    let module = context
        .load_module(
            compile_ptx_with_opts(
                kernels(),
                CompileOptions {
                    arch: Some("compute_80"),
                    ..Default::default()
                },
            )
            .expect("the kernels compile"),
        )
        .unwrap();

    // The benchmark model's attention shape: batch 4, sequence 1024, 12 query
    // heads over 4 key/value heads, head dimension 64.
    let (sequences, seq_len, heads, kv_heads, head_dim) = (4usize, 1024, 12, 4, 64);
    let rows = sequences * seq_len;
    let query_width = heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let qkv_width = query_width + 2 * kv_width;
    let group = heads / kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let tiles = seq_len.div_ceil(64) as u32;

    // BF16, which is the only precision the fused path runs in.
    let qkv = stream.alloc_zeros::<u16>(rows * qkv_width).unwrap();
    let grad_qkv = stream.alloc_zeros::<u16>(rows * qkv_width).unwrap();
    let out = stream.alloc_zeros::<u16>(rows * query_width).unwrap();
    let grad_out = stream.alloc_zeros::<f32>(rows * query_width).unwrap();
    let lse = stream.alloc_zeros::<f32>(heads * rows).unwrap();
    let delta = stream.alloc_zeros::<f32>(heads * rows).unwrap();

    let shared = |tiles: u32| 2 * tiles * (64 * 72 * 2);
    let mut best = [f64::MAX; 3];
    let names = ["forward", "grad query", "grad key/value"];
    for round in 0..5 {
        for (index, name) in names.iter().enumerate() {
            let (function, grid_y, shared_mem_bytes) = match index {
                0 => ("flash_attention_fwd", heads, shared(1)),
                1 => ("flash_attention_dq", heads, shared(1) + 64 * 72 * 2),
                _ => (
                    "flash_attention_dkv",
                    kv_heads,
                    shared(2) + 2 * 64 * 4,
                ),
            };
            let kernel = module.load_function(function).unwrap();
            kernel
                .set_attribute(
                    cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                )
                .unwrap();
            let config = LaunchConfig {
                grid_dim: (tiles, grid_y as u32, sequences as u32),
                block_dim: (128, 1, 1),
                shared_mem_bytes,
            };
            let (seq, width, qwidth) = (seq_len as i32, qkv_width as i32, query_width as i32);
            let (kbase, vbase) = (query_width as i32, (query_width + kv_width) as i32);
            let (grouped, stride) = (group as i32, rows as i32);
            let run = |iterations: usize| {
                for _ in 0..iterations {
                    let mut launch = stream.launch_builder(&kernel);
                    launch.arg(&qkv);
                    if index == 0 {
                        launch.arg(&out).arg(&lse);
                    } else {
                        launch.arg(&grad_out).arg(&lse).arg(&delta).arg(&grad_qkv);
                    }
                    launch
                        .arg(&seq)
                        .arg(&width)
                        .arg(&qwidth)
                        .arg(&kbase)
                        .arg(&vbase)
                        .arg(&grouped)
                        .arg(&stride)
                        .arg(&scale);
                    unsafe { launch.launch(config) }.unwrap();
                }
                stream.synchronize().unwrap();
            };
            run(3);
            let started = std::time::Instant::now();
            run(20);
            best[index] = best[index].min(started.elapsed().as_secs_f64() * 1000.0 / 20.0);
            if round == 4 {
                println!("{name:<16} {:>7.3} ms", best[index]);
            }
        }
    }
}
