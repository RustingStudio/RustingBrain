//! What the fused attention forward costs on its two hot shapes, for one or
//! more versions of `src/cuda_flash.rs`.
//!
//! A full image or training run has noise about as large as a kernel change.
//! This runs the forward kernel on its own, interleaved across the sources
//! given on the command line (default: this crate's) and best of five, so a
//! change can be measured in seconds:
//! `git show HEAD:src/cuda_flash.rs > /tmp/old.rs` then
//! `cargo run --release --features cuda --example flash_probe -- /tmp/old.rs src/cuda_flash.rs`.

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

/// The kernel string out of a copy of `src/cuda_flash.rs`, which keeps it
/// private.
fn kernels(path: &str) -> String {
    let source = std::fs::read_to_string(path).expect("the kernel source");
    let body = source.split("r#\"").nth(1).expect("the kernel string");
    body.rsplit_once("\"#")
        .expect("the kernel string ends")
        .0
        .to_string()
}

/// One attention call over a packed `[rows, q | k | v]` buffer.
struct Shape {
    name: &'static str,
    sequences: usize,
    seq_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
}

fn main() {
    let mut sources: Vec<String> = std::env::args().skip(1).collect();
    if sources.is_empty() {
        sources.push(concat!(env!("CARGO_MANIFEST_DIR"), "/src/cuda_flash.rs").to_string());
    }
    let context = CudaContext::new(0).expect("no device");
    let stream = context.default_stream();
    let modules: Vec<_> = sources
        .iter()
        .map(|path| {
            let options = CompileOptions {
                arch: Some("compute_80"),
                ..Default::default()
            };
            let ptx = compile_ptx_with_opts(kernels(path), options).expect("the kernels compile");
            context.load_module(ptx).unwrap()
        })
        .collect();

    let shapes = [
        // The language-model benchmark: batch 4, 1024 tokens, 12 query heads
        // over 4 key/value heads of 64.
        Shape {
            name: "lm 4x1024 h64 causal",
            sequences: 4,
            seq_len: 1024,
            heads: 12,
            kv_heads: 4,
            head_dim: 64,
            causal: true,
        },
        // FLUX.2-klein at 1024x1024: 512 text and 4096 image tokens, 24 heads
        // of 128.
        Shape {
            name: "flux2 4608 h128",
            sequences: 1,
            seq_len: 4608,
            heads: 24,
            kv_heads: 24,
            head_dim: 128,
            causal: false,
        },
    ];
    for shape in &shapes {
        let rows = shape.sequences * shape.seq_len;
        let query_width = shape.heads * shape.head_dim;
        let kv_width = shape.kv_heads * shape.head_dim;
        let qkv_width = query_width + 2 * kv_width;
        // BF16 in [-1, 1): zeros would let the exponentials run on constants.
        let host: Vec<u16> = (0..rows * qkv_width)
            .map(|i| {
                let x = ((i as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32;
                ((2.0 * x - 1.0).to_bits() >> 16) as u16
            })
            .collect();
        let qkv = stream.clone_htod(&host).unwrap();
        let mut out = stream.alloc_zeros::<u16>(rows * query_width).unwrap();
        let mut lse = stream.alloc_zeros::<f32>(shape.heads * rows).unwrap();
        // Keys and values, a `[64][head_dim + 8]` BF16 tile each; the older
        // transposed value tile took the same room.
        let shared_mem_bytes = (2 * 64 * (shape.head_dim + 8) * 2) as u32;
        let config = LaunchConfig {
            grid_dim: (
                shape.seq_len.div_ceil(64) as u32,
                shape.heads as u32,
                shape.sequences as u32,
            ),
            block_dim: (128, 1, 1),
            shared_mem_bytes,
        };
        let kernels: Vec<CudaFunction> = modules
            .iter()
            .map(|module| {
                let kernel = module
                    .load_function(&format!("flash_attention_fwd_{}", shape.head_dim))
                    .unwrap();
                kernel
                    .set_attribute(
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        shared_mem_bytes as i32,
                    )
                    .unwrap();
                kernel
            })
            .collect();
        // q_len, kv_len, q_width, kv_width, query_width, key_base,
        // value_base, group, lse_head_stride.
        let ints = [
            shape.seq_len,
            shape.seq_len,
            qkv_width,
            qkv_width,
            query_width,
            query_width,
            query_width + kv_width,
            shape.heads / shape.kv_heads,
            rows,
        ]
        .map(|value| value as i32);
        let scale = (shape.head_dim as f32).sqrt().recip();
        let causal = i32::from(shape.causal);
        let mut best = vec![f64::MAX; kernels.len()];
        for _ in 0..5 {
            for (index, kernel) in kernels.iter().enumerate() {
                let mut run = |iterations: usize| {
                    for _ in 0..iterations {
                        let mut launch = stream.launch_builder(kernel);
                        launch.arg(&qkv).arg(&qkv).arg(&mut out).arg(&mut lse);
                        for value in &ints {
                            launch.arg(value);
                        }
                        launch.arg(&scale).arg(&causal);
                        unsafe { launch.launch(config) }.unwrap();
                    }
                    stream.synchronize().unwrap();
                };
                run(3);
                let started = std::time::Instant::now();
                run(20);
                best[index] = best[index].min(started.elapsed().as_secs_f64() * 1000.0 / 20.0);
            }
        }
        // Two matmuls of 2 * keys * head_dim per query row and head, about
        // half of them skipped under a causal mask.
        let causal_share = if shape.causal { 0.5 } else { 1.0 };
        let flops = 4.0
            * (shape.sequences * shape.heads * shape.seq_len * shape.seq_len * shape.head_dim)
                as f64
            * causal_share;
        for (path, ms) in sources.iter().zip(&best) {
            println!(
                "{:<22} {:>8.3} ms {:>6.1} TFLOPS  {path}",
                shape.name,
                ms,
                flops / ms / 1e9
            );
        }
    }
}
