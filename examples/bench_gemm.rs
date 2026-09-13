//! Ceilings for a CPU step: GEMM throughput at the shapes this library uses,
//! and `exp` throughput, which is what every softmax and SiLU pays per element.
//!
//! Run it before blaming a layer for a slow step. If the whole step is already
//! close to the rates printed here, the layer is not the problem.
//!
//! `cargo run --release --example bench_gemm`

use rusting_brain::matrix::Matrix;
use std::time::Instant;

fn bench(label: &str, m: usize, k: usize, n: usize) {
    let a = Matrix::random(m, k);
    let b = Matrix::random(k, n);
    let mut c = Matrix::new(m, n);

    // One untimed pass, so the measurement does not include first-touch page
    // faults on the output.
    a.dot(&b, &mut c);

    let repeats = (2_000_000_000f64 / (m * k * n) as f64).ceil().max(1.0) as usize;
    let start = Instant::now();
    for _ in 0..repeats {
        a.dot(&b, &mut c);
    }
    let seconds = start.elapsed().as_secs_f64() / repeats as f64;
    let gflops = 2.0 * (m * k * n) as f64 / seconds / 1e9;
    println!("{label:<28} {m:>6} x {k:>5} x {n:>6}   {gflops:7.1} GFLOP/s");
}

/// `exp` is not a vector instruction. If the compiler cannot inline and
/// vectorize it, every softmax and every SiLU pays a scalar libm call per
/// element, which is easily worth more than the GEMMs it sits between.
fn bench_exp() {
    let mut values: Vec<f32> = (0..1 << 20).map(|i| (i as f32 % 17.0) - 8.0).collect();

    let start = Instant::now();
    for _ in 0..64 {
        for value in values.iter_mut() {
            *value = value.exp().min(1e30);
        }
    }
    let seconds = start.elapsed().as_secs_f64();
    let per_second = (64 * values.len()) as f64 / seconds;
    println!(
        "{:<28} {:>25}   {:7.1} Mexp/s",
        "exp, one core",
        "",
        per_second / 1e6
    );
}

fn main() {
    bench_exp();
    bench("lm head forward", 2048, 128, 32000);
    bench("lm head backward (input)", 2048, 32000, 128);
    bench("square reference", 1024, 1024, 1024);
    bench("block projection", 2048, 128, 128);
    bench("swiglu up", 2048, 128, 308);
}
