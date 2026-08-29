#[cfg(feature = "cuda")]
fn main() {
    rusting_brain::gpu_test::run_benchmarks();
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("Re-run with `--features cuda` on a machine with the CUDA toolkit installed.");
}
