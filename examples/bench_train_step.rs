//! Times `TransformerLm::train_step` on the default ~55M/36M parameter model.
//!
//! `cargo run --release --features cuda --example bench_train_step -- <batch> <seq_len> <cuda|cpu>`

use rusting_brain::{Optimizer, TransformerLm};
use std::time::Instant;

fn main() {
    let batch_size: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(4);
    let seq_len: usize = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(128);
    let device: bool = std::env::args().nth(3).map(|a| a == "cuda").unwrap_or(true);

    let mut model = TransformerLm::builder()
        .optimizer(Optimizer::adam(1e-4))
        .max_seq_len(seq_len.max(256))
        .seed(7)
        .build()
        .unwrap();
    if device {
        #[cfg(feature = "cuda")]
        model.to_cuda(0, 16384).unwrap();
        #[cfg(not(feature = "cuda"))]
        panic!("rebuild with --features cuda to benchmark the device path");
    }

    let batch: Vec<Vec<u32>> = (0..batch_size)
        .map(|s| (0..seq_len).map(|t| ((s * 131 + t * 17) % 32000) as u32).collect())
        .collect();

    model.train_step(&batch).unwrap();
    let start = Instant::now();
    let steps = 3;
    for _ in 0..steps {
        model.train_step(&batch).unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64() / steps as f64;
    println!(
        "batch {batch_size} x seq {seq_len} on {}: {elapsed:.3}s/step, {:.0} tokens/s",
        if device { "cuda" } else { "cpu" },
        (batch_size * seq_len) as f64 / elapsed
    );
}
