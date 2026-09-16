//! Times one training step for a single architecture, so a long run can be
//! costed before it starts. One configuration per process: the mixture-of-
//! experts router is data dependent, and a model left resident on the device
//! skews the next one's memory headroom.
//!
//! `cargo run --release --features cuda --example sweep_arch -- <vocab> <d_model> <layers> <batch> <seq> [moe] [experts] [top_k]`

use rusting_brain::{Optimizer, TransformerLm};
use std::time::Instant;

fn arg<T: std::str::FromStr>(index: usize, default: T) -> T {
    std::env::args()
        .nth(index)
        .and_then(|a| a.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let (vocab, d_model, layers) = (arg(1, 32_000), arg(2, 512usize), arg(3, 8usize));
    let (batch, seq) = (arg(4, 16usize), arg(5, 512usize));
    let moe: usize = arg(6, 1usize);
    let (experts, top_k) = (arg(7, 8usize), arg(8, 2usize));
    let heads = d_model / 64;

    let builder = || {
        TransformerLm::builder()
            .vocab_size(vocab)
            .d_model(d_model)
            .n_layers(layers)
            .heads(heads, (heads / 4).max(1), 64)
            .d_ff(d_model * 11 / 4)
            .moe_d_ff(d_model * 11 / 16)
            .experts(experts, top_k)
            .moe_layers(if moe == 1 { 2..layers } else { 0..0 })
            .max_seq_len(seq.max(256))
            .optimizer(Optimizer::adam(1e-4))
            .seed(7)
    };
    let counts = builder().parameter_counts();
    let label = format!(
        "v{vocab} d{d_model} L{layers} {batch}x{seq} {}",
        if moe == 1 {
            format!("moe{experts}x{top_k}")
        } else {
            "dense".into()
        }
    );

    let mut model = builder().build().unwrap();
    #[cfg(feature = "cuda")]
    if let Err(error) = model.to_cuda(0, 11_000) {
        return println!("{label:<30} to_cuda failed: {error}");
    }

    // Ids spread over the whole vocabulary: a narrow range routes every token
    // to the same few experts and times a model that is not the one being run.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % vocab as u64) as u32
    };
    let tokens: Vec<Vec<u32>> = (0..batch)
        .map(|_| (0..seq).map(|_| next()).collect())
        .collect();

    for _ in 0..2 {
        if let Err(error) = model.train_step(&tokens) {
            return println!("{label:<30} step failed: {error}");
        }
    }
    let start = Instant::now();
    let steps = 5;
    for _ in 0..steps {
        model.train_step(&tokens).unwrap();
    }
    let per_step = start.elapsed().as_secs_f64() / steps as f64;
    let rate = (batch * seq) as f64 / per_step;
    println!(
        "{label:<30} {:>5.1}M tot {:>5.1}M act  {rate:>7.0} tok/s  {:>5.1} days for 50B",
        counts.total as f64 / 1e6,
        counts.active as f64 / 1e6,
        50e9 / rate / 86_400.0,
    );
}
