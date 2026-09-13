//! Times cached single-token decode, the inference hot loop.
//!
//! `cargo run --release --example bench_decode -- <prompt_len> <new_tokens>`

use rusting_brain::TransformerLm;
use std::time::Instant;

fn main() {
    let prompt: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(512);
    let new_tokens: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(128);

    let model = TransformerLm::builder()
        .vocab_size(32_000)
        .d_model(128)
        .n_layers(6)
        .heads(4, 4, 32)
        .d_ff(308)
        .moe_layers([])
        .max_seq_len(prompt + new_tokens + 1)
        .seed(7)
        .build()
        .unwrap();

    let mut caches = model.new_kv_caches();
    let ids: Vec<u32> = (0..prompt).map(|t| (t * 17 % 32_000) as u32).collect();
    model.forward_cached(&ids, &mut caches).unwrap();

    let start = Instant::now();
    for t in 0..new_tokens {
        model
            .forward_cached(&[(t * 31 % 32_000) as u32], &mut caches)
            .unwrap();
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "prompt {prompt} + {new_tokens} decoded: {:.2} ms/token, {:.0} tokens/s",
        1e3 * elapsed / new_tokens as f64,
        new_tokens as f64 / elapsed
    );
}
