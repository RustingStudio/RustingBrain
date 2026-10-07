//! Times cached single-token decode, the inference hot loop.
//!
//! `cargo run --release --example bench_decode -- <prompt_len> <new_tokens> [cuda|cpu] [sequences] [int8]`
//!
//! The host run uses a small model; `cuda` decodes the default model on the
//! device instead. `sequences` decodes that many in step, and tokens/s counts
//! the tokens of all of them. `int8` turns on
//! [`TransformerLm::set_int8_decode`] for the device run.

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

    let device = std::env::args().nth(3).is_some_and(|a| a == "cuda");
    let sequences: usize = std::env::args()
        .nth(4)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1);

    let builder = TransformerLm::builder()
        .max_seq_len(prompt + new_tokens + 1)
        .seed(7);
    let builder = if device {
        builder
    } else {
        builder
            .vocab_size(32_000)
            .d_model(128)
            .n_layers(6)
            .heads(4, 4, 32)
            .d_ff(308)
            .moe_layers([])
    };
    let mut model = builder.build().unwrap();
    model.set_int8_decode(std::env::args().nth(5).is_some_and(|a| a == "int8"));
    if device {
        #[cfg(feature = "cuda")]
        model.to_cuda(0, 16384).unwrap();
        #[cfg(not(feature = "cuda"))]
        panic!("rebuild with --features cuda to benchmark the device path");
    }

    let mut decoder = model.batch_decoder(sequences).unwrap();
    let ids: Vec<u32> = (0..prompt).map(|t| (t * 17 % 32_000) as u32).collect();
    decoder.feed(&vec![&ids[..]; sequences]).unwrap();

    let start = Instant::now();
    for t in 0..new_tokens {
        let step: Vec<[u32; 1]> = (0..sequences)
            .map(|s| [((t * 31 + s) % 32_000) as u32])
            .collect();
        let step: Vec<&[u32]> = step.iter().map(|id| &id[..]).collect();
        decoder.feed(&step).unwrap();
    }
    model.synchronize().unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    let tokens = (new_tokens * sequences) as f64;
    println!(
        "prompt {prompt} + {new_tokens} x {sequences} decoded on {}: {:.2} ms/step, {:.0} tokens/s",
        if device { "cuda" } else { "cpu" },
        1e3 * elapsed / new_tokens as f64,
        tokens / elapsed
    );
}
