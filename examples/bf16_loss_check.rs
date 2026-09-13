//! Loss trajectory of the BF16 head against FP32 on a real vocabulary.
//!
//! The parity test runs a 24-token vocabulary; this runs 32k, where the
//! softmax denominator sums far more terms and BF16's 8-bit mantissa has the
//! most room to drift.
//!
//! `cargo run --release --features cuda --example bf16_loss_check`

use rusting_brain::{Optimizer, TokenBatch, TransformerLm};

fn main() {
    let steps: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(20);
    let ids: Vec<Vec<u32>> = (0..64)
        .map(|s| (0..128).map(|t| ((s * 131 + t * 17) % 32_000) as u32).collect())
        .collect();
    let batch = TokenBatch::new(&ids).unwrap();

    for mixed in [false, true] {
        let mut model = TransformerLm::builder()
            .vocab_size(32_000)
            .d_model(128)
            .n_layers(6)
            .heads(4, 4, 32)
            .d_ff(308)
            .moe_layers([])
            .max_seq_len(256)
            .mixed_precision(mixed)
            .optimizer(Optimizer::adam(1e-4))
            .seed(7)
            .build()
            .unwrap();
        #[cfg(feature = "cuda")]
        model.to_cuda(0, 0).unwrap();
        let mut losses = Vec::new();
        for _ in 0..steps {
            losses.push(model.train_step_batch(&batch).unwrap().lm_loss);
        }
        let label = if mixed { "bf16" } else { "fp32" };
        println!(
            "{label}: first {:.5} last {:.5} | {}",
            losses[0],
            losses[steps - 1],
            losses.iter().map(|l| format!("{l:.4}")).collect::<Vec<_>>().join(" ")
        );
    }
}
