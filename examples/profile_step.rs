//! Phase timing for one `train_step_batch`, so an optimization targets the
//! phase that actually costs time.
//!
//! `cargo run --release --features cuda --example profile_step -- <dense|moe> <batch> <seq> [cpu]`

use rusting_brain::{Optimizer, TokenBatch, TransformerLm, causal_lm_loss_batch};
use std::time::Instant;

/// ~5.2M parameters, dense, tied embeddings: the small regression config.
fn dense() -> rusting_brain::TransformerBuilder {
    TransformerLm::builder()
        .vocab_size(32_000)
        .d_model(128)
        .n_layers(6)
        .heads(4, 4, 32)
        .d_ff(308)
        .moe_layers([])
}

/// The library default: ~55M total, ~36M active, MoE from layer 2 on.
fn moe() -> rusting_brain::TransformerBuilder {
    TransformerLm::builder().moe_layers(2..8)
}

fn main() {
    let preset = std::env::args().nth(1).unwrap_or_else(|| "dense".into());
    let batch_size: usize = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(256);
    let seq_len: usize = std::env::args()
        .nth(3)
        .and_then(|a| a.parse().ok())
        .unwrap_or(128);
    let cuda = std::env::args().nth(4).map(|a| a != "cpu").unwrap_or(true);
    // Reduced precision is the library default; `nomp` is the opt-out, so the
    // plain invocation measures what an ordinary caller gets.
    let mixed = !std::env::args().any(|a| a == "nomp");

    let builder = match preset.as_str() {
        "dense" => dense(),
        "moe" => moe(),
        // Same size and shape with every layer dense, so the MoE machinery's
        // own cost is the difference between the two runs.
        "nomoe" => TransformerLm::builder().moe_layers([]),
        other => panic!("unknown preset {other}, expected dense or moe"),
    };
    let mut model = builder
        .max_seq_len(seq_len.max(256))
        .mixed_precision(mixed)
        .optimizer(Optimizer::adam(1e-4))
        .seed(7)
        .build()
        .unwrap();
    println!("preset {preset}: {}", model.parameter_counts());

    if cuda {
        #[cfg(feature = "cuda")]
        model.to_cuda(0, 0).unwrap();
        #[cfg(not(feature = "cuda"))]
        panic!("rebuild with --features cuda");
    }

    let ids: Vec<Vec<u32>> = (0..batch_size)
        .map(|s| {
            (0..seq_len)
                .map(|t| ((s * 131 + t * 17) % 32_000) as u32)
                .collect()
        })
        .collect();
    let batch = TokenBatch::new(&ids).unwrap();

    // Warm up: the first step pays kernel compilation and allocator growth.
    if std::env::var_os("RB_PHASES").is_none() {
        model.train_step_batch(&batch).unwrap();
    } else {
        let (logits, cache) = model.forward_batch(&batch).unwrap();
        let loss = causal_lm_loss_batch(&logits, &batch).unwrap();
        model.zero_grad();
        model.backward(&cache, &loss.grad_logits).unwrap();
        model.step(1.0);
    }
    model.synchronize().unwrap();

    let steps = 3;
    // `RB_PHASES=1` times the unfused path instead: forward, host loss,
    // backward and step as four separate calls. That is the comparison the
    // fused `train_step_batch` below has to beat to be worth its complexity.
    if std::env::var_os("RB_PHASES").is_none() {
        let fused = Instant::now();
        for _ in 0..steps {
            model.train_step_batch(&batch).unwrap();
        }
        model.synchronize().unwrap();
        let fused = fused.elapsed().as_secs_f64() / steps as f64;
        println!(
            "train_step_batch {fused:.3} s/step, {:.0} tokens/s",
            (batch_size * seq_len) as f64 / fused
        );
        return;
    }

    let mut totals = [0f64; 6];
    let labels = [
        "forward",
        "loss(host)",
        "zero_grad",
        "backward",
        "step",
        "total",
    ];

    for _ in 0..steps {
        let t0 = Instant::now();
        let (logits, cache) = model.forward_batch(&batch).unwrap();
        model.synchronize().unwrap();
        let t1 = Instant::now();
        let loss = causal_lm_loss_batch(&logits, &batch).unwrap();
        let t2 = Instant::now();
        model.zero_grad();
        model.synchronize().unwrap();
        let t3 = Instant::now();
        model.backward(&cache, &loss.grad_logits).unwrap();
        model.synchronize().unwrap();
        let t4 = Instant::now();
        model.step(1.0);
        model.synchronize().unwrap();
        let t5 = Instant::now();

        for (slot, dt) in
            totals
                .iter_mut()
                .zip([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0])
        {
            *slot += dt.as_secs_f64();
        }
    }

    let total = totals[5] / steps as f64;
    for (label, sum) in labels.iter().zip(totals) {
        let seconds = sum / steps as f64;
        println!(
            "{label:<12} {:>8.3} s  {:>5.1}%",
            seconds,
            100.0 * seconds / total
        );
    }
    println!(
        "batch {batch_size} x seq {seq_len}: {total:.3} s/step, {:.0} tokens/s",
        (batch_size * seq_len) as f64 / total
    );
}
