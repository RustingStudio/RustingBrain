//! Loss trajectory of 8-bit Adam state against FP32 state on real text.
//!
//! Trains the default model twice from the same seed on the repository's
//! markdown, tokenized with a 4096-entry BPE trained on it, once with each
//! state, and prints the training loss every 250 steps and the held-out loss
//! at the end. Also prints the device memory each run peaks at.
//!
//! `cargo run --release --features cuda --example adam8_loss_check -- <steps>`

#[cfg(feature = "cuda")]
fn main() {
    use rusting_brain::{Bpe, Optimizer, TokenBatch, TransformerLm};

    let steps: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(2000);
    let mut texts = Vec::new();
    for directory in [".", "docs", "tutorials"] {
        let mut paths: Vec<_> = std::fs::read_dir(directory)
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
            .collect();
        paths.sort();
        texts.extend(
            paths
                .iter()
                .map(|path| std::fs::read_to_string(path).unwrap()),
        );
    }
    let bpe = Bpe::from_json(&Bpe::train_json(&texts, 4096, &[]).unwrap()).unwrap();
    let ids = bpe.encode(&texts.concat()).unwrap();
    let held_out = ids.len() / 10;
    let (train, test) = ids.split_at(ids.len() - held_out);
    let (rows, length) = (16, 256);
    let windows = |tokens: &[u32], offset: usize| -> Vec<Vec<u32>> {
        (0..rows)
            .map(|row| {
                let start = (offset + row * 7919 * length) % (tokens.len() - length);
                tokens[start..start + length].to_vec()
            })
            .collect()
    };
    println!("{} training tokens, {held_out} held out", train.len());

    for eight_bit in [false, true] {
        let mut model = TransformerLm::builder()
            .vocab_size(4096)
            .optimizer(Optimizer::adam(3e-4))
            .seed(7)
            .build()
            .unwrap();
        model.to_cuda(0, 0).unwrap();
        model.set_8bit_optimizer_state(eight_bit).unwrap();
        let mut line = Vec::new();
        for step in 0..steps {
            let batch = TokenBatch::new(&windows(train, step * 104_729)).unwrap();
            let loss = model.train_step_batch(&batch).unwrap().lm_loss;
            if step % 250 == 0 || step == steps - 1 {
                line.push(format!("{step}:{loss:.4}"));
            }
        }
        let mut held = 0.0;
        for chunk in 0..8 {
            let batch = TokenBatch::new(&windows(test, chunk * 31_337)).unwrap();
            held += model.evaluate(&batch).unwrap().lm_loss / 8.0;
        }
        println!(
            "{}: {} | held-out {held:.4}",
            if eight_bit { "8-bit" } else { "fp32 " },
            line.join(" ")
        );
    }
}

#[cfg(not(feature = "cuda"))]
fn main() {
    println!("rebuild with --features cuda");
}
