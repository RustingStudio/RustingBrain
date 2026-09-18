//! Trains a character-level transformer language model, then generates from it.
//!
//! ```bash
//! cargo run --release --example language_model -- train
//! cargo run --release --example language_model -- train --moe
//! cargo run --release --example language_model -- generate "the borrow checker"
//! ```
//!
//! `train` writes `lm.rbw` and `lm.vocab`; `generate` reads them back, so the
//! two halves run in separate processes and the checkpoint is really exercised.
//!
//! Character-level because the crate ships no tokenizer: tokenization is a
//! text problem rather than a neural-network one, and the `tokenizers` crate
//! already solves it. Swap a BPE vocabulary in and nothing else here changes.
//!
//! Tutorial chapters 14 to 16 explain every decision in this file.

use rusting_brain::{Optimizer, Precision, TokenBatch, TransformerLm};

const WEIGHTS: &str = "lm.rbw";
const VOCABULARY: &str = "lm.vocab";

const SEQ_LEN: usize = 64;
const STRIDE: usize = 8;
const MICRO_BATCH: usize = 8;
const ACCUMULATE: usize = 2;
const STEPS: usize = 600;
const WARMUP: usize = 60;
const PEAK_LR: f32 = 3e-3;

const CORPUS: &str = "\
ownership moves, borrows do not. a value has exactly one owner, and when the \
owner goes out of scope the value is dropped. a reference borrows the value \
without taking ownership, and the compiler checks that no reference outlives \
the value it points at. shared references are read only and there may be many. \
a mutable reference is exclusive and there may be only one. these two rules \
together are what make data races impossible to write, not merely unlikely. \
the borrow checker is not your enemy. it is a colleague who has read the code \
more carefully than you have. when it rejects a program it is telling you that \
two parts of the program disagree about who owns a value, and that the \
disagreement would have been a crash. the fix is almost never to fight it. the \
fix is to decide who owns the value and say so.";

/// One token per distinct character. The vocabulary is saved beside the
/// weights, because a checkpoint and the tokenizer that produced its ids are
/// only meaningful together.
struct CharTokenizer {
    vocabulary: Vec<char>,
}

impl CharTokenizer {
    fn fit(text: &str) -> Self {
        let mut vocabulary: Vec<char> = text.chars().collect();
        vocabulary.sort_unstable();
        vocabulary.dedup();
        Self { vocabulary }
    }

    fn load(path: &str) -> std::io::Result<Self> {
        Ok(Self {
            vocabulary: std::fs::read_to_string(path)?.chars().collect(),
        })
    }

    fn save(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.vocabulary.iter().collect::<String>())
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.vocabulary.iter().position(|&v| v == c))
            .map(|index| index as u32)
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocabulary.get(id as usize))
            .collect()
    }

    fn len(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Warmup, then cosine decay to a tenth of the peak.
///
/// Full-size steps before Adam's moment estimates have settled can knock the
/// model somewhere it takes a long time to leave; full-size steps at the end
/// stop it settling at all.
fn learning_rate(step: usize) -> f32 {
    if step < WARMUP {
        PEAK_LR * step as f32 / WARMUP as f32
    } else {
        let progress = (step - WARMUP) as f32 / (STEPS - WARMUP) as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        PEAK_LR * (0.1 + 0.9 * cosine)
    }
}

fn train(moe: bool) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = CharTokenizer::fit(CORPUS);
    let ids = tokenizer.encode(CORPUS);
    let windows: Vec<Vec<u32>> = ids
        .windows(SEQ_LEN)
        .step_by(STRIDE)
        .map(<[u32]>::to_vec)
        .collect();

    let mut builder = TransformerLm::builder()
        .vocab_size(tokenizer.len())
        .d_model(128)
        .n_layers(4)
        .heads(4, 2, 32)
        .d_ff(256)
        // Generation runs past the training window, and the limit is baked
        // into the checkpoint, so it is set for the longest sequence the model
        // will ever see rather than the longest it trains on.
        .max_seq_len(4 * SEQ_LEN)
        .optimizer(Optimizer::adam(PEAK_LR))
        .seed(42);

    builder = if moe {
        builder
            .moe_d_ff(64)
            .experts(8, 2)
            .moe_layers(1..4)
            .shared_expert(true)
    } else {
        builder.moe_layers([])
    };

    let mut model = builder.build()?;

    // On a GPU, these two lines and nothing else:
    // model.set_mixed_precision(true);
    // model.to_cuda(0, 9_000)?;

    println!(
        "{} windows of {SEQ_LEN} tokens, vocabulary {}",
        windows.len(),
        tokenizer.len()
    );
    println!("{}\n", model.parameter_counts());
    println!(" step      lr     loss    ppl     aux");

    let mut cursor = 0;
    for step in 1..=STEPS {
        // The optimizer is read fresh at every step, so a schedule is one
        // assignment and there is nothing to register.
        let rate = learning_rate(step);
        model.optimizer = Optimizer::adam(rate);

        // Gradient accumulation: the effective batch is not bounded by the
        // memory one forward pass needs.
        model.zero_grad();
        let (mut lm, mut aux) = (0.0, 0.0);
        for _ in 0..ACCUMULATE {
            let micro: Vec<Vec<u32>> = (0..MICRO_BATCH)
                .map(|_| {
                    let window = windows[cursor % windows.len()].clone();
                    cursor += 1;
                    window
                })
                .collect();
            let loss = model.accumulate_step(&TokenBatch::new(&micro)?)?;
            lm += loss.lm_loss;
            aux += loss.auxiliary_loss;
        }
        // The averaging belongs on the step: Adam normalizes by the gradient's
        // own second moment, so scaling every accumulation identically would
        // cancel out and change nothing.
        model.step(1.0 / ACCUMULATE as f32);

        if step == 1 || step % 100 == 0 {
            let lm = lm / ACCUMULATE as f32;
            println!(
                "{step:>5}  {rate:.5}  {lm:.4}  {:>6.2}  {:.4}",
                lm.exp(),
                aux / ACCUMULATE as f32
            );
        }
    }

    // F32, not Q8: int8 rounding is invisible for inference and shows up as a
    // step in the loss curve of a run that resumes from the file.
    #[cfg(feature = "cuda")]
    model.sync_from_device()?;
    model.save_bin(WEIGHTS, Precision::F32)?;
    model.save_optimizer_state(format!("{WEIGHTS}.opt"))?;
    tokenizer.save(VOCABULARY)?;

    println!("\nwrote {WEIGHTS}, {WEIGHTS}.opt and {VOCABULARY}");
    println!("now: cargo run --release --example language_model -- generate \"the borrow\"");
    Ok(())
}

/// Temperature sharpens or flattens the distribution; top-k cuts the long tail.
///
/// Without top-k the thirty thousand individually-impossible tokens carry
/// enough probability between them that one is eventually drawn, and a single
/// wrong token derails everything after it.
fn sample(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    let mut scaled: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(id, &value)| (id, value / temperature.max(1e-6)))
        .collect();
    scaled.sort_by(|a, b| b.1.total_cmp(&a.1));
    scaled.truncate(top_k.clamp(1, logits.len()));

    // Subtracting the maximum before `exp` is not a nicety: logits reach 30 or
    // more, `exp(30)` overflows to infinity, and every weight becomes NaN.
    let max = scaled[0].1;
    let weights: Vec<f32> = scaled.iter().map(|&(_, v)| (v - max).exp()).collect();
    let total: f32 = weights.iter().sum();

    let mut threshold = rand::random::<f32>() * total;
    for (&(id, _), &weight) in scaled.iter().zip(&weights) {
        threshold -= weight;
        if threshold <= 0.0 {
            return id as u32;
        }
    }
    scaled[0].0 as u32
}

fn generate(prompt: &str, new_tokens: usize) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = CharTokenizer::load(VOCABULARY)
        .map_err(|_| format!("{VOCABULARY} not found - run `language_model -- train` first"))?;
    let model = TransformerLm::load_bin(WEIGHTS)?;

    let ids = tokenizer.encode(prompt);
    if ids.is_empty() {
        return Err("the prompt has no characters this model was trained on".into());
    }

    // Two temperatures from the same prompt, because the difference is the
    // whole point: low is repetitive and safe, high is varied and wrong more
    // often.
    for (temperature, top_k) in [(0.2, 8), (0.8, 8)] {
        let mut generated = ids.clone();

        // Prefill the caches with the whole prompt in one pass, then append one
        // token at a time. The caches carry the position, so nothing tracks it.
        let mut caches = model.new_kv_caches();
        let mut logits = model.forward_cached(&generated, &mut caches)?;

        for _ in 0..new_tokens {
            let next = sample(logits.row(logits.rows - 1), temperature, top_k);
            generated.push(next);
            logits = model.forward_cached(&[next], &mut caches)?;
        }

        println!(
            "T={temperature:.1} k={top_k}\n  {}\n",
            tokenizer.decode(&generated)
        );
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("train") => train(args.iter().any(|a| a == "--moe")),
        Some("generate") => {
            let prompt = args
                .get(1)
                .map(String::as_str)
                .unwrap_or("the borrow checker");
            generate(prompt, 120)
        }
        _ => {
            eprintln!("usage:");
            eprintln!("  language_model -- train [--moe]");
            eprintln!("  language_model -- generate \"<prompt>\"");
            Ok(())
        }
    }
}
