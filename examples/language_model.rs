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
//! The corpus is a paragraph and the model is 0.6M parameters, so the held-out
//! loss printed beside the training loss turns back up within a hundred steps:
//! there is nothing here to generalize to and memorizing is the only thing left
//! to do. That gap is the point of printing it.
//!
//! Character-level because the crate ships no tokenizer: tokenization is a
//! text problem rather than a neural-network one, and the `tokenizers` crate
//! already solves it. Swap a BPE vocabulary in and nothing else here changes.
//!
//! Tutorial chapters 14 to 16 explain every decision in this file.

use rusting_brain::{Optimizer, Precision, Sampler, Schedule, TokenBatch, TransformerLm};

const WEIGHTS: &str = "lm.rbw";
const VOCABULARY: &str = "lm.vocab";

const SEQ_LEN: usize = 64;
const STRIDE: usize = 8;
const MICRO_BATCH: usize = 8;
const ACCUMULATE: usize = 2;
const STEPS: usize = 600;
const WARMUP: usize = 60;
const PEAK_LR: f32 = 3e-3;
const MAX_GRAD_NORM: f32 = 1.0;

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

fn train(moe: bool) -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = CharTokenizer::fit(CORPUS);
    let ids = tokenizer.encode(CORPUS);

    // The held-out split is a slice of the text, not a sample of the windows:
    // windows overlap, so holding out every n-th one would put most of a
    // validation window inside a training window and report a loss that means
    // nothing.
    let split = ids.len() * 85 / 100;
    let window = |ids: &[u32]| -> Vec<Vec<u32>> {
        ids.windows(SEQ_LEN)
            .step_by(STRIDE)
            .map(<[u32]>::to_vec)
            .collect()
    };
    let windows = window(&ids[..split]);
    let held_out = TokenBatch::new(&window(&ids[split..]))?;

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
        "{} training windows of {SEQ_LEN} tokens, {} held out, vocabulary {}",
        windows.len(),
        held_out.batch(),
        tokenizer.len()
    );
    println!("{}\n", model.parameter_counts());
    println!(" step      lr     loss    ppl     aux    |g|      val");

    // Warmup then cosine decay: full-size steps before Adam's moments have
    // settled knock the model somewhere it takes a long time to leave, and
    // full-size steps at the end stop it settling at all.
    let schedule = Schedule::warmup_cosine(PEAK_LR, WARMUP, STEPS);

    let mut cursor = 0;
    for step in 0..STEPS {
        // The optimizer is read fresh at every step, so a schedule is one
        // assignment and there is nothing to register.
        let rate = schedule.rate(step);
        model.optimizer.set_learning_rate(rate);

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
        //
        // Clipping is not an optimization. A single batch whose gradient is an
        // order of magnitude larger than usual moves the weights far enough to
        // undo thousands of steps, and the norm printed below is the warning
        // that it happened.
        let norm = model.step_clipped(1.0 / ACCUMULATE as f32, MAX_GRAD_NORM)?;

        if step == 0 || (step + 1) % 100 == 0 {
            let lm = lm / ACCUMULATE as f32;
            // Training loss falls whether or not the model is learning anything
            // general. The held-out loss is what says which, and the step where
            // it turns back up is the step to have stopped at.
            let validation = model.evaluate(&held_out)?;
            println!(
                "{:>5}  {rate:.5}  {lm:.4}  {:>6.2}  {:.4}  {norm:.3}  {:.4}",
                step + 1,
                lm.exp(),
                aux / ACCUMULATE as f32,
                validation.lm_loss
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
    // often. Top-k matters as much as the temperature: without it the thirty
    // thousand individually-impossible tokens carry enough probability between
    // them that one is eventually drawn, and a single wrong token derails
    // everything after it.
    for (temperature, top_k) in [(0.2, 8), (0.8, 8)] {
        // A fixed seed, so re-running the example on the same checkpoint prints
        // the same text; pass `None` for a different sample each run.
        let mut sampler = Sampler::temperature(temperature, Some(42)).top_k(top_k);

        // Printed as it decodes rather than at the end: each token costs a
        // forward pass, and on a real model that is several seconds of nothing.
        // `flush`, because stdout is line buffered and none of this is a line.
        print!("T={temperature:.1} k={top_k}\n  {}", tokenizer.decode(&ids));
        model.generate_with(&ids, new_tokens, &mut sampler, |id| {
            print!("{}", tokenizer.decode(&[id]));
            let _ = std::io::Write::flush(&mut std::io::stdout());
            true
        })?;
        println!("\n");
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
