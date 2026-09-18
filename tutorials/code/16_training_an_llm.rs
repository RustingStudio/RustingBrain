//! Chapter 16: a complete training program, at a size that finishes.
//!
//! Everything a week-long run needs, in the order a week-long run needs it:
//! a held-out split, gradient accumulation, a warmup-cosine learning-rate
//! schedule, periodic evaluation, checkpointing, and a resume that restores
//! the optimizer state in the right order.
//!
//! The only differences between this and a real run are the size of the model,
//! the size of the corpus, and the two commented lines that move it to a GPU.
//!
//! Run with `cargo run --release`.

use rusting_brain::{
    NetworkError, Optimizer, Precision, TokenBatch, TransformerLm, causal_lm_loss_batch,
};

const SEQ_LEN: usize = 48;
const MICRO_BATCH: usize = 4;
const ACCUMULATE: usize = 4; // effective batch of 16
const TOTAL_STEPS: usize = 400;
const WARMUP_STEPS: usize = 40;
const PEAK_LR: f32 = 3e-3;
const EVAL_EVERY: usize = 50;
const CHECKPOINT_STEP: usize = 200;

const CORPUS: &str = "\
ownership moves, borrows do not. a value has exactly one owner, and when the \
owner goes out of scope the value is dropped. a reference borrows the value \
without taking ownership, and the compiler checks that no reference outlives \
the value it points at. shared references are read only and there may be many. \
a mutable reference is exclusive and there may be only one. these two rules \
together are what make data races impossible to write, not merely unlikely. \
the cost is that some correct programs are rejected, and the reward is that \
no incorrect program of this kind is accepted.";

/// Warmup, then cosine decay to 10% of peak.
///
/// Full-size steps early are dangerous because Adam's moment estimates are
/// still noise; full-size steps late stop the model settling. Decaying to 10%
/// rather than 0 leaves it still learning if the run is extended.
fn learning_rate(step: usize) -> f32 {
    if step < WARMUP_STEPS {
        PEAK_LR * step as f32 / WARMUP_STEPS as f32
    } else {
        let progress = (step - WARMUP_STEPS) as f32 / (TOTAL_STEPS - WARMUP_STEPS) as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        PEAK_LR * (0.1 + 0.9 * cosine)
    }
}

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

    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.vocabulary.iter().position(|&v| v == c))
            .map(|index| index as u32)
            .collect()
    }

    fn len(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Fixed-length windows out of one flat token array, which is what a real
/// pre-tokenized corpus file looks like.
fn windows(ids: &[u32], stride: usize) -> Vec<Vec<u32>> {
    ids.windows(SEQ_LEN).step_by(stride).map(<[u32]>::to_vec).collect()
}

/// Mean loss over data the model has never trained on.
///
/// Training loss says the model is moving. Only this says it is learning.
fn evaluate(model: &TransformerLm, batches: &[TokenBatch]) -> Result<f32, NetworkError> {
    let mut total = 0.0;
    for batch in batches {
        let (logits, _cache) = model.forward_batch(batch)?;
        total += causal_lm_loss_batch(&logits, batch)?.loss;
    }
    Ok(total / batches.len() as f32)
}

/// Weights and optimizer state together. Either alone is not a resumable run.
fn checkpoint(model: &mut TransformerLm, path: &str) -> Result<(), NetworkError> {
    // On a device the weights live there; pull them back before writing.
    #[cfg(feature = "cuda")]
    model.sync_from_device()?;
    // F32, never Q8: int8 rounding is invisible for inference and shows up as
    // a step in the loss curve of a resumed run.
    model.save_bin(path, Precision::F32)?;
    model.save_optimizer_state(&format!("{path}.opt"))?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = CharTokenizer::fit(CORPUS);
    let ids = tokenizer.encode(CORPUS);

    // Hold back the tail and never train on it.
    let split = ids.len() * 9 / 10;
    let train_windows = windows(&ids[..split], 4);
    let validation: Vec<TokenBatch> = windows(&ids[split..], SEQ_LEN)
        .chunks(MICRO_BATCH)
        .map(TokenBatch::new)
        .collect::<Result<_, _>>()?;

    println!(
        "vocabulary {}, {} training windows, {} validation batches",
        tokenizer.len(),
        train_windows.len(),
        validation.len()
    );

    let mut model = TransformerLm::builder()
        .vocab_size(tokenizer.len())
        .d_model(128)
        .n_layers(4)
        .heads(4, 2, 32)
        .d_ff(256)
        .moe_layers([]) // dense at this size; chapter 15 for the alternative
        .max_seq_len(SEQ_LEN)
        .optimizer(Optimizer::adam(PEAK_LR))
        .seed(42)
        .build()?;

    // On a GPU, these two lines and nothing else:
    // model.set_mixed_precision(true);
    // model.to_cuda(0, 9_000)?;

    println!("{}\n", model.parameter_counts());
    println!(" step      lr     train    held-out");

    let mut cursor = 0;
    for step in 1..=TOTAL_STEPS {
        // The optimizer is read fresh every step, so a schedule is one
        // assignment. There is no schedule object to register.
        let rate = learning_rate(step);
        model.optimizer = Optimizer::adam(rate);

        // Gradient accumulation: the effective batch is not bounded by memory.
        model.zero_grad();
        let mut train_loss = 0.0;
        for _ in 0..ACCUMULATE {
            let micro: Vec<Vec<u32>> = (0..MICRO_BATCH)
                .map(|_| {
                    let window = train_windows[cursor % train_windows.len()].clone();
                    cursor += 1;
                    window
                })
                .collect();
            train_loss += model.accumulate_step(&TokenBatch::new(&micro)?)?.lm_loss;
        }
        // The averaging goes on the step, not on each backward pass: Adam
        // normalizes by the gradient's own second moment, so scaling every
        // accumulation identically would cancel out.
        model.step(1.0 / ACCUMULATE as f32);

        if step % EVAL_EVERY == 0 {
            let train = train_loss / ACCUMULATE as f32;
            let held_out = evaluate(&model, &validation)?;
            println!("{step:>5}  {rate:.5}   {train:.4}      {held_out:.4}");
        }

        if step == CHECKPOINT_STEP {
            checkpoint(&mut model, "run.rbw")?;
        }
    }

    // Resume, in the order that works.
    println!("\nresuming from the step-{CHECKPOINT_STEP} checkpoint");
    let mut resumed = TransformerLm::load_bin("run.rbw")?;
    // resumed.to_cuda(0, 9_000)?;   <- uploading zeroes the moments, so it
    //                                  goes here, BEFORE the next line
    resumed.load_optimizer_state("run.rbw.opt")?;
    println!(
        "  optimizer step counter restored: {} (checkpoint was at {CHECKPOINT_STEP})",
        resumed.optimizer_step()
    );
    println!("  held-out loss at the checkpoint: {:.4}", evaluate(&resumed, &validation)?);

    std::fs::remove_file("run.rbw")?;
    std::fs::remove_file("run.rbw.opt")?;
    Ok(())
}
