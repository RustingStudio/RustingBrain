//! Chapter 14: a character-level transformer language model, end to end.
//!
//! Trains on one paragraph until it can reproduce it, then generates from a
//! prompt. Deliberately tiny: the point is to watch perplexity fall from
//! `vocab_size` to near 1 and to see every stage of the pipeline.
//!
//! Run with `cargo run --release --example 14_language_model`.

use rusting_brain::{Optimizer, Precision, TransformerLm};

const CORPUS: &str = "\
the borrow checker is not your enemy. it is a colleague who has read the code \
more carefully than you have. when it rejects a program it is telling you that \
two parts of the program disagree about who owns a value, and that the \
disagreement would have been a crash. the fix is almost never to fight it. the \
fix is to decide who owns the value and say so.";

const SEQ_LEN: usize = 64;
const STRIDE: usize = 16;
const STEPS: usize = 600;

/// A character-level tokenizer: every distinct character in the corpus is one
/// token. Real models use subword BPE (see 14.2), but the model does not care
/// where the ids come from.
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

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| self.vocabulary.get(id as usize))
            .collect()
    }

    fn len(&self) -> usize {
        self.vocabulary.len()
    }
}

/// Pick the next token from one row of logits.
///
/// `temperature` below 1.0 sharpens toward the top choice; `top_k` cuts the
/// long tail, which is what stops one unlucky token derailing the rest.
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

fn generate(
    model: &TransformerLm,
    tokenizer: &CharTokenizer,
    prompt: &str,
    new_tokens: usize,
    temperature: f32,
    top_k: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut ids = tokenizer.encode(prompt);
    let mut caches = model.new_kv_caches();

    // Prefill: the whole prompt in one pass.
    let mut logits = model.forward_cached(&ids, &mut caches)?;

    for _ in 0..new_tokens {
        let next = sample(logits.row(logits.rows - 1), temperature, top_k);
        ids.push(next);
        // Decode: one token, attending to everything already cached.
        logits = model.forward_cached(&[next], &mut caches)?;
    }

    Ok(tokenizer.decode(&ids))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = CharTokenizer::fit(CORPUS);
    let corpus = tokenizer.encode(CORPUS);
    println!(
        "{} characters, {} distinct -> vocabulary of {}",
        CORPUS.len(),
        tokenizer.len(),
        tokenizer.len()
    );

    // Fixed-length windows. Overlapping them by `STRIDE` gives the model every
    // position as a prediction target without padding anything.
    let windows: Vec<Vec<u32>> = corpus
        .windows(SEQ_LEN)
        .step_by(STRIDE)
        .map(<[u32]>::to_vec)
        .collect();
    println!("{} training windows of {} tokens", windows.len(), SEQ_LEN);

    let mut model = TransformerLm::builder()
        .vocab_size(tokenizer.len())
        .d_model(128)
        .n_layers(4)
        .heads(4, 2, 32)
        .d_ff(256)
        .moe_layers([]) // dense; chapter 15 turns this on
        // Room for the prompt plus everything generated after it, not just one
        // training window.
        .max_seq_len(4 * SEQ_LEN)
        .tie_embeddings(true)
        .optimizer(Optimizer::adam(3e-3))
        .seed(42)
        .build()?;

    println!("{}\n", model.parameter_counts());

    // An untrained model is guessing uniformly, so its perplexity should start
    // at roughly the vocabulary size. If it does not, the ids are wrong.
    println!("step    loss   perplexity   (uniform = {:.0})", tokenizer.len());
    for step in 1..=STEPS {
        let loss = model.train_step(&windows)?;
        if step == 1 || step % 100 == 0 {
            println!(
                "{step:>4}  {:.4}   {:>8.2}",
                loss.lm_loss,
                loss.lm_loss.exp()
            );
        }
    }

    let prompt = "the borrow checker";
    println!("\nprompt: {prompt:?}");
    for (temperature, top_k) in [(0.2, 8), (0.8, 8), (1.5, tokenizer.len())] {
        let text = generate(&model, &tokenizer, prompt, 80, temperature, top_k)?;
        println!("  T={temperature:.1} k={top_k:<3} {text:?}");
    }

    model.save_bin("char_lm.rbw", Precision::F32)?;
    let reloaded = TransformerLm::load_bin("char_lm.rbw")?;
    println!(
        "\nreloaded checkpoint agrees: {}",
        reloaded.parameter_counts().total == model.parameter_counts().total
    );
    std::fs::remove_file("char_lm.rbw")?;

    Ok(())
}
