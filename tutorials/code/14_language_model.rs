//! Chapter 14: a character-level transformer language model, end to end.
//!
//! Trains on one paragraph until it can reproduce it, then generates from a
//! prompt. Deliberately tiny: the point is to watch perplexity fall from
//! `vocab_size` to near 1 and to see every stage of the pipeline.
//!
//! Run with `cargo run --release --example 14_language_model`.

use rusting_brain::{Optimizer, Precision, Sampler, TransformerLm};

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

fn generate(
    model: &TransformerLm,
    tokenizer: &CharTokenizer,
    prompt: &str,
    new_tokens: usize,
    temperature: f32,
    top_k: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    let ids = tokenizer.encode(prompt);

    // `Sampler` holds the decoding knobs: temperature reshapes the
    // distribution, top-k cuts the long tail that would otherwise derail
    // everything after one unlucky token. A fixed seed so the output above is
    // the output you get.
    let mut sampler = Sampler::temperature(temperature, Some(7)).top_k(top_k);

    // `generate` runs the prefill-then-decode loop over a KV cache and returns
    // the new tokens only, so the prompt goes back in front for printing.
    let continuation = model.generate(&ids, new_tokens, &mut sampler)?;
    Ok(tokenizer.decode(&[ids, continuation].concat()))
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
