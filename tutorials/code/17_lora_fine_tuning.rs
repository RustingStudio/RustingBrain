//! Chapter 17: fine-tune a trained model with a LoRA adapter.
//!
//! A base model is trained on one text, then adapted to a second text by
//! training a few small matrices beside its frozen weights. The adapter is
//! saved on its own, loaded back onto a fresh copy of the base, and finally
//! merged into the weights.
//!
//! Run with `cargo run --release`.

use rusting_brain::{LoraConfig, NetworkError, Optimizer, Precision, TokenFile, TransformerLm};

const SEQ_LEN: usize = 48;
const BATCH: usize = 8;
const BASE_STEPS: usize = 300;
const ADAPTER_STEPS: usize = 200;

/// What the base model learns.
const GENERAL: &str = "\
ownership moves, borrows do not. a value has exactly one owner, and when the \
owner goes out of scope the value is dropped. a reference borrows the value \
without taking ownership, and the compiler checks that no reference outlives \
the value it points at. shared references are read only and there may be many. \
a mutable reference is exclusive and there may be only one.";

/// What the adapter teaches it.
const DOMAIN: &str = "\
a lifetime names the region of code in which a reference is valid. most \
lifetimes are inferred, and the compiler asks for one only when a function \
returns a reference and could have borrowed it from more than one argument. \
the annotation does not change how long anything lives. it tells the compiler \
which argument the returned reference borrows from.";

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

/// Mean loss over every window of `file`, in order.
fn evaluate(model: &TransformerLm, file: &mut TokenFile) -> Result<f32, NetworkError> {
    let (mut total, mut batches) = (0.0, 0);
    while let Some(batch) = file.chunk(batches, 1, SEQ_LEN)? {
        total += model.evaluate(&batch)?.lm_loss;
        batches += 1;
    }
    Ok(total / batches as f32)
}

fn train(
    model: &mut TransformerLm,
    file: &mut TokenFile,
    steps: usize,
) -> Result<(), NetworkError> {
    for step in 0..steps {
        let batch = file.batch(step as u64, BATCH, SEQ_LEN)?;
        model.train_step_batch(&batch)?;
    }
    Ok(())
}

fn file_size(path: &str) -> std::io::Result<u64> {
    Ok(std::fs::metadata(path)?.len())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // One vocabulary for both texts: an adapter cannot add tokens.
    let tokenizer = CharTokenizer::fit(&format!("{GENERAL}{DOMAIN}"));
    TokenFile::write("general.bin", &tokenizer.encode(GENERAL))?;
    TokenFile::write("domain.bin", &tokenizer.encode(DOMAIN))?;
    let mut general = TokenFile::open("general.bin", 1)?;
    let mut domain = TokenFile::open("domain.bin", 2)?;

    // 1. The base model: every weight trains.
    let mut base = TransformerLm::builder()
        .vocab_size(tokenizer.len())
        .d_model(128)
        .n_layers(4)
        .heads(4, 2, 32)
        .d_ff(256)
        .moe_layers([])
        .max_seq_len(SEQ_LEN)
        .optimizer(Optimizer::adam(3e-3))
        .seed(7)
        .build()?;
    // base.to_cuda(0, 9_000)?;
    train(&mut base, &mut general, BASE_STEPS)?;
    base.save_bin("base.rbw", Precision::F32)?;
    println!(
        "base model    general {:.4}   domain {:.4}",
        evaluate(&base, &mut general)?,
        evaluate(&base, &mut domain)?
    );

    // 2. The adapter. `add_lora` freezes the base and runs on the host, so it
    //    goes before `to_cuda`. A fresh optimizer: the base's Adam moments
    //    belong to weights that no longer move.
    let mut model = TransformerLm::load_bin("base.rbw")?;
    model.add_lora(LoraConfig::new(8).alpha(16.0).seed(3))?;
    model.optimizer = Optimizer::adam(1e-3);
    let total = model.parameter_counts().total;
    let trainable = model.trainable_parameters();
    println!(
        "adapter       {trainable} of {total} weights train ({:.1}%)",
        100.0 * trainable as f32 / total as f32
    );
    // model.to_cuda(0, 9_000)?;
    train(&mut model, &mut domain, ADAPTER_STEPS)?;
    let general_after = evaluate(&model, &mut general)?;
    let domain_after = evaluate(&model, &mut domain)?;
    println!("fine-tuned    general {general_after:.4}   domain {domain_after:.4}");

    // 3. Ship the adapter alone.
    model.save_lora("domain.rbl")?;
    println!(
        "files         base.rbw {} bytes, domain.rbl {} bytes",
        file_size("base.rbw")?,
        file_size("domain.rbl")?
    );

    // 4. Anyone with the base can rebuild the fine-tuned model: load the base,
    //    attach an adapter of the same shape, read the weights into it.
    let mut rebuilt = TransformerLm::load_bin("base.rbw")?;
    rebuilt.add_lora(LoraConfig::new(8).alpha(16.0).seed(3))?;
    rebuilt.load_lora("domain.rbl")?;
    println!(
        "reloaded      domain {:.4}",
        evaluate(&rebuilt, &mut domain)?
    );

    // 5. Merge for deployment: an ordinary model, no adapter left to carry.
    rebuilt.merge_lora()?;
    println!(
        "merged        domain {:.4}   has adapter: {}",
        evaluate(&rebuilt, &mut domain)?,
        rebuilt.has_lora()
    );

    for file in ["general.bin", "domain.bin", "base.rbw", "domain.rbl"] {
        std::fs::remove_file(file)?;
    }
    Ok(())
}
