//! Chapter 13 — the four failure shapes, reproduced on purpose.
//!
//!     cargo run --release --bin 13_diagnostics
//!
//! Each run prints a curve. Learn to recognize them here, where you know the
//! answer, so you recognize them later when you do not.

use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig, TrainingHistory};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = xor();

    report("healthy", train(&data, Activation::Tanh, 0.05)?);
    report("rate too high (13.2)", train(&data, Activation::Tanh, 50.0)?);
    report("rate too low (13.3)", train(&data, Activation::Tanh, 1e-9)?);
    report("no non-linearity (13.3)", train(&data, Activation::Linear, 0.05)?);

    // 13.6 — two seeds, and the run is reproducible to the bit.
    let first = train(&data, Activation::Tanh, 0.05)?;
    let second = train(&data, Activation::Tanh, 0.05)?;
    assert_eq!(first.losses, second.losses, "seeded runs must be identical");
    println!("\nseeded runs are identical: yes");

    // 13.9 step 2 — the most useful test there is. A correct pipeline can
    // memorize a handful of rows; one with a wiring bug cannot.
    let memorized = train(&data, Activation::Tanh, 0.05)?;
    let final_loss = *memorized.losses.last().unwrap();
    println!(
        "overfit test on {} rows: final loss {final_loss:.6} ({})",
        data.len(),
        if final_loss < 0.01 { "pipeline works" } else { "something is wrong" }
    );

    Ok(())
}

fn train(
    data: &Dataset,
    hidden: Activation,
    learning_rate: f32,
) -> Result<TrainingHistory, Box<dyn std::error::Error>> {
    let mut model = Network::builder()
        .input_size(2)
        .dense(8, hidden)
        .dense(1, Activation::Sigmoid)
        .loss(Loss::BinaryCrossEntropy)
        .optimizer(Optimizer::adam(learning_rate))
        .seed(42)
        .build();
    Ok(model.fit(
        data,
        TrainConfig {
            epochs: 400,
            batch_size: 4,
            shuffle: true,
            seed: Some(7),
        },
    )?)
}

/// 13.1 — one line per run, sampled, plus the diagnosis the shape implies.
fn report(label: &str, history: TrainingHistory) {
    let sampled: Vec<String> = history
        .losses
        .iter()
        .step_by(history.losses.len() / 8)
        .map(|l| format!("{l:.4}"))
        .collect();
    let first = history.losses[0];
    let last = *history.losses.last().unwrap();

    let diagnosis = if !last.is_finite() {
        // 13.2 — find the epoch it died on, and whether it was climbing first.
        let bad = history.losses.iter().position(|l| !l.is_finite()).unwrap();
        format!("diverged at epoch {bad}; previous loss {:.4}", history.losses[bad - 1])
    } else if last > first {
        // 13.2 again — the step overshoots, so each correction is bigger than
        // the error it corrects. One more order of magnitude and this is NaN.
        "unstable: rate too high".into()
    } else if (first - last).abs() < 0.01 {
        "flat: not learning".into()
    } else if last < 0.05 {
        "converged".into()
    } else {
        "learning, but not finished".into()
    };

    println!("{label:>24}: {}  -> {diagnosis}", sampled.join(" "));
}

fn xor() -> Dataset {
    Dataset::new(
        vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]],
        vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
    )
}
