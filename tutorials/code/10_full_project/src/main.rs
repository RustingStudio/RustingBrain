//! Churn prediction, end to end.
//!
//!   cargo run --release -- train
//!   cargo run --release -- evaluate
//!   cargo run --release -- predict <months> <hours|?> <tickets> <plan>

mod data;
mod metrics;
mod prep;

use data::{read_csv, Row, FEATURE_NAMES};
use metrics::Confusion;
use prep::Preprocessing;
use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};
use std::error::Error;

const DATA: &str = "data/subscriptions.csv";
const MODEL: &str = "churn_model.json";
const PREP: &str = "churn_model.prep";
const THRESHOLD: &str = "churn_model.threshold";

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("train") => train(),
        Some("evaluate") => evaluate(),
        Some("predict") => predict(&args[1..]),
        _ => {
            eprintln!("usage: churn <train|evaluate|predict>");
            eprintln!("  predict <months> <hours|?> <tickets> <basic|plus|pro>");
            std::process::exit(2);
        }
    }
}

/// Split rows the same way every time, so train/evaluate/predict agree.
fn split(rows: &[Row]) -> (Vec<Row>, Vec<Row>, Vec<Row>) {
    // A fixed shuffle: same permutation on every run and in every subcommand.
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    let mut state: u64 = 20260905;
    for i in (1..idx.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        idx.swap(i, j);
    }
    let n_train = rows.len() * 70 / 100;
    let n_val = rows.len() * 15 / 100;
    let take = |r: &[usize]| r.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>();
    (
        take(&idx[..n_train]),
        take(&idx[n_train..n_train + n_val]),
        take(&idx[n_train + n_val..]),
    )
}

fn to_dataset(rows: &[Row], prep: &Preprocessing) -> Dataset {
    Dataset::new(
        rows.iter().map(|r| prep.transform(r)).collect(),
        rows.iter().map(|r| vec![if r.cancelled { 1.0 } else { 0.0 }]).collect(),
    )
}

fn scores(model: &Network, rows: &[Row], prep: &Preprocessing) -> Result<Vec<f32>, Box<dyn Error>> {
    let inputs: Vec<Vec<f32>> = rows.iter().map(|r| prep.transform(r)).collect();
    Ok(model.predict_batch(&inputs)?.into_iter().map(|o| o[0]).collect())
}

fn build() -> Network {
    Network::builder()
        .input_size(FEATURE_NAMES.len())
        .dense(16, Activation::Relu)
        .dense(8, Activation::Relu)
        .dense(1, Activation::Sigmoid)
        .loss(Loss::BinaryCrossEntropy)
        .optimizer(Optimizer::adam(0.004))
        .seed(42)
        .build()
}

fn train() -> Result<(), Box<dyn Error>> {
    let rows = read_csv(DATA)?;
    let churn = rows.iter().filter(|r| r.cancelled).count();
    let missing = rows.iter().filter(|r| r.monthly_hours.is_none()).count();
    println!("{} rows, {} cancelled ({:.1}%), {} missing monthly_hours",
        rows.len(), churn, churn as f32 / rows.len() as f32 * 100.0, missing);

    let (train_rows, val_rows, test_rows) = split(&rows);
    println!("split: {} train / {} validation / {} test\n",
        train_rows.len(), val_rows.len(), test_rows.len());

    let prep = Preprocessing::fit(&train_rows);
    println!("filling {} blank hours cells with the training median: {:.1}",
        missing, prep.hours_median);

    let train_ds = to_dataset(&train_rows, &prep);
    let val_ds = to_dataset(&val_rows, &prep);

    // Early stopping (chapter 7): keep the best validation model, not the last.
    let mut model = build();
    let mut best = model.clone();
    let mut best_loss = f32::INFINITY;
    let mut best_epoch = 0;
    let mut since = 0;
    const PATIENCE: usize = 30;

    println!("\n epoch     train      val");
    for epoch in 1..=400 {
        model.fit(&train_ds, TrainConfig {
            epochs: 1, batch_size: 32, shuffle: true, seed: Some(epoch as u64),
        })?;
        let v = model.evaluate_loss(&val_ds)?;
        if v < best_loss - 1e-5 {
            best_loss = v;
            best = model.clone();
            best_epoch = epoch;
            since = 0;
        } else {
            since += 1;
        }
        if epoch <= 10 || epoch % 10 == 0 {
            println!("  {epoch:>4}   {:.4}   {:.4}{}",
                model.evaluate_loss(&train_ds)?, v,
                if since == 0 { "  *" } else { "" });
        }
        if since >= PATIENCE {
            println!("\nstopped early at epoch {epoch}: no improvement for {PATIENCE} epochs");
            break;
        }
    }
    println!("best epoch {best_epoch}, validation loss {best_loss:.4}");
    let model = best;

    // Pick the decision threshold on VALIDATION, never on test.
    let val_scores = scores(&model, &val_rows, &prep)?;
    let val_truth: Vec<bool> = val_rows.iter().map(|r| r.cancelled).collect();
    println!("\nchoosing a decision threshold on validation:");
    println!("  threshold   accuracy   precision   recall      F1");
    let mut best_t = 0.5;
    let mut best_f1 = -1.0;
    let mut t = 0.20;
    while t <= 0.801 {
        let c = Confusion::count(&val_scores, &val_truth, t);
        let f1 = c.f1();
        println!("      {:.2}      {:.1}%      {:.1}%      {:.1}%   {:.3}{}",
            t, c.accuracy()*100.0, c.precision()*100.0, c.recall()*100.0, f1,
            if f1 > best_f1 { "  <-" } else { "" });
        if f1 > best_f1 { best_f1 = f1; best_t = t; }
        t += 0.10;
    }
    println!("chose threshold {best_t:.2} (F1 {best_f1:.3})");

    model.save_json(MODEL)?;
    prep.save(PREP)?;
    std::fs::write(THRESHOLD, format!("{best_t}\n"))?;
    println!("\nsaved {MODEL}, {PREP}, {THRESHOLD}");
    println!("({} test rows left untouched - run `evaluate` next)", test_rows.len());
    Ok(())
}

fn evaluate() -> Result<(), Box<dyn Error>> {
    let rows = read_csv(DATA)?;
    let (_, _, test_rows) = split(&rows);
    let model = Network::load_json(MODEL)?;
    let prep = Preprocessing::load(PREP)?;
    let threshold: f32 = std::fs::read_to_string(THRESHOLD)?.trim().parse()?;

    let truth: Vec<bool> = test_rows.iter().map(|r| r.cancelled).collect();
    let s = scores(&model, &test_rows, &prep)?;
    let churn = truth.iter().filter(|&&t| t).count();

    println!("test set: {} rows, {} actually cancelled\n", test_rows.len(), churn);
    let majority = (truth.len() - churn) as f32 / truth.len() as f32;
    println!("  baseline (predict nobody churns): {:.1}% accuracy, 0% recall\n", majority * 100.0);

    Confusion::count(&s, &truth, threshold)
        .print(&format!("at the chosen threshold {threshold:.2} - THIS is the reported result:"));

    // We may REPORT how other thresholds would have done. We may not go back and
    // pick one from this table - that choice was made on validation, and reusing
    // the test set to decide would spend it (chapter 8).
    println!("\n  for reference only, other thresholds on the test set:");
    println!("  threshold   accuracy   precision   recall      F1");
    let mut t = 0.20;
    while t <= 0.801 {
        let c = Confusion::count(&s, &truth, t);
        println!("      {:.2}      {:.1}%      {:.1}%      {:.1}%   {:.3}{}",
            t, c.accuracy()*100.0, c.precision()*100.0, c.recall()*100.0, c.f1(),
            if (t - threshold).abs() < 0.001 { "  <- chosen" } else { "" });
        t += 0.10;
    }

    let mut ranked: Vec<(usize, f32)> = s.iter().cloned().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let hits = ranked[..20].iter().filter(|(i, _)| truth[*i]).count();
    println!("\n  top 20 highest-risk customers: {hits}/20 really did cancel");
    Ok(())
}

fn predict(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() != 4 {
        return Err("predict needs: <months> <hours|?> <tickets> <basic|plus|pro>".into());
    }
    let row = Row {
        months_active: args[0].parse()?,
        monthly_hours: if args[1] == "?" { None } else { Some(args[1].parse()?) },
        support_tickets: args[2].parse()?,
        plan: args[3].clone(),
        cancelled: false, // unknown - that is what we are predicting
    };
    if !data::PLANS.contains(&row.plan.as_str()) {
        return Err(format!("unknown plan {:?}", row.plan).into());
    }

    let model = Network::load_json(MODEL)?;
    let prep = Preprocessing::load(PREP)?;
    let threshold: f32 = std::fs::read_to_string(THRESHOLD)?.trim().parse()?;

    let risk = model.predict(&prep.transform(&row))?[0];
    println!("churn risk: {:.1}%", risk * 100.0);
    println!("decision at threshold {:.2}: {}", threshold,
        if risk >= threshold { "AT RISK - worth an intervention" } else { "likely to stay" });
    Ok(())
}
