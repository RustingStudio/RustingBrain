use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};
use std::error::Error;
use std::fs;

fn read_numeric_csv(path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty")?.split(',').count();
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != header {
            return Err(format!("line {}: bad column count", i + 2).into());
        }
        rows.push(cells.iter().map(|c| c.trim().parse::<f32>().unwrap()).collect());
    }
    Ok(rows)
}

struct Scaler { min: Vec<f32>, max: Vec<f32> }
impl Scaler {
    fn fit(rows: &[Vec<f32>]) -> Self {
        let w = rows[0].len();
        let mut min = vec![f32::INFINITY; w];
        let mut max = vec![f32::NEG_INFINITY; w];
        for r in rows { for (i, &v) in r.iter().enumerate() {
            min[i] = min[i].min(v); max[i] = max[i].max(v); } }
        Self { min, max }
    }
    fn tr(&self, row: &[f32]) -> Vec<f32> {
        row.iter().enumerate().map(|(i, &v)| {
            let s = self.max[i] - self.min[i];
            if s.abs() < f32::EPSILON { 0.0 } else { (v - self.min[i]) / s }
        }).collect()
    }
    fn inputs(&self, d: &mut Dataset) { for r in d.inputs.iter_mut() { *r = self.tr(r); } }
    fn targets(&self, d: &mut Dataset) { for r in d.targets.iter_mut() { *r = self.tr(r); } }
}

fn bar(value: f32, scale: f32) -> String {
    let n = ((value / scale) * 40.0).round().max(0.0) as usize;
    "#".repeat(n.min(60))
}

fn main() -> Result<(), Box<dyn Error>> {
    let rows = read_numeric_csv("data/houses.csv")?;
    let inputs: Vec<Vec<f32>> = rows.iter().map(|r| r[..4].to_vec()).collect();
    let targets: Vec<Vec<f32>> = rows.iter().map(|r| vec![r[4]]).collect();
    let mut dataset = Dataset::new(inputs, targets);
    dataset.shuffle(Some(7));

    let (train_full, rest) = dataset.split(0.70);
    let (validation, _test) = rest.split(0.50);
    let (mut train_full, mut validation) = (train_full, validation);

    let xs = Scaler::fit(&train_full.inputs);
    let ys = Scaler::fit(&train_full.targets);
    for d in [&mut train_full, &mut validation] { xs.inputs(d); ys.targets(d); }

    // ---------------------------------------------------------------
    // 1. A manual training loop that watches validation loss
    // ---------------------------------------------------------------
    println!("=== 1. manual loop: watching both losses ===\n");

    let mut model = Network::builder()
        .input_size(4)
        .dense(16, Activation::Relu)
        .dense(8, Activation::Relu)
        .dense(1, Activation::Linear)
        .loss(Loss::Mse)
        .optimizer(Optimizer::adam(0.01))
        .seed(42)
        .build();

    let one_epoch = TrainConfig { epochs: 1, batch_size: 16, shuffle: true, seed: Some(1) };

    println!("  epoch   train      val");
    for epoch in 1..=200 {
        model.fit(&train_full, one_epoch)?;
        if epoch % 25 == 0 || epoch == 1 {
            println!("  {:>5}   {:.5}   {:.5}", epoch,
                model.evaluate_loss(&train_full)?, model.evaluate_loss(&validation)?);
        }
    }

    // ---------------------------------------------------------------
    // 2. Overfitting on purpose
    // ---------------------------------------------------------------
    println!("\n=== 2. overfitting on purpose ===");
    let small = Dataset::new(
        train_full.inputs[..25].to_vec(),
        train_full.targets[..25].to_vec(),
    );
    println!("25 training rows, a 4->64->64->64->1 network ({} params)\n",
        4*64+64 + 64*64+64 + 64*64+64 + 64*1+1);

    let mut big = Network::builder()
        .input_size(4)
        .dense(64, Activation::Relu)
        .dense(64, Activation::Relu)
        .dense(64, Activation::Relu)
        .dense(1, Activation::Linear)
        .loss(Loss::Mse)
        .optimizer(Optimizer::adam(0.01))
        .seed(42)
        .build();

    let cfg = TrainConfig { epochs: 1, batch_size: 8, shuffle: true, seed: Some(2) };
    println!("  epoch     train       val   val curve");
    let mut best = f32::INFINITY;
    let mut best_epoch = 0;
    for epoch in 1..=300 {
        big.fit(&small, cfg)?;
        let v = big.evaluate_loss(&validation)?;
        if v < best { best = v; best_epoch = epoch; }
        if epoch % 25 == 0 || epoch == 1 {
            println!("  {:>5}   {:.5}   {:.5}   {}", epoch,
                big.evaluate_loss(&small)?, v, bar(v, 0.05));
        }
    }
    println!("\n  best validation loss {:.5} was at epoch {} - everything after was wasted",
        best, best_epoch);

    // ---------------------------------------------------------------
    // 3. Early stopping
    // ---------------------------------------------------------------
    println!("\n=== 3. early stopping ===\n");
    let mut model = Network::builder()
        .input_size(4)
        .dense(64, Activation::Relu)
        .dense(64, Activation::Relu)
        .dense(64, Activation::Relu)
        .dense(1, Activation::Linear)
        .loss(Loss::Mse)
        .optimizer(Optimizer::adam(0.01))
        .seed(42)
        .build();

    let patience = 20;
    let mut best_loss = f32::INFINITY;
    let mut best_model = model.clone();
    let mut best_epoch = 0;
    let mut since_improved = 0;

    for epoch in 1..=300 {
        model.fit(&small, cfg)?;
        let v = model.evaluate_loss(&validation)?;
        if v < best_loss {
            best_loss = v;
            best_model = model.clone();
            best_epoch = epoch;
            since_improved = 0;
        } else {
            since_improved += 1;
            if since_improved >= patience {
                println!("  stopped at epoch {} - no improvement for {} epochs", epoch, patience);
                break;
            }
        }
    }
    println!("  best epoch: {}   best validation loss: {:.5}", best_epoch, best_loss);
    println!("  restored model validation loss: {:.5}", best_model.evaluate_loss(&validation)?);
    println!("  (the still-training model was at {:.5})", model.evaluate_loss(&validation)?);

    // ---------------------------------------------------------------
    // 4. Learning rate sweep
    // ---------------------------------------------------------------
    println!("\n=== 4. learning rate matters ===\n");
    println!("  {:>10}   {:>10}   {:>10}", "lr", "train", "val");
    for lr in [0.0001f32, 0.001, 0.01, 0.1, 1.0] {
        let mut m = Network::builder()
            .input_size(4)
            .dense(16, Activation::Relu)
            .dense(8, Activation::Relu)
            .dense(1, Activation::Linear)
            .loss(Loss::Mse)
            .optimizer(Optimizer::adam(lr))
            .seed(42)
            .build();
        m.fit(&train_full, TrainConfig { epochs: 100, batch_size: 16, shuffle: true, seed: Some(1) })?;
        println!("  {:>10}   {:>10.5}   {:>10.5}", lr,
            m.evaluate_loss(&train_full)?, m.evaluate_loss(&validation)?);
    }

    // ---------------------------------------------------------------
    // 5. Batch size
    // ---------------------------------------------------------------
    println!("\n=== 5. batch size ===\n");
    println!("  {:>10}   {:>8}   {:>10}   {:>10}", "batch", "updates", "train", "val");
    for bs in [1usize, 8, 16, 64, 140] {
        let mut m = Network::builder()
            .input_size(4)
            .dense(16, Activation::Relu)
            .dense(8, Activation::Relu)
            .dense(1, Activation::Linear)
            .loss(Loss::Mse)
            .optimizer(Optimizer::adam(0.01))
            .seed(42)
            .build();
        m.fit(&train_full, TrainConfig { epochs: 100, batch_size: bs, shuffle: true, seed: Some(1) })?;
        let updates = 100 * train_full.len().div_ceil(bs);
        println!("  {:>10}   {:>8}   {:>10.5}   {:>10.5}", bs, updates,
            m.evaluate_loss(&train_full)?, m.evaluate_loss(&validation)?);
    }

    Ok(())
}
