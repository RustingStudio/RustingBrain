use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};
use std::error::Error;
use std::fs;

fn read_numeric_csv(path: &str) -> Result<(Vec<String>, Vec<Vec<f32>>), Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<String> = lines.next().ok_or("empty file")?
        .split(',').map(|s| s.trim().to_string()).collect();
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != header.len() {
            return Err(format!("line {}: expected {} columns, found {}",
                i + 2, header.len(), cells.len()).into());
        }
        let mut row = Vec::with_capacity(cells.len());
        for c in cells {
            row.push(c.trim().parse::<f32>()
                .map_err(|e| format!("line {}: {:?}: {e}", i + 2, c.trim()))?);
        }
        rows.push(row);
    }
    Ok((header, rows))
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
    fn transform(&self, row: &[f32]) -> Vec<f32> {
        row.iter().enumerate().map(|(i, &v)| {
            let span = self.max[i] - self.min[i];
            if span.abs() < f32::EPSILON { 0.0 } else { (v - self.min[i]) / span }
        }).collect()
    }
    fn apply_inputs(&self, d: &mut Dataset) {
        for r in d.inputs.iter_mut() { *r = self.transform(r); }
    }
    fn apply_targets(&self, d: &mut Dataset) {
        for r in d.targets.iter_mut() { *r = self.transform(r); }
    }
    fn invert_one(&self, scaled: f32) -> f32 {
        scaled * (self.max[0] - self.min[0]) + self.min[0]
    }
}

/// Root mean squared error and mean absolute error, in real units.
fn metrics(model: &Network, data: &Dataset, ty: &Scaler) -> Result<(f32, f32), Box<dyn Error>> {
    let mut se = 0.0f32;
    let mut ae = 0.0f32;
    for (input, target) in data.inputs.iter().zip(&data.targets) {
        let pred = ty.invert_one(model.predict(input)?[0]);
        let truth = ty.invert_one(target[0]);
        se += (pred - truth).powi(2);
        ae += (pred - truth).abs();
    }
    let n = data.len() as f32;
    Ok(((se / n).sqrt(), ae / n))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (header, rows) = read_numeric_csv("data/houses.csv")?;
    println!("columns: {:?}", header);
    println!("rows:    {}\n", rows.len());

    // last column (price_k) is the target
    let inputs: Vec<Vec<f32>> = rows.iter().map(|r| r[..r.len()-1].to_vec()).collect();
    let targets: Vec<Vec<f32>> = rows.iter().map(|r| vec![r[r.len()-1]]).collect();

    let mut dataset = Dataset::new(inputs, targets);
    dataset.shuffle(Some(7));
    let (train, rest) = dataset.split(0.70);
    let (validation, test) = rest.split(0.50);
    let (mut train, mut validation, mut test) = (train, validation, test);
    println!("split: {} train / {} validation / {} test\n",
        train.len(), validation.len(), test.len());

    let x_scaler = Scaler::fit(&train.inputs);
    let y_scaler = Scaler::fit(&train.targets);
    println!("price range in training data: {:.1}k .. {:.1}k\n",
        y_scaler.min[0], y_scaler.max[0]);

    for d in [&mut train, &mut validation, &mut test] {
        x_scaler.apply_inputs(d);
        y_scaler.apply_targets(d);
    }

    // Baseline: always predict the mean training price.
    let mean_price: f32 = train.targets.iter().map(|t| y_scaler.invert_one(t[0])).sum::<f32>()
        / train.len() as f32;
    let mut base_se = 0.0f32;
    let mut base_ae = 0.0f32;
    for t in &test.targets {
        let truth = y_scaler.invert_one(t[0]);
        base_se += (mean_price - truth).powi(2);
        base_ae += (mean_price - truth).abs();
    }
    let base_rmse = (base_se / test.len() as f32).sqrt();
    let base_mae = base_ae / test.len() as f32;
    println!("BASELINE (always predict the mean, {:.1}k):", mean_price);
    println!("  test RMSE {:.2}k   MAE {:.2}k\n", base_rmse, base_mae);

    let mut model = Network::builder()
        .input_size(4)
        .dense(16, Activation::Relu)
        .dense(8, Activation::Relu)
        .dense(1, Activation::Linear)
        .loss(Loss::Mse)
        .optimizer(Optimizer::adam(0.01))
        .seed(42)
        .build();

    let (rmse0, mae0) = metrics(&model, &test, &y_scaler)?;
    println!("BEFORE training: test RMSE {:.2}k   MAE {:.2}k\n", rmse0, mae0);

    let history = model.fit(&train, TrainConfig {
        epochs: 400, batch_size: 16, shuffle: true, seed: Some(1),
    })?;

    println!("training loss (scaled units):");
    for e in [0, 24, 99, 199, 399] {
        println!("  epoch {:>3}: {:.6}", e + 1, history.losses[e]);
    }

    let (train_rmse, train_mae) = metrics(&model, &train, &y_scaler)?;
    let (val_rmse, val_mae) = metrics(&model, &validation, &y_scaler)?;
    let (test_rmse, test_mae) = metrics(&model, &test, &y_scaler)?;
    println!("\nAFTER training (real units, thousands):");
    println!("  {:<12} RMSE {:>7.2}k   MAE {:>7.2}k", "train", train_rmse, train_mae);
    println!("  {:<12} RMSE {:>7.2}k   MAE {:>7.2}k", "validation", val_rmse, val_mae);
    println!("  {:<12} RMSE {:>7.2}k   MAE {:>7.2}k", "test", test_rmse, test_mae);
    println!("  {:<12} RMSE {:>7.2}k   MAE {:>7.2}k", "baseline", base_rmse, base_mae);
    println!("\nimprovement over baseline: {:.0}% lower MAE",
        (1.0 - test_mae / base_mae) * 100.0);

    println!("\nsome individual test predictions:");
    println!("  {:>8} {:>10} {:>10} {:>9}", "actual", "predicted", "error", "");
    for i in 0..8 {
        let pred = y_scaler.invert_one(model.predict(&test.inputs[i])?[0]);
        let truth = y_scaler.invert_one(test.targets[i][0]);
        println!("  {:>7.1}k {:>9.1}k {:>9.1}k", truth, pred, pred - truth);
    }

    Ok(())
}
