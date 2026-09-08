use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};
use std::error::Error;
use std::fs;

fn read_numeric_csv(path: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let n = lines.next().ok_or("empty")?.split(',').count();
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != n { return Err(format!("line {}: bad columns", i + 2).into()); }
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
    fn inv(&self, v: f32) -> f32 { v * (self.max[0] - self.min[0]) + self.min[0] }
}

fn build(h1: usize, h2: usize, lr: f32, seed: u64) -> Network {
    Network::builder()
        .input_size(4)
        .dense(h1, Activation::Relu)
        .dense(h2, Activation::Relu)
        .dense(1, Activation::Linear)
        .loss(Loss::Mse)
        .optimizer(Optimizer::adam(lr))
        .seed(seed)
        .build()
}

fn mae(model: &Network, d: &Dataset, ys: &Scaler) -> Result<f32, Box<dyn Error>> {
    let mut s = 0.0;
    for (i, t) in d.inputs.iter().zip(&d.targets) {
        s += (ys.inv(model.predict(i)?[0]) - ys.inv(t[0])).abs();
    }
    Ok(s / d.len() as f32)
}

fn main() -> Result<(), Box<dyn Error>> {
    let rows = read_numeric_csv("data/houses.csv")?;
    let inputs: Vec<Vec<f32>> = rows.iter().map(|r| r[..4].to_vec()).collect();
    let targets: Vec<Vec<f32>> = rows.iter().map(|r| vec![r[4]]).collect();
    let mut dataset = Dataset::new(inputs, targets);
    dataset.shuffle(Some(7));

    let (train, rest) = dataset.split(0.70);
    let (validation, test) = rest.split(0.50);
    let (mut train, mut validation, mut test) = (train, validation, test);
    let xs = Scaler::fit(&train.inputs);
    let ys = Scaler::fit(&train.targets);
    for d in [&mut train, &mut validation, &mut test] { xs.inputs(d); ys.targets(d); }

    let cfg = TrainConfig { epochs: 150, batch_size: 16, shuffle: true, seed: Some(1) };

    // ---------------------------------------------------------------
    // 1. Tuning against validation, then the honest test score
    // ---------------------------------------------------------------
    println!("=== 1. searching 12 architectures on the validation set ===\n");
    println!("  {:>4} {:>4} {:>8} {:>10} {:>10}", "h1", "h2", "lr", "val MAE", "");
    let mut best = (f32::INFINITY, 0usize, 0usize, 0.0f32);
    let mut results = Vec::new();
    for (h1, h2) in [(8, 4), (16, 8), (32, 16), (64, 32)] {
        for lr in [0.003f32, 0.01, 0.03] {
            let mut m = build(h1, h2, lr, 42);
            m.fit(&train, cfg)?;
            let v = mae(&m, &validation, &ys)?;
            results.push((v, h1, h2, lr));
            let star = if v < best.0 { best = (v, h1, h2, lr); "  <- best so far" } else { "" };
            println!("  {:>4} {:>4} {:>8} {:>10.3} {:>10}", h1, h2, lr, v, star);
        }
    }
    println!("\n  winner: {}x{} lr={}  validation MAE {:.3}k", best.1, best.2, best.3, best.0);

    let mut chosen = build(best.1, best.2, best.3, 42);
    chosen.fit(&train, cfg)?;
    let test_mae = mae(&chosen, &test, &ys)?;
    println!("  the SAME model on the untouched test set: {:.3}k", test_mae);
    println!("  optimism from picking the winner: {:.3}k ({:.0}%)",
        test_mae - best.0, (test_mae / best.0 - 1.0) * 100.0);

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mean_val: f32 = results.iter().map(|r| r.0).sum::<f32>() / results.len() as f32;
    println!("\n  best validation MAE:    {:.3}k", results[0].0);
    println!("  average validation MAE: {:.3}k", mean_val);
    println!("  worst validation MAE:   {:.3}k", results[results.len()-1].0);

    // ---------------------------------------------------------------
    // 2. Seed variance
    // ---------------------------------------------------------------
    println!("\n=== 2. the same architecture, 8 different seeds ===\n");
    let mut vals = Vec::new();
    for seed in 1..=8u64 {
        let mut m = build(16, 8, 0.01, seed);
        m.fit(&train, cfg)?;
        let v = mae(&m, &validation, &ys)?;
        vals.push(v);
        println!("  seed {:>2}: validation MAE {:.3}k", seed, v);
    }
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    let sd = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
    let lo = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let hi = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("\n  mean {:.3}k   std dev {:.3}k   range {:.3}k .. {:.3}k (spread {:.3}k)",
        mean, sd, lo, hi, hi - lo);

    // ---------------------------------------------------------------
    // 3. k-fold cross validation
    // ---------------------------------------------------------------
    println!("\n=== 3. 5-fold cross validation ===\n");
    let mut pool = Dataset::new(
        [train.inputs.clone(), validation.inputs.clone()].concat(),
        [train.targets.clone(), validation.targets.clone()].concat(),
    );
    pool.shuffle(Some(99));
    let k = 5;
    let fold_size = pool.len() / k;
    let mut fold_scores = Vec::new();
    for fold in 0..k {
        let start = fold * fold_size;
        let end = if fold == k - 1 { pool.len() } else { start + fold_size };

        let mut tr_in = Vec::new(); let mut tr_tg = Vec::new();
        let mut va_in = Vec::new(); let mut va_tg = Vec::new();
        for i in 0..pool.len() {
            if i >= start && i < end {
                va_in.push(pool.inputs[i].clone()); va_tg.push(pool.targets[i].clone());
            } else {
                tr_in.push(pool.inputs[i].clone()); tr_tg.push(pool.targets[i].clone());
            }
        }
        let tr = Dataset::new(tr_in, tr_tg);
        let va = Dataset::new(va_in, va_tg);
        let mut m = build(16, 8, 0.01, 42);
        m.fit(&tr, cfg)?;
        let v = mae(&m, &va, &ys)?;
        fold_scores.push(v);
        println!("  fold {}: trained on {:>3}, tested on {:>2}  ->  MAE {:.3}k",
            fold + 1, tr.len(), va.len(), v);
    }
    let fm: f32 = fold_scores.iter().sum::<f32>() / k as f32;
    let fsd = (fold_scores.iter().map(|v| (v - fm).powi(2)).sum::<f32>() / k as f32).sqrt();
    println!("\n  cross-validated MAE: {:.3}k +/- {:.3}k", fm, fsd);
    println!("  (every row was used for testing exactly once)");

    // ---------------------------------------------------------------
    // 4. Error analysis
    // ---------------------------------------------------------------
    println!("\n=== 4. where does it go wrong? ===\n");
    let mut errs: Vec<(f32, f32, f32)> = Vec::new();
    for (i, t) in test.inputs.iter().zip(&test.targets) {
        let p = ys.inv(chosen.predict(i)?[0]);
        let a = ys.inv(t[0]);
        errs.push((( p - a).abs(), a, p));
    }
    errs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("  5 worst predictions:");
    for (e, a, p) in errs.iter().take(5) {
        println!("    actual {:>7.1}k  predicted {:>7.1}k  off by {:>6.1}k", a, p, e);
    }
    println!("  5 best predictions:");
    for (e, a, p) in errs.iter().rev().take(5) {
        println!("    actual {:>7.1}k  predicted {:>7.1}k  off by {:>6.1}k", a, p, e);
    }

    let mut cheap = (0.0f32, 0usize);
    let mut pricey = (0.0f32, 0usize);
    for (e, a, _) in &errs {
        if *a < 200.0 { cheap.0 += e; cheap.1 += 1; } else { pricey.0 += e; pricey.1 += 1; }
    }
    println!("\n  MAE on houses under 200k: {:.2}k  ({} houses)", cheap.0 / cheap.1 as f32, cheap.1);
    println!("  MAE on houses over  200k: {:.2}k  ({} houses)", pricey.0 / pricey.1 as f32, pricey.1);

    Ok(())
}
