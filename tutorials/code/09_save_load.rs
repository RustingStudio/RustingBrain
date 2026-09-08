use rusting_brain::{
    Activation, CudaTrainingCheckpoint, Dataset, Loss, Network, Optimizer, TrainConfig,
};
use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::time::Instant;

struct Row { features: Vec<f32>, label: String }

fn read_csv(path: &str) -> Result<Vec<Row>, Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let n = lines.next().ok_or("empty")?.split(',').count();
    let mut rows = Vec::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() { continue; }
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != n { return Err(format!("line {}: bad columns", i+2).into()); }
        let features = cells[..n-1].iter()
            .map(|c| c.trim().parse::<f32>())
            .collect::<Result<Vec<_>, _>>()?;
        rows.push(Row { features, label: cells[n-1].trim().to_string() });
    }
    Ok(rows)
}

/// Everything a saved model needs besides the weights.
struct Preprocessing {
    min: Vec<f32>,
    max: Vec<f32>,
    classes: Vec<String>,
}

impl Preprocessing {
    fn fit(rows: &[Vec<f32>], classes: Vec<String>) -> Self {
        let w = rows[0].len();
        let mut min = vec![f32::INFINITY; w];
        let mut max = vec![f32::NEG_INFINITY; w];
        for r in rows { for (i, &v) in r.iter().enumerate() {
            min[i] = min[i].min(v); max[i] = max[i].max(v); } }
        Self { min, max, classes }
    }

    fn transform(&self, row: &[f32]) -> Vec<f32> {
        row.iter().enumerate().map(|(i, &v)| {
            let s = self.max[i] - self.min[i];
            if s.abs() < f32::EPSILON { 0.0 } else { (v - self.min[i]) / s }
        }).collect()
    }

    fn apply(&self, d: &mut Dataset) {
        for r in d.inputs.iter_mut() { *r = self.transform(r); }
    }

    /// A tiny line-based format: no extra dependencies needed.
    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let mut out = String::new();
        out.push_str("version 1\n");
        out.push_str(&format!("min {}\n",
            self.min.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")));
        out.push_str(&format!("max {}\n",
            self.max.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")));
        out.push_str(&format!("classes {}\n", self.classes.join(",")));
        fs::write(path, out)?;
        Ok(())
    }

    fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let text = fs::read_to_string(path)?;
        let mut min = Vec::new();
        let mut max = Vec::new();
        let mut classes = Vec::new();
        for line in text.lines() {
            let (key, value) = line.split_once(' ').ok_or("malformed line")?;
            match key {
                "version" => if value.trim() != "1" {
                    return Err(format!("unsupported preprocessing version {value}").into());
                },
                "min" => min = value.split(',').map(|v| v.parse()).collect::<Result<_,_>>()?,
                "max" => max = value.split(',').map(|v| v.parse()).collect::<Result<_,_>>()?,
                "classes" => classes = value.split(',').map(|s| s.to_string()).collect(),
                _ => return Err(format!("unknown key {key}").into()),
            }
        }
        if min.is_empty() || min.len() != max.len() || classes.is_empty() {
            return Err("incomplete preprocessing file".into());
        }
        Ok(Self { min, max, classes })
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi,bv),(i,&x)| if x>bv {(i,x)} else {(bi,bv)}).0
}

fn accuracy(m: &Network, d: &Dataset) -> Result<f32, Box<dyn Error>> {
    let mut c = 0;
    for (i, t) in d.inputs.iter().zip(&d.targets) {
        if argmax(&m.predict(i)?) == argmax(t) { c += 1; }
    }
    Ok(c as f32 / d.len() as f32)
}

fn main() -> Result<(), Box<dyn Error>> {
    let rows = read_csv("data/flowers.csv")?;
    let classes: Vec<String> = rows.iter().map(|r| r.label.clone())
        .collect::<BTreeSet<_>>().into_iter().collect();

    let inputs: Vec<Vec<f32>> = rows.iter().map(|r| r.features.clone()).collect();
    let targets: Vec<Vec<f32>> = rows.iter()
        .map(|r| classes.iter().map(|c| if *c == r.label {1.0} else {0.0}).collect())
        .collect();
    let mut dataset = Dataset::new(inputs, targets);
    dataset.shuffle(Some(42));
    let (train, rest) = dataset.split(0.70);
    let (_validation, test) = rest.split(0.50);
    let (mut train, mut test) = (train, test);

    let prep = Preprocessing::fit(&train.inputs, classes.clone());
    prep.apply(&mut train);
    prep.apply(&mut test);

    let mut model = Network::builder()
        .input_size(4)
        .dense(12, Activation::Relu)
        .dense(3, Activation::Softmax)
        .loss(Loss::CrossEntropy)
        .optimizer(Optimizer::adam(0.02))
        .seed(42)
        .build();
    model.fit(&train, TrainConfig { epochs: 300, batch_size: 16, shuffle: true, seed: Some(3) })?;

    let acc_before = accuracy(&model, &test)?;
    println!("trained model test accuracy: {:.1}%\n", acc_before * 100.0);

    // ---- save ----
    let dir = std::env::temp_dir().join("rb_tutorial");
    fs::create_dir_all(&dir)?;
    let model_path = dir.join("flowers.json");
    let prep_path = dir.join("flowers.prep");
    model.save_json(&model_path)?;
    prep.save(prep_path.to_str().unwrap())?;
    println!("saved:");
    println!("  {}  ({} bytes)", model_path.display(), fs::metadata(&model_path)?.len());
    println!("  {}  ({} bytes)", prep_path.display(), fs::metadata(&prep_path)?.len());

    let json = fs::read_to_string(&model_path)?;
    println!("\nfirst 260 characters of the model file:");
    println!("{}", &json[..260.min(json.len())]);
    println!("  ...");

    // ---- load ----
    println!("\n--- new process would start here ---\n");
    let loaded = Network::load_json(&model_path)?;
    let loaded_prep = Preprocessing::load(prep_path.to_str().unwrap())?;
    println!("loaded model: {} inputs, {} outputs, {} layers",
        loaded.input_size(), loaded.output_size(), loaded.layers().len());
    println!("loaded classes: {:?}", loaded_prep.classes);

    let acc_after = accuracy(&loaded, &test)?;
    println!("\nloaded model test accuracy: {:.1}%", acc_after * 100.0);
    println!("identical to before saving: {}", (acc_before - acc_after).abs() < 1e-6);

    let mut max_diff = 0.0f32;
    for input in &test.inputs {
        for (a, b) in model.predict(input)?.iter().zip(&loaded.predict(input)?) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    println!("largest difference in any output value: {:e}", max_diff);

    // ---- inference on brand new raw data ----
    println!("\n--- predicting on brand new, unscaled measurements ---\n");
    let new_flowers = [
        ("looks like rosetta",  vec![1.4f32, 0.2, 5.0, 3.4]),
        ("looks like valentia", vec![4.3, 1.3, 5.9, 2.8]),
        ("looks like borealis", vec![5.7, 2.1, 6.7, 3.0]),
        ("ambiguous",           vec![4.9, 1.7, 6.2, 2.9]),
    ];
    for (note, raw) in &new_flowers {
        let scaled = loaded_prep.transform(raw);      // ← same scaling as training
        let probs = loaded.predict(&scaled)?;
        let best = argmax(&probs);
        println!("  {:?}  ({})", raw, note);
        println!("     -> {} at {:.1}% confidence", loaded_prep.classes[best], probs[best] * 100.0);
        let mut ranked: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let all: Vec<String> = ranked.iter()
            .map(|(i, p)| format!("{} {:.1}%", loaded_prep.classes[*i], p * 100.0)).collect();
        println!("     all: {}", all.join(", "));
    }

    // ---- forgetting the scaler ----
    println!("\n--- what happens if you forget to scale ---\n");
    let raw = vec![1.4f32, 0.2, 5.0, 3.4];
    let right = loaded.predict(&loaded_prep.transform(&raw))?;
    let wrong = loaded.predict(&raw)?;
    println!("  scaled correctly: {} at {:.1}%",
        loaded_prep.classes[argmax(&right)], right[argmax(&right)] * 100.0);
    println!("  raw, unscaled:    {} at {:.1}%",
        loaded_prep.classes[argmax(&wrong)], wrong[argmax(&wrong)] * 100.0);

    // ---- batch inference ----
    println!("\n--- batch inference ---\n");
    let many: Vec<Vec<f32>> = (0..20000).map(|i| test.inputs[i % test.len()].clone()).collect();

    let t0 = Instant::now();
    let mut one_by_one = Vec::with_capacity(many.len());
    for input in &many { one_by_one.push(loaded.predict(input)?); }
    let single = t0.elapsed();

    let t1 = Instant::now();
    let batched = loaded.predict_batch(&many)?;
    let batch = t1.elapsed();

    println!("  {} predictions", many.len());
    println!("  predict() in a loop: {:?}", single);
    println!("  predict_batch():     {:?}", batch);
    println!("  speedup: {:.2}x", single.as_secs_f64() / batch.as_secs_f64());
    println!("  identical results: {}", one_by_one == batched);

    // ---- the optimizer state gotcha ----
    // Use a deliberately under-trained model (25 epochs, not 300) so that
    // "20 more epochs" has something visible left to do.
    println!("\n--- gotcha: save_json does NOT save the optimizer ---\n");
    let mut short = Network::builder()
        .input_size(4)
        .dense(12, Activation::Relu)
        .dense(3, Activation::Softmax)
        .loss(Loss::CrossEntropy)
        .optimizer(Optimizer::adam(0.02))
        .seed(42)
        .build();
    let first = TrainConfig { epochs: 25, batch_size: 16, shuffle: true, seed: Some(3) };
    let more = TrainConfig { epochs: 20, batch_size: 16, shuffle: true, seed: Some(7) };
    short.fit(&train, first.clone())?;
    println!("  after 25 epochs: train loss {:.4}", short.evaluate_loss(&train)?);

    let weights_path = dir.join("resume_weights.json");
    let ckpt_path = dir.join("resume_full.json");
    short.save_json(&weights_path)?;
    // Despite the name, a checkpoint is plain CPU state - no GPU, no feature flag.
    short.cuda_checkpoint(25, Some(3)).save_json(&ckpt_path)?;
    println!("  weights only   : {} bytes", fs::metadata(&weights_path)?.len());
    println!("  full checkpoint: {} bytes", fs::metadata(&ckpt_path)?.len());

    // (a) resuming from save_json: the optimizer falls back to Sgd(0.01)
    let mut weights_only = Network::load_json(&weights_path)?;
    weights_only.fit(&train, more.clone())?;
    println!("\n  resumed from save_json    -> train loss {:.4}",
        weights_only.evaluate_loss(&train)?);

    // (b) resuming from a checkpoint: optimizer and Adam moments come back too
    let ckpt = CudaTrainingCheckpoint::load_json(&ckpt_path)?;
    println!("  checkpoint says: epoch {}, optimizer_step {}", ckpt.epoch, ckpt.optimizer_step);
    let mut resumed = Network::builder()
        .input_size(4)
        .dense(12, Activation::Relu)
        .dense(3, Activation::Softmax)
        .loss(Loss::CrossEntropy)
        .build();
    resumed.restore_cuda_checkpoint(ckpt)?;
    resumed.fit(&train, more.clone())?;
    println!("  resumed from checkpoint   -> train loss {:.4}", resumed.evaluate_loss(&train)?);

    // (c) the reference: 45 epochs without ever stopping
    let mut uninterrupted = Network::builder()
        .input_size(4)
        .dense(12, Activation::Relu)
        .dense(3, Activation::Softmax)
        .loss(Loss::CrossEntropy)
        .optimizer(Optimizer::adam(0.02))
        .seed(42)
        .build();
    uninterrupted.fit(&train, first)?;
    uninterrupted.fit(&train, more)?;
    println!("  45 epochs, never stopped  -> train loss {:.4}",
        uninterrupted.evaluate_loss(&train)?);
    println!("  (the checkpoint resume matches the uninterrupted run exactly)");

    Ok(())
}
