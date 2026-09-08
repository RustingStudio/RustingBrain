use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};
use std::collections::BTreeSet;
use std::error::Error;
use std::fs;

struct Row { features: Vec<f32>, label: String }

fn read_csv(path: &str) -> Result<(Vec<String>, Vec<Row>), Box<dyn Error>> {
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
        let mut features = Vec::with_capacity(cells.len() - 1);
        for c in &cells[..cells.len() - 1] {
            features.push(c.trim().parse::<f32>()
                .map_err(|e| format!("line {}: {:?}: {e}", i + 2, c.trim()))?);
        }
        rows.push(Row { features, label: cells[cells.len()-1].trim().to_string() });
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
    fn apply(&self, d: &mut Dataset) {
        for r in d.inputs.iter_mut() {
            *r = r.iter().enumerate().map(|(i, &v)| {
                let s = self.max[i] - self.min[i];
                if s.abs() < f32::EPSILON { 0.0 } else { (v - self.min[i]) / s }
            }).collect();
        }
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}

fn accuracy(model: &Network, d: &Dataset) -> Result<f32, Box<dyn Error>> {
    let mut correct = 0;
    for (input, target) in d.inputs.iter().zip(&d.targets) {
        if argmax(&model.predict(input)?) == argmax(target) { correct += 1; }
    }
    Ok(correct as f32 / d.len() as f32)
}

fn confusion(model: &Network, d: &Dataset, n: usize) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let mut m = vec![vec![0usize; n]; n];
    for (input, target) in d.inputs.iter().zip(&d.targets) {
        m[argmax(target)][argmax(&model.predict(input)?)] += 1;
    }
    Ok(m)
}

fn main() -> Result<(), Box<dyn Error>> {
    let (_header, rows) = read_csv("data/flowers.csv")?;
    let classes: Vec<String> = rows.iter().map(|r| r.label.clone())
        .collect::<BTreeSet<_>>().into_iter().collect();
    println!("classes: {:?}\n", classes);

    let inputs: Vec<Vec<f32>> = rows.iter().map(|r| r.features.clone()).collect();
    let targets: Vec<Vec<f32>> = rows.iter()
        .map(|r| classes.iter().map(|c| if *c == r.label {1.0} else {0.0}).collect())
        .collect();

    let mut dataset = Dataset::new(inputs, targets);
    dataset.shuffle(Some(42));
    let (train, rest) = dataset.split(0.70);
    let (validation, test) = rest.split(0.50);
    let (mut train, mut validation, mut test) = (train, validation, test);

    let scaler = Scaler::fit(&train.inputs);
    for d in [&mut train, &mut validation, &mut test] { scaler.apply(d); }
    println!("split: {} train / {} validation / {} test\n",
        train.len(), validation.len(), test.len());

    let mut model = Network::builder()
        .input_size(4)
        .dense(12, Activation::Relu)
        .dense(3, Activation::Softmax)
        .loss(Loss::CrossEntropy)
        .optimizer(Optimizer::adam(0.02))
        .seed(42)
        .build();

    println!("untrained output for one flower: {:?}",
        model.predict(&test.inputs[0])?.iter().map(|v| (v*1000.0).round()/1000.0).collect::<Vec<_>>());
    println!("  (they sum to {:.4} - softmax always does)\n",
        model.predict(&test.inputs[0])?.iter().sum::<f32>());
    println!("accuracy before training: {:.1}%\n", accuracy(&model, &test)? * 100.0);

    let history = model.fit(&train, TrainConfig {
        epochs: 300, batch_size: 16, shuffle: true, seed: Some(3),
    })?;

    println!("training loss:");
    for e in [0, 24, 99, 299] { println!("  epoch {:>3}: {:.6}", e+1, history.losses[e]); }

    println!("\naccuracy:");
    println!("  train       {:>6.1}%", accuracy(&model, &train)? * 100.0);
    println!("  validation  {:>6.1}%", accuracy(&model, &validation)? * 100.0);
    println!("  test        {:>6.1}%", accuracy(&model, &test)? * 100.0);

    let m = confusion(&model, &test, classes.len())?;
    println!("\nconfusion matrix on the test set (rows = actual, cols = predicted):");
    print!("  {:<12}", "");
    for c in &classes { print!("{:>10}", c); }
    println!("{:>8}", "total");
    for (i, c) in classes.iter().enumerate() {
        print!("  {:<12}", c);
        for j in 0..classes.len() { print!("{:>10}", m[i][j]); }
        println!("{:>8}", m[i].iter().sum::<usize>());
    }

    println!("\nper-class precision / recall:");
    for (i, c) in classes.iter().enumerate() {
        let tp = m[i][i];
        let actual: usize = m[i].iter().sum();
        let predicted: usize = (0..classes.len()).map(|r| m[r][i]).sum();
        let recall = if actual == 0 { 0.0 } else { tp as f32 / actual as f32 };
        let precision = if predicted == 0 { 0.0 } else { tp as f32 / predicted as f32 };
        let f1 = if precision + recall == 0.0 { 0.0 }
                 else { 2.0 * precision * recall / (precision + recall) };
        println!("  {:<12} precision {:>6.1}%   recall {:>6.1}%   F1 {:.3}",
            c, precision * 100.0, recall * 100.0, f1);
    }

    println!("\nconfidence on test examples:");
    for i in 0..6 {
        let p = model.predict(&test.inputs[i])?;
        let pi = argmax(&p);
        let ti = argmax(&test.targets[i]);
        println!("  predicted {:<10} {:>5.1}%   actual {:<10} {}",
            classes[pi], p[pi] * 100.0, classes[ti],
            if pi == ti { "OK" } else { "WRONG" });
    }

    // The accuracy trap.
    println!("\n--- the accuracy trap ---");
    let mut imbal_inputs = Vec::new();
    let mut imbal_targets = Vec::new();
    for (input, target) in test.inputs.iter().zip(&test.targets) {
        let k = if argmax(target) == 0 { 1 } else { 12 };
        for _ in 0..k { imbal_inputs.push(input.clone()); imbal_targets.push(target.clone()); }
    }
    let imbal = Dataset::new(imbal_inputs, imbal_targets);
    let mut counts = vec![0usize; classes.len()];
    for t in &imbal.targets { counts[argmax(t)] += 1; }
    println!("an imbalanced test set: {:?} -> {:?}", classes, counts);
    let majority = argmax(&counts.iter().map(|&c| c as f32).collect::<Vec<_>>());
    let dumb = counts[majority] as f32 / imbal.len() as f32;
    println!("a model that ALWAYS says \"{}\" scores {:.1}% accuracy",
        classes[majority], dumb * 100.0);
    println!("and it is completely useless.");

    Ok(())
}
