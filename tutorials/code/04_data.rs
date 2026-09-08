use rusting_brain::Dataset;
use std::collections::BTreeSet;
use std::error::Error;
use std::fs;

struct Row {
    features: Vec<f32>,
    label: String,
}

fn read_csv(path: &str) -> Result<(Vec<String>, Vec<Row>), Box<dyn Error>> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();

    let header: Vec<String> = lines
        .next()
        .ok_or("file is empty")?
        .split(',')
        .map(|name| name.trim().to_string())
        .collect();

    let mut rows = Vec::new();
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != header.len() {
            return Err(format!(
                "line {}: expected {} columns, found {}",
                index + 2,
                header.len(),
                cells.len()
            )
            .into());
        }

        let mut features = Vec::with_capacity(cells.len() - 1);
        for cell in &cells[..cells.len() - 1] {
            let value = cell.trim().parse::<f32>().map_err(|e| {
                format!("line {}: cannot parse {:?} as a number: {e}", index + 2, cell.trim())
            })?;
            features.push(value);
        }

        rows.push(Row {
            features,
            label: cells[cells.len() - 1].trim().to_string(),
        });
    }
    Ok((header, rows))
}

struct Scaler {
    min: Vec<f32>,
    max: Vec<f32>,
}

impl Scaler {
    fn fit(rows: &[Vec<f32>]) -> Self {
        let width = rows[0].len();
        let mut min = vec![f32::INFINITY; width];
        let mut max = vec![f32::NEG_INFINITY; width];
        for row in rows {
            for (i, &v) in row.iter().enumerate() {
                min[i] = min[i].min(v);
                max[i] = max[i].max(v);
            }
        }
        Self { min, max }
    }

    fn transform(&self, row: &[f32]) -> Vec<f32> {
        row.iter()
            .enumerate()
            .map(|(i, &v)| {
                let span = self.max[i] - self.min[i];
                if span.abs() < f32::EPSILON { 0.0 } else { (v - self.min[i]) / span }
            })
            .collect()
    }

    fn apply_to(&self, dataset: &mut Dataset) {
        for row in dataset.inputs.iter_mut() {
            *row = self.transform(row);
        }
    }
}

fn class_counts(dataset: &Dataset, classes: &[String]) -> Vec<usize> {
    let mut counts = vec![0usize; classes.len()];
    for target in &dataset.targets {
        if let Some(i) = target.iter().position(|&v| v == 1.0) {
            counts[i] += 1;
        }
    }
    counts
}

fn main() -> Result<(), Box<dyn Error>> {
    // ---- 1. read ----
    let (header, rows) = read_csv("data/flowers.csv")?;
    println!("columns: {:?}", header);
    println!("rows:    {}", rows.len());

    println!("\nfirst 3 raw rows:");
    for row in rows.iter().take(3) {
        println!("  {:?} -> {}", row.features, row.label);
    }

    // ---- 2. find the classes ----
    let classes: Vec<String> = rows
        .iter()
        .map(|r| r.label.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    println!("\nclasses: {:?}", classes);

    println!("\nraw feature ranges:");
    for i in 0..header.len() - 1 {
        let vals: Vec<f32> = rows.iter().map(|r| r.features[i]).collect();
        let mn = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("  {:<14} {:>5.1} .. {:>5.1}", header[i], mn, mx);
    }

    // ---- 3. build a Dataset (raw features, one-hot targets) ----
    let inputs: Vec<Vec<f32>> = rows.iter().map(|r| r.features.clone()).collect();
    let targets: Vec<Vec<f32>> = rows
        .iter()
        .map(|r| classes.iter().map(|c| if *c == r.label { 1.0 } else { 0.0 }).collect())
        .collect();
    let mut dataset = Dataset::new(inputs, targets);

    println!("\none-hot encoding:");
    for (i, class) in classes.iter().enumerate() {
        let mut v = vec![0.0; classes.len()];
        v[i] = 1.0;
        println!("  {:<10} -> {:?}", class, v);
    }

    // ---- 4. shuffle, THEN split ----
    dataset.shuffle(Some(42));

    let (train_pool, rest) = dataset.split(0.70);
    let (validation, test) = rest.split(0.50);
    let mut train = train_pool;
    let mut validation = validation;
    let mut test = test;

    println!("\nsplit: {} train / {} validation / {} test",
        train.len(), validation.len(), test.len());

    println!("\nclass balance per split (want them similar):");
    println!("  {:<12} {:?}", "train", class_counts(&train, &classes));
    println!("  {:<12} {:?}", "validation", class_counts(&validation, &classes));
    println!("  {:<12} {:?}", "test", class_counts(&test, &classes));

    // ---- 5. scale, fitting on TRAIN only ----
    let scaler = Scaler::fit(&train.inputs);
    println!("\nscaler learned from training data:");
    println!("  min: {:?}", scaler.min);
    println!("  max: {:?}", scaler.max);

    scaler.apply_to(&mut train);
    scaler.apply_to(&mut validation);
    scaler.apply_to(&mut test);

    println!("\nfirst 3 training examples after scaling:");
    for i in 0..3 {
        let pretty: Vec<f32> = train.inputs[i].iter().map(|v| (v * 100.0).round() / 100.0).collect();
        let name = classes.iter().zip(&train.targets[i])
            .find(|(_, &t)| t == 1.0).map(|(c, _)| c.as_str()).unwrap_or("?");
        println!("  {:?} -> {:?}  ({})", pretty, train.targets[i], name);
    }

    // ---- 6. batching ----
    let batches = train.batches(32);
    println!("\ntrain.batches(32) -> {} batches; sizes {:?}",
        batches.len(), batches.iter().map(|b| b.inputs.len()).collect::<Vec<_>>());

    Ok(())
}
