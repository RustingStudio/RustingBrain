# 4. Working With Real Data

**You need:** chapters 1–3.

**Time:** 45 minutes.

**Full code:** [`code/04_data.rs`](code/04_data.rs) · **Data:** [`data/flowers.csv`](data/flowers.csv)

Chapter 3 had four hand-typed examples. Real data arrives in files, in the wrong
units, with text where you need numbers. This chapter builds the pipeline that
every remaining chapter uses:

```
CSV file ─▶ parse ─▶ encode labels ─▶ shuffle ─▶ split ─▶ scale ─▶ Dataset
```

Get this wrong and no architecture will save you. Two of the steps below are
places where beginners routinely, silently, ruin their results — they're marked.

---

## 4.1 The dataset

`data/flowers.csv` holds 150 measurements of three flower species:

```
petal_length,petal_width,sepal_length,sepal_width,species
5.6,1.8,6.0,3.1,borealis
1.7,0.2,5.1,3.3,rosetta
5.1,1.9,7.2,2.8,borealis
```

Four numeric **features** and one text **label**. The task: given the four
measurements, predict the species.

Copy the file into your project so the path `data/flowers.csv` works:

```bash
mkdir -p data
cp /path/to/RustingBrain/tutorials/data/flowers.csv data/
```

> This is a synthetic dataset modelled on the classic iris dataset. It's
> realistic in the way that matters: one species is easy to separate and the
> other two overlap.

---

## 4.2 Reading a CSV

This exact file shape — numeric columns, a text label last — has a one-line
reader:

```rust
use rusting_brain::Dataset;

let (dataset, classes) = Dataset::from_csv_labeled("data/flowers.csv")?;
println!("{} rows, classes {classes:?}", dataset.len());
```

`classes` comes back sorted (`["borealis", "cascade", "rosetta"]`) and each
target is already the one-hot row section 4.3 explains. If the last columns are
numbers rather than a name, `Dataset::from_csv("prices.csv", 1)` reads the file
with the last column as the target instead.

The rest of this section parses the same file by hand anyway. Not because the
loader is lacking, but because every real dataset eventually has a column the
loader does not understand, and a parser you have written once is a parser you
can change. For simple files, Rust's standard library is enough:

```rust
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
            continue;                       // tolerate a trailing newline
        }
        let cells: Vec<&str> = line.split(',').collect();
        if cells.len() != header.len() {
            return Err(format!(
                "line {}: expected {} columns, found {}",
                index + 2, header.len(), cells.len()
            ).into());
        }

        // Every column except the last is a number.
        let mut features = Vec::with_capacity(cells.len() - 1);
        for cell in &cells[..cells.len() - 1] {
            let value = cell.trim().parse::<f32>().map_err(|e| {
                format!("line {}: cannot parse {:?} as a number: {e}",
                        index + 2, cell.trim())
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
```

The error handling is the point. `index + 2` converts a zero-based data index
into the line number you'd see in a text editor (`+1` for zero-based, `+1` for
the header), so a bad file tells you *exactly* where it's bad:

```
line 84: cannot parse "N/A" as a number: invalid float literal
```

That beats a panic thirty lines later.

> **Real-world CSVs** have quoted fields containing commas, missing values,
> and inconsistent encodings. When `split(',')` stops being enough, add the
> `csv` crate (`cargo add csv`). The rest of this chapter is unchanged — you're
> only swapping the reader.

---

## 4.3 Turning labels into numbers

A network outputs numbers, so `"rosetta"` has to become numbers. First collect
the distinct labels:

```rust
use std::collections::BTreeSet;

let classes: Vec<String> = rows
    .iter()
    .map(|r| r.label.clone())
    .collect::<BTreeSet<_>>()   // deduplicate...
    .into_iter()
    .collect();                 // ...and BTreeSet keeps them sorted
```

```
classes: ["borealis", "rosetta", "valentia"]
```

`BTreeSet` matters: it gives a **deterministic, sorted** order. With a `HashSet`
the order changes between runs, so class 0 might be `borealis` today and
`valentia` tomorrow — and a saved model would decode its own predictions wrongly.

### One-hot encoding

The obvious idea is `borealis = 0, rosetta = 1, valentia = 2`. **Don't.** That
tells the network `valentia` is three times `rosetta`, and that `rosetta` sits
between the other two. None of that is true, and the model will act on it.

Instead give each class its own output neuron, and mark the right one with a 1:

```rust
let one_hot: Vec<f32> = classes
    .iter()
    .map(|c| if *c == row.label { 1.0 } else { 0.0 })
    .collect();
```

```
one-hot encoding:
  borealis   -> [1.0, 0.0, 0.0]
  rosetta    -> [0.0, 1.0, 0.0]
  valentia   -> [0.0, 0.0, 1.0]
```

Now no class is "bigger" than another — they're just three separate answers.
This is why chapter 6's classifier will have exactly 3 output neurons and a
`Softmax` activation: the network outputs a probability per class, and the
one-hot target says which one should have been 1.

> **Two classes only?** You can one-hot into 2 outputs with softmax, or use a
> single sigmoid output with `0.0`/`1.0` targets like XOR. The single-output
> version is simpler and slightly cheaper.

---

## 4.4 Building the `Dataset`

```rust
use rusting_brain::Dataset;

let inputs: Vec<Vec<f32>> = rows.iter().map(|r| r.features.clone()).collect();
let targets: Vec<Vec<f32>> = rows
    .iter()
    .map(|r| classes.iter()
        .map(|c| if *c == r.label { 1.0 } else { 0.0 })
        .collect())
    .collect();

let mut dataset = Dataset::new(inputs, targets);
```

`Dataset` is deliberately plain — two public `Vec`s that line up by index:

```rust
pub struct Dataset {
    pub inputs:  Vec<Vec<f32>>,
    pub targets: Vec<Vec<f32>>,
}
```

They're public, so you can inspect and modify them directly. We'll use that in
4.6 to scale in place.

---

## 4.5 Shuffle, then split

### ⚠️ Gotcha #1: `split` does not shuffle

`Dataset::split(ratio)` takes the **first** `ratio` of the rows and leaves the
rest. It does not reorder anything.

So if your file is sorted by species — and exported data very often is — then
`split(0.7)` hands you a training set with no `valentia` in it at all, and a
test set that is *entirely* `valentia`. Your model scores 0% and you have no
idea why.

**Always shuffle first:**

```rust
dataset.shuffle(Some(42));      // Some(seed) = reproducible; None = truly random
```

### Making three splits

Chapter 1.10 called for three sets. `split` gives two at a time, so call it
twice:

```rust
let (train, rest) = dataset.split(0.70);   // 70% train, 30% left over
let (validation, test) = rest.split(0.50); // halve the leftovers: 15% / 15%
```

```
split: 105 train / 23 validation / 22 test
```

The three sets and their jobs:

| Set | Share | What it's for | How often you look |
|-----|-------|---------------|--------------------|
| **train** | 70% | the model learns from this | constantly |
| **validation** | 15% | tuning: layer sizes, learning rate, when to stop | every epoch |
| **test** | 15% | the final honest score | **once**, at the very end |

Why validation *and* test? Because if you tune your architecture until the
validation score is great, you've partly fitted your *decisions* to the
validation set. It's no longer an unbiased estimate. The test set is the one you
never touched, so it's the one you can trust. Chapter 8 covers this properly.

### Checking the balance

```rust
println!("  {:<12} {:?}", "train", class_counts(&train, &classes));
```

```
class balance per split (want them similar):
  train        [39, 30, 36]
  validation   [3, 10, 10]
  test         [8, 10, 4]
```

Training looks fine — roughly even thirds. But **validation has only 3
`borealis`**. Random splitting of a small dataset does this.

It matters: with 3 examples of a class, your validation accuracy for that class
moves in jumps of 33 percentage points. One lucky prediction looks like a huge
improvement. Always print this table — if a class is nearly missing from a
split, treat that split's numbers with suspicion.

The proper fix is a **stratified split**: split each class separately and
concatenate, so every split has the same class proportions. `Dataset` does it
for you:

```rust
let (train, rest) = dataset.split_stratified(0.7);
let (validation, test) = rest.split_stratified(0.5);
```

Same call shape as `split`, and the class balance above stops jumping around.
It reads the class off the target — the largest entry of a one-hot row, or the
value itself for a single-column target — so it applies to classification only;
a continuous target has no class and stays on plain `split`.

Shuffle before either one. `split_stratified` keeps the row order it is given
inside each class, so an unshuffled dataset splits into whatever order the
loader produced.

---

## 4.6 Scaling the features

Look at the raw ranges:

```
raw feature ranges:
  petal_length     1.1 ..   6.9
  petal_width      0.1 ..   2.5
  sepal_length     3.8 ..   7.9
  sepal_width      2.2 ..   4.2
```

`petal_width` spans 2.4 units; `petal_length` spans 5.8. That's a mild
difference and the model would cope. But imagine a house dataset with
`bedrooms` (1–5) and `price` (50,000–2,000,000). The price feature is
400,000× larger, so it produces vastly larger gradients, so it dominates every
update — and `bedrooms` is effectively ignored no matter how predictive it is.

The fix is **min-max scaling**: squash every feature into 0..1.

```
scaled = (value − min) / (max − min)
```

```rust
struct Scaler {
    min: Vec<f32>,
    max: Vec<f32>,
}

impl Scaler {
    /// Learn the range of every column.
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
                // A constant column would divide by zero. Map it to 0.0.
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
```

### ⚠️ Gotcha #2: fit the scaler on training data only

This is the one that quietly inflates people's results.

```rust
let scaler = Scaler::fit(&train.inputs);   // ← TRAIN ONLY. Never the full dataset.

scaler.apply_to(&mut train);
scaler.apply_to(&mut validation);          // transform with the TRAINING min/max
scaler.apply_to(&mut test);
```

If you compute min and max over the whole dataset, then the minimum and maximum
of your *test* set have influenced how your *training* data was scaled.
Information leaked backwards from data the model was supposed to have never
seen. Your test score comes out better than the truth, and you find out in
production.

This is called **data leakage**, it's one of the most common real-world ML bugs,
and it is completely invisible — nothing errors, the numbers just lie.

You can see the separation in the output:

```
scaler learned from training data:
  min: [1.1, 0.1, 4.1, 2.4]
  max: [6.9, 2.5, 7.9, 4.1]
```

Compare `sepal_length`: the scaler says `4.1 .. 7.9`, but the full dataset spans
`3.8 .. 7.9`. The 3.8 lives in validation or test — and the scaler correctly
knows nothing about it. That's the proof it worked.

A consequence: scaled validation/test values can land slightly outside 0..1
(a value below the training minimum goes negative). That is fine and expected.
Don't "fix" it by clamping.

### The result

```
first 3 training examples after scaling:
  [0.71, 0.75, 0.71, 0.47] -> [1.0, 0.0, 0.0]  (borealis)
  [0.41, 0.54, 0.26, 0.29] -> [0.0, 0.0, 1.0]  (valentia)
  [0.1, 0.04, 0.26, 0.53]  -> [0.0, 1.0, 0.0]  (rosetta)
```

Every feature now lives on the same scale, and the labels are one-hot vectors.
**This is exactly the shape a network wants.**

> **Save your scaler.** When you deploy the model in chapter 9, incoming data
> must be scaled with the *same* min and max. A model fed unscaled input returns
> confident nonsense. The scaler is part of your model, even though it isn't
> part of the `Network`.

### The other scaler: standardisation

The common alternative maps each feature to mean 0, standard deviation 1:

```
scaled = (value − mean) / std_dev
```

| | Min-max | Standardisation |
|---|---|---|
| Output range | exactly 0..1 on train | unbounded, mostly −3..3 |
| Outliers | one huge value squashes everything else | handled much better |
| Use when | bounded data (pixels, percentages) | roughly bell-shaped data |

Min-max is a fine default. Switch to standardisation when you have outliers.

`Dataset` has standardisation built in, and it hands back the statistics it
measured so later data goes through the same transform:

```rust
let statistics = train.standardize();     // train is now mean 0, deviation 1
statistics.apply(&mut validation);        // the SAME numbers, not validation's own
statistics.apply(&mut test);

let scaled = statistics.apply_row(&[5.6, 1.8, 6.0, 3.1]);   // one new sample
```

`Standardizer` is `serde`-serializable, so it saves next to the model — which
is the point of the box above, and chapter 9 does exactly that. A constant
column has no deviation to divide by; it is left alone rather than turned into
`NaN`.

---

## 4.6b Files that are not CSV

CSV is where tutorials start and rarely where real data lives. Four more
formats read straight into a `Dataset`:

```rust
// NumPy arrays, which is how data leaves PyTorch, TensorFlow and scikit-learn
let mut dataset = Dataset::from_npy("x.npy", "y.npy")?;
dataset.one_hot_targets(10)?;             // y held class indices, not one-hot rows

// MNIST and everything shaped like it (gunzip the archives first)
let mut train = Dataset::from_idx("train-images-idx3-ubyte", "train-labels-idx1-ubyte")?;
train.one_hot_targets(10)?;

// A directory per class, images inside it  (--features images)
let (mut images, classes) = Dataset::from_image_folder("photos", 32, 32, true)?;
```

`from_npy` reads `float32`, `float64`, `int32`, `int64` and `uint8`;
`from_idx` reads the IDX type codes and scales `u8` pixels to `0.0..=1.0`;
`from_image_folder` decodes PNG and JPEG in parallel, resizes, and flattens each
image into one row with a one-hot target and the class names beside it.

Images get one augmentation for free:

```rust
let (mut train, test) = images.split_stratified(0.8);
train.flip_horizontal(32)?;               // the image width, in pixels
```

That mirrors every image and appends it with its label — twice the training
data for one decode. Run it on the training split **only**: augment before the
split and an image and its mirror land on opposite sides, which turns the test
score into a memory test.

---

## 4.7 Batching

```rust
let batches = train.batches(32);
```

```
train.batches(32) -> 4 batches; sizes [32, 32, 32, 9]
```

105 rows in groups of 32 gives three full batches and a final short one of 9.
The last batch being smaller is normal — `batches` never pads or drops rows.

You rarely call this yourself; `fit` does it for you from
`TrainConfig::batch_size`. You'll want it directly in chapter 7, when we write
the training loop by hand.

---

## 4.8 A reusable checklist

Every dataset, every time:

1. **Read** the file; fail loudly with a line number.
2. **Encode** text labels as one-hot vectors, with a deterministic class order.
3. **Shuffle** — before splitting, always.
4. **Split** into train / validation / test — `split_stratified` for classes.
5. **Check** class balance in each split.
6. **Scale** features, fitting on **train only** (`standardize` on train,
   `apply` on the rest).
7. **Keep** the scaler; you need it at prediction time.

---

## 4.9 Things to try

1. Comment out `dataset.shuffle(Some(42))` and print the class balance. Then
   sort the CSV by species first and try again — watch a split go to zero.
2. Fit the scaler on the whole dataset instead of train, and diff the min/max.
3. Write `Scaler::fit_standard` using mean and standard deviation.
4. Implement a stratified split and compare the balance table.
5. Break the CSV: put `N/A` in a cell, delete a comma. Check the errors are
   useful.

---

## Recap

- `Dataset::from_csv_labeled`, `from_csv`, `from_npy`, `from_idx` and
  `from_image_folder` read the common formats; the standard library is enough
  for anything they do not, and a hand-written parser should report line numbers.
- One-hot encode categorical labels — never integer codes.
- Use a sorted, deterministic class order.
- **Shuffle before splitting.** `split` slices, it does not shuffle;
  `split_stratified` keeps each class's share on both sides.
- Three splits: train (learn), validation (tune), test (final honest score).
- Check class balance; tiny per-class counts make a split's metrics unreliable.
- Scale your features, and **fit the scaler on training data only** — otherwise
  you leak information and your scores lie.
- The scaler is part of your model. Save it with the weights.

Data is ready. Next we point a network at it and predict an actual number.

---

**Next:** [5. Regression — Predicting Numbers](5_regression.md)
