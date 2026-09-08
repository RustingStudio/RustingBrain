# 5. Regression — Predicting Numbers

**You need:** chapters 1–4.

**Time:** 45 minutes.

**Full code:** [`code/05_regression.rs`](code/05_regression.rs) · **Data:** [`data/houses.csv`](data/houses.csv)

XOR answered yes/no. Now we predict a **quantity**: a house price, which could
be 20 or 450 or anything between. That's **regression**, and it changes three
things — the output activation, the loss, and (new this chapter) what you do
with the target values.

---

## 5.1 The dataset

`data/houses.csv` has 200 houses:

```
area_m2,bedrooms,age_years,distance_km,price_k
94.4,5,27,21.6,142.1
170.4,4,44,24.6,251.7
```

Four features → one price in thousands. Copy it next to `flowers.csv`:

```bash
cp /path/to/RustingBrain/tutorials/data/houses.csv data/
```

Every column is numeric here, so the reader is simpler than chapter 4's — no
label column to peel off:

```rust
fn read_numeric_csv(path: &str) -> Result<(Vec<String>, Vec<Vec<f32>>), Box<dyn Error>> {
    // ...same structure as chapter 4, but every cell parses as f32
}
```

Split the last column off as the target:

```rust
let inputs: Vec<Vec<f32>> = rows.iter().map(|r| r[..r.len()-1].to_vec()).collect();
let targets: Vec<Vec<f32>> = rows.iter().map(|r| vec![r[r.len()-1]]).collect();
```

Then the chapter 4 ritual, unchanged: shuffle, split, scale.

---

## 5.2 The new thing: scale the target too

Chapter 4 scaled the inputs. For regression you must think about the **output**
as well.

Prices here run from 35 to 446. If the network must produce a number near 446,
and its final weights start out small and random, it has a long way to travel —
and MSE on values that size produces gradients in the tens of thousands. That
tends to blow up or crawl.

So scale the target into 0..1 as well, using its own scaler:

```rust
let x_scaler = Scaler::fit(&train.inputs);     // for features
let y_scaler = Scaler::fit(&train.targets);    // for the price

for d in [&mut train, &mut validation, &mut test] {
    x_scaler.apply_inputs(d);
    y_scaler.apply_targets(d);
}
```

Same rule as before, and it matters just as much here: **both scalers are fitted
on training data only.**

Now the model learns to output values in 0..1. Which raises the obvious problem:
a prediction of `0.31` is meaningless to a human. So the scaler needs to run
backwards:

```rust
fn invert_one(&self, scaled: f32) -> f32 {
    scaled * (self.max[0] - self.min[0]) + self.min[0]
}
```

Plain algebra, undoing `(v − min) / (max − min)`. **Every prediction you show a
user goes through this.** Forgetting it is the most common regression bug, and
it's easy to spot: your model confidently predicts that a house costs 0.31.

> **Alternative:** leave the target unscaled and use a much smaller learning
> rate. It can work, but you'll fight it. Scaling the target is the simpler
> habit.

---

## 5.3 The architecture

```rust
let mut model = Network::builder()
    .input_size(4)                          // four features
    .dense(16, Activation::Relu)            // hidden
    .dense(8,  Activation::Relu)            // hidden
    .dense(1,  Activation::Linear)          // ← output: ONE number, no squashing
    .loss(Loss::Mse)                        // ← the regression loss
    .optimizer(Optimizer::adam(0.01))
    .seed(42)
    .build();
```

Three lines carry the entire difference from chapter 3:

**`.dense(1, Activation::Linear)`** — one output neuron, no activation. This is
*the* regression signature. `Linear` means the output can be any value at all.
Put a `Sigmoid` here and your model can never predict above 1.0, so it would be
permanently wrong about every expensive house.

**`.loss(Loss::Mse)`** — mean squared error, the regression loss from chapter
1.6. It punishes big misses much harder than small ones, which is what you want
when predicting quantities.

**Two hidden layers, 16 and 8** — a gentle funnel from 4 inputs to 1 output. The
first layer builds combinations of the raw features; the second combines those
into higher-level ones. Two hidden layers is a reasonable default for tabular
data; going deeper rarely helps and starts overfitting.

Parameter count:

```
4 → 16:  4×16 + 16 =  80
16 → 8: 16×8  +  8 = 136
8 → 1:   8×1  +  1 =   9
                     ───
                     225 parameters
```

225 knobs, 140 training examples. That ratio should make you slightly nervous
about overfitting — and we'll check for it in 5.6 rather than assume.

---

## 5.4 Always build a baseline first

Before training anything, answer this: **how good is "no model at all"?**

For regression the dumbest reasonable predictor is "always guess the average
training price". If your network can't beat that, it has learned nothing —
regardless of how nice its loss curve looks.

```rust
let mean_price: f32 = train.targets.iter()
    .map(|t| y_scaler.invert_one(t[0]))
    .sum::<f32>() / train.len() as f32;
```

```
BASELINE (always predict the mean, 250.5k):
  test RMSE 121.46k   MAE 104.92k
```

**That's the number to beat: 104.92k average error.** Write it down.

Skipping this step is how people ship models that are worse than an average.
It happens constantly.

---

## 5.5 Metrics you can actually explain

The training loss is MSE on *scaled* values. Here it ends around `0.0015`, which
tells you nothing about houses. For results you can report to a human, convert
back to real units.

Two standard regression metrics:

**MAE — mean absolute error.** The average size of a miss.

```
MAE = mean(|prediction − actual|)
```

Reads directly: "on average we're off by 18.8 thousand". This is the one to
show people.

**RMSE — root mean squared error.** Square the errors, average, square-root.

```
RMSE = sqrt(mean((prediction − actual)²))
```

Same units, but big misses count disproportionately. Use it when occasional
large errors are much worse than steady small ones.

**RMSE is always ≥ MAE.** The gap between them tells you about your error
distribution: close together means errors are uniform; far apart means a few
predictions are badly wrong.

```rust
fn metrics(model: &Network, data: &Dataset, ty: &Scaler) -> Result<(f32, f32), Box<dyn Error>> {
    let mut se = 0.0f32;
    let mut ae = 0.0f32;
    for (input, target) in data.inputs.iter().zip(&data.targets) {
        let pred  = ty.invert_one(model.predict(input)?[0]);   // ← back to real units
        let truth = ty.invert_one(target[0]);
        se += (pred - truth).powi(2);
        ae += (pred - truth).abs();
    }
    let n = data.len() as f32;
    Ok(((se / n).sqrt(), ae / n))
}
```

---

## 5.6 Results

```
BEFORE training: test RMSE 266.68k   MAE 230.45k
```

Worse than the baseline, as it should be — random parameters.

```
training loss (scaled units):
  epoch   1: 0.195557
  epoch  25: 0.002121
  epoch 100: 0.001756
  epoch 200: 0.002113
  epoch 400: 0.001489
```

Down by a factor of ~130, and note it is *not* perfectly monotonic — epoch 200
is slightly worse than epoch 100. That's normal. Mini-batch training is noisy
because each batch is a slightly different sample. Look at the trend, not
individual epochs.

```
AFTER training (real units, thousands):
  train        RMSE   16.53k   MAE   13.92k
  validation   RMSE   23.21k   MAE   19.40k
  test         RMSE   21.64k   MAE   18.81k
  baseline     RMSE  121.46k   MAE  104.92k

improvement over baseline: 82% lower MAE
```

### Reading this table properly

**We beat the baseline by 82%.** Average error dropped from 105k to 18.8k. The
model learned something real.

**Train (13.9k) is better than validation (19.4k) and test (18.8k).** That gap
is mild overfitting — the model is slightly better on data it has seen. With 225
parameters and 140 training rows, that's expected and acceptable. If train were
2k and test were 60k, you'd have a serious problem and chapter 7 would be your
next stop.

**Validation and test agree** (19.4k vs 18.8k). Good sign — it means our
validation set is a fair proxy for unseen data, so tuning against it is
trustworthy.

### How good is 18.8k, really?

Here's the most useful question in this chapter, and the answer is genuinely
surprising.

This dataset is synthetic. It was generated from a formula plus random noise
with a standard deviation of **18k**. That noise is not predictable — it isn't a
pattern, it's a dice roll baked into the data.

So the best *conceivable* model, one that recovered the generating formula
exactly, would still have an RMSE of about 18k.

**We got 21.6k.** We are within a few thousand of the theoretical floor. This
model is close to as good as any model could ever be on this data.

That reframes everything. If you spent the next week tuning architectures, you
could win maybe 3k — and never more. Knowing where the floor is tells you when
to stop working, and most people never work it out and tune forever.

Real datasets don't come with a noise level printed on the box. But you can
estimate the floor: how consistently can a human expert do this task? If two
appraisers value the same house 15k apart, a model with 18k error is at the
limit of the signal in the data.

### Individual predictions

```
some individual test predictions:
    actual  predicted      error
    142.1k     155.8k      13.7k
     84.7k     106.9k      22.2k
    266.9k     260.3k      -6.6k
     20.0k      47.9k      27.9k
    238.3k     232.7k      -5.6k
    352.2k     372.9k      20.7k
    187.6k     178.4k      -9.2k
    333.7k     292.0k     -41.7k
```

Always look at individual rows, not just averages. Averages hide things.

Errors go both ways (no systematic bias — good). But notice the 20.0k house
predicted at 47.9k: **139% off**. Look at the generator and you'll see why —
prices were clamped to a minimum of 20k, so that row is an artificial floor the
model has no way to know about. Edge cases at the extremes of your data are
routinely the worst predictions, because there are fewest examples there.

---

## 5.7 Things to try

1. **Don't scale the target.** Remove `y_scaler.apply_targets` and the
   `invert_one` calls. Watch the loss explode.
2. **`Sigmoid` on the output** instead of `Linear`. Predictions cap out. This is
   the single most instructive break in the chapter.
3. **One hidden layer, then three.** Does deeper help here? (Probably not — find
   out.)
4. **`.dense(256, ...)` twice.** With 140 training rows, watch the train/test gap
   widen. That's overfitting, on demand.
5. **`Optimizer::sgd(0.01)`.** More epochs needed for the same result.
6. **Drop a feature** — train on area alone. How much does each column
   contribute?
7. **Add MAPE** (mean absolute percentage error) and see the 20k house wreck it.

---

## Recap

- Regression = predicting a number. Output layer is `.dense(1, Activation::Linear)`
  with `Loss::Mse`.
- Scale the **target** as well as the features, both fitted on training data only.
- **Invert the scaling before showing any prediction to a human.**
- Build a baseline (predict the mean) before training. Beat it, or you have
  nothing.
- Report MAE and RMSE in real units — training loss in scaled units is not a
  result.
- Compare train vs validation vs test to see overfitting.
- Estimate your noise floor. It tells you when to stop tuning.
- Read individual predictions; averages hide the worst failures.

Next: the same pipeline, but choosing between categories — and a metric that
can lie to you far more convincingly than MAE ever could.

---

**Next:** [6. Classification — Choosing Between Categories](6_classification.md)
