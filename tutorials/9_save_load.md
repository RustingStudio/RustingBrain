# 9. Saving, Loading, and Using a Model

**You need:** chapters 1–8.

**Time:** 40 minutes.

**Full code:** [`code/09_save_load.rs`](code/09_save_load.rs)

Everything so far has happened inside one program: train, measure, exit. The
model died with the process.

A model that only exists during training is useless. The whole point is to train
**once** and then use those weights thousands of times — in a web server, a CLI
tool, a game, a background job. That is called **instancing** or **serving** a
model, and this chapter covers the whole round trip:

```
train  →  save to disk  →  (program exits)  →  load  →  predict on new data
```

The single most common beginner bug in this chapter is not about weights at all.
It's forgetting that **the scaler is part of your model**. We're going to prove
that with a prediction that is 100% confident and completely wrong.

---

## 9.1 What `save_json` actually stores

RustingBrain writes a model as plain JSON:

```rust
model.save_json("flowers.json")?;
```

That's it. No custom binary format, no version-locked pickle file you can never
open again. You can read it in a text editor:

```
first 260 characters of the model file:
{
  "version": 1,
  "input_size": 4,
  "layers": [
    {
      "weights": {
        "rows": 12,
        "cols": 4,
        "data": [
          -0.5184369,
          0.03755772,
          -0.3553377,
          0.06042254,
          2.3527298,
          1.775454
  ...
```

Our 4→12→3 flower classifier is **2680 bytes**. That's the entire model.

Here is the honest inventory — what is inside that file and what is not:

| Stored in the JSON | **Not** stored in the JSON |
|---|---|
| every weight and bias | your scaler's min/max |
| each layer's size | your class names |
| each layer's activation | the optimizer and its state (`m`, `v` for Adam) |
| the loss function | the learning rate you used |
| the input size | which columns of your CSV were features |

The left column is the *model*. The right column is everything you need to
**use** the model — and losing it makes the weights meaningless.

> **The mental model:** `save_json` saves the brain. It does not save the eyes.
> The scaler is how your model sees the world; without the identical scaler,
> the same flower looks like a different flower.

---

## 9.2 Saving the preprocessing alongside it

So we save a second, tiny file next to the model. It needs the scaler bounds
(from chapter 4) and the class names in the exact order used to build the
one-hot targets (from chapter 6).

```rust
/// Everything a saved model needs besides the weights.
struct Preprocessing {
    min: Vec<f32>,
    max: Vec<f32>,
    classes: Vec<String>,
}
```

`fit` and `transform` are exactly the scaler from chapter 4 — fitted on the
**training set only**, no leakage:

```rust
impl Preprocessing {
    fn fit(rows: &[Vec<f32>], classes: Vec<String>) -> Self {
        let w = rows[0].len();
        let mut min = vec![f32::INFINITY; w];
        let mut max = vec![f32::NEG_INFINITY; w];
        for r in rows {
            for (i, &v) in r.iter().enumerate() {
                min[i] = min[i].min(v);
                max[i] = max[i].max(v);
            }
        }
        Self { min, max, classes }
    }

    fn transform(&self, row: &[f32]) -> Vec<f32> {
        row.iter().enumerate().map(|(i, &v)| {
            let s = self.max[i] - self.min[i];
            if s.abs() < f32::EPSILON { 0.0 } else { (v - self.min[i]) / s }
        }).collect()
    }
}
```

> **If you standardise instead of min-max scaling**, half of this is already
> written: `Dataset::standardize` returns a `Standardizer` that is
> `serde`-serializable, so it saves as JSON in two lines and scales a later row
> with `apply_row`:
>
> ```rust
> let statistics = train.standardize();
> std::fs::write("scaler.json", serde_json::to_string(&statistics)?)?;
> // later, beside the model:
> let statistics: Standardizer = serde_json::from_str(&std::fs::read_to_string("scaler.json")?)?;
> let scaled = statistics.apply_row(&raw_row);
> ```
>
> The class names still have to travel with it, which is what the struct below
> is for.

For the file format we'll use one line per field — no serialisation crate
needed, and you can read and edit it by hand:

```rust
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
```

The whole file is **84 bytes**:

```
version 1
min 1.1,0.1,4.1,2.4
max 6.9,2.5,7.9,4.1
classes borealis,rosetta,valentia
```

Three things worth copying from this design:

1. **`version 1` on the first line.** In six months you will change the format
   — maybe you switch to standardisation and need `mean`/`std` instead. A
   version number lets a future loader say "I can't read this" instead of
   silently producing garbage numbers.
2. **Loading validates.** The loader refuses unknown versions, unknown keys,
   mismatched `min`/`max` lengths and empty class lists:

   ```rust
   if min.is_empty() || min.len() != max.len() || classes.is_empty() {
       return Err("incomplete preprocessing file".into());
   }
   ```
   A corrupt sidecar should fail loudly at startup, not produce quietly wrong
   predictions for a year.
3. **The class order is stored explicitly.** Chapter 4 used a `BTreeSet` so the
   order is deterministic — but "deterministic" isn't "guaranteed to be the same
   as last week's CSV". Saving the list means output index 0 always means what
   it meant at training time.

Save both together:

```rust
model.save_json(&model_path)?;
prep.save(prep_path.to_str().unwrap())?;
```

```
saved:
  /tmp/rb_tutorial/flowers.json  (2680 bytes)
  /tmp/rb_tutorial/flowers.prep  (84 bytes)
```

> **Treat them as one unit.** Same directory, same base name, copied together,
> deployed together. If you ever ship `flowers.json` without `flowers.prep`,
> you have shipped a broken model that still runs.

---

## 9.3 Loading it back

In a real project this happens in a completely different program. Loading is
two lines:

```rust
let loaded = Network::load_json(&model_path)?;
let loaded_prep = Preprocessing::load(prep_path.to_str().unwrap())?;
```

```
--- new process would start here ---

loaded model: 4 inputs, 3 outputs, 2 layers
loaded classes: ["borealis", "rosetta", "valentia"]
```

No architecture code needed. You don't rebuild the network with
`Network::builder()` and hope the shape matches — the shape is in the file.

### Is it really the same model?

Do not take this on faith. Measure it, both ways:

```rust
let acc_after = accuracy(&loaded, &test)?;
println!("loaded model test accuracy: {:.1}%", acc_after * 100.0);
println!("identical to before saving: {}", (acc_before - acc_after).abs() < 1e-6);

let mut max_diff = 0.0f32;
for input in &test.inputs {
    for (a, b) in model.predict(input)?.iter().zip(&loaded.predict(input)?) {
        max_diff = max_diff.max((a - b).abs());
    }
}
println!("largest difference in any output value: {:e}", max_diff);
```

```
loaded model test accuracy: 95.5%
identical to before saving: true
largest difference in any output value: 0e0
```

Accuracy is a weak check — two different models can score 95.5% by coincidence.
The strong check is the second one: across every test row and every output
neuron, the largest disagreement is **exactly zero**. The round trip is
bit-exact, because the JSON stores full `f32` precision.

Make this comparison a test in your own projects. It's four lines and it catches
an entire class of "the deployed model behaves differently" bugs.

---

## 9.4 Instancing: predicting on brand new data

This is the payoff. Someone hands you four raw measurements of a flower nobody
has ever measured before. The pipeline is always the same three steps:

```rust
let scaled = loaded_prep.transform(raw);   // 1. same scaling as training
let probs  = loaded.predict(&scaled)?;     // 2. forward pass
let best   = argmax(&probs);               // 3. interpret the outputs
println!("-> {} at {:.1}%", loaded_prep.classes[best], probs[best] * 100.0);
```

```
--- predicting on brand new, unscaled measurements ---

  [1.4, 0.2, 5.0, 3.4]  (looks like rosetta)
     -> rosetta at 100.0% confidence
     all: rosetta 100.0%, valentia 0.0%, borealis 0.0%
  [4.3, 1.3, 5.9, 2.8]  (looks like valentia)
     -> valentia at 100.0% confidence
     all: valentia 100.0%, borealis 0.0%, rosetta 0.0%
  [5.7, 2.1, 6.7, 3.0]  (looks like borealis)
     -> borealis at 100.0% confidence
     all: borealis 100.0%, valentia 0.0%, rosetta 0.0%
  [4.9, 1.7, 6.2, 2.9]  (ambiguous)
     -> valentia at 68.7% confidence
     all: valentia 68.7%, borealis 31.3%, rosetta 0.0%
```

Note that fourth flower. Its petals sit right on the boundary between valentia
and borealis — exactly the overlap chapter 6's confusion matrix found — and the
model reports `68.7% / 31.3%` instead of pretending to be sure.

**Ship the full distribution, not just the winner.** A caller that only sees
`"valentia"` cannot tell a 100% answer from a 69% coin-flip. Chapter 6 showed
that our one misclassification was also our one low-confidence prediction. A
threshold like "below 80% → ask a human" is free accuracy, and you only get it
if you keep the probabilities.

> **`argmax` needs no threshold with Softmax.** The outputs already sum to 1, so
> the largest one is the prediction. Thresholds are for `Sigmoid` binary
> outputs (chapter 3), where 0.5 is a choice you make.

---

## 9.5 The bug this chapter exists for

Here's what happens if you skip `transform` and feed raw measurements straight
in — the model still runs, no error, no warning:

```rust
let raw = vec![1.4f32, 0.2, 5.0, 3.4];
let right = loaded.predict(&loaded_prep.transform(&raw))?;
let wrong = loaded.predict(&raw)?;
```

```
--- what happens if you forget to scale ---

  scaled correctly: rosetta at 100.0%
  raw, unscaled:    borealis at 100.0%
```

Same flower. Same model. **Confidently, completely wrong.**

Why: the model has only ever seen inputs in `0.0 – 1.0`. A raw petal length of
`5.0` is not "a bit big" to it, it's five times the largest value in its entire
universe. The hidden neurons saturate and the answer is meaningless — but
`Softmax` still normalises to a tidy 100%, so nothing looks broken.

This is the reason data pipelines are shipped as code, not as notebook cells:

- ❌ scaling done inline in your training script, retyped from memory at
  inference time
- ✅ one `Preprocessing` type, `transform` called in exactly one place, saved
  and loaded from a file

If you take one habit from this chapter: **the only path from raw input to
`predict` goes through `transform`.** Make it structurally impossible to skip —
wrap them together:

```rust
fn classify(&self, raw: &[f32]) -> Result<(String, f32), NetworkError> {
    let probs = self.model.predict(&self.prep.transform(raw))?;
    let i = argmax(&probs);
    Ok((self.prep.classes[i].clone(), probs[i]))
}
```

Now nobody — including you at 2am — can call `predict` without the scaler.

---

## 9.6 Batch inference

`predict()` handles one row. When you have many, `predict_batch()` does the
whole set as matrix operations instead of one small multiplication at a time:

```rust
let batched = loaded.predict_batch(&many)?;
```

```
--- batch inference ---

  20000 predictions
  predict() in a loop: 2.431684ms
  predict_batch():     1.514791ms
  speedup: 1.61x
  identical results: true
```

Two things to notice:

- **`identical results: true`.** Batching is a performance optimisation, not a
  different computation. Same numbers, fewer allocations and better cache use.
- **The speedup depends on the model.** 1.61× here on a tiny 4→12→3 network
  where per-call overhead is most of the cost; on wider layers the gap grows.
  Your numbers will differ from these — timings vary from run to run, and by
  machine and build profile. Which brings us to the most important performance point in this
  chapter:

> **Always benchmark inference with `cargo run --release`.** Debug builds run
> unoptimised arithmetic and are routinely 10× slower or worse. If your model
> "feels too slow", check the build profile before you change a single line of
> code.

Rule of thumb: one item at a time (a web request) → `predict`. A file, a
queue, a table → `predict_batch`.

---

## 9.7 Gotcha: `save_json` does not save the optimizer

Load a model and continue training it, and you get a surprise.

Here is a flower classifier stopped early, at 25 epochs, so it still has room to
improve:

```
after 25 epochs: test accuracy 100.0%, train loss 0.0684
```

Save it with `save_json`, load it in a fresh process, and train 20 more epochs:

```
--- resuming from save_json (weights only) ---
  loaded: 100.0%, train loss 0.0684
  after 20 more epochs: 100.0%, train loss 0.0660
```

Twenty epochs of training moved the loss by 0.0024. Something is wrong.

The weights loaded perfectly — that's why the starting loss matches exactly. But
`load_json` has no optimizer to restore, so it falls back to a default
`Optimizer::sgd(0.01)`. Our model was trained with `Optimizer::adam(0.02)`.
Plain SGD at a fifth of the learning rate, with none of Adam's accumulated
momentum, is a nudge rather than training.

Worse, it fails *silently*. Nothing errors. The model just quietly stops
improving, and you go hunting through your data for a problem that isn't there.

> **`save_json` is for inference.** If all you do with the loaded model is
> `predict`, none of this matters — the optimizer is never touched. This gotcha
> only bites when you want to *resume training*.

### The fix: a full checkpoint

For resuming, RustingBrain has a second format that stores the complete training
state. It's called `CudaTrainingCheckpoint`, but ignore the name — it's ordinary
CPU-side state, it needs no GPU and no feature flag:

```rust
// saving: the 25 is the epoch you stopped at, Some(3) the shuffle seed
model.cuda_checkpoint(25, Some(3)).save_json("resume_full.json")?;
```

```
  weights only   : 2701 bytes
  full checkpoint: 7589 bytes
```

Nearly 3× larger, and the extra bytes are exactly what was missing — the
optimizer, its step counter, and Adam's two moment tensors (`m` and `v`, from
chapter 7) for every weight and bias.

To restore, build a network of the right shape and hand it the checkpoint:

```rust
let ckpt = CudaTrainingCheckpoint::load_json("resume_full.json")?;

let mut model = Network::builder()
    .input_size(4)
    .dense(12, Activation::Relu)
    .dense(3, Activation::Softmax)
    .loss(Loss::CrossEntropy)
    .build();

model.restore_cuda_checkpoint(ckpt)?;   // weights + optimizer + moments
model.fit(&train, config)?;             // genuinely resumes
```

Now the same 20 epochs:

```
--- resuming from a full checkpoint ---
  checkpoint says: epoch 25, optimizer_step 175,
                   optimizer Adam { learning_rate: 0.02, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }
  restored: 100.0%, train loss 0.0684
  after 20 more epochs: 100.0%, train loss 0.0272
```

0.0684 → **0.0272**, versus 0.0660 the other way. That is real training.

And here is the proof that the resume is genuinely seamless. Train the same
model for all 45 epochs in one uninterrupted run, never saving anything:

```
--- reference: 45 epochs in one uninterrupted run ---
  100.0%, train loss 0.0272
```

**Identical.** Stopping, saving a checkpoint, exiting the process, and resuming
produced exactly the same model as never stopping at all. That is what a correct
checkpoint means, and it's worth testing in your own projects the same way:
compare a resumed run against an uninterrupted one and expect the same number.

`restore_cuda_checkpoint` also validates before it touches your model — layer
shapes, moment tensor sizes, and a check that no moment value is `NaN` or
infinite. A corrupt checkpoint returns an error instead of half-restoring.

### Which one to save

| You want to… | Save |
|---|---|
| deploy for inference | `save_json` + your `Preprocessing` sidecar |
| run the model outside Rust | `save_onnx` + your `Preprocessing` sidecar |
| stop and resume a long training run | `cuda_checkpoint(...).save_json(...)` |
| survive a crash mid-training | a checkpoint every N epochs |
| keep the best model during early stopping (ch. 7) | `model.clone()` in memory |

### Saving for a different runtime: ONNX

`save_json` is RustingBrain's own format — nothing else reads it. ONNX is the
format everything else reads:

```rust
model.save_onnx("model.onnx")?;
```

That writes the network as a graph: one `Gemm` node per layer and one
activation node after it, with the weights as initializers. ONNX Runtime,
TensorRT, or a browser through onnxruntime-web will all load it, and so will
this crate's own reader:

```bash
cargo run --example onnx_inference --features onnx -- model.onnx 1,4 5.1,3.5,1.4,0.2
```

Three things to know:

- **The input is a `[batch, input_size]` tensor named `input`.** The batch
  dimension is symbolic, so one row and a thousand both load.
- **Writing needs no feature flag; reading needs `--features onnx`.** Export is
  a few hundred lines of protobuf; import pulls in `tract-onnx`.
- **Dense networks only.** A transformer's RMSNorm, SwiGLU and MoE routing have
  no equally short representation — `TransformerLm::save_bin` is how you move
  one of those.

The `Preprocessing` sidecar still travels with it. ONNX carries the graph, not
your scaler.

A common setup does both: a checkpoint every 10 epochs while training, then one
final `save_json` for the thing you actually ship. The checkpoints are working
files you delete afterwards; the small JSON is the artifact.

> **Practical advice:** save your training configuration next to the model too —
> learning rate, optimizer, epochs, batch size. Your `Preprocessing` sidecar is
> already the right place for it. Future you will want to know how this file was
> made.

---

## Try it yourself

1. Save a model, delete the `.prep` file, and try to make a prediction. What do
   you have to guess at, and how wrong is the answer?
2. Add `learning_rate`, `optimizer` and `epochs` lines to the `Preprocessing`
   file. Bump the version to 2 and make the loader accept both 1 and 2.
3. Write a test that trains for 1 epoch, saves, loads, and asserts the max
   output difference is 0. Put it in your CI.
4. Build a tiny CLI: `cargo run -- 1.4 0.2 5.0 3.4` loads the model and prints
   the class. Read the numbers from `std::env::args()`.
5. Time `predict` vs `predict_batch` on a 64→64→3 network. Does the speedup
   grow?
6. Corrupt the `.prep` file (delete the `max` line) and confirm the loader
   rejects it instead of predicting nonsense.
7. Train 25 epochs, save a checkpoint, restore it in a second program and
   train 20 more. Compare the final loss against one 45-epoch run — they should
   match exactly.

---

## Recap

- `save_json` / `load_json` store the **complete architecture and weights** as
  readable JSON — 2680 bytes for our classifier.
- They do **not** store your scaler, class names, or optimizer. Save those
  yourself.
- Keep a small versioned sidecar file next to the model and validate it on load.
- Verify the round trip by comparing outputs, not just accuracy. Ours matched to
  `0e0` — bit-exact.
- Inference is always: `transform` → `predict` → interpret.
- **Forgetting to scale doesn't error — it lies.** Our rosetta became a
  confident borealis. Wrap the scaler and the model together so it can't happen.
- Return the whole probability distribution; low confidence is a useful signal.
- Use `predict_batch` for many rows, and always benchmark in `--release`.
- `save_json` drops the optimizer, so a loaded model **silently stops learning**
  if you resume training on it. Use `cuda_checkpoint` / `restore_cuda_checkpoint`
  (CPU, no GPU needed) — a restored run matched an uninterrupted one exactly.

You can now train, evaluate, save, and serve a model. Next we put all nine
chapters together into one complete project, start to finish.

---

**Next:** [10. A Complete Project](10_full_project.md)
