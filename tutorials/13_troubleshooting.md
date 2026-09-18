# 13. When Things Go Wrong

**You need:** chapters 1–10.

**Time:** read it once, then come back to it.

**Full code:** [`code/13_diagnostics.rs`](code/13_diagnostics.rs)

This is a reference chapter, not a lesson. Nothing here is new material; it is
the list of things that actually go wrong and what each one means.

Training failures are unusually hard to debug because a broken model does not
crash. It produces numbers. Plausible-looking, confidently wrong numbers. The
skill this chapter teaches is telling the four failure shapes apart before you
start changing things.

---

## 13.1 Read the loss curve first

Almost every problem announces itself in the loss curve, and the shape tells
you which problem it is.

```
loss                                   loss
 │╲                                     │╲
 │ ╲___________  healthy                │ ╲____  training
 │                                       │  ╲
 └──────────── epoch                     │   ╲___
                                         │ ___/     validation
loss                                     └──────────── epoch
 │──────────── flat: not learning              overfitting
 │
 └──────────── epoch                    loss
                                         │╲  ╱╲    ╱╲
loss                                     │ ╲╱  ╲  ╱  ╲   unstable:
 │╲                                      │      ╲╱        rate too high
 │ ╲___
 │     ╳  NaN                            └──────────── epoch
 └──────────── epoch
```

So: print it. Every time, in every run.

```rust
let history = model.fit(&data, config)?;
for (epoch, loss) in history.losses.iter().enumerate() {
    if epoch % 10 == 0 {
        println!("{epoch:4} {loss:.6}");
    }
}
```

A training run you did not watch is a training run you cannot debug.

---

## 13.2 The loss is NaN

Once a single `NaN` appears it spreads through every weight on the next update
and the model is dead. No amount of further training recovers it.

In order of likelihood:

**The learning rate is too high.** By a long way the most common cause. A large
step overshoots, the next gradient is larger, the step after that is larger
still, and within a dozen updates you are at infinity. Divide the rate by ten
and try again. If `0.001` NaNs, try `0.0001`.

**The inputs are not normalized.** A feature in the thousands (chapter 4:
house prices, populations, timestamps) multiplied through two layers overflows
`f32` quickly. Standardize every input column.

**A target is out of the activation's range.** Sigmoid can only output
`(0, 1)`. Ask binary cross entropy to explain a target of `5.0` and it takes
the logarithm of a negative number.

**Division by zero in your own preprocessing.** A constant column has zero
standard deviation, and standardizing it divides by that zero. Guard it:

```rust
let scale = if std_dev < 1e-8 { 1.0 } else { std_dev };
```

Find the first bad epoch:

```rust
if let Some(bad) = history.losses.iter().position(|l| !l.is_finite()) {
    println!("diverged at epoch {bad}, previous loss {}", history.losses[bad - 1]);
}
```

If the previous loss was already large and growing, it is the learning rate. If
the previous loss was fine and it went straight to `NaN`, it is the data.

---

## 13.3 The loss does not move

It starts around its initial value and stays there.

**The learning rate is too low.** The mirror image of 13.2, and it looks like a
model that cannot learn rather than one that is barely trying. Multiply by ten.
A flat curve at `1e-6` and a healthy one at `1e-3` are the same model.

**There is no non-linearity.** `Linear` layers stacked on `Linear` layers
collapse into one linear layer, whatever the depth; it cannot learn XOR. Every
hidden layer needs `Relu`, `Tanh`, or `Sigmoid`. This is chapter 1.4, and it is
worth re-reading when a model refuses to learn something you know is learnable.

**Dead ReLUs.** ReLU outputs zero for every negative input, and the gradient
through a zero output is zero, so a unit that goes negative for every row in
your data never comes back. If a whole layer does this the network below it
stops receiving gradient. Symptoms: the loss drops a little, then flatlines
high. Try `Tanh`, or lower the learning rate — dead ReLUs are usually caused by
one enormous early update.

**The features do not contain the answer.** Not every dataset supports every
question. Build the baseline from chapter 10 first: if predicting the majority
class gets 68% and your model gets 68%, the model learned nothing and the
features may be why.

---

## 13.4 Training loss falls, validation loss rises

Textbook overfitting: the model is memorizing rows rather than learning the
pattern. Chapter 7 covers the fix; the short list, in the order worth trying:

1. **Early stopping.** Keep the weights from the best validation epoch, not the
   last. Cheapest and most effective.
2. **More data.** Beats every other intervention when you can get it.
3. **A smaller model.** Fewer parameters, less capacity to memorize.
4. **Weight decay.** `Optimizer::adam_with_weight_decay(rate, 0.01)` pulls
   weights toward zero and makes large ones cost something.

The gap itself is not the problem — a small, stable gap is normal. Validation
loss *rising* is the problem.

---

## 13.5 It works in training and fails in production

You trained it, you validated it, you shipped it, and the predictions are
garbage. Three suspects.

**Preprocessing drift.** The model was trained on standardized inputs and
production is feeding it raw ones. Chapter 10 saves the normalization
constants next to the weights for exactly this reason. Do that, and load both
together.

**A different feature order.** Your training code built `[age, price, plan]`
and your service builds `[price, age, plan]`. Nothing errors — the shapes
match. The answers are simply wrong. Name your columns in one place and index
them by name everywhere else.

**A missing category.** One-hot encoding built from the training set alone has
no column for a category that first appears in production. Decide explicitly
what an unseen category does; silently encoding it as all-zeros is a choice,
and it should be a deliberate one.

---

## 13.6 Results change between runs

Weights are randomly initialized and batches are randomly shuffled, so two runs
differ unless you say otherwise:

```rust
let model = Network::builder().input_size(4).dense(8, Activation::Relu).seed(42).build();
let config = TrainConfig { epochs: 100, batch_size: 32, shuffle: true, seed: Some(7) };
```

Both seeds matter — one for initialization, one for shuffling. With both fixed,
a CPU run is reproducible to the bit.

Two things still legitimately vary:

- **CPU versus GPU.** Float addition is not associative, and the two backends
  sum in different orders. Sixth-decimal differences are arithmetic, not bugs.
- **Thread count.** Rayon's parallel reductions combine partial sums in
  completion order. Set `RAYON_NUM_THREADS=1` if you need a fixed order.

If you want reproducibility to be a property of your project rather than a
thing you remember, assert it in a test: train twice with the same seeds and
compare the final losses exactly.

---

## 13.7 Every error this library returns

| Message | Meaning |
|---|---|
| `network needs an input size and at least one dense layer` | `build()` called on an empty builder |
| `input length N does not match expected length M` | `predict` got the wrong number of features |
| `target length N does not match expected length M` | A target row's width differs from the output layer |
| `dataset is empty` | `fit` on zero rows; usually a CSV parsed to nothing |
| `invalid network snapshot: ...` | A JSON model file is corrupt or from an incompatible version |
| `io error: ...` | The path is wrong, or the directory does not exist. Create parents before saving |
| `serialization error: ...` | Malformed JSON |
| `invalid transformer configuration: ...` | A `TransformerLm` knob is inconsistent; the message names it |
| `token id N is outside the vocabulary of M` | Token ids must be `< vocab_size`; usually a tokenizer/model mismatch |
| `sequence length N exceeds the configured maximum M` | Raise `max_seq_len` — but it is baked into the checkpoint, so this means retraining |
| `CUDA backend is unavailable: built without the cuda feature` | Add `--features cuda` |
| `CUDA memory budget exceeded: estimated N MiB exceeds budget M MiB` | Lower `batch_size` or raise the budget |
| `CUDA backend error: ...` | The driver or a kernel failed; the string is from the driver |
| `CUDA backend does not support ...` | That configuration has no device path. Train it on the CPU |
| `invalid CUDA checkpoint: ...` | Checkpoint version or layer count does not match the model |
| `Metal backend is unavailable: ...` | No `metal` feature, or not macOS |
| `accelerator backend error: ...` | Backend-independent failure; the string says which |

None of these are warnings. There is no path in this library where a failure
downgrades itself into a slower or less accurate run.

---

## 13.8 Training is slower than it should be

**Not in release mode.** A debug build is ten to fifty times slower. `cargo run
--release`. This is the answer more often than everything below combined.

**Batch size of one.** One row at a time means one tiny matrix multiply per
row. Chapter 7 covers why 32–256 is usually right.

**The dataset is being cloned every epoch.** `Dataset::batches` borrows; if
your own loop is calling `.clone()` inside it, that is your time.

**A GPU that is not being fed.** `dataset_resident: false` in
`AcceleratorStats` means batches are crossing PCIe every step. Raise the memory
budget.

**Too many threads for the work.** Rayon's overhead can exceed the work for
small models. Compare against `RAYON_NUM_THREADS=1`; if single-threaded is
faster, your model is too small to parallelize.

---

## 13.9 A method that works

When something is wrong and you do not know what:

1. **Make it smaller.** 100 rows, 2 epochs. A five-second reproduction is worth
   an hour of staring at a five-minute one.
2. **Make it overfit.** Train on ten rows with no validation. A correct
   pipeline drives the loss to nearly zero — it is *supposed* to memorize ten
   rows. If it cannot, the bug is in the model or the data, not in the
   training. This is the single most useful test in machine learning.
3. **Print the intermediate values.** The first batch's inputs, its targets,
   the first prediction. Most data bugs are visible in the first three rows.
4. **Change one thing.** Learning rate *or* architecture *or* features. Two at
   once and you learn nothing from the result.
5. **Compare against the baseline.** Chapter 10. Without it you cannot tell a
   working model from a broken one.

---

## Exercises

1. Deliberately break it four ways: learning rate `100.0`, learning rate
   `1e-12`, all-`Linear` activations, and an unnormalized input column in the
   thousands. Recognize each from the curve alone.
2. Run the ten-row overfit test on the chapter 10 pipeline. Confirm the loss
   goes to nearly zero, then introduce a bug (shuffle the targets) and confirm
   it does not.
3. Write a test that trains twice with fixed seeds and asserts the final losses
   are exactly equal.

---

## Recap

- The loss curve tells you which failure you have. Print it every run.
- `NaN` is nearly always the learning rate, then unnormalized inputs.
- A flat curve is a learning rate too low, a missing non-linearity, dead ReLUs,
  or features that do not contain the answer.
- Rising validation loss is overfitting: early stopping first, more data
  second.
- Working in training and failing in production is preprocessing, feature
  order, or an unseen category — in that order.
- Two seeds for reproducibility. CPU/GPU differences in the sixth decimal are
  arithmetic.
- Nothing in this library degrades silently. Every failure is an error value.
- When lost: shrink it, overfit ten rows, print the first batch, change one
  thing.

---

**Next:** [14. Tokens and a Transformer Language Model](14_language_model.md)
