# 3. The XOR Problem — Your First Trained Network

**You need:** chapters 1 and 2.

**Time:** 30 minutes.

**Full code:** [`code/03_xor.rs`](code/03_xor.rs)

XOR is the "hello world" of neural networks, and not because it's cute. It is
the smallest problem that *proves* you need a hidden layer. We'll solve it, and
then we'll deliberately fail to solve it, because the failure teaches more than
the success.

---

## 3.1 The problem

XOR ("exclusive or") takes two 0/1 inputs and answers 1 when they **differ**:

| `x1` | `x2` | XOR |
|------|------|-----|
| 0 | 0 | **0** |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | **0** |

Four examples. That's the entire dataset.

### Why it's hard

Draw the four points, filled for output 1, hollow for output 0:

```
  x2
   1 │  ●        ○          ● = output 1
     │                      ○ = output 0
     │
   0 │  ○        ●
     └──────────────  x1
        0        1
```

Now try to separate the filled dots from the hollow dots with **one straight
line**. Go on, try. Any line you draw gets at least one point wrong. The two
filled points are diagonally opposite, and so are the two hollow ones.

This matters because **a single layer of neurons can only draw straight lines.**
A neuron computes `w1·x1 + w2·x2 + b` and splits the space based on whether
that's above or below zero — that's the definition of a straight line. One layer,
one line, no XOR.

In 1969 this observation nearly killed neural network research for a decade.
The fix — stack layers with a non-linear activation between them, and train them
with backpropagation — is what chapter 1 described. A hidden layer lets the
network bend the space so a line *can* separate the points.

Let's watch it happen.

---

## 3.2 Building it, line by line

Start a fresh project (or reuse the one from chapter 2):

```bash
cargo new xor-demo
cd xor-demo
cargo add rusting_brain
```

### The imports

```rust
use rusting_brain::{Activation, Dataset, Loss, Network, Optimizer, TrainConfig};
```

Six types, and they map exactly onto chapter 1's concepts:

| Type | What it is |
|------|-----------|
| `Network` | the model itself |
| `Activation` | the bend: `Relu`, `Sigmoid`, `Tanh`, `Softmax`, `Linear` |
| `Loss` | how wrong we are: `Mse`, `BinaryCrossEntropy`, `CrossEntropy` |
| `Optimizer` | the update rule: `sgd(lr)` or `adam(lr)` |
| `Dataset` | inputs paired with correct answers |
| `TrainConfig` | epochs, batch size, shuffling |

### The data

```rust
let data = Dataset::new(
    // inputs: one Vec<f32> per example
    vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ],
    // targets: the correct answer for each input, in the same order
    vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]],
);
```

A `Dataset` is two lists that line up by position: `inputs[0]` is the question,
`targets[0]` is its answer.

Note that a target is a `Vec<f32>`, not a bare `f32` — even with one output.
That's because networks can have many outputs, so a target is always a list. Here
it's a list of one.

> `Dataset::new` panics if the two lists have different lengths. That's a
> deliberate loud failure: silently mismatched data would give you a model that
> trains happily and predicts garbage.

### The architecture

```rust
let mut model = Network::builder()
    .input_size(2)                          // XOR takes two numbers
    .dense(8, Activation::Tanh)             // hidden layer
    .dense(1, Activation::Sigmoid)          // output layer
    .loss(Loss::BinaryCrossEntropy)
    .optimizer(Optimizer::adam(0.05))
    .seed(42)
    .build();
```

Every line is a decision. Here's the reasoning for each:

**`.input_size(2)`** — XOR has two inputs. This must match the length of each
`Vec` in `data.inputs`, or `predict` returns an error.

**`.dense(8, Activation::Tanh)`** — the hidden layer that makes XOR possible.

Why 8 neurons? XOR provably needs only 2, but 2 is a razor-thin margin: with an
unlucky random start, training gets stuck. 8 gives the optimiser room to find a
solution reliably. This is a real and general trade-off — slightly bigger than
strictly necessary trains more reliably, much bigger overfits.

Why `Tanh`? Its output is centred on zero and it's smooth everywhere. On a
network this tiny, `Relu` risks killing neurons (chapter 1.4) and losing a
meaningful fraction of an 8-neuron layer. On big networks you'd use `Relu`.

**`.dense(1, Activation::Sigmoid)`** — one output because there's one answer, and
`Sigmoid` because we want a probability between 0 and 1.

Remember: **the last `.dense()` is your output layer.**

**`.loss(Loss::BinaryCrossEntropy)`** — the standard partner for a sigmoid
output on a yes/no question, straight from chapter 1.6's table. It punishes
confident wrong answers hard.

**`.optimizer(Optimizer::adam(0.05))`** — Adam adapts the step size per
parameter, so it just works more often than plain SGD. `0.05` is a big learning
rate, which is fine here: four examples and a tiny model. Real datasets want
`0.001`–`0.01`.

**`.seed(42)`** — fixes the random starting parameters so your run matches this
page. In real work, leave it off (or vary it) so you don't accidentally tune your
model to one lucky initialisation.

**`mut`** — training mutates the model in place, so it must be mutable.

---

## 3.3 Predicting before training

```rust
println!("Before training:");
for input in &data.inputs {
    println!("  {:?} -> {:.4}", input, model.predict(input)?[0]);
}
```

```
Before training:
  [0.0, 0.0] -> 0.5000
  [0.0, 1.0] -> 0.4505
  [1.0, 0.0] -> 0.5017
  [1.0, 1.0] -> 0.4434
```

Everything sits near 0.5 — the network is shrugging at every question. It has
17 + parameters set to random values and no information. Exactly as expected.

We print this because it makes the "after" meaningful. **Always look at your
model before training.** If the before and after look the same, training silently
did nothing, and you want to notice that.

---

## 3.4 Training

```rust
let history = model.fit(
    &data,
    TrainConfig {
        epochs: 2_000,
        batch_size: 4,
        shuffle: true,
        seed: Some(11),
    },
)?;
```

One call runs the entire loop from chapter 1.9. The settings:

**`epochs: 2_000`** — 2,000 full passes over the data. That sounds enormous, but
each pass is 4 examples, so it's 8,000 example-visits total, which is nothing. On
a dataset of 100,000 rows you'd use 10–50 epochs.

**`batch_size: 4`** — the whole dataset in one batch. With 4 examples there's no
reason to split. On real data, 32 is the standard starting point.

**`shuffle: true`** — reorder each epoch. Barely matters for 4 examples, but it's
a good habit and it costs nothing.

**`seed: Some(11)`** — makes the shuffling deterministic, so this run is
reproducible. Use `None` for genuine randomness.

### The loss curve

`fit` returns a `TrainingHistory` holding one average loss per epoch —
`history.losses` is 2,000 numbers long. It is the single most useful diagnostic
you have.

```rust
for epoch in [0, 99, 499, 999, 1999] {
    println!("  epoch {:>4}: {:.6}", epoch + 1, history.losses[epoch]);
}
```

```
Loss during training:
  epoch    1: 0.691553
  epoch  100: 0.015627
  epoch  500: 0.001511
  epoch 1000: 0.000504
  epoch 2000: 0.000150
```

Read this as a story:

- **0.6916** at the start. Hold onto that number — it's about to become
  important.
- **0.0156** by epoch 100. It has already essentially solved XOR. Everything
  after this is polish.
- **0.000150** at the end. Confidently correct.

A healthy loss curve falls fast, then flattens. If yours is flat from the
beginning, the learning rate is too low or the model is too small. If it jumps
around or hits `NaN`, the learning rate is too high.

---

## 3.5 The results

```rust
for (input, target) in data.inputs.iter().zip(&data.targets) {
    let raw = model.predict(input)?[0];
    let decision = if raw > 0.5 { 1.0 } else { 0.0 };
    let mark = if (decision - target[0]).abs() < 0.001 { "OK" } else { "WRONG" };
    println!("  {:?} -> {:.4}  decision: {}  expected: {}  {}",
        input, raw, decision, target[0], mark);
}
```

```
After training:
  [0.0, 0.0] -> 0.0000  decision: 0  expected: 0  OK
  [0.0, 1.0] -> 1.0000  decision: 1  expected: 1  OK
  [1.0, 0.0] -> 0.9997  decision: 1  expected: 1  OK
  [1.0, 1.0] -> 0.0003  decision: 0  expected: 0  OK
```

**Four out of four.** The pile of random numbers learned XOR.

Two things to notice, because both generalise to every classifier you'll ever
write:

**The model outputs 0.9997, not 1.** A sigmoid can never *quite* reach 0 or 1 —
it approaches them asymptotically. Getting to 0.9997 already required driving the
pre-activation to about +8. This is normal and healthy; a model that outputs
exactly 1.0 is usually a model that has broken.

**You need a threshold to get a decision.** The network gives you a confidence,
and *you* choose where to cut. `> 0.5` is the obvious default, but it's a
choice you own. A medical screening tool might use `> 0.1` — far more false
alarms, but it stops missing real cases. Chapter 6 comes back to this.

---

## 3.6 Now let's break it on purpose

This is the important part of the chapter. Same data, same loss, same optimizer,
same 2,000 epochs — one change. Delete the hidden layer:

```rust
let mut flat = Network::builder()
    .input_size(2)
    .dense(1, Activation::Sigmoid)     // output only. no hidden layer.
    .loss(Loss::BinaryCrossEntropy)
    .optimizer(Optimizer::adam(0.05))
    .seed(42)
    .build();
```

```
No hidden layer, same 2000 epochs:
  final loss: 0.693147  (with hidden layer: 0.000150)
  [0.0, 0.0] -> 0.5000
  [0.0, 1.0] -> 0.5000
  [1.0, 0.0] -> 0.5000
  [1.0, 1.0] -> 0.5000
```

**It answers 0.5000 to everything.** After two thousand epochs it has learned
precisely nothing — it just shrugs at every input.

And look at that loss: `0.693147`. That is `ln(2)`, and it is not a coincidence.
`ln(2)` is *exactly* the binary-cross-entropy loss of a model that answers "50/50,
no idea" to every question. The network didn't fail to train. It trained
perfectly — and the best possible straight line for XOR is the one that gives up
and guesses.

The optimiser worked. The architecture was impossible.

> **Keep this.** When a model refuses to learn, the instinct is to train longer
> or turn the learning rate up. Often the real problem is that the architecture
> cannot express the answer, and no amount of training fixes that. Compare
> against the loss of a model that always guesses: for binary cross entropy on a
> balanced problem that's `ln(2) ≈ 0.693`; for MSE it's the variance of your
> targets. If you're sitting at that number, your model is guessing.

---

## 3.7 Things to try

Learning happens when you break things. Each of these takes one line:

1. **Two hidden neurons instead of eight** (`.dense(2, Activation::Tanh)`). Does
   it still work? Now change `.seed(42)` to `.seed(7)`, `.seed(123)`. Some seeds
   solve it, some get stuck — that's the razor-thin margin from 3.2.
2. **Learning rate `5.0`.** Watch the loss explode or go `NaN`.
3. **Learning rate `0.0001`.** Watch it crawl and not finish in 2,000 epochs.
4. **`Optimizer::sgd(0.05)` instead of Adam.** Much slower. Now try `sgd(0.5)`.
5. **`Loss::Mse` with the sigmoid output.** It still works, but slower — this is
   why the loss/activation pairings in chapter 1.6 exist.
6. **`.dense(8, Activation::Relu)`** for the hidden layer. Try several seeds. When
   it fails, you're watching dead neurons.
7. **Only 50 epochs.** Underfitting, live.

---

## Recap

- One layer of neurons can only draw a straight line, so it cannot do XOR.
- A hidden layer with a non-linear activation fixes that.
- `Network::builder()` describes the architecture; the last `.dense()` is the
  output layer.
- `fit` runs the whole training loop and hands back the per-epoch loss curve.
- Read the loss curve. Falling fast then flattening = healthy.
- A model stuck at the "always guess" loss has an architecture problem, not a
  training problem.
- `predict` returns confidences; thresholding them into decisions is your call.

Everything so far used four hand-typed examples. Real machine learning is mostly
about data — getting it, cleaning it, and splitting it correctly. That's next.

---

**Next:** [4. Working With Real Data](4_data.md)
