# 6. Classification — Choosing Between Categories

**You need:** chapters 1–5.

**Time:** 45 minutes.

**Full code:** [`code/06_classification.rs`](code/06_classification.rs) · **Data:** [`data/flowers.csv`](data/flowers.csv)

Chapter 5 predicted a number. Now we pick **one option out of three**: which
species is this flower? Along the way we'll meet the metric that lies to more
beginners than any other in machine learning.

---

## 6.1 The output layer

Regression used one `Linear` neuron. Classification uses **one neuron per
class** with `Softmax`:

```rust
let mut model = Network::builder()
    .input_size(4)                          // four measurements
    .dense(12, Activation::Relu)            // hidden
    .dense(3, Activation::Softmax)          // ← one neuron per class
    .loss(Loss::CrossEntropy)               // ← the partner loss
    .optimizer(Optimizer::adam(0.02))
    .seed(42)
    .build();
```

**3 output neurons** because there are 3 species. This must match the length of
your one-hot targets from chapter 4.

**`Softmax`** turns the three raw scores into probabilities that sum to 1:

```rust
println!("untrained output: {:?}", model.predict(&test.inputs[0])?);
```

```
untrained output for one flower: [0.324, 0.353, 0.323]
  (they sum to 1.0000 - softmax always does)
```

Read that as "32% borealis, 35% rosetta, 32% valentia" — an untrained model
spreading its bet evenly. The summing-to-1 property is what makes the outputs
interpretable as probabilities, and it's why softmax looks at the whole layer at
once instead of each neuron separately.

**`Loss::CrossEntropy`** is softmax's partner. Chapter 1.6 explained why: it
punishes confident wrong answers savagely. Predicting 1% for the true class
costs far more than a squared error would.

The three pairings one more time, because getting this wrong is the most common
setup mistake:

| Task | Output layer | Loss |
|------|--------------|------|
| Predict a number | `.dense(1, Linear)` | `Mse` |
| Yes / no | `.dense(1, Sigmoid)` | `BinaryCrossEntropy` |
| Pick 1 of N | `.dense(N, Softmax)` | `CrossEntropy` |

---

## 6.2 From probabilities to an answer

The model outputs three numbers. To name a species, take the index of the
largest — **argmax**:

```rust
fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv { (i, x) } else { (bi, bv) }
        })
        .0
}
```

It works on targets too — a one-hot vector's argmax is its class index. So
comparing prediction to truth is just `argmax(pred) == argmax(target)`.

---

## 6.3 Accuracy

The obvious metric: what fraction did we get right?

```rust
fn accuracy(model: &Network, d: &Dataset) -> Result<f32, Box<dyn Error>> {
    let mut correct = 0;
    for (input, target) in d.inputs.iter().zip(&d.targets) {
        if argmax(&model.predict(input)?) == argmax(target) {
            correct += 1;
        }
    }
    Ok(correct as f32 / d.len() as f32)
}
```

```
accuracy before training: 45.5%
```

Untrained, on 3 classes. Chance would be ~33%; we got 45% because the random
model happens to favour a class that's common in this small test set. Which is
already a hint that accuracy alone is a slippery number.

After 300 epochs:

```
training loss:
  epoch   1: 1.040006
  epoch  25: 0.074809
  epoch 100: 0.008683
  epoch 300: 0.001394

accuracy:
  train        100.0%
  validation   100.0%
  test          95.5%
```

Good result. But "95.5%" tells you almost nothing about *what the model does*.
For that you need the next tool.

---

## 6.4 The confusion matrix

A single accuracy number throws away all the structure. The **confusion matrix**
keeps it: rows are what the answer actually was, columns are what the model
said.

```rust
fn confusion(model: &Network, d: &Dataset, n: usize) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
    let mut m = vec![vec![0usize; n]; n];
    for (input, target) in d.inputs.iter().zip(&d.targets) {
        m[argmax(target)][argmax(&model.predict(input)?)] += 1;
    }
    Ok(m)
}
```

```
confusion matrix on the test set (rows = actual, cols = predicted):
                borealis   rosetta  valentia   total
  borealis             7         0         1       8
  rosetta              0        10         0      10
  valentia             0         0         4       4
```

How to read it:

- **The diagonal is correct predictions.** 7 + 10 + 4 = 21 right out of 22.
- **Everything off the diagonal is a mistake, and it tells you which mistake.**
  There's exactly one: a `borealis` the model called `valentia`.
- **`rosetta` is perfect** — 10 out of 10, never confused with anything.

That last point is the payoff. The model isn't uniformly 95.5% good; it is
*flawless* at `rosetta` and occasionally confuses `borealis` with `valentia`.
Accuracy hid that completely.

And the confusion is the *right* one to have: this dataset was generated with
`rosetta` well separated and the other two overlapping. The model discovered the
true structure of the data.

**A confusion matrix turns "it's 95% accurate" into "it mixes up these two
specific things", which is the difference between a number and an action.** If
you needed to improve this model, you now know exactly where to spend effort.

---

## 6.5 Precision, recall, F1

The confusion matrix gives three per-class numbers.

For one class, with `tp` = correct predictions of that class:

**Recall** — of all the real ones, how many did we catch?

```
recall = tp / (everything in that class's ROW)
```

**Precision** — when we said this class, how often were we right?

```
precision = tp / (everything in that class's COLUMN)
```

**F1** — their harmonic mean, one number balancing both:

```
F1 = 2 × precision × recall / (precision + recall)
```

```
per-class precision / recall:
  borealis     precision  100.0%   recall   87.5%   F1 0.933
  rosetta      precision  100.0%   recall  100.0%   F1 1.000
  valentia     precision   80.0%   recall  100.0%   F1 0.889
```

Trace the single mistake through both numbers:

- **`borealis` recall is 87.5%** — there were 8, we found 7. We *missed* one.
- **`valentia` precision is 80%** — we said `valentia` 5 times, only 4 were. We
  raised a *false alarm*.

One error, two different symptoms, in two different classes. That's why you
report both.

### Which one do you care about?

It depends entirely on the cost of each mistake — and they're almost never
equal:

- **Cancer screening:** recall matters most. A false alarm costs a follow-up
  test. A miss costs a life.
- **Spam filter:** precision matters most. Spam in the inbox is annoying. A job
  offer in the spam folder is a disaster.

You can trade one for the other by moving the decision threshold instead of
using plain argmax — the same choice chapter 3.5 raised. Requiring 90%
confidence before saying "spam" raises precision and lowers recall.

---

## 6.6 Confidence is a signal — use it

```rust
let p = model.predict(&test.inputs[i])?;
let pi = argmax(&p);
println!("predicted {} at {:.1}%", classes[pi], p[pi] * 100.0);
```

```
confidence on test examples:
  predicted rosetta    100.0%   actual rosetta    OK
  predicted valentia    53.5%   actual borealis   WRONG
  predicted borealis   100.0%   actual borealis   OK
  predicted borealis   100.0%   actual borealis   OK
  predicted valentia   100.0%   actual valentia   OK
  predicted rosetta    100.0%   actual rosetta    OK
```

**Look at the one it got wrong. It was only 53.5% confident.** Every correct
prediction here is at 100%; the single error is the single uncertain one.

That is enormously useful. The model didn't just fail — it *told you* it was
unsure, and you can act on that:

```rust
if p[pi] < 0.70 {
    println!("uncertain - flagging for human review");
}
```

This is how ML systems work in practice. The model handles the confident 95%
automatically and routes the uncertain 5% to a person. You get high accuracy on
the automated portion *and* you catch the hard cases.

> Caveat: neural networks are often **overconfident** — a model saying 99% is
> usually right less than 99% of the time. Treat confidence as a useful ranking
> signal, not a calibrated probability. (Fixing this is called *calibration*.)

---

## 6.7 ⚠️ The accuracy trap

Now the most important section in this chapter.

**Accuracy is a dangerously misleading metric on imbalanced data.**

Watch. Take our test set and make it lopsided — few `borealis`, many `rosetta`:

```
an imbalanced test set: ["borealis", "rosetta", "valentia"] -> [8, 120, 48]
a model that ALWAYS says "rosetta" scores 68.2% accuracy
and it is completely useless.
```

A model consisting of `fn predict() -> "rosetta"` — no inputs, no parameters,
no training — scores **68.2%**.

Report "68% accurate" and it sounds like it works. It has literally never looked
at a flower.

This gets much worse with rarer events. Fraud detection where 1 in 1,000
transactions is fraudulent: "always say legitimate" scores **99.9% accuracy** and
catches zero fraud. Rare disease screening: 99.99% accuracy, every patient
missed. These are not hypotheticals — models like this get deployed.

### How to not get fooled

1. **Always print the class distribution** of your test set (chapter 4.5).
2. **Always compute the majority-class baseline** — the accuracy of always
   guessing the most common class. That's your real floor, exactly like the mean
   baseline in chapter 5.4.
3. **Always look at the confusion matrix.** A useless model's matrix has one
   fully populated column and everything else empty. It's unmissable.
4. **Report per-class recall.** The trap model has 100% recall on `rosetta` and
   **0%** on everything else. That's the number that exposes it instantly.

> **The general rule from chapters 5 and 6:** never report a metric without
> reporting what a trivial model scores on it. In chapter 5 that was predicting
> the mean; here it's predicting the majority class. A metric without a baseline
> is not a result.

---

## 6.8 Things to try

1. Print the confusion matrix on the *training* set. It'll be perfect — which is
   exactly why you never evaluate on training data.
2. Change the model to always predict class 0 and run the full metric suite.
   Watch accuracy stay respectable while recall collapses.
3. Threshold instead of argmax: only answer when confidence > 0.9, count the
   rest as "unknown". How do precision and coverage trade off?
4. Train on only `petal_length`. How much accuracy survives on one feature?
5. Swap to `Loss::Mse` with the softmax output. It trains, but worse — that's
   chapter 1.6's pairing table earning its keep.
6. Reduce the hidden layer to 2 neurons. Where does it break first?
7. Deliberately imbalance the *training* set (drop most `borealis`) and see what
   the confusion matrix does.

---

## Recap

- Classification: `.dense(N, Activation::Softmax)` + `Loss::CrossEntropy`, with
  N = number of classes and one-hot targets.
- Softmax outputs probabilities summing to 1; argmax turns them into a decision.
- **Accuracy alone is not enough** and is actively misleading on imbalanced data.
- The **confusion matrix** shows *which* mistakes happen — the difference between
  a number and an action.
- **Recall** = did we find them all. **Precision** = were we right when we said
  so. Which matters depends on the cost of each error type.
- Low confidence flags likely errors — route those to a human.
- Always compare against the majority-class baseline.

You now have working regression and classification models. Chapter 7 opens up
`fit` and takes control of the training loop — which is how you stop
overfitting.

---

**Next:** [7. The Training Loop — Taking Control](7_training_loop.md)
