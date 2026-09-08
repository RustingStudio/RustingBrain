# 7. The Training Loop — Taking Control

**You need:** chapters 1–6.

**Time:** 1 hour.

**Full code:** [`code/07_training_loop.rs`](code/07_training_loop.rs)

So far `fit(...)` has been a black box: hand it a dataset, get a trained model.
That's fine until training goes wrong — and you can't fix what you can't see.

This chapter opens the box. You'll watch a model overfit in real time, stop it
automatically, and see what learning rate and batch size actually do.

---

## 7.1 The loop, by hand

`fit` with `epochs: 1` runs exactly one pass. Call it in your own loop and you
get to run code between epochs:

```rust
let one_epoch = TrainConfig {
    epochs: 1,
    batch_size: 16,
    shuffle: true,
    seed: Some(1),
};

for epoch in 1..=200 {
    model.fit(&train, one_epoch)?;                       // train one pass

    let train_loss = model.evaluate_loss(&train)?;       // measure
    let val_loss   = model.evaluate_loss(&validation)?;  // ← the important one

    println!("{epoch}: train {train_loss:.5}  val {val_loss:.5}");
}
```

`evaluate_loss` runs the loss over a dataset **without training on it** — no
gradients, no updates. That's what makes it safe on validation data.

The reason this matters: `fit`'s returned history only contains *training* loss.
Training loss almost always goes down. It will happily keep going down while
your model gets worse at its actual job. **Validation loss is the number that
tells you the truth**, and the only way to see it every epoch is to run the loop
yourself.

### What it shows

```
  epoch   train      val
      1   0.05536   0.08320
     25   0.00200   0.00265
     50   0.00187   0.00236     ← validation bottoms out here
     75   0.00174   0.00244
    100   0.00168   0.00250
    125   0.00160   0.00253
    150   0.00152   0.00260
    175   0.00148   0.00269
    200   0.00144   0.00272
```

Read the two columns separately:

- **Train** falls the whole way: `0.00200 → 0.00144`. Uninterrupted progress.
- **Val** falls until epoch 50 (`0.00236`), then **rises** for the next 150
  epochs.

That divergence is the overfitting signature from chapter 1.10, in real numbers.
From epoch 50 onward the model is memorising the training set. Every epoch after
that made it *worse* at the only thing that matters — and the training loss
never gave a hint.

**Epochs 50–200 were not merely wasted. They were harmful.**

This is why "train for 200 epochs" is not a plan. The right number of epochs is
"until validation stops improving", and you can only know that by measuring.

---

## 7.2 Overfitting on purpose

Let's make it unmissable. Two changes designed to overfit hard: shrink the
training set to 25 rows, and use a network with 8,705 parameters.

```rust
let small = Dataset::new(
    train.inputs[..25].to_vec(),
    train.targets[..25].to_vec(),
);

let mut big = Network::builder()
    .input_size(4)
    .dense(64, Activation::Relu)
    .dense(64, Activation::Relu)
    .dense(64, Activation::Relu)
    .dense(1, Activation::Linear)
    .loss(Loss::Mse)
    .optimizer(Optimizer::adam(0.01))
    .seed(42)
    .build();
```

**8,705 parameters to fit 25 examples — 348 knobs per data point.** The model
has more than enough capacity to store the answers outright.

```
  epoch     train       val
      1   0.07982   0.03860
     25   0.00114   0.00296
     50   0.00073   0.00314
     75   0.00054   0.00335
    100   0.00066   0.00426
    150   0.00070   0.00390
    200   0.00027   0.00456
    300   0.00517   0.00957

  best validation loss 0.00270 was at epoch 18 - everything after was wasted
```

Training loss reaches `0.00027`. Validation loss is `0.00456` — **17× worse**.

The model has essentially memorised all 25 rows. It is superb at questions it
has already seen the answers to, and it has learned very little that transfers.

**Peak performance was epoch 18.** The remaining 282 epochs made it worse. And
if you'd only watched training loss, you'd have concluded it was improving the
whole time.

> Notice the late instability too — epoch 225 spikes to `0.00629` train,
> `0.01063` val. A heavily overfitted model sits in a sharp, fragile minimum, so
> single batches can knock it around. Wild late-training swings are a symptom.

### The three cures

| Cure | How | When |
|------|-----|------|
| **Stop earlier** | early stopping (next section) | always — it's free |
| **Smaller model** | fewer layers/neurons | when params ≫ data |
| **More data** | get more rows | best fix, usually impossible |

Chapter 5's model overfitted mildly (225 params, 140 rows) and that was fine.
This one is pathological. The ratio of parameters to data is the thing to watch.

---

## 7.3 Early stopping

The fix writes itself: track the best validation loss, keep a copy of the model
at its best, and stop when it hasn't improved for a while.

```rust
let patience = 20;                       // how many bad epochs to tolerate
let mut best_loss = f32::INFINITY;
let mut best_model = model.clone();      // Network implements Clone
let mut best_epoch = 0;
let mut since_improved = 0;

for epoch in 1..=300 {
    model.fit(&small, cfg)?;
    let v = model.evaluate_loss(&validation)?;

    if v < best_loss {
        best_loss = v;
        best_model = model.clone();      // ← snapshot the good one
        best_epoch = epoch;
        since_improved = 0;
    } else {
        since_improved += 1;
        if since_improved >= patience {
            println!("stopped at epoch {epoch}");
            break;
        }
    }
}
```

```
  stopped at epoch 38 - no improvement for 20 epochs
  best epoch: 18   best validation loss: 0.00270
  restored model validation loss: 0.00270
  (the still-training model was at 0.00302)
```

Two wins in one:

**It stopped at epoch 38 instead of 300** — 8× less compute for a better result.

**It kept the epoch-18 model.** This is the part people forget. Stopping alone
leaves you with the epoch-38 model (`0.00302`); the snapshot gives you the
epoch-18 model (`0.00270`). Validation loss wanders, so the last epoch is
essentially never the best one. `model.clone()` is cheap. Always keep the best.

### Choosing patience

Too low and you stop on noise. Too high and you waste time. **10–20 is a good
default.** Noisier validation curves want more patience.

Early stopping is the highest value-per-line technique in this entire course.
It costs ten lines, needs no tuning, and prevents the most common failure mode
in applied ML.

---

## 7.4 Learning rate

Chapter 1.7 said the learning rate is your step size and the most important
setting you choose. Here it is measured — same model, same 100 epochs, only the
learning rate changes:

```
          lr        train          val
      0.0001      0.04308      0.04456     ← way too small
       0.001      0.00193      0.00261
        0.01      0.00184      0.00260
         0.1      0.00222      0.00207     ← best here
           1      0.06225      0.06416     ← too big
```

- **0.0001** — barely moved in 100 epochs. Loss is 20× worse than the good runs.
  It would get there eventually; you don't have that long.
- **0.001 – 0.1** — all work. There's a broad valley of acceptable values, which
  is why this is tunable rather than terrifying.
- **1.0** — worse than `0.0001`. Steps overshoot the minimum and bounce.

The failure modes look different, which is how you diagnose them:

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss barely moves | too small | multiply by 10 |
| Loss falls then plateaus high | slightly small | raise it, or train longer |
| Loss jumps around | too big | divide by 10 |
| Loss becomes `NaN` or `inf` | far too big | divide by 100 |

**Practical recipe:** start at `0.01` with Adam. If the loss doesn't move, try
`0.1`. If it explodes, try `0.001`. Change by factors of 10 — fine-tuning to
`0.023` is not where your gains are.

> `NaN` deserves a note. Once a single parameter becomes `NaN`, it spreads to
> every parameter within a few updates and the model is permanently dead — no
> amount of further training recovers it. If you see `NaN`, stop, lower the
> learning rate, and restart. Chapter 13 covers the other causes.

---

## 7.5 Batch size

```
       batch    updates        train          val
           1      14000      0.00195      0.00235
           8       1800      0.00180      0.00326
          16        900      0.00184      0.00260
          64        300      0.00210      0.00312
         140        100      0.00221      0.00302
```

The `updates` column explains most of this. With 140 training rows and 100
epochs:

- **batch 1** → 140 updates per epoch → 14,000 total. Slow per epoch, but each
  epoch learns a lot.
- **batch 140** → 1 update per epoch → 100 total. Fast per epoch, but only 100
  parameter updates all run — and it shows in the worst training loss.

The trade-off:

| | Small batches | Large batches |
|---|---|---|
| Updates per epoch | many | few |
| Gradient quality | noisy | smooth, well-averaged |
| Speed per epoch | slow | fast (better parallelism) |
| Memory | low | high |
| Side effect | noise acts a bit like regularisation | can settle into worse minima |

The differences here are small — 140 rows is a tiny dataset. On real data batch
size matters much more, mostly for speed.

**Practical advice: use 32.** Use 8 or 16 if your dataset is tiny. Go bigger only
when you need the throughput. It is not a parameter worth agonising over — spend
that effort on the learning rate.

---

## 7.6 A training loop worth reusing

Everything from this chapter, assembled:

```rust
let mut best_loss = f32::INFINITY;
let mut best_model = model.clone();
let mut best_epoch = 0;
let mut since_improved = 0;
let patience = 20;

for epoch in 1..=max_epochs {
    model.fit(&train, one_epoch)?;

    let train_loss = model.evaluate_loss(&train)?;
    let val_loss   = model.evaluate_loss(&validation)?;

    if epoch % 10 == 0 {
        println!("epoch {epoch:>4}  train {train_loss:.5}  val {val_loss:.5}");
    }

    if val_loss < best_loss {
        best_loss = val_loss;
        best_model = model.clone();
        best_epoch = epoch;
        since_improved = 0;
    } else {
        since_improved += 1;
        if since_improved >= patience {
            println!("early stop at epoch {epoch}");
            break;
        }
    }
}

let model = best_model;    // ← use the best one, not the last one
println!("using epoch {best_epoch}, validation loss {best_loss:.5}");
```

That's the professional version of `fit`. Copy it.

> **One cost to know about:** `evaluate_loss` on both sets every epoch adds real
> time on large datasets. If it hurts, evaluate every 5 epochs instead. Just
> don't skip validation entirely — that's trading your only honest signal for a
> little speed.

---

## 7.7 Things to try

1. Run the section 1 loop for 1,000 epochs. How much worse does validation get?
2. Set `patience = 3`. Does it stop too early? Try `100`.
3. Remove `best_model = model.clone()` and use the final model. Measure the
   difference.
4. Shrink the big network to `.dense(4, ...)` on 25 rows. Does it still overfit?
5. Grow `small` from 25 rows to 50, 100. Watch the train/val gap shrink.
6. Add a `TrainConfig` with `shuffle: false` and compare.
7. Use early stopping on chapter 6's classifier, stopping on validation
   *accuracy* instead of loss.

---

## Recap

- `fit` with `epochs: 1` inside your own loop gives you control between epochs.
- `evaluate_loss` measures without training — safe on validation data.
- **Training loss is not a progress report.** It falls even as the model gets
  worse. Validation loss is the honest signal.
- Training loss ↓ while validation loss ↑ = overfitting. Stop.
- **Early stopping + keeping the best snapshot** is the best ten lines you can
  add to any training script.
- The last epoch is almost never the best epoch. `model.clone()` your best.
- Learning rate: start at `0.01`, adjust by factors of 10. It matters more than
  anything else you'll tune.
- Batch size: use 32 and move on.

You can now train a model properly and stop at the right time. Next: measuring
how good it really is, without fooling yourself.

---

**Next:** [8. Evaluation — Measuring Honestly](8_evaluation.md)
