# 8. Evaluation — Measuring Honestly

**You need:** chapters 1–7.

**Time:** 45 minutes.

**Full code:** [`code/08_evaluation.rs`](code/08_evaluation.rs)

You can train a model. Now the harder skill: knowing how good it *actually* is.

Almost every inflated ML claim comes from a measurement mistake, not a lie. This
chapter runs four experiments on real numbers — including one whose result
contradicts the textbook, which turns out to be the most useful part.

---

## 8.1 Why there are three splits

Chapter 4 made train/validation/test without fully justifying the third one.
Here's the reason.

You train dozens of models — different sizes, learning rates, epochs — and pick
the one that scores best on validation. That *choice* is itself a form of
fitting. You are optimising your decisions against the validation set, so
validation stops being a neutral judge.

The test set is the one you never looked at while deciding. That's what makes it
trustworthy.

| Set | Used for | How often |
|-----|----------|-----------|
| train | learning parameters | every batch |
| validation | choosing hyperparameters, early stopping | every epoch |
| test | the final reported number | **once, at the end** |

> **The rule:** once you've looked at the test set, you can't use it to make
> decisions any more. If you tweak your model after seeing a test score, that
> score is spent. Be disciplined — this is the professional standard, and it's
> what separates a real result from a self-flattering one.

---

## 8.2 Experiment: searching 12 architectures

Let's do what everyone does — grid search on validation:

```rust
for (h1, h2) in [(8, 4), (16, 8), (32, 16), (64, 32)] {
    for lr in [0.003, 0.01, 0.03] {
        let mut m = build(h1, h2, lr, 42);
        m.fit(&train, cfg)?;
        let v = mae(&m, &validation, &ys)?;
        // keep the best
    }
}
```

```
    h1   h2       lr    val MAE
     8    4    0.003     16.039   <- best so far
     8    4     0.01     16.164
     8    4     0.03     15.646   <- best so far
    16    8    0.003     17.723
    16    8     0.01     17.717
    16    8     0.03     15.840
    32   16    0.003     18.675
    32   16     0.01     18.585
    32   16     0.03     19.000
    64   32    0.003     21.193
    64   32     0.01     23.222
    64   32     0.03     19.780
```

One clear trend: **bigger is worse.** 64×32 is consistently the worst block. With
140 training rows, extra capacity buys overfitting, exactly as chapter 7
predicted. That trend spans 12 runs and is worth believing.

The winner is 8×4 at `lr = 0.03`, validation MAE **15.646k**. Now the honest
test:

```
  winner: 8x4 lr=0.03  validation MAE 15.646k
  the SAME model on the untouched test set: 14.200k
  optimism from picking the winner: -1.446k (-9%)
```

### The test score came out *better* than validation

The textbook says selection bias should make the test score **worse**. It didn't.
So either the textbook is wrong, or something else is going on.

Something else is going on, and the next two experiments prove exactly what.

---

## 8.3 Experiment: how much is just luck?

Same architecture. Same data. Same everything — except the random seed that sets
the initial weights.

```rust
for seed in 1..=8u64 {
    let mut m = build(16, 8, 0.01, seed);
    m.fit(&train, cfg)?;
    println!("seed {seed}: validation MAE {:.3}k", mae(&m, &validation, &ys)?);
}
```

```
  seed  1: validation MAE 18.369k
  seed  2: validation MAE 18.566k
  seed  3: validation MAE 17.347k
  seed  4: validation MAE 17.084k
  seed  5: validation MAE 16.543k
  seed  6: validation MAE 16.908k
  seed  7: validation MAE 17.582k
  seed  8: validation MAE 18.423k

  mean 17.603k   std dev 0.718k   range 16.543k .. 18.566k (spread 2.023k)
```

**A 2k spread from nothing but the starting random numbers.**

Now go back and look at the grid search. The winner beat the runner-up by
`15.840 − 15.646 = 0.194k`. Random seed noise is **ten times larger than that
gap**.

So the "winner" was not meaningfully better than the runner-up. Re-run the search
with a different seed and a different architecture wins. **Most of that grid
search was measuring noise, not quality.**

That also explains 8.2. The validation set has 30 rows and the test set has 30
rows. The difference between them (1.4k) is smaller than the noise (2.0k). The
selection bias is real, but it's buried under sampling noise on splits this
small — so it showed up with the opposite sign this time. It's a tendency, not a
law.

**The lesson:** before believing that model A beats model B, check whether the
gap is bigger than the noise. Run several seeds. If the difference is within the
spread, you have no result.

> This is a real and widespread problem. Published papers have claimed
> improvements later shown to be entirely within seed variance. The fix is
> cheap: report mean ± standard deviation over several seeds, never a single
> number.

---

## 8.4 K-fold cross validation

A 30-row validation set is too small to trust. But we only have 200 rows total —
we can't afford a bigger one.

**K-fold cross validation** solves this. Split the data into k parts. Train k
times, each time holding out a different part. Average the scores.

```
5-fold, 170 rows:

fold 1:  [TEST][ train ][ train ][ train ][ train ]
fold 2:  [train][ TEST ][ train ][ train ][ train ]
fold 3:  [train][ train][ TEST  ][ train ][ train ]
fold 4:  [train][ train][ train ][ TEST  ][ train ]
fold 5:  [train][ train][ train ][ train ][ TEST  ]
```

Every row gets tested exactly once, and every row is trained on k−1 times.

```rust
let k = 5;
let fold_size = pool.len() / k;
for fold in 0..k {
    let start = fold * fold_size;
    let end = if fold == k - 1 { pool.len() } else { start + fold_size };

    // rows in [start, end) validate; everything else trains
    let mut tr_in = Vec::new(); /* ... */
    for i in 0..pool.len() {
        if i >= start && i < end { /* push to validation */ }
        else                     { /* push to train */ }
    }
    // train a fresh model, score it, keep the score
}
```

Note `end` for the last fold takes everything remaining, so a dataset that
doesn't divide evenly doesn't silently drop rows.

```
  fold 1: trained on 136, tested on 34  ->  MAE 18.888k
  fold 2: trained on 136, tested on 34  ->  MAE 19.728k
  fold 3: trained on 136, tested on 34  ->  MAE 14.886k
  fold 4: trained on 136, tested on 34  ->  MAE 18.159k
  fold 5: trained on 136, tested on 34  ->  MAE 15.971k

  cross-validated MAE: 17.526k +/- 1.816k
```

**Look at the fold spread: 14.9k to 19.7k.** Same model, same procedure — the
only difference is which 34 rows got held out. A 4.8k swing.

That is the definitive answer to 8.2. A single split's score is a coin flip
worth ±2k. Reporting "14.200k" as *the* performance was never justified.

**`17.5k ± 1.8k` is the honest answer.** It has an uncertainty attached, which is
what a real measurement looks like.

| | Single split | K-fold |
|---|---|---|
| Cost | 1 training run | k training runs |
| Uses all data for testing | no | yes |
| Gives an uncertainty | no | yes |
| Use when | data is plentiful, training is slow | data is scarce |

**Rule of thumb: under a few thousand rows, use k-fold.** k = 5 or 10. Your
dataset here has 200 rows — k-fold is clearly right, and the single-split number
was misleading us.

---

## 8.5 Error analysis: look at the failures

Aggregate metrics tell you *how much* you're wrong. To improve, you need to know
*where*. Sort by error and read the extremes:

```rust
errs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
```

```
  5 worst predictions:
    actual    20.0k  predicted    58.9k  off by   38.9k
    actual   333.7k  predicted   298.5k  off by   35.2k
    actual   163.4k  predicted   131.6k  off by   31.8k
    actual   139.6k  predicted   171.4k  off by   31.8k
    actual   378.3k  predicted   407.5k  off by   29.2k

  5 best predictions:
    actual   407.7k  predicted   407.4k  off by    0.3k
    actual   303.2k  predicted   305.1k  off by    1.9k
    actual   266.9k  predicted   269.0k  off by    2.1k
    actual   193.7k  predicted   191.5k  off by    2.2k
```

The worst one is the 20.0k house again (chapter 5.6). It's the artificial price
floor — the model can't know about a clamp that isn't in the features. That's a
**data** problem, and no architecture change will fix it.

Now slice the errors by price band:

```
  MAE on houses under 200k: 18.62k  (14 houses)
  MAE on houses over  200k: 10.33k  (16 houses)
```

**The model is nearly twice as accurate on expensive houses.** A single MAE of
14.2k hid two very different behaviours.

That's actionable in a way no aggregate number is:

- If cheap houses matter to your users, you have a specific problem to fix.
- Likely cause: fewer training examples at the low end, plus that 20k clamp
  distorting the region.
- Possible fixes: more cheap-house data, or predicting `log(price)` so relative
  errors are weighted evenly.

**Always slice your errors** — by class, by value range, by any feature you have.
Aggregate metrics are for reporting; sliced metrics are for improving.

---

## 8.6 The honest evaluation checklist

- [ ] Test set untouched until the very end
- [ ] Compared against a trivial baseline (mean / majority class)
- [ ] Metrics in real units, not scaled ones
- [ ] Several seeds run; spread reported
- [ ] Improvements confirmed larger than seed noise
- [ ] K-fold used if the dataset is small
- [ ] Errors sliced by group, not just averaged
- [ ] Worst predictions actually read
- [ ] Uncertainty reported: `17.5k ± 1.8k`, not `17.526k`

That last one is a good habit in general. Quoting `14.200k` from a 30-row test
set implies a precision you do not have.

---

## 8.7 Things to try

1. Re-run the grid search with `seed = 7` instead of `42`. Does the same
   architecture win?
2. Run the winner over 8 seeds and report mean ± std. Does it still beat 16×8?
3. Change k to 10, then to 2. How does the reported uncertainty move?
4. Wrap k-fold around the grid search. Slower, far more trustworthy — which
   architecture wins now?
5. Slice errors by `bedrooms` instead of price.
6. Train on `log(price)` and compare relative errors on cheap houses.
7. Add early stopping (chapter 7) inside each fold.

---

## Recap

- Train learns, validation tunes, test is spent **once**.
- Random seeds alone moved our score by 2k — **check that a difference is bigger
  than the noise before believing it.**
- A grid search whose winner beats the runner-up by less than seed noise has
  selected noise.
- Single-split scores on small data are unreliable; folds here ranged 14.9k to
  19.7k.
- **K-fold cross validation** gives a mean and an uncertainty. Use it under a
  few thousand rows.
- Report `mean ± std`, not a single over-precise number.
- Slice your errors. Our model was 2× better on expensive houses, and the
  aggregate hid it.
- Read your worst predictions — some failures are data problems that no model
  can fix.

You can now train a model and state honestly how good it is. Next: saving it so
something else can use it.

---

**Next:** [9. Saving, Loading, and Using a Model](9_save_load.md)
