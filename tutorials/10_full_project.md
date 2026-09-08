# 10. A Complete Project

**You need:** chapters 1–9.

**Time:** 90 minutes.

**Full code:** [`code/10_full_project/`](code/10_full_project/) — a real
multi-file cargo project, not a script.

Every chapter so far has been one `main.rs` that does one thing and exits. Real
projects aren't shaped like that. This chapter builds the thing you'd actually
ship:

```
cargo run --release -- train                        # learn, and save
cargo run --release -- evaluate                     # the honest final number
cargo run --release -- predict 3 45.0 6 basic       # use it
```

Three subcommands, four modules, one saved model. The data is also harder than
anything you've seen so far — it has a **categorical column**, **missing
values**, and **class imbalance**, none of which the flowers or houses had.

---

## 10.1 The problem

`data/subscriptions.csv` — 600 customers of a subscription service. For each
one, did they cancel?

```
months_active,monthly_hours,support_tickets,plan,cancelled
38,58.9,0,plus,no
42,13.7,0,pro,no
37,13.5,2,pro,no
14,38.5,4,pro,no
7,45.3,3,basic,yes
```

This is worth doing well, because the business use is obvious: find the
customers about to leave *before* they leave, and do something about it.

Three things here are new:

| Problem | Where it appears | What we'll do |
|---|---|---|
| `plan` is a word, not a number | input column | one-hot encode it (3 columns) |
| `monthly_hours` is sometimes blank | 30 of 600 rows | fill with the training median |
| only 35.7% cancelled | the label | baselines and a tuned threshold |

Chapter 6 one-hot encoded the *output*. Here we one-hot encode an *input* — same
idea, same reason: `basic=0, plus=1, pro=2` would tell the network that plus is
"between" basic and pro and that pro is three times basic, which is nonsense.
Three separate 0/1 columns say only "it is this one".

---

## 10.2 Project layout

```
churn/
├── Cargo.toml
├── data/
│   └── subscriptions.csv
└── src/
    ├── main.rs      the three subcommands, and the training loop
    ├── data.rs      reading and validating the CSV
    ├── prep.rs      encoding + imputation + scaling, saved to disk
    └── metrics.rs   confusion matrix, precision, recall, F1
```

The split isn't decoration. Each file answers one question:

- **`data.rs` — what does the file contain?** Nothing here knows about neural
  networks. If your CSV changes, you edit this file and nothing else.
- **`prep.rs` — how does a row become numbers?** This is the file that gets
  saved next to the model. Chapter 9's rule, enforced by the module boundary.
- **`metrics.rs` — how good is it?** No training code, so you can score a model
  you didn't train.
- **`main.rs` — what do we do with all that?**

To follow along:

```bash
cargo new churn && cd churn
cargo add rusting_brain
mkdir data && cp .../tutorials/data/subscriptions.csv data/
```

---

## 10.3 `data.rs` — read the file, and complain loudly

The type says what the data *is*, including the part that's missing:

```rust
pub struct Row {
    pub months_active: f32,
    pub monthly_hours: Option<f32>,   // genuinely blank in some rows
    pub support_tickets: f32,
    pub plan: String,
    pub cancelled: bool,
}
```

`Option<f32>` is the important line. A missing reading is not zero hours — zero
hours is a real, meaningful value that means "logged in and did nothing". If you
parse blanks as `0.0` you have silently invented data, and your model will learn
that inactive users don't churn. Rust's type system won't let you forget the
difference; use that.

The reader checks everything and reports the **line number**:

```rust
if header != expected {
    return Err(format!("unexpected header: {header:?}").into());
}
...
if !PLANS.contains(&plan.as_str()) {
    return Err(format!("line {line_no}: unknown plan {plan:?}").into());
}
let cancelled = match c[4].trim() {
    "yes" => true,
    "no" => false,
    other => return Err(format!("line {line_no}: cancelled must be yes/no, got {other:?}").into()),
};
```

That's more validation than the earlier chapters, and it's deliberate. A typo in
row 400 of a CSV does not announce itself — an unvalidated reader turns it into
a slightly worse model and you never find out. Ten minutes of `match` arms here
saves a week of "why is the model bad".

One more small thing that pays off later:

```rust
pub const FEATURE_NAMES: [&str; 6] = [
    "months_active", "monthly_hours", "support_tickets",
    "plan=basic", "plan=plus", "plan=pro",
];
```

The column order is written down **once**. `input_size(FEATURE_NAMES.len())`
then can't drift out of sync with the encoder.

---

## 10.4 `prep.rs` — the part that ships with the model

Chapter 9's lesson, now with a third thing to remember:

```rust
pub struct Preprocessing {
    pub hours_median: f32,   // for the blanks
    pub min: Vec<f32>,       // for the scaling
    pub max: Vec<f32>,
}
```

### Imputation

Filling a blank with a stand-in value is called **imputation**. The median is
the safe default for a numeric column — unlike the mean, one absurd outlier
can't drag it.

```rust
let mut hours: Vec<f32> = rows.iter().filter_map(|r| r.monthly_hours).collect();
hours.sort_by(|a, b| a.partial_cmp(b).unwrap());
let hours_median = hours[hours.len() / 2];
```

**Fitted on the training rows only** — the median is something learned from
data, so computing it over the whole file is leakage, exactly like the scaler in
chapter 4. And it must be *saved*, because a prediction six months from now with
a missing reading has to be filled with the same number the model was trained
with.

> An honest caveat: imputation is a guess, and the model can't tell an imputed
> value from a measured one. When a lot of values are missing, a common
> improvement is to add a `hours_was_missing` 0/1 column so the network can learn
> that missingness itself is a signal. With 30 rows out of 600 it isn't worth it
> here — but it's the first thing to try if blanks become common.

### Encoding

```rust
fn encode(r: &Row, hours_median: f32) -> Vec<f32> {
    let mut v = vec![
        r.months_active,
        r.monthly_hours.unwrap_or(hours_median),
        r.support_tickets,
    ];
    v.extend(PLANS.iter().map(|p| if *p == r.plan { 1.0 } else { 0.0 }));
    v
}
```

Then min-max scaling on top, so all six columns land in `0.0–1.0`. The one-hot
columns are already 0/1 and pass through unchanged.

`transform` is the only public way in, and it does encode → impute → scale in
one call:

```rust
pub fn transform(&self, row: &Row) -> Vec<f32>
```

There is deliberately no way to get a half-processed row out of this module.
That's how you make chapter 9's "confidently wrong" bug impossible rather than
merely unlikely.

---

## 10.5 `main.rs` — training

The architecture:

```rust
Network::builder()
    .input_size(FEATURE_NAMES.len())    // 6
    .dense(16, Activation::Relu)
    .dense(8, Activation::Relu)
    .dense(1, Activation::Sigmoid)      // one probability
    .loss(Loss::BinaryCrossEntropy)     // the partner of Sigmoid
    .optimizer(Optimizer::adam(0.004))
    .seed(42)
    .build()
```

Two hidden layers this time, because six inputs with a categorical among them
has more structure than the flowers did. One `Sigmoid` output with
`BinaryCrossEntropy` is the standard yes/no setup from chapter 3 — the output is
"probability this customer cancels".

The split is 70/15/15, and it uses a **fixed permutation** so `train`,
`evaluate` and `predict` all agree about which rows are the test set:

```rust
let mut state: u64 = 20260905;
for i in (1..idx.len()).rev() {
    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let j = (state >> 33) as usize % (i + 1);
    idx.swap(i, j);
}
```

This matters more than it looks. If `evaluate` reshuffled differently, it would
score the model on rows it was trained on and report a wonderful, meaningless
number.

The loop is chapter 7's early stopping, verbatim:

```
600 rows, 214 cancelled (35.7%), 30 missing monthly_hours
split: 420 train / 90 validation / 90 test

filling 30 blank hours cells with the training median: 30.8

 epoch     train      val
     1   0.6219   0.6501  *
     2   0.5904   0.6303  *
     3   0.5624   0.6019  *
     4   0.5345   0.5731  *
     5   0.5039   0.5343  *
     6   0.4612   0.4929  *
     7   0.4110   0.4373  *
     8   0.3677   0.3896  *
     9   0.3270   0.3452  *
    10   0.2974   0.3157  *
    20   0.2565   0.2918
    30   0.2510   0.2917
    40   0.2493   0.2874

stopped early at epoch 46: no improvement for 30 epochs
best epoch 16, validation loss 0.2828
```

Read that carefully. Validation loss falls fast for ten epochs, then flattens
completely. The best model was at **epoch 16** — but training ran to 46 before
patience expired, and the weights at epoch 46 are *not* the ones we keep. This
is exactly why chapter 7 insisted on snapshotting with `model.clone()` instead
of just stopping.

Meanwhile train loss keeps creeping down (0.2565 → 0.2493) while validation
doesn't. That gap is the beginning of overfitting, caught early.

---

## 10.6 Choosing the threshold

Here's the step that has no equivalent in the earlier chapters, and it's the one
that turns a model into a decision.

`Sigmoid` gives a probability. Turning it into "call this customer" needs a
cutoff, and 0.5 is a *default*, not a right answer. So we sweep it — **on the
validation set**:

```
choosing a decision threshold on validation:
  threshold   accuracy   precision   recall      F1
      0.20      84.4%      75.0%      94.7%   0.837
      0.30      87.8%      80.0%      94.7%   0.867
      0.40      87.8%      80.0%      94.7%   0.867
      0.50      90.0%      83.7%      94.7%   0.889  <-
      0.60      88.9%      86.8%      86.8%   0.868
      0.70      87.8%      90.9%      78.9%   0.845
      0.80      86.7%      96.4%      71.1%   0.818
chose threshold 0.50 (F1 0.889)
```

The whole precision/recall trade-off from chapter 6 laid out in one table:

- **Low threshold (0.20):** catches 94.7% of leavers, but a quarter of your
  alerts are false alarms.
- **High threshold (0.80):** 96.4% of your alerts are real — and you miss 29%
  of the customers who leave.

**Which one is right depends on what an error costs you, not on the maths.** If
an intervention is a cheap automated email, take the low threshold and accept
the false alarms. If it's a phone call from a human, or a discount you have to
honour, false alarms are expensive and you want the high one. F1 (the balance of
the two) picked 0.50 here, but F1 is only the right objective when the two
mistakes cost about the same.

The threshold is a learned parameter like any other, so it gets saved:

```
saved churn_model.json, churn_model.prep, churn_model.threshold
```

Three files, one model. All three ship together.

---

## 10.7 `evaluate` — the number you're allowed to report

Fresh process. Loads the three files, touches the test set for the first time:

```
test set: 90 rows, 28 actually cancelled

  baseline (predict nobody churns): 68.9% accuracy, 0% recall

  at the chosen threshold 0.50 - THIS is the reported result:
                 predicted stay   predicted churn
  actual stay                58                 4
  actual churn                5                23
  accuracy 90.0%   precision 85.2%   recall 82.1%   F1 0.836
```

Start with the baseline, always. A model that predicts "nobody churns" scores
**68.9% accuracy** — and is completely worthless, because it finds zero of the
customers you actually wanted to find. Our 90.0% is only meaningful next to that
68.9%, and the recall of 82.1% is the number that actually says the model works.

The confusion matrix names the two mistakes:

- **4 false alarms** — customers we'd contact who were never going to leave.
  Cost: a wasted email.
- **5 missed** — customers who left without us noticing. Cost: the customer.

Then, printed but explicitly fenced off:

```
  for reference only, other thresholds on the test set:
  threshold   accuracy   precision   recall      F1
      0.20      84.4%      69.4%      89.3%   0.781
      0.30      83.3%      68.6%      85.7%   0.762
      0.40      84.4%      70.6%      85.7%   0.774
      0.50      90.0%      85.2%      82.1%   0.836  <- chosen
      0.60      90.0%      85.2%      82.1%   0.836
      0.70      90.0%      91.3%      75.0%   0.824
      0.80      86.7%      90.0%      64.3%   0.750
```

> **You may report this table. You may not choose from it.** The threshold was
> decided on validation and is now fixed. Going back and picking 0.70 because it
> looks better here would spend the test set (chapter 8) — and would be exactly
> the kind of quiet self-flattery that makes published ML numbers untrustworthy.
>
> It happens that validation's choice, 0.50, is also the best F1 on test. That's
> a reassuring sign that the tuning generalised. It is not something you get to
> take credit for after the fact.

Finally, the metric a business would actually ask for:

```
  top 20 highest-risk customers: 18/20 really did cancel
```

Nobody has budget to call all 90 customers. They have budget for 20. Ranking by
risk score and taking the top 20 gives **18 real saves out of 20 calls** — and
that framing sidesteps the threshold question entirely. When your model feeds a
limited-capacity process, "how good is the top N" is usually the honest metric.

---

## 10.8 `predict` — the model in use

```bash
$ cargo run --release -- predict 3 45.0 6 basic
churn risk: 98.8%
decision at threshold 0.50: AT RISK - worth an intervention
```

Three months in, six support tickets, on the cheapest plan. The model is nearly
certain, and it's the profile you'd expect.

```bash
$ cargo run --release -- predict 40 8.0 0 pro
churn risk: 1.7%
decision at threshold 0.50: likely to stay
```

Long-standing pro customer with no complaints. Note that it *is* a low-usage
customer — 8 hours a month — and the model still says stay. It learned that
tenure and plan outweigh usage, which matches how the data was generated.

Missing values work at prediction time too, using the saved median:

```bash
$ cargo run --release -- predict 12 ? 3 plus
churn risk: 76.8%
```

And bad input is refused rather than guessed at:

```bash
$ cargo run --release -- predict 12 25.0 3 gold
Error: "unknown plan \"gold\""
```

That error is worth more than it looks. Without the check, `gold` would encode
as `[0,0,0]` — a customer on no plan at all, a row the model has never seen —
and it would return a confident number anyway. **Validate inputs at the edge of
your system.** A model will always answer; it's your job to only ask it
answerable questions.

---

## 10.9 Adapting this to your own data

The skeleton transfers. What changes, and where:

| Your situation | Change |
|---|---|
| different columns | `data.rs`: `Row`, the header check, `FEATURE_NAMES` |
| a new categorical column | `prep.rs`: `encode`, plus its constant list in `data.rs` |
| predicting a number, not yes/no | `.dense(1, Linear)` + `Loss::Mse`; swap `metrics.rs` for MAE/RMSE (chapter 5) |
| more than two classes | `.dense(N, Softmax)` + `Loss::CrossEntropy`; `argmax` instead of a threshold (chapter 6) |
| model underfits (train loss high) | wider/deeper layers, more epochs, higher learning rate |
| model overfits (val ≫ train) | smaller network, more data, stop earlier |
| under ~2000 rows | k-fold cross validation instead of one split (chapter 8) |

What does **not** change, in any project:

1. Validate the input file and fail on the bad line.
2. Fit every learned transform on training data only.
3. Save the preprocessing next to the weights.
4. Keep the best model, not the last one.
5. Tune on validation, report on test, once.
6. Compare against a baseline before believing any number.

---

## Try it yourself

1. Change the threshold objective from F1 to "the highest recall with precision
   above 80%". Which threshold wins, and how does the test result change?
2. Add a `hours_was_missing` feature column. Does it help? (Be honest — compare
   against the seed noise from chapter 8 before you claim it did.)
3. Add a `--seed` argument to `train` and run five seeds. Report `mean ± std`
   like chapter 8, not a single number.
4. Add a `rank` subcommand that reads a CSV of customers with no `cancelled`
   column and prints them sorted by risk. This is the shape most real deployments
   actually take.
5. Replace `save_json` in `train` with a checkpoint every 10 epochs (chapter 9)
   and add a `--resume` flag.
6. Make `evaluate` refuse to run twice by writing a marker file. Silly? It's a
   real discipline problem, and you now understand why.
7. Slice the test errors by `plan`. Is the model worse on one of the three?

---

## Recap

You've now built the whole thing:

- A **project layout** where reading data, preparing it, training, and scoring
  are four separate files with four separate jobs.
- **Categorical inputs** one-hot encoded, never integer-coded.
- **Missing values** as `Option`, imputed with a training-set median that gets
  saved with the model.
- **Early stopping** keeping the epoch-16 model even though training ran to 46.
- A **decision threshold** chosen on validation, saved, and reported honestly —
  with the precision/recall trade-off made explicit rather than defaulted away.
- A **baseline** (68.9%) that gives the result (90.0%) its meaning.
- A **ranking metric** (18/20) that matches how the model would really be used.
- Three files shipped together: weights, preprocessing, threshold.

That's a complete machine learning project. The remaining chapters are
extras — running on a GPU, importing models from other frameworks, and what to
do when things break.

---

**Next:** [11. Training on the GPU with CUDA](11_gpu_cuda.md)
