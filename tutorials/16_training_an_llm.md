# 16. Training a Language Model End to End

**You need:** chapters 14 and 15, and chapter 11 if you have a GPU.

**Time:** an afternoon to set up. Days to run.

**Full code:** [`code/16_training_an_llm.rs`](code/16_training_an_llm.rs)

Chapter 14 trained a model on one paragraph in a few seconds. This chapter is
about the version that runs for a week, and everything that only becomes a
problem at that length.

Nothing in the API changes. What changes is that you can no longer afford to
get anything wrong, because a mistake costs days rather than seconds.

---

## 16.1 Decide the shape before you start

The single most expensive mistake is starting a long run with the wrong
architecture. `load_bin` takes the model's shape from the checkpoint, so
`d_model` and `n_layers` cannot be changed on a resume. Changing them means
starting over.

Work top down.

**Start from the data.** The working rule is about twenty tokens of training
text per *active* parameter. Below that the model memorizes instead of
generalizing; far above it you are leaving capacity unused.

```
3.5B tokens / 20  ≈  175M active parameters
```

This is where mixture of experts earns its place. You can hold 300M total — more
knowledge in the weights — while paying for 175M per token, which is what the
corpus supports.

**Then the shape.** With an active budget fixed, two numbers decide whether the
model is wide or deep:

```
aspect ratio = d_model / n_layers        working range: 50–100
```

Outside that range models train worse at the same parameter count. Inside it,
the choice is mostly about hardware — wider is faster per parameter on a GPU
because the GEMMs are larger.

**Then the rest follows.** These are conventions, not tuning opportunities:

| Knob | Value |
|---|---|
| `head_dim` | 64 |
| `n_heads` | `d_model / 64` |
| `n_kv_heads` | `n_heads / 4` |
| `d_ff` | ≈ `2.67 × d_model`, rounded to a multiple of 64 |
| `moe_d_ff` | ≈ `d_ff / 4` |
| `experts` | `(8, 2)` |
| `moe_layers` | `n_layers / 4 .. n_layers` |

A 300M-total model on a 12 GB card:

```rust
let mut model = TransformerLm::builder()
    .vocab_size(32_000)
    .d_model(1024)
    .n_layers(20)            // aspect ratio 51
    .heads(16, 4, 64)
    .d_ff(2816)
    .moe_d_ff(512)
    .experts(8, 2)
    .moe_layers(5..20)
    .shared_expert(true)
    .max_seq_len(1024)
    .optimizer(Optimizer::adam(3e-4))
    .seed(42)
    .build()?;
```

**Then measure it, before you start.** `sweep_arch` times one step and reports
peak memory for exactly one configuration, in its own process:

```bash
cargo run --release --features cuda --example sweep_arch -- 32000 1024 20 8 512 1 8 2
```

[`docs/baseline.md`](../docs/baseline.md) has the measured table for an RTX
3060, including a "days for 50B tokens" column. Read that column before
committing to a shape. The difference between 13 days and 37 days is one
architecture decision.

---

## 16.2 Tokenize once, not every epoch

Tokenizing is slow, deterministic, and you will do it many times if you put it
in the training loop. Do it once and write the ids to disk.

```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let text = std::fs::read_to_string("corpus.txt")?;
let ids: Vec<u32> = tokenizer.encode(text, false)?.get_ids().to_vec();

let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
std::fs::write("train.u32", bytes)?;
```

With a vocabulary under 65,536, `u16` halves the file. A 3.5B-token corpus is
14 GB as `u32` and 7 GB as `u16`.

Then the loop reads windows out of that flat array:

```rust
fn window(ids: &[u32], start: usize, seq_len: usize) -> Vec<u32> {
    ids[start..start + seq_len].to_vec()
}
```

Hold back a slice — the last 1% is enough — and never train on it. Chapter 8's
argument has not changed: without held-out data you cannot tell learning from
memorizing, and at this scale you will not be able to tell by looking at the
samples either.

---

## 16.3 The loop

```rust
model.set_mixed_precision(true);
model.to_cuda(0, 9_000)?;

for step in 1..=total_steps {
    model.optimizer = Optimizer::adam(learning_rate(step, total_steps));

    model.zero_grad();
    let mut lm = 0.0;
    for _ in 0..accumulate {
        let batch = TokenBatch::new(&next_micro_batch())?;
        lm += model.accumulate_step(&batch)?.lm_loss;
    }
    model.step(1.0 / accumulate as f32);

    if step % 100 == 0 {
        println!("{step}  loss {:.4}  ppl {:.2}", lm / accumulate as f32,
                 (lm / accumulate as f32).exp());
    }
    if step % 2_000 == 0 {
        checkpoint(&mut model, step)?;
    }
}
```

Four things in there deserve their own explanation.

### Mixed precision

`set_mixed_precision(true)` runs the GEMMs with BF16 operands while keeping
FP32 master weights. It is roughly 40% less activation memory and a substantial
speedup on tensor cores, and because the weights that accumulate updates stay
FP32, it does not cost accuracy in practice.

Turn it on. The only reason not to is a card with no BF16 support.

### Gradient accumulation

Device memory bounds the batch you can run at once, not the batch you can train
with. `zero_grad`, several `accumulate_step` calls, one `step(1.0 / n)` gives
you the gradient of the whole group.

```rust
model.zero_grad();
for part in parts {
    model.accumulate_step(part)?;
}
model.step(1.0 / parts.len() as f32);
```

The scaling belongs on the step rather than on each backward pass. Adam
normalizes by the gradient's own second moment, so scaling every accumulation
identically would cancel out and change nothing.

Batch 4 with 4 accumulations and batch 16 with 1 are the same effective batch
and the same gradient — with one exception. **A MoE model's load-balancing loss
is not exactly reproduced.** It is computed from routing fractions over
whatever batch the layer sees, so four sequences in two accumulations balance
the experts against two different halves rather than against the whole. The
language-modelling gradient is unaffected; only the auxiliary term shifts. In
practice this does not matter, but it is why two runs with different
accumulation settings will not match to the last digit.

### The learning-rate schedule

A constant rate is wrong at both ends of a long run. Early on, Adam's moment
estimates are still garbage and a full-size step on a garbage direction can
knock the model somewhere it takes thousands of steps to recover from. Late on,
large steps prevent the model from settling.

Warmup then cosine decay:

```rust
fn learning_rate(step: usize, total: usize) -> f32 {
    let peak = 3e-4;
    let warmup = 2_000;

    if step < warmup {
        peak * step as f32 / warmup as f32
    } else {
        let progress = (step - warmup) as f32 / (total - warmup) as f32;
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        peak * (0.1 + 0.9 * cosine)     // decay to 10% of peak, not to zero
    }
}
```

Assigning `model.optimizer` is all this needs. The optimizer is read fresh at
every `step`, so there is no schedule object and nothing to register.

Decaying to 10% rather than 0 leaves the model still learning at the end, which
matters if you decide to extend the run.

### Logging

Log the loss, not a smoothed version of it. A smoothed curve hides exactly the
spike you need to see. If you want a smooth line, smooth it in the plot.

---

## 16.4 Checkpoints

A week-long run will be interrupted. Plan for it.

```rust
fn checkpoint(model: &mut TransformerLm, step: usize) -> Result<(), NetworkError> {
    model.sync_from_device()?;                              // weights back to host
    model.save_bin("checkpoint.rbw", Precision::F32)?;
    model.save_optimizer_state("checkpoint.rbw.opt")?;
    println!("checkpoint at step {step}");
    Ok(())
}
```

Three rules, each of which costs a day if broken.

**`F32`, always, for a checkpoint you might resume.** `Q8` rounding is
invisible for inference and visible in a resumed loss curve. Quantize at the
end, to a separate file, for shipping.

**Save the optimizer state.** Without it Adam's moments restart at zero while
the step counter carries on, so bias correction no longer compensates and the
first updates after the resume are several times larger than the ones the run
was taking before it stopped. That shows up as a loss spike exactly at the
resume point.

**Order matters on resume.** Uploading a parameter to the device zeroes its
moments, so `load_optimizer_state` must come *after* `to_cuda`:

```rust
let mut model = TransformerLm::load_bin("checkpoint.rbw")?;
model.set_mixed_precision(true);
model.to_cuda(0, 9_000)?;                        // this zeroes the moments
model.load_optimizer_state("checkpoint.rbw.opt")?;   // so restore them after
```

Get it backwards and everything succeeds, nothing errors, and you have silently
thrown away the optimizer state you just carefully saved.

If you only have weights and no moments — an older checkpoint, or one someone
sent you — tell the model how many steps it has taken so bias correction
matches:

```rust
model.set_optimizer_step(48_000);
```

Keep more than one checkpoint. A run that diverges takes its last checkpoint
with it.

---

## 16.5 Evaluation

Loss on the training batch tells you the model is still moving. It does not
tell you it is learning language.

```rust
let mut held_out_loss = 0.0;
for batch in &validation_batches {
    held_out_loss += model.forward_batch(batch)
        .and_then(|(logits, _)| causal_lm_loss_batch(&logits, batch))?
        .loss;
}
let perplexity = (held_out_loss / validation_batches.len() as f32).exp();
```

Run it every few thousand steps. Two curves, training and held-out, is chapter
8's whole method applied to a much longer run.

| What you see | What it means |
|---|---|
| Both falling | Working. Keep going |
| Training falls, held-out flat | Memorizing. More data, or a smaller model |
| Held-out rises | Overfitting from here on. The best checkpoint is behind you |
| Both flat from the start | Something is broken. Chapter 13 |
| Sudden spike in both | Loss spike. See below |

Also generate samples. A held-out perplexity of 18 and a sample that reads like
word salad means something is wrong with your tokenizer or your decoding, and
no number will tell you that.

---

## 16.6 What goes wrong at scale

### It does not fit in memory

In order, cheapest first:

1. Smaller `batch_size`, more accumulation. Mathematically almost the same,
   half the memory.
2. `set_mixed_precision(true)` if it is not already on. About 40% of activation
   memory.
3. `seq_len` 512 instead of 1024. Attention memory is quadratic in it.
4. `n_kv_heads` 4 → 2. Halves the KV cache.
5. More experts at a smaller `moe_d_ff`. Same total, less active.
6. Only then: fewer layers or a smaller `d_model` — and that is a different
   model, so it is a restart.

### A loss spike

The curve jumps and either recovers over a few hundred steps or does not.

Usually a bad batch — a run of repeated tokens, a chunk of binary that survived
your data cleaning, a document in a script the tokenizer handles badly. If it
recovers, ignore it. If it does not, restore the last checkpoint, skip past
that region of the corpus, and continue.

A spike exactly at a resume point is not a bad batch. That is the missing
optimizer state from 16.4.

### Throughput drops partway through

Almost always thermal. Check the clock speed, not the code.

Failing that, check whether something else started using the GPU.
[`docs/baseline.md`](../docs/baseline.md) records the same configurations
measured on a contended card and on an idle one: **2.14x**, with nothing else
changed.

### Everything works and the model is bad

The most common cause is data, not architecture. Deduplicated, cleaned,
reasonably diverse text beats a cleverer model on scraped noise every time, and
it is much cheaper to fix.

---

## 16.7 The script

[`code/16_training_an_llm.rs`](code/16_training_an_llm.rs) is a complete
training program at a size that finishes: a held-out split, gradient
accumulation, a warmup-cosine schedule, periodic evaluation, a checkpoint, and
a resume that restores the optimizer state in the right order.

It runs on a CPU in under a minute. The only differences between it and a real
run are the size of the model, the size of the corpus, and two commented lines.

```text
vocabulary 27, 120 training windows, 1 validation batches
0.6M total / 0.6M active (1.00x)

 step      lr     train    held-out
   50  0.00299   2.0141      2.5248
  100  0.00282   0.8525      2.4063
  150  0.00242   0.3388      2.7173
  200  0.00188   0.0612      3.5383
  250  0.00130   0.0578      3.6876
  300  0.00078   0.0544      3.7992
  350  0.00043   0.0477      3.8710
  400  0.00030   0.0513      3.9082

resuming from the step-200 checkpoint
  optimizer step counter restored: 200 (checkpoint was at 200)
  held-out loss at the checkpoint: 3.5383
```

That is textbook overfitting, and it is worth sitting with, because it is the
second row of 16.5's table drawn from life. Training loss falls from 2.01 to
0.05 — a beautiful curve, and if it were the only number you logged you would
call the run a success. Held-out loss bottoms out at step 100 and climbs from
there. **The best model this run ever produced existed at step 100**, and every
step after that made it worse while the training curve kept improving.

A 0.6M-parameter model on 400 characters of text has nothing else it can do. At
real scale the shape is the same and the turn is later, which is the entire
reason to log both curves rather than one.

The resume is also doing more than it looks. The held-out loss after loading
the step-200 checkpoint is 3.5383 — bit for bit the value the live model had at
step 200. That is what a correct checkpoint looks like: `F32` weights, restored
optimizer moments, restored step counter, and no discontinuity.

## 16.8 If something went wrong

| Symptom | Cause |
|---|---|
| Loss spike exactly at a resume | Optimizer state not loaded, or loaded before `to_cuda` |
| Resumed loss is slightly worse than saved | Checkpoint written with `Q8` |
| `optimizer state holds N parameters, this model has M` | The `.opt` file belongs to a different architecture |
| CUDA out of memory partway through | Peak is at the longest sequence, not the average. Budget for the longest |
| Held-out perplexity rises while training falls | Overfitting. Stop, take the earlier checkpoint |
| Two runs with the same seed differ | Different accumulation settings, on a MoE model. 16.3 |
| Throughput halved overnight | Thermal throttling, or another process on the card |
| Generation is fluent but wrong | Normal. That is what a model this size does |

---

## Exercises

1. Run `sweep_arch` on three architectures at your active-parameter budget.
   Which is fastest, and is it the one you expected?
2. Train 2,000 steps with warmup and 2,000 without. Compare the first 200 steps
   of each.
3. Checkpoint, then resume twice: once with `load_optimizer_state` after
   `to_cuda` and once before. Plot both.
4. Train a model deliberately too large for your corpus. Find the step where
   held-out loss turns up.
5. Shrink `max_seq_len` from 1024 to 512 and measure both tokens per second and
   peak memory. Is the trade worth it for your data?

---

## Recap

- Decide the architecture before the run. `load_bin` takes the shape from the
  checkpoint; `d_model` and `n_layers` cannot change on a resume.
- Size from the data: roughly 20 tokens per active parameter, aspect ratio
  50–100, everything else conventional.
- Measure with `sweep_arch` and read the days-per-50B column before committing.
- Tokenize once to a flat file. Hold back a validation slice.
- Mixed precision on. Gradient accumulation for effective batch. Warmup then
  cosine decay, set by assigning `model.optimizer`.
- Checkpoint in `F32`, with the optimizer state, and load that state *after*
  `to_cuda`.
- Track held-out perplexity, not just training loss, and read samples as well
  as numbers.
- When memory runs out, work down the list: batch, precision, sequence length,
  KV heads, expert width. Architecture last.
- When the model is disappointing, suspect the data before the architecture.

---

That is the course. You have a library that trains dense networks, transformer
language models, and sparse mixtures of experts, on a CPU or a GPU, from first
principles with no Python in the loop.

The reference documentation is in the source — every public type carries the
reasoning behind it, which is the part that does not fit in a tutorial.

**Back to:** [the index](README.md)
