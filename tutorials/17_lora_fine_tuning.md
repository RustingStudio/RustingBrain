# 17. Fine-Tuning With LoRA

**You need:** chapter 16, and chapter 11 if you have a GPU.

**Time:** 45 minutes.

**Full code:** [`code/17_lora_fine_tuning.rs`](code/17_lora_fine_tuning.rs)

Chapter 16 trained every weight of a model from nothing. Most real work starts
from a model somebody else already trained and teaches it one more thing: a
domain, a house style, a task. Training every weight again for that is
wasteful, and on a 12 GB card it is often impossible.

LoRA (low-rank adaptation) trains a few small matrices beside the frozen
weights instead. This chapter adapts a base model to new text, saves the
adapter on its own, loads it back, and merges it into the weights.

---

## 17.1 What an adapter is

Every attention and feed-forward projection holds a weight `W` of shape
`[out, in]`. An adapter adds two matrices next to it:

```text
down : [rank, in]
up   : [out, rank]

output = W x + (alpha / rank) * up (down x)
```

`W` is frozen. Only `down` and `up` train. With `rank` 8 and a `1024 x 1024`
projection, the adapter has `8 * (1024 + 1024) = 16,384` weights against the
projection's 1,048,576: one and a half percent.

The saving is bigger than the weight count. A frozen weight needs no gradient
and no Adam moments, so a parameter that cost four weight-sized buffers to
train now costs one. That is what lets a model that only just fits for
inference be fine-tuned on the same card.

`up` starts at zero, so the adapted model's first forward pass returns exactly
what the base model returned. Training moves it away from there.

## 17.2 Attach

```rust
let mut model = TransformerLm::load_bin("base.rbw")?;
model.add_lora(LoraConfig::new(8).alpha(16.0).seed(3))?;
model.optimizer = Optimizer::adam(1e-3);
// model.to_cuda(0, 9_000)?;
```

Three rules, in the order they bite:

- **`add_lora` runs on the host.** Freezing releases gradient and moment
  buffers; a model already on the device has allocated them. Attach first,
  then `to_cuda`.
- **Give it a fresh optimizer.** A learning rate around `1e-4` to `1e-3` is
  usual, lower than pretraining. `load_bin` restores the base's optimizer
  settings, which were chosen for a different job.
- **The vocabulary is fixed.** An adapter changes how the model uses its
  tokens, not which tokens exist. Build the tokenizer from everything the
  model will ever see before you train the base. The example fits one
  character vocabulary over both texts for that reason.

Everything else in the training loop is unchanged: `train_step_batch`,
`step`, `step_clipped`, `Schedule`, `TokenFile`. The frozen parameters ignore
the step.

`trainable_parameters()` reports what the optimizer will move:

```rust
let total = model.parameter_counts().total;
let trainable = model.trainable_parameters();
```

## 17.3 Rank and alpha

`rank` is the bottleneck width. 8 to 32 is the usual range. More rank can
learn more, and costs more memory and more chance of overfitting a small
dataset.

`alpha / rank` scales the adapter's output. Keeping `alpha` proportional to
`rank` means raising the rank does not, by itself, make the adapter push
harder. The conventions are `alpha == rank` (the default of
`LoraConfig::new`) and `alpha == 2 * rank`.

`seed` fixes the random start of `down`. Two runs with the same seed and data
start from the same adapter.

## 17.4 Save, load, merge

```rust
model.save_lora("domain.rbl")?;
```

writes the adapter alone. A rank-16 adapter over a 50M-parameter model is a
few megabytes, and several adapters can share one base checkpoint.

To rebuild the fine-tuned model, prepare the base the same way and read the
adapter into it:

```rust
let mut rebuilt = TransformerLm::load_bin("base.rbw")?;
rebuilt.add_lora(LoraConfig::new(8).alpha(16.0).seed(3))?;
rebuilt.load_lora("domain.rbl")?;
```

`load_lora` checks the rank, alpha and every shape, and refuses a mismatch
rather than loading part of an adapter.

For deployment, fold the adapter into the weights:

```rust
rebuilt.merge_lora()?;   // host only: call to_cpu first if needed
```

What comes out is an ordinary model with no adapter to carry and no extra
inference cost. It is also the only way to get a trained adapter through
`quantize`.

## 17.5 The script

[`code/17_lora_fine_tuning.rs`](code/17_lora_fine_tuning.rs) trains a base
model for 300 steps on a paragraph about ownership, then adapts it for 200
steps to a paragraph about lifetimes. The output below came from the GPU path,
with the two commented `to_cuda` lines enabled; your digits will differ a
little:

```text
base model    general 0.1115   domain 4.8584
adapter       65536 of 594304 weights train (11.0%)
fine-tuned    general 1.7315   domain 0.1529
files         base.rbw 2378369 bytes, domain.rbl 263096 bytes
reloaded      domain 0.1528
merged        domain 0.1528   has adapter: false
```

Read it line by line:

- The base model knows its own text (loss 0.11) and nothing of the new one
  (4.86).
- Eleven percent of the weights train. That share is high only because the
  model is tiny and its embedding table is a large part of it; on a model
  with `d_model` 1024 the same rank is closer to one percent.
- After 200 steps the adapter has learned the new text: 4.86 down to 0.15.
- The adapter file is a ninth of the base file, for the same reason as the
  eleven percent.
- The rebuilt model and the merged model give the loss the trained model
  gave. The last digit moves because the trained model ran on the device in
  BF16 and the rebuilt one runs on the host in FP32.

## 17.6 Forgetting

Look at the `general` column. Loss on the base model's own text went from 0.11
to 1.73. The base weights never changed, and the model still got worse at the
old text, because the adapter changes what every projection outputs.

LoRA limits how much the model can change, which makes forgetting slower and
smaller than in a full fine-tune. It does not prevent it. Three things help:

- **Measure it.** Keep a held-out slice of the original data and evaluate it
  next to the new task, exactly as chapter 16 evaluates a validation set.
- **Mix in old data.** A small share of the original corpus in every batch,
  with `TokenMix`, holds the old behaviour in place.
- **Keep the adapter separate.** An unmerged adapter can be removed. Ship
  the base and the adapter, and the original model is one `load_bin` away.

---

## Exercises

1. Set the rank to 2, then 32. How do the trainable share, the adapter file
   size and the final `domain` loss change?
2. Train the adapter for 50 steps instead of 200. Does `general` loss rise as
   much? Plot both columns every 10 steps.
3. Load `domain.rbl` onto a base prepared with `LoraConfig::new(8)` (alpha 8).
   Read the error.
4. Build a mixed corpus with `TokenMix`: nine parts `domain.bin` to one part
   `general.bin`. Fine-tune again and compare both losses with the table
   above.
5. Merge, then `quantize`, and evaluate. How much does int8 cost on top of
   the merge?

---

## Recap

- LoRA trains `down` and `up` beside each frozen projection:
  `W x + (alpha / rank) * up (down x)`.
- Frozen weights need no gradient and no moments, which is most of the
  memory saving.
- `add_lora` on the host, before `to_cuda`. Fresh optimizer, smaller rate.
- `save_lora` writes the adapter alone; `load_bin`, `add_lora` with the same
  config, then `load_lora` rebuilds the model.
- `merge_lora` folds the adapter in for deployment and quantization.
- The adapter can still make the model forget. Evaluate the old data, and mix
  some of it into the new.

**Back to:** [the index](README.md)
