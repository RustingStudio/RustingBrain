# 14. Tokens and a Transformer Language Model

**You need:** chapters 1–10. Chapter 11 if you want it to be fast.

**Time:** 2 hours.

**Full code:** [`code/14_language_model.rs`](code/14_language_model.rs)

Everything so far predicted one thing from a fixed set of features: a price
from four columns, a class from three measurements. A language model predicts
the next piece of text from all the text before it, and then does it again with
its own output appended.

That is the whole idea. Everything else in this chapter is machinery for making
it work at a scale where it is useful.

```
"fn main() {"  →  model  →  "\n"
"fn main() {\n"  →  model  →  "    "
"fn main() {\n    "  →  model  →  "println"
```

---

## 14.1 Next-token prediction is still classification

A language model is a classifier with a very large number of classes. Chapter 6
built a classifier over three species of flower; this one is a classifier over
32,000 possible next tokens, run once per position in the text.

Everything you learned there still applies. Softmax over the outputs. Cross
entropy against the correct answer. The same gradient.

Two things are genuinely new:

- **The input is a sequence, and order matters.** `let x = y` and `let y = x`
  contain the same tokens and mean different things.
- **Every position is a training example.** A 512-token sequence is not one
  training row; it is 511 of them — predict token 2 from token 1, token 3 from
  tokens 1–2, and so on. This is why language models learn from so much less
  human labelling than everything else: the label is the next token, and the
  text labels itself.

The second point is also why the architecture has to be *causal*. Position 5
must not see position 6, or predicting token 6 becomes a lookup rather than a
prediction, and the model learns nothing while reporting a wonderful loss.

---

## 14.2 Tokens

Models work on numbers, so text has to become numbers first. There are three
ways to do it and only one of them is used in practice.

**Characters.** Vocabulary of about 100. Simple, and every string encodes.
But `println` becomes seven separate decisions, sequences get very long, and
attention cost grows with the square of sequence length. Expensive.

**Words.** Vocabulary of hundreds of thousands, and it still cannot spell
`unwrap_or_else` if that word was not in the training data. Unbounded and
brittle.

**Subwords (BPE).** The compromise everything uses. Start from bytes, then
repeatedly merge the most frequent adjacent pair into a new token. Common
things become single tokens — `fn`, `impl`, `String`, `-> Result<` — and rare
things fall back to pieces. Fixed vocabulary, nothing is unrepresentable.

```
"fn main() {"  →  [1023, 887, 12, 13, 45]
```

RustingBrain does not ship a tokenizer. That is a deliberate boundary: a
tokenizer is a text-processing problem, not a neural-network one, and the
[`tokenizers`](https://crates.io/crates/tokenizers) crate already does it
properly with the Hugging Face format.

```bash
cargo add tokenizers
```

```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let ids: Vec<u32> = tokenizer.encode("fn main() {", false)?.get_ids().to_vec();
let text = tokenizer.decode(&ids, true)?;
```

For this chapter's code we will use a character-level tokenizer built in
fifteen lines, so the example runs with no extra dependency and you can read
every part of it. Swap in BPE when you move to real text; nothing else in the
model changes.

The model's only requirement is that every id is less than `vocab_size`. Break
that and you get:

```text
token id 41023 is outside the vocabulary of 32000
```

which almost always means the tokenizer and the checkpoint disagree.

---

## 14.3 Building the model

```rust
use rusting_brain::{Optimizer, TransformerLm};

let mut model = TransformerLm::builder()
    .vocab_size(32_000)
    .d_model(512)
    .n_layers(8)
    .heads(8, 2, 64)
    .d_ff(1408)
    .moe_layers([])            // dense; chapter 15 turns this on
    .max_seq_len(1024)
    .tie_embeddings(true)
    .optimizer(Optimizer::adam(3e-4))
    .seed(42)
    .build()?;

println!("{}", model.parameter_counts());
```

Each knob, and what it costs you:

| Knob | What it is | Effect |
|---|---|---|
| `vocab_size` | Number of distinct tokens | Sets embedding size: `vocab_size × d_model` parameters, often the largest single matrix |
| `d_model` | Width of the vector carrying each token | The main capacity dial. Cost grows with its square |
| `n_layers` | How many blocks stack | The main depth dial. Cost grows linearly |
| `heads(q, kv, dim)` | Query heads, key/value heads, size of each | See below |
| `d_ff` | Hidden width inside the feed-forward | Usually 2.5–4× `d_model`. Most of the parameters live here |
| `max_seq_len` | Longest context ever supported | Baked into the checkpoint. Raising it later means retraining |
| `rope_base` | Rotary frequency base | 10000 is standard. Larger stretches positional resolution over longer contexts |
| `tie_embeddings` | Reuse the embedding matrix as the output projection | Saves `vocab_size × d_model` parameters, costs a little quality |

### Heads, and why there are two counts

Attention runs in parallel "heads", each looking at a different relationship —
one might track matching brackets, another the subject of a sentence. `d_model`
is split across them.

`heads(8, 2, 64)` means eight query heads, two key/value heads, 64 dimensions
each. That is **grouped-query attention**: four query heads share each
key/value head.

Why bother? Because at generation time the keys and values of every previous
token have to be kept in memory (14.6). That cache is the dominant memory cost
of serving a model, and it is proportional to `n_kv_heads`, not `n_heads`. Going
from 8 to 2 cuts it by four with very little quality loss. `n_heads` must be a
multiple of `n_kv_heads`; setting them equal gives you ordinary multi-head
attention.

`head_dim` must be even, because rotary embeddings rotate dimensions in pairs.

---

## 14.4 What one block does

Eight of these, in a row:

```
input ──┬─────────────────────────────┐
        ↓                             │
    RMSNorm                           │
        ↓                             │
    attention (RoPE + causal mask)    │
        ↓                             ↓
        └───────────── + ─────────────┘   residual
                       ↓
                 ──────┬──────────────┐
                       ↓              │
                   RMSNorm            │
                       ↓              │
                   SwiGLU             │
                       ↓              ↓
                       └────── + ─────┘   residual
                              ↓
                           output
```

**RMSNorm** rescales each vector to a consistent magnitude so the next layer
sees inputs in a predictable range. It is LayerNorm without the mean
subtraction — one pass over the row instead of two, and empirically just as
good.

**Attention** lets each position look at earlier positions and pull in what it
needs. This is the part that makes order matter. The *causal mask* is what
stops position 5 seeing position 6.

**RoPE** (rotary position embedding) tells attention where each token is, by
rotating the query and key vectors by an angle proportional to position. The
useful property is that the rotation cancels in the dot product except for the
*difference* in positions, so attention naturally sees relative distance rather
than absolute index.

**SwiGLU** is the feed-forward: `(xW₁ ⊙ swish(xW₂))W₃`. It is where most of the
parameters and most of the arithmetic are, and the multiplicative gate lets the
layer suppress its own channels — which a plain ReLU MLP cannot do.

**Residuals** — the `+` arrows — are what make eight layers trainable at all.
The gradient reaches layer 0 through the additions without passing through
every transformation on the way.

---

## 14.5 Training

```rust
let batch: Vec<Vec<u32>> = vec![
    vec![10, 22, 7, 31, 4],
    vec![9, 18, 44, 2, 61],
];

let loss = model.train_step(&batch)?;
println!("loss {:.4}, perplexity {:.2}", loss.lm_loss, loss.lm_loss.exp());
```

`train_step` does the forward pass, computes the next-token cross entropy,
backpropagates, and updates the weights. One call, one optimizer step.

Sequences in one batch should be the same length. Different lengths are
allowed — shorter ones are padded on the right and the loss ignores the padded
positions — but padding is wasted arithmetic, so chop your corpus into
fixed-length windows instead.

### Read perplexity, not loss

`exp(loss)` is **perplexity**: roughly, how many tokens the model is choosing
between. It is the number worth watching, because it has a meaning. Every loss
the crate returns carries it:

```rust
let loss = model.train_step(&windows)?;
println!("loss {:.4}  ppl {:.2}", loss.lm_loss, loss.perplexity());
```

| Perplexity | What it means |
|---|---|
| = vocab_size | Uniform guessing. An untrained model. |
| 100 | Narrowed 32,000 options to about 100 |
| 20 | Respectable for a small model on narrow text |
| 5 | Very confident; check you are not evaluating on training data |
| 1 | Perfect prediction. On real text this means a bug |

A fresh 32,000-token model starts at a loss near `ln(32000) ≈ 10.4`. If your
first step prints something far from that, the model or the data is wrong
before training has begun.

### TokenBatch, for control

`train_step` builds a `TokenBatch` internally. Build it yourself when you want
padding control or a loss mask:

```rust
use rusting_brain::TokenBatch;

let batch = TokenBatch::new(&sequences)?;
println!("{} sequences × {} tokens", batch.batch(), batch.seq_len());
let loss = model.train_step_batch(&batch)?;
```

`with_loss_mask` marks positions that should not contribute to the loss — the
prompt half of a prompt/answer pair, for instance, when you want the model
graded only on its answer.

---

## 14.6 Generating text

Generation is: run the model, pick a token from the last position's
probabilities, append it, repeat. That loop is written for you:

```rust
use rusting_brain::Sampler;

let mut sampler = Sampler::temperature(0.8, None).top_k(40);
let continuation = model.generate(&prompt_ids, 200, &mut sampler)?;
```

`generate` returns the new tokens only, not the prompt. It builds the KV caches
itself and throws them away when it returns.

### Sampling

The model gives one score per vocabulary entry for the next position. Turning
that row into a token is a separate decision, and `Sampler` holds every knob for
it:

```rust
let mut sampler = Sampler::temperature(0.8, Some(42))
    .top_k(40)
    .top_p(0.95)
    .repetition_penalty(1.1);
```

**Greedy** (`Sampler::greedy()`) — always the highest score. Deterministic, and
it loops: "the the the". Use it when you are measuring something, not reading
it.

**Temperature** — divides the logits by `T` before the softmax. `T < 1` sharpens
toward the top choice, `T > 1` flattens toward uniform. `0.7–0.9` is the usual
range; `T = 0` is greedy. The `seed` argument is `Some` for a reproducible run
and `None` for a different continuation every time.

**Top-k** — keeps the `k` highest and samples among those. The point is to cut
the long tail: individually the 30,000 worst tokens are each nearly impossible,
but together they carry enough probability that one eventually gets picked, and
one wrong token derails everything after it. `k = 40` is a common default; `0`
keeps all of them.

**Top-p** (nucleus) — keeps the smallest set of tokens whose probabilities reach
`p`, so the number kept follows how confident the model is: a handful where the
model is sure, hundreds where it is not. `0.95` is a common default; `1.0` keeps
all of them.

**Repetition penalty** — divides the logit of every token already generated by
`penalty`, which is how a small model is stopped from repeating one phrase for
as long as you let it run. `1.0` leaves the logits alone; `1.1` is gentle.

Temperature reshapes the distribution and the two truncations cut it. They
compose, and they are applied in that order.

### Tokens as they arrive

A hundred tokens take as long as a hundred forward passes, so a program that
prints the result at the end prints nothing for several seconds.
`generate_with` hands over each token at the moment it exists, which is also
where you notice an end-of-text id or a stop sequence:

```rust
let continuation = model.generate_with(&prompt_ids, 200, &mut sampler, |id| {
    print!("{}", tokenizer.decode(&[id]));
    use std::io::Write;
    std::io::stdout().flush().ok();
    id != end_of_text
})?;
```

Returning `false` stops the generation. The token that stopped it is still in
the returned `Vec` — it was sampled, and hiding it would make `generate` and
`generate_with` disagree about what the model produced.

### More than one turn

`generate` drops its caches on return, so a chat loop that calls it once per
turn re-reads the whole conversation every time — quadratic in the number of
turns. A `Decoder` keeps them:

```rust
let mut decoder = model.decoder();

decoder.feed(&prompt_ids)?;                 // the user's first message
for _ in 0..100 {
    let id = decoder.next(&mut sampler)?;
    if id == end_of_text { break; }
}

decoder.feed(&next_message_ids)?;           // the second turn
let id = decoder.next(&mut sampler)?;       // nothing above is read again
```

`feed` reads only the ids you hand it. `decoder.history()` is everything the
session has seen, prompts and generated tokens alike — which is what you write
back to the screen or save as the transcript.

### Why any of that is fast

Done naively, generation is quadratic. To produce token 500 you would re-run all
499 previous tokens through all eight layers, and you have already done that
work 499 times.

The **KV cache** is what all three functions above use to avoid it, and if you
want the loop open — a custom stopping rule, a beam search, two models in
lockstep — it is three lines:

```rust
let mut caches = model.new_kv_caches();

// Prefill: the whole prompt in one pass.
let mut logits = model.forward_cached(&prompt_ids, &mut caches)?;

let mut generated = prompt_ids.clone();
for _ in 0..200 {
    let next = sampler.pick(logits.row(logits.rows - 1), &generated);
    generated.push(next);

    // Decode: one token, attending to everything cached.
    logits = model.forward_cached(&[next], &mut caches)?;
}
```

The caches carry the position, so you never track it yourself. Prefill is one
large matrix multiply over the prompt; each decode step is one small one. On
the default model that is roughly 2 ms per token on a CPU instead of 2 ms × the
sequence length.

Cost: memory. The cache holds `2 × n_layers × n_kv_heads × head_dim` floats per
token — which is precisely why `n_kv_heads` is 2 and not 8.

---

## 14.7 Saving

Dense networks used JSON (chapter 9). A transformer should not: JSON costs
about ten bytes per weight, so a 100M-parameter model becomes a gigabyte of
text.

```rust
model.save_bin("model.rbw", Precision::F32)?;   // 4 bytes per weight
model.save_bin("model.rbw", Precision::Q8)?;    // 1 byte per weight, lossy

let model = TransformerLm::load_bin("model.rbw")?;
```

The file carries the architecture as well as the weights, so `load_bin` needs
no builder — it reconstructs the model it was saved from.

**`F32` for anything you will resume.** `Q8` rounds each weight to 256 levels
with one scale per row, costing around 0.4% per weight. For inference that is
invisible. For a resumed Adam run it is a visible step in the loss curve, and
you will spend an hour looking for a bug that is just rounding.

Resuming also needs the optimizer:

```rust
model.save_optimizer_state("model.rbw.opt")?;
// later
model.load_optimizer_state("model.rbw.opt")?;
```

Without it, Adam's moment estimates restart at zero and the first few hundred
steps take much larger effective steps than they should. Chapter 16.4 covers
the ordering rule that makes this actually work on a GPU.

---

## 14.8 The whole thing, small

[`code/14_language_model.rs`](code/14_language_model.rs) trains a character-level
model on one paragraph until it can reproduce it, then generates from a prompt.
It is deliberately tiny — 0.6M parameters, a few seconds on a CPU — because the
point is to watch the machinery work end to end.

```bash
cargo run --release
```

```text
354 characters, 27 distinct -> vocabulary of 27
19 training windows of 64 tokens
0.6M total / 0.6M active (1.00x)

step    loss   perplexity   (uniform = 27)
   1  3.2626      26.12
 100  0.0165       1.02
 600  0.0124       1.01

prompt: "the borrow checker"
  T=0.2 k=8   "the borrow checker is not your enemy. it is a colleague who has read the code mo thaoref e de cisa"
  T=0.8 k=8   "the borrow checker is not your enemy. it is a colleague who has read the code mo thaoref e de cisa"
  T=1.5 k=27  "the borrow checker is not your enemy. it is a colleague who has read the code mo thaoref e de cisa"
```

Three things in that output are worth stopping on.

**Step 1 perplexity is 26.12 against a vocabulary of 27.** The untrained model
is guessing almost uniformly, exactly as 14.5 said it would. This is the
cheapest sanity check you have, and it catches a wrong `vocab_size` or
out-of-range ids before you waste an hour.

**Perplexity 1.01 means memorization, not language.** The model has 0.6M
parameters and 354 characters of text. It is not learning English; it is
storing the paragraph. That is the chapter 13.9 overfit test — proof the
pipeline works — and nothing more. On real text at this perplexity you have a
bug, almost always training data leaking into evaluation.

**Generation is verbatim for about 64 characters, then falls apart.** That is
`SEQ_LEN`. The model never saw a position past 64 during training, so rotary
embeddings are being asked to extrapolate to angles they were never fit for,
and the output degrades into plausible-looking rubble. Train on longer windows
and the boundary moves. This is the same effect people hit when they prompt a
real model past its context length.

Notice too that all three temperatures produce identical text — even `T=1.5`
with no top-k cut at all. When a model has memorized its corpus the probability
mass is a single spike: the top logit is twenty or more above the next, and
dividing by 1.5 does not close that gap. **Temperature only matters when the
model is genuinely uncertain.** On a real model, trained on more text than it
can store, the same three settings are three visibly different paragraphs.

## 14.9 If something went wrong

| Symptom | Cause |
|---|---|
| `token id N is outside the vocabulary of M` | Tokenizer and model disagree. They must be saved and loaded together |
| `sequence length N exceeds the configured maximum M` | `max_seq_len` is baked in at build time; this means retraining |
| `head_dim must be even for rotary embeddings` | RoPE rotates pairs |
| `n_heads must be a multiple of n_kv_heads` | Grouped-query attention needs whole groups |
| First loss is not near `ln(vocab_size)` | Wrong `vocab_size`, or ids out of range |
| Loss drops to near zero on real text | Evaluating on training data, or your windows overlap |
| Generation repeats one token forever | Greedy sampling. Add temperature and top-k, or a `repetition_penalty` |
| Generation is fluent nonsense | Undertrained. This is what a small model on little data does |
| Generation is `NaN` or one id | Overflow in a hand-written softmax — subtract the max, or use `Sampler` |
| Decoding slows down as it goes | Not using `forward_cached`, or rebuilding the caches each step |

---

## Exercises

1. Print the first `train_step` loss for `vocab_size` 100, 1000, and 32000.
   Confirm each is near `ln(vocab_size)`.
2. Generate the same prompt at temperature 0.1, 0.8, and 1.5. Then at
   `top_k` 1, 40, and `vocab_size`. Then at `top_p` 0.5 and 0.95, and explain
   why the second one changes less than you expected on a certain model.
3. Time 100 tokens with `forward_cached` against 100 tokens re-running the
   whole prefix through `forward_train` each step.
4. Set `tie_embeddings(false)` and compare `parameter_counts()` and the loss
   after 200 steps. Was the saving worth it?
5. Replace the character tokenizer with a real BPE one from the `tokenizers`
   crate. Note how many fewer tokens the same text becomes.
6. Raise `SEQ_LEN` from 64 to 128 and retrain. Confirm the coherent stretch of
   generated text grows with it, and explain why.

---

## Recap

- A language model is a classifier over the vocabulary, run at every position.
  Every position is a training example, which is why text labels itself.
- The causal mask is what keeps it a prediction rather than a lookup.
- Subword (BPE) tokenization is the practical middle between characters and
  words. RustingBrain leaves it to the `tokenizers` crate.
- `d_model` and `n_layers` are the capacity dials; `d_ff` holds most of the
  parameters; `max_seq_len` is baked into the checkpoint.
- Grouped-query attention exists to shrink the KV cache, which is the dominant
  memory cost of serving.
- A block is RMSNorm → attention+RoPE → residual → RMSNorm → SwiGLU → residual.
- Watch perplexity, not loss. An untrained model sits at `vocab_size`.
- `generate` is the whole decode loop; `generate_with` hands you each token as
  it arrives; `Decoder` keeps the caches between turns of a conversation.
- `forward_cached` plus a KV cache turns generation from quadratic to linear,
  and is there when you want the loop open.
- `Sampler` holds temperature, top-k, top-p and the repetition penalty — the
  knobs that stop generation looping or derailing.
- `save_bin` with `F32` to resume, `Q8` to ship. JSON is for dense networks.

---

**Next:** [15. Mixture of Experts](15_mixture_of_experts.md)
