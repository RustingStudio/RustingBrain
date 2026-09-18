# 15. Mixture of Experts

**You need:** chapter 14.

**Time:** 90 minutes.

**Full code:** [`code/15_mixture_of_experts.rs`](code/15_mixture_of_experts.rs)

Chapter 14's model ran every parameter for every token. A 300M-parameter model
did 300M parameters' worth of arithmetic per token, and a 3B one did ten times
that. Capacity and cost move together, which is a problem when you have one
GPU.

A mixture-of-experts layer breaks that link. It holds many feed-forward
networks and runs two of them per token.

```
dense:   token → [        one big FFN        ] → output

MoE:     token → router → expert 3 ─┐
                       → expert 7 ─┴─ weighted sum → output
                          (experts 1,2,4,5,6,8 idle)
```

Same latency per token, several times the parameters.

---

## 15.1 Total and active

Two numbers now describe a model instead of one.

**Total parameters** are everything in the checkpoint. This sets how much the
model can know, and how much disk and VRAM it occupies.

**Active parameters** are what actually run for one token. This sets the speed
and, in practice, how much training data you need.

```rust
println!("{}", model.parameter_counts());
// 55.2M total / 35.7M active (1.54x)
```

That ratio is the entire reason the layer exists. The default configuration
carries 55% more parameters than it pays for per token. Push it further — 16
experts instead of 8 — and the ratio grows while the speed does not change.

The rule of thumb for how much text you need is roughly twenty tokens per
*active* parameter. MoE is what lets you hold more knowledge than your corpus
would otherwise support: the total can exceed what the data justifies as long
as the active count does not.

---

## 15.2 Routing

The router is one bias-free `d_model → num_experts` projection. For each token
it produces one score per expert, softmax turns those into probabilities, and
the top `k` win.

```
token vector ──→ router ──→ [0.02, 0.61, 0.05, 0.28, 0.01, 0.01, 0.01, 0.01]
                                     ↑            ↑
                                 expert 1     expert 3       (top-2)

output = 0.69 × expert_1(token) + 0.31 × expert_3(token)
```

The two surviving gates are renormalized to sum to one, so the output stays on
the same scale regardless of how confident the router was.

`experts_per_token` is almost always 2. One is possible — that is Switch-style
routing — but with a single expert the router gets a weak training signal,
because there is no second choice to compare against. More than two erodes the
saving the layer exists for.

Routing is per token, not per sequence. The same sentence can send its verbs to
one expert and its punctuation to another, and it will, because nothing tells
the experts what to specialize in. They differentiate on their own, and what
they end up specializing in is usually not interpretable.

### Dropless

Many implementations give each expert a fixed capacity and throw away tokens
that overflow it, because fixed shapes are easier to batch. RustingBrain does
not: every token reaches every expert it was routed to, and the expert batches
are ragged.

That means no token is ever silently dropped, and training loss never has an
unexplained floor caused by discarded tokens. The cost is that the expert GEMMs
have varying shapes, which 15.6 comes back to.

---

## 15.3 The load-balancing loss

Left alone, routing collapses. An expert that is slightly better early gets
more tokens, trains faster on them, becomes more attractive, and takes more.
Within a few thousand steps one or two experts receive everything and the rest
are dead weight in the checkpoint — you have paid for eight experts and trained
two.

The fix is a second loss term that punishes imbalance:

```
aux_loss = num_experts × Σᵢ (fraction routed to expert i) × (mean gate for expert i)
```

It is minimized at 1.0 when every expert gets an equal share, and grows toward
`num_experts` as routing concentrates. It reaches the loss multiplied by a small
weight:

```rust
.aux_loss_weight(0.01)      // Switch Transformer default
```

0.01 is the value from the Switch Transformer paper and it is a good default.
Too low and routing collapses anyway; too high and the router spreads tokens
evenly regardless of which expert would actually handle them best, which
defeats the specialization.

### The z-loss

A second, smaller term penalizes the squared log-sum-exp of the router logits:

```rust
.router_z_loss_weight(1e-3)   // ST-MoE default
```

This keeps the router's raw logits from growing without bound. They can, since
nothing else constrains their scale, and in `bf16` large logits lose precision
in exactly the place where a small difference decides which expert runs. The
z-loss is cheap insurance against a routing instability that is very hard to
diagnose after the fact. Set it to `0.0` to disable it.

### Reading them

`TotalLoss` splits the two apart:

```rust
let loss = model.train_step(&batch)?;
println!(
    "lm {:.4}  aux {:.4}  total {:.4}",
    loss.lm_loss, loss.auxiliary_loss, loss.total()
);
```

Watch `lm_loss` to see whether the model is learning. Watch `auxiliary_loss` to
see whether routing is healthy. A rising `auxiliary_loss` while `lm_loss` falls
means the router is collapsing and you are about to lose most of your
parameters.

---

## 15.4 The shared expert

One expert that runs for every token, in addition to the routed ones:

```rust
.shared_expert(true)
```

Some things every token needs — basic syntax, common words, whatever the
equivalent of "how English works" is. Without a shared expert, every routed
expert has to learn those things separately, and eight copies of the same
knowledge is eight times the parameters doing one expert's job.

The shared expert holds them once. The routed experts are then free to
specialize, which is what you are paying them for. Qwen-MoE and DeepSeek-MoE
both do this, it costs one expert's worth of active parameters, and it is on by
default here.

---

## 15.5 Building one

```rust
let mut model = TransformerLm::builder()
    .vocab_size(32_000)
    .d_model(512)
    .n_layers(8)
    .heads(8, 2, 64)
    .d_ff(1408)              // the dense layers
    .moe_d_ff(352)           // one expert, d_ff / 4
    .experts(8, 2)           // 8 experts, top-2
    .moe_layers(2..8)        // layers 0 and 1 stay dense
    .shared_expert(true)
    .aux_loss_weight(0.01)
    .router_z_loss_weight(1e-3)
    .optimizer(Optimizer::adam(3e-4))
    .seed(42)
    .build()?;
```

Every MoE knob, and how to pick it:

| Knob | Default | How to choose |
|---|---|---|
| `experts(num, per_token)` | `(8, 2)` | 8 is the working default. More experts means more total at the same active, but each expert sees proportionally less data and trains worse |
| `moe_d_ff` | 352 | About `d_ff / 4`. With 2 routed plus 1 shared expert active, that is `0.75 × d_ff` of work — *less* arithmetic than the dense layer it replaces |
| `moe_layers` | `2..8` | Not all of them. See below |
| `shared_expert` | `true` | Leave it on |
| `aux_loss_weight` | 0.01 | Leave it alone unless you see collapse |
| `router_z_loss_weight` | 1e-3 | Leave it alone. `0.0` disables |

### Why the first layers stay dense

`moe_layers(2..8)` makes layers 0 and 1 ordinary dense blocks. The convention
is roughly the first quarter.

Early layers do work that is the same for every token: finding token
boundaries, tracking brackets and indentation, the mechanics of the text.
There is nothing to specialize in, so a router there only adds noise and
routing overhead. Specialization becomes useful closer to the output, where the
representation is about meaning rather than form.

An index at or past `n_layers` is an error:

```text
moe_layers names layer 8, but the model has 8 layers
```

---

## 15.6 The honest performance note

More total parameters at the same active count should mean more capability at
the same speed. In practice, on one consumer GPU, the layer does not come free.

From [`docs/baseline.md`](../docs/baseline.md), same vocabulary, width, depth,
and batch shape, on an idle RTX 3060:

| Configuration | Total | Active | tok/s |
|---|---|---|---|
| `v16384 d512 L8 16x512 dense` | 30.9M | 30.9M | 43704 |
| `v16384 d512 L8 16x512 moe` | 47.2M | 27.7M | 40115 |

The MoE model has **fewer** active parameters and still runs **8% slower**.

Two reasons, both structural:

**The expert GEMMs are too small.** One expert at `moe_d_ff` 352 handling its
share of a 16×512 batch is a matrix multiply that does not fill 28 streaming
multiprocessors. Eight small GEMMs do less work per second than one large GEMM
of the same total size, and the gap widens as the card gets bigger.

**Routing is a host round-trip.** Deciding which tokens go to which expert is
a sort and a gather, and it synchronizes.

What this means practically:

- On a **consumer GPU**, MoE buys capacity, not speed. Use it when you want a
  larger model to fit in the data and VRAM you have, not when you want the same
  model to run faster.
- The advantage grows with **batch size and sequence length**, because the
  expert GEMMs grow with them. The `128x128` row in the same table shows the
  effect from the other direction.
- On a **datacenter GPU** with more SMs and enough batch to fill them, this
  reverses, which is why every large open-weight MoE model is trained on one.

The layer is correct, tested against finite differences, and does what it
claims for parameter count. The speed claim is hardware-dependent, and on a
3060 it does not hold.

---

## 15.7 Watching routing

[`code/15_mixture_of_experts.rs`](code/15_mixture_of_experts.rs) does two
things a loss curve cannot show you.

First, it prints total and active as the expert count changes, straight from
`TransformerConfig::parameter_counts` — no model is built, so it costs nothing:

```text
experts   total    active    ratio
      1   32.5M    32.5M    1.00x
      2   35.7M    35.7M    1.00x
      4   42.2M    35.7M    1.18x
      8   55.2M    35.7M    1.54x
     16   81.2M    35.7M    2.27x
     32  133.1M    35.8M    3.72x
```

Total quadruples. Active moves by 0.1M — the router projection growing from 4
outputs to 32 — because top-2 of 32 experts is the same work per token as top-2
of 4. That column is the whole argument for the layer.

Second, it shows what the routing statistics look like in both states. A bare
`MoeLayer` rather than a whole transformer, because `forward_train` returns a
cache with the routing in it and `TransformerLm` does not expose one:

```rust
let (_output, cache) = layer.forward_train(&tokens, Layout::default())?;

for (expert, share) in cache.load_fractions().iter().enumerate() {
    println!("expert {expert}: {:>5.1}%", share * 100.0);
}
println!("aux {:.4}  z {:.4}", cache.aux_loss(), cache.z_loss());
```

`load_fractions` is the fraction of routed token slots each expert received,
and it sums to one. With 8 experts, balanced is 12.5% each.

```text
varied tokens, balanced routing:
  routing   13.3%  14.2%  13.7%  10.6%  12.0%  10.4%  14.3%  11.6%
  busiest 14.3%   dead experts 0/8   aux 0.0100   z 0.0047

identical tokens, collapsed routing:
  routing   50.0%  50.0%   0.0%   0.0%   0.0%   0.0%   0.0%   0.0%
  busiest 50.0%   dead experts 6/8   aux 0.0144   z 0.0053
```

The collapsed case is produced by feeding the layer 512 copies of the *same*
token vector. The router has nothing to tell them apart by, so every one of
them lands on the same two experts and the other six never run. That is the
end state the load-balancing loss exists to keep a real run away from — where
it arrives gradually, over thousands of steps, from a small early advantage
compounding.

Note what did and did not move. The routing distribution is unmistakable; the
aux loss went from 0.0100 to 0.0144, a 44% rise. In a real run you are watching
that second number, on a layer you cannot inspect directly, so learn what its
healthy value looks like early and treat a sustained climb as the alarm it is.

## 15.8 If something went wrong

| Symptom | Cause |
|---|---|
| `auxiliary_loss` climbs steadily | Routing collapsing. Raise `aux_loss_weight`, or check it is not 0.0 |
| `load_fractions` shows one expert at 90% | Same thing, further along |
| `auxiliary_loss` is exactly 0.0 | Both weights are zero, or the model has no MoE layers |
| MoE is slower than dense | Expected on a small GPU. 15.6 |
| Loss is worse than the dense model at the same total | Compare at the same *active*, not the same total. MoE trades capacity for compute, not for quality at fixed size |
| `experts_per_token must be between 1 and num_experts` | Top-k cannot exceed the expert count |
| `moe_layers names layer N, but the model has N layers` | `moe_layers` names a layer that does not exist |
| Experts all learn the same thing | `aux_loss_weight` too high — the router is being forced to spread evenly regardless of fit |

---

## Exercises

1. Extend the parameter table to 64 and 128 experts. At what point does the
   ratio stop being worth the VRAM the total occupies?
2. Feed the layer tokens that are 90% identical and 10% varied. Where does the
   aux loss land between the two values the example prints?
3. Set `experts_per_token` to 1, then 4. What happens to active parameters, and
   to the loss after 300 steps?
4. Turn `shared_expert` off and compare loss at the same active parameter
   count.
5. Time a dense model against an MoE model with the same active count on your
   own hardware. Does 15.6 hold for your GPU?

---

## Recap

- MoE separates total parameters from active ones. Capacity stops costing
  speed.
- A router scores every expert per token; the top 2 run and their gates are
  renormalized.
- Routing is dropless here: no token is ever discarded, so no unexplained loss
  floor.
- Without a load-balancing loss, routing collapses onto one or two experts and
  the rest of the checkpoint is wasted. `aux_loss_weight` 0.01 is the defence.
- The z-loss keeps router logits from growing large enough to lose precision in
  `bf16`.
- A shared expert holds what every token needs, so the routed experts can
  specialize.
- Early layers stay dense. There is nothing to specialize in there.
- Watch `lm_loss` for learning and `auxiliary_loss` for routing health.
- On a consumer GPU, MoE buys capacity, not throughput. Measured: 8% slower
  than dense at fewer active parameters.

---

**Next:** [16. Training a Language Model End to End](16_training_an_llm.md)
