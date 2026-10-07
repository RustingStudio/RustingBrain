# Training throughput baseline

Measured with `examples/sweep_arch.rs`, one configuration per process, on an
otherwise idle RTX 3060 12 GB (desktop applications holding 1258 MiB).

Every earlier measurement in this repository's history was taken while a
second training job held 5754 MiB and 100% of the GPU. Those numbers are
superseded by this table and should not be compared against it.

## Regression check

`scripts/bench_check.sh` compares the best tokens/s of five runs against these
rows and fails below 97%. Measured 2026-10-07 at `cd8b514` on the idle RTX 3060
(train step) and the Ryzen 5 7600X (decode, CPU). One run spreads about 3.5% on
both. `bench_decode` is sensitive to anything else on the CPU: with one
parallel `rustc` it measured 1464-2088 tokens/s, so run it on a quiet machine.
Re-record with `scripts/bench_check.sh --record` after a change that is meant
to move these numbers. The train-step row was re-recorded alone on
2026-10-07 at `6f48659` (idle GPU, best of 46927, 52707, 52890, 52783,
52815) after A7's BF16 MoE work; the CPU was busy, so the decode row was kept.

| Benchmark | Best tokens/s |
|---|---|
| bench_train_step 4 1024 cuda | 52890 |
| bench_decode 512 128 | 2434 |

## Throughput

FLOPs per token counts the parameter term and the attention term:

```
FLOPs/token = 6 * active_params + 12 * n_layers * seq_len * d_model
MFU         = tok/s * FLOPs/token / 25e12     (RTX 3060 bf16 peak, fp32 accumulate)
```

| Configuration | Total | Active | tok/s | TFLOPS | MFU | Days for 50B |
|---|---|---|---|---|---|---|
| v32000 d512 L8 16x512 moe | 55.2M | 35.7M | 36302 | 8.69 | 34.8% | 15.9 |
| v16384 d512 L8 16x512 moe | 47.2M | 27.7M | 40115 | 7.68 | 30.7% | 14.4 |
| v16384 d512 L8 16x512 dense | 30.9M | 30.9M | 43704 | 9.21 | 36.8% | 13.2 |
| v16384 d512 L8 32x512 dense | 30.9M | 30.9M | 43952 | 9.26 | 37.0% | 13.2 |
| v16384 d512 L8 16x1024 dense | 30.9M | 30.9M | 34089 | 8.03 | 32.1% | 17.0 |
| v16384 d512 L12 16x512 dense | 42.2M | 42.2M | 29189 | 8.49 | 34.0% | 19.8 |
| v16384 d768 L12 16x512 dense | 88.7M | 88.7M | 15620 | 9.20 | 36.8% | 37.0 |
| v32000 d512 L8 128x128 moe | 55.2M | 35.7M | 39988 | 7.87 | 31.5% | 14.5 |

Contended figures for the same configurations ran at 16989, 18651, 21295 and
14453 tok/s: **the clean card is 2.14x faster**, and nothing else in this
repository changed between the two sets.

## Peak VRAM

| Configuration | Peak (including 1258 MiB desktop) |
|---|---|
| v16384 d512 L8 32x512 dense | 6779 MiB |
| v16384 d512 L8 16x1024 dense | 7355 MiB |

No configuration in the sweep ran out of memory, including the three that
did under contention (batch 24, 12 layers, seq 1024 at batch 16). The 12 GB
card has roughly 4.5 GB of headroom at the largest shape measured.

## What this rules out

The plan in `optimizePlan.md` gated two tasks on this measurement:

- **Task 6, bf16 activation storage** — justified only below about 25% MFU,
  where the step is starved of bandwidth rather than arithmetic.
- **Task 7, pointer-array batched attention GEMM** — justified only if the
  per-launch overhead is a visible share of step time.

MFU sits between 30.7% and 37.0% across every shape measured, including the
128-sequence batch that issues the fewest launches per token and the d768
model that issues the largest GEMMs. A device this well fed does not have
1.3x sitting in its activation traffic, and the launch count is plainly not
what bounds it. **Both tasks are closed as not worth their complexity.**

Re-open them only if a later change (much longer sequences, a much smaller
`d_model`) moves MFU back below 25%.

---

## Attention softmax profile (Task 9, Step 1)

`nsys profile --stats=true` on an idle card, over

```
./target/release/bench --gpu --mixed-precision --d-model 768 --n-layers 16 \
  --n-heads 12 --n-kv-heads 4 --head-dim 64 --d-ff 1408 --seq-len 1024 \
  --batch-size 4 --steps 20 --gpu-memory-budget-mib 9000
```

Total GPU kernel time across the run is 7.00 s. The `cuda_gpu_kern_sum`
table reproduces the numbers Task 9 was written against:

| Kernel | Instances | Total | Avg | Share of GPU time |
|---|---|---|---|---|
| `causal_softmax_bwd` | 336 | 627.3 ms | 1.867 ms | 9.0% |
| `causal_softmax_lse` | 336 | 528.5 ms | 1.573 ms | 7.6% |
| `causal_probs_from_lse` | 336 | 336.1 ms | 1.000 ms | 4.8% |
| **Softmax total** | | **1491.9 ms** | | **21.3%** |

336 instances is 16 layers x 21 steps, once per layer per step, so none of it
is setup cost. The nine `cutlass::Kernel2<...>` GEMM variants together account
for roughly 60% of GPU time; the two 64x64 tile variants among them, 1186 ms
and 17%, are the per-head attention GEMMs.

### Why these three kernels are not slow kernels

At this shape the score matrix is `heads * sequences * seq_len` = 49152 rows
of 1024 floats, of which the causal mask leaves about half visible: 25.2M
elements, 100.8 MB in FP32. Counting the passes each kernel makes over that
data against the card's 360 GB/s:

| Kernel | Traffic | Time | Achieved |
|---|---|---|---|
| `causal_softmax_lse` | 3 reads + 2 writes of the visible half, plus one write zeroing the masked half — 605 MB | 1.573 ms | 384 GB/s |
| `causal_softmax_bwd` | reads of both `g` and `p` twice, one write of each half — 604 MB | 1.867 ms | 324 GB/s |
| `causal_probs_from_lse` | one read of the visible half, one write of the whole row — 302 MB | 1.000 ms | 302 GB/s |

All three run at or above 84% of the card's nominal bandwidth, and
`causal_softmax_lse` exceeds it on L2 hits. **There is no meaningful headroom
inside these kernels.** Tightening the pass count would move the total from
21.3% to perhaps 15%, and no rewrite that still materializes a
`[seq_len, seq_len]` FP32 matrix in global memory can do better than that,
because the traffic, not the arithmetic, is what costs.

That leaves two ways forward, and they are not the same size:

1. **Store the score matrix in BF16.** Every byte of the traffic above halves,
   and so does the FP32 traffic in the two attention GEMMs that read and write
   the same buffer. The softmax kernels keep their structure and their FP32
   arithmetic; only the loads and stores narrow. The existing
   `gemm_strided_batched_dispatch` already computes in
   `CUBLAS_COMPUTE_32F_FAST_16BF` and would need `CUDA_R_16BF` for the score
   operand rather than a new code path. Expected: most of a 10% reduction in
   total GPU time, and the score buffer drops from 201 MB to 100 MB.
2. **Never materialize the matrix — flash attention.** This is what Task 9's
   Step 2 asks for and what PyTorch's SDPA and TF's XLA fusion do. It removes
   `causal_probs_from_lse` outright, folds the softmax into the QK^T and AV
   passes, and is the only path that recovers the full 21%. It also replaces
   two cuBLAS tensor-core GEMMs with hand-written NVRTC kernels that have to
   beat them, which is the largest piece of work anywhere in this plan.

Option 1 is a precondition for neither and a fair fraction of the gain, so it
is worth measuring before committing to option 2.

---

## Fused attention (Task 9, Steps 2-5)

Five kernels became three, and the `[seq_len, seq_len]` matrix stopped being
written at all.

| Pass | Before | After |
|---|---|---|
| Forward | `causal_softmax_lse` plus two batched GEMMs per head | `flash_attention_fwd` |
| Backward | `causal_probs_from_lse`, `causal_softmax_bwd` and four batched GEMMs per head | `flash_attention_delta`, `flash_attention_dq`, `flash_attention_dkv` |

The forward kernel is blocked over query tiles. The backward pass needs two
kernels because the query gradient is complete inside a query tile and the key
and value gradients are complete inside a key tile; blocking the second one on
the key/value head rather than the query head is what makes grouped-query
attention exact without atomics. Both rebuild the scores from the stored
log-sum-exp, which costs one extra matmul per tile pair and saves the whole
round trip.

### Throughput

Same command as the profile above. **The card was not idle: a game was running
on it throughout, so treat these as a paired comparison, not as absolute
numbers.** `RUSTING_BRAIN_NO_FLASH=1` is what selects the old path.

| Path | Run 1 | Run 2 |
|---|---|---|
| Fused | 12077 tok/s | 12064 tok/s |
| Three-kernel | 7087 tok/s | 7187 tok/s |

**1.70x**, and the final loss agrees to 0.006 across both paths (6.711-6.716
fused, 6.717-6.718 split) after 20 steps. For reference the three-kernel path
measured 11905 tok/s on an idle card, so the fused path on a contended one is
already past the old idle figure; the idle figure for the fused path is still
to be taken.

Of that 1.70x, the forward fusion alone was 1.15x, measured the same way
before the backward kernels existed. The backward half is the larger share, as
the profile said it would be.

### What the profile looks like now

| Kernel | Share of GPU time |
|---|---|
| `cutlass ... 128x256_16x3_nt` | 17.3% |
| `cutlass ... 256x128_16x3_nn` | 17.3% |
| `cutlass ... 256x128_16x3_tn` | 16.5% |
| `flash_attention_dkv` | 8.0% |
| `flash_attention_dq` | 5.8% |
| `flash_attention_fwd` | 4.0% |

`causal_softmax_lse`, `causal_softmax_bwd` and `causal_probs_from_lse` are
gone, and so are the 8064-instance per-head attention GEMMs that used to run
beside them. Attention is now 17.8% of GPU time in three kernels, against
21.3% in the softmax kernels alone plus 17% in the per-head GEMMs before.

The three remaining `cutlass` entries are the model's own projections, and at
51% of GPU time between them they are what stands between this and PyTorch.
`flash_attention_dkv` is the slowest of the three fused kernels at a 1.04 ms
median against `flash_attention_dq`'s 0.68 ms, which is what four matmuls per
tile pair and a tile count that falls from 16 to 1 across the grid look like.

### Accuracy

Both device paths round their matmul operands to BF16, so neither is a
reference for the other; the FP32 host path is. On a two-layer, four-head
model at 100 tokens the fused forward lands 3.7e-3 from the host where the
three-kernel path lands 2.5e-3, on logits of magnitude 0.475. After one full
gradient-descent step the two land 1.0 to 1.4 times apart, the difference
being that the fused backward takes the softmax row sum from the output
gradient and the output rather than from probabilities it no longer keeps.
`fused_attention_matches_the_three_kernel_path_or_skips_without_device` and
its backward twin hold that ratio.

---

## BF16 activations (the GEMM operand pipeline)

The fused-attention profile left the block's projection GEMMs at roughly half
of GPU time, all of them on `cutlass_80_tensorop_s1688bf16gemm_*_align4`. That
is the m16n8k8 tensor-core shape, which is what cuBLAS picks for FP32 operands
with a `32F_FAST_16BF` compute type. The language-model head, whose operands
are already BF16, lands on `s16816 ... align8` instead: twice the `k` per
instruction.

`examples/gemm_probe.rs` measured the three ways to close that gap on the four
projection shapes at batch 4, sequence 1024:

| Shape | units/inner | FP32 operands | BF16 operands | BF16 plus a per-call cast |
|---|---|---|---|---|
| qkv projection | 1280/768 | 12.82 TF | 13.07 TF | 9.22 TF |
| attention output | 768/768 | 9.83 TF | 11.67 TF | 7.36 TF |
| feed forward in | 2816/768 | 12.55 TF | 14.35 TF | 11.48 TF |
| feed forward out | 768/1408 | 12.49 TF | 12.85 TF | 8.09 TF |

**Casting the operands inside the GEMM wrapper is a net loss** - 0.65x to 0.91x
- so the kernel that produces an activation has to write it narrow. That is
what this change does: `rmsnorm_fwd`, `swiglu_fwd`, `swiglu_bwd` and
`flash_attention_fwd` take a `narrow` flag and store through `store_act`, and
`Act` in `gpu_model` holds an activation as untyped bytes plus that flag. One
byte buffer serves both precisions because a kernel launch only ever passes a
device pointer and cuBLAS takes the element type as a runtime argument.

Weights and gradients, Adam moments and every accumulator stay FP32. The routed
feed-forward stays FP32 end to end, because its gathers, scatters and row-wise
reductions are FP32 kernels and it is not what these shapes exercise.

### Throughput

Idle card, same command as the profile above.

| Path | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| BF16 activations | 21516 tok/s | 21529 tok/s | 21484 tok/s |
| FP32 activations (commit `1f3ad13`) | 20152 tok/s | 20180 tok/s | |

**1.067x.** Final loss 6.707 against 6.713, on logits the two paths compute
with the same FP32 accumulators and different operand rounding.

### Where the step goes now

185.6 ms of kernel time per step against 190.4 ms of wall clock, so launch
overhead is 2.5% and not worth attacking.

| Group | Per step | Share |
|---|---|---|
| Projection and head GEMMs | 108.0 ms | 58.2% |
| Fused attention (three kernels) | 36.2 ms | 19.5% |
| Elementwise and the optimizer | 41.5 ms | 22.3% |

The GEMMs move roughly 2498 GFLOP per step - 1894 in the blocks, 604 in the
head - which at 108.0 ms is **23.1 TFLOPS against the card's 25.5 TFLOPS BF16
peak, 91%.** There is nothing left in them. Half of them still land on
`ampere_s1688gemm_bf16_128x128`, the k8 kernel, but at that fraction of peak
the heuristic is not what bounds the step.

The three flash kernels move about 360 GFLOP per step at 36.2 ms, which is
10.0 TFLOPS, 39% of peak. That is where the remaining headroom is.

For reference: PyTorch 23121 tok/s, TensorFlow 20140 tok/s, this repository's
pre-optimization idle baseline 11905 tok/s. At 21516 tok/s the step is 1.81x
the baseline, past TensorFlow, and 7% short of PyTorch.

## Step 12: the fused attention buffers in BF16 (`579b040`)

The previous step narrowed the activations that feed cuBLAS. `qkv` was left
FP32, even though its only readers were the three fused attention kernels and
they rounded every tile to BF16 as they loaded it. This step makes `qkv` and
`grad_qkv` narrow too, so the rounding happens once at the projection's store
instead of on every tile load.

`eligible(mixed_precision, head_dim) = mixed_precision && head_dim == TILE`, so
whenever the fused kernels run at all, every buffer they touch is BF16. The
runtime `narrow` flag is therefore gone from all four kernels in
`cuda_flash.rs` - the type is known at compile time. cuBLAS writes the
projection straight to BF16 (`Ctype = CUDA_R_16BF`, via `linear_packed_act`),
which removes a `cast_act` launch as well as the FP32 round trip.

### The first version was 2.6% slower

20960 tok/s against 21524, all of it in `flash_attention_dkv`. It was not
occupancy: `nvcc -arch=sm_86 -cubin -Xptxas -v` reports 198 registers for dkv
both before and after, and deleting the `narrow` branch changed nothing.

The cause is that these kernels are latency bound, not bandwidth bound. Halving
the bytes buys nothing; issuing the same number of loads at 16 bits each,
plus a shift and an `__uint_as_float` per element, costs.

The fix is that **two adjacent BF16 elements are exactly the 32-bit register an
`m16n8k16` mma operand takes**, so one 32-bit load fills a fragment with no
conversion at all:

```cuda
__device__ __forceinline__ unsigned bf16_pair(const unsigned short* p, size_t i){
  return *(const unsigned*)(p + i);   // i must be even
}
```

dkv now loads K and V straight into `kf`/`vf`, and tiles Q and dO two elements
a thread. dq tiles K and V the same way, and every `grad_qkv` and `out` store
is a packed 32-bit write. The forward kernel's K/V tile was tried this way and
measured worse - 8.0 ms to 10.9 ms - because its value tile is stored
transposed and the wider load buys bank conflicts; it stays scalar, with a
comment saying so.

One incidental exactness note: the attention scale moved off the queries and
onto the scores and the `dk` store. For head dim 64 the scale is 1/8, a power
of two, so this is bit-exact and it saves 64 multiplies per query row.

### Throughput

Idle card, 20 steps of the 102M model, same command as above.

| Path | Tokens/sec | Final loss |
|---|---|---|
| BF16 `qkv` (`579b040`) | 21863 | 6.715 |
| FP32 `qkv` (`8c318ae`) | 21524 | 6.710 |

**1.016x.** 144 library tests pass.

### Where the step goes now

| Kernel | Per step |
|---|---|
| `ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_nt` | 20.38 ms |
| `cutlass_80_tensorop_s16816gemm_bf16_256x128` | 19.37 ms |
| `flash_attention_dkv` | 17.16 ms |
| `ampere_s1688gemm_bf16_128x128_ldg8_tn` | 13.09 ms |
| `cutlass_80_tensorop_s16816gemm_bf16_128x256` | 12.79 ms |
| `flash_attention_dq` | 10.96 ms |
| `adam` | 9.04 ms |
| `flash_attention_fwd` | 8.04 ms |
| `add_inplace` | 5.76 ms |
| `swiglu_bwd` | 4.68 ms |
| `rmsnorm_bwd` | 4.28 ms |
| `cast_act` | 4.04 ms |
| `swiglu_fwd` | 3.41 ms |
| `rmsnorm_fwd` | 2.80 ms |
| `rope_rotate` | 2.66 ms |

`rope_rotate` is 32% cheaper than it was and `cast_act` 31%, both because they
now move half the bytes. The GEMMs are 59% of the step, the three flash kernels
19%, everything else 21%.

`examples/flash_probe.rs` runs the three kernels standalone on the benchmark
shape - forward 0.449 ms, grad query 0.656 ms, grad key/value 1.014 ms per
launch - which is the fast loop for working on them without a 20-step
benchmark in between.

PyTorch is 23121 tok/s. At 21863 the gap is 5.7%.

## Steps 13-16: the passes that were not carrying their weight

Four changes, each of them a pass over a whole activation or a whole parameter
set that some other kernel was already positioned to absorb. None of them makes
a kernel faster; they delete work.

### The residual add inside the RMSNorm backward (`12c1700`)

A residual branch's backward pass produced two gradients for the same tensor and
added them in a separate kernel. `rmsnorm_bwd` already writes that tensor, so it
takes the other addend as an argument and folds it in:

```cuda
float g_in=up[c]*w[c]*t-src[c]*shared;
if(add_res)g_in+=res[base+c];
dst[c]=g_in;
```

**21863 -> 22240 tok/s, 1.017x.**

### The operand copy inside the RMSNorm backward (`81c3e9e`)

The gradient a norm hands down is a GEMM operand for the layer below it, and a
mixed-precision GEMM wants it in BF16. That narrowing was its own pass. The same
kernel writes it, because the value is in a register at that point:

```cuda
if(copy_mode)store_act(copy,base+c,g_in,copy_mode-1);
```

Which precision the copy wants is the *receiving* feed-forward's, not the
context flag: a routed layer's GEMMs are FP32 and ask for no copy at all, and
the fused attention path's `merged` is narrow only when flash attention runs.
Deriving it from the consumer (`cache.merged.is_narrow()`, and
`cache.blocks[i].feed_forward_normed.is_narrow()` for the norm above a block)
is what makes the FP32 and flash-disabled tests pass.

**22240 -> 22383 tok/s, 1.006x.**

### A BF16 output gradient for the fused attention backward (`1f596d5`)

The three flash backward kernels took `grad_out` in FP32 and converted on every
tile load. They now take it in BF16, which is both half the traffic and, because
two adjacent BF16 elements *are* the 32-bit register an `m16n8k16` operand takes,
zero conversion:

```cuda
df[kk][0]=bf16_pair(grad_out,e+c);
```

**22383 -> 22692 tok/s, 1.014x.**

### The residual duplicate inside the RMSNorm forward (`345c46b`)

A residual branch's projection accumulates over the block input with `beta = 1`,
so that input had to be duplicated into the destination first: two
`clone_dtod` calls a block, 32 copies of 12.58 MB a step, 2.5 ms.

The first attempt removed them the other way round -- projection writes its own
buffer with `beta = 0`, and the next norm sums the two inputs -- and measured
*neutral*: 22875, 22747, 22681 tok/s against a 22692 baseline. The profile said
exactly why. The copies went from 2.52 ms a step to 0.08, and `rmsnorm_fwd` went
from 2.17 ms to 4.63. The two are the same number because the scheme still
materializes two tensors per branch; the `beta = 1` read it also removed was
never costing anything, because it is hidden under a GEMM running at 22 TFLOPS.

What does pay is leaving the accumulate alone and moving only the copy. The norm
above a branch reads the row it would have copied, so it writes the duplicate
itself, out of registers:

```cuda
for(int c=tid;c<cols;c+=ROW_THREADS){float v=src[c];if(do_copy)dup[c]=v;s+=v*v;}
```

That is a store where there was a read-modify-write pass: 12.58 MB less per
site.

**22758 -> 22944 tok/s, 1.008x** (mean of two 60-step runs each).

### Stale gradients instead of zeroed ones (`0c5d951`)

Every optimizer step ended by memsetting each parameter's gradient, and every
weight-gradient GEMM then accumulated into those zeros with `beta = 1`. That is
467 MB of zeroing a step.

A gradient buffer holding the last step's value is as good as a zeroed one, so
long as the first writer of the next step overwrites rather than accumulates.
Each `DeviceParam` carries a `grad_dirty` flag; `grad_beta()` hands out `0.0`
once and `1.0` after, and the optimizer step clears the flag instead of the
memory. The callers that can only accumulate -- the embedding scatter's atomic
adds, the tied head's BF16 accumulate, and the optimizer itself for a parameter
that took no gradient -- call `clear_grad()`, which memsets only if nothing has
claimed the buffer yet.

The win is larger than the memsets alone, because `beta = 0` also spares each
weight-gradient GEMM the read of its destination.

**22944 -> 23411 tok/s, 1.020x** (23399 and 23423 over two 60-step runs).

### Throughput

Idle card, 60 steps of the 102M model, same command as above.

| Path | Tokens/sec | Final loss |
|---|---|---|
| This branch (`0c5d951`) | 23411 | 1.188 / 1.201 |
| Before these four (`579b040`) | 21863 | 6.715 (20 steps) |
| PyTorch | 23121 | |
| TensorFlow | 20140 | |
| Idle-GPU baseline | 11905 | |

**1.97x the baseline, and 1.013x PyTorch.** 144 library tests pass.

### Where the step goes now

174.9 ms of kernel time a step.

| Kernel | Per step | Launches |
|---|---|---|
| `ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_nt` | 19.57 ms | 48 |
| `cutlass_80_tensorop_s16816gemm_bf16_256x128_32x3_nn` | 18.51 ms | 32 |
| `flash_attention_dkv` | 15.45 ms | 16 |
| `ampere_s1688gemm_bf16_128x128_ldg8_tn` | 11.58 ms | 16 |
| `cutlass_80_tensorop_s16816gemm_bf16_128x256_32x3_tn` | 11.40 ms | 32 |
| `flash_attention_dq` | 10.45 ms | 16 |
| `adam` | 8.96 ms | 113 |
| LM head GEMMs (three kernels) | 22.72 ms | 3 |
| `flash_attention_fwd` | 7.24 ms | 16 |
| `rmsnorm_bwd` | 6.66 ms | 33 |
| `swiglu_bwd` | 4.52 ms | 16 |
| memsets | 0.75 ms | 113 |

GEMMs are 60% of the step, the three flash kernels 19%, everything else 20%.

### What is left

The GEMMs have no headroom. Mapping each cuBLAS and cutlass kernel to its shape
by grid dimensions gives 24.9 TFLOPS for the `gate_up` weight gradient, 24.0 for
`qkv`, 23.5 for the `gate_up` forward, 22.1 for the `qkv` and `down` forwards,
21.8 for the attention-output weight gradient and 19.9 for `down`'s, against a
25.5 TFLOPS dense BF16 peak. 78% to 98% of the card.

The flash kernels run at roughly half of peak, which makes `flash_attention_dkv`
the largest single target left. Four ablations on it, each measured with
`examples/flash_probe.rs` against a 0.995 ms baseline:

| What was removed | Grad key/value |
|---|---|
| The two global tile loads | 1.023 ms |
| The `__expf` epilogue | 1.027 ms |
| The `dk`/`dv` mma pair | 0.556 ms |
| The score/`dpt` mma pair | 0.825 ms |

Global traffic and the softmax epilogue cost nothing measurable: only removing
mma work moves the number, so the kernel is bound by the tensor-core and
shared-operand pipeline, not by bandwidth. With `DKV_SHARED_BYTES` at 37376 and
198 registers per thread the kernel is capped at two blocks per SM either way,
so the fix is not occupancy. The asymmetry between the two mma loops -- 0.44 ms
against 0.17 ms for identical mma and `LDS.32` counts -- points at the second
loop's fragment packing, which is where to look next.

## Tokenizer merge (J8, 2026-10-07)

`Bpe::encode` on a sentencepiece-style `tokenizer.json` (Metaspace), release
build, one paragraph of English repeated. CPU, not a throughput row: it
only shows the shape of the cost.

| Characters | Ids | Time |
|---:|---:|---:|
| 84 | 22 | 0.146 ms |
| 336 | 85 | 1.733 ms |
| 1,344 | 337 | 27.1 ms |
| 5,376 | 1,345 | 432 ms |

Four times the text costs about sixteen times the time: the merge loop is
quadratic in the length of the piece, and Metaspace hands it the whole text.
Roadmap item E8.

After E8 (heap merge), same file and machine, load average about 3.8. The
50 KB text is this README repeated. Ids hash the same before and after.

| Characters | Before | After | Per character after |
|---:|---:|---:|---:|
| 1,008 | 17.2 ms | 0.258 ms | 0.26 µs |
| 5,376 | 451 ms | 1.06 ms | 0.20 µs |
| 50,000 | 43.1 s | 12.6 ms | 0.25 µs |

## LM-head z-loss (B7, 2026-10-07)

Not measured: another process held the GPU. The term adds one FP32 multiply
to the head kernel's softmax scale and, when the weight is nonzero, one atomic
per row. At the default weight of 0 the host path skips its extra pass.

## Byte-level BPE pre-tokenizer (E7, 2026-10-07)

CPU, release build, one run each. The contraction check built a `String` of
the whole remaining text at every character. The text is
"The cat's hat, isn't it? 42 cats sat.  " repeated 2,000 times (78,000
characters, 28,000 words). The words are the same before and after; the
tokenizer tests cover the split.

| Characters | Before | After |
|---:|---:|---:|
| 78,000 | 617 ms | 1.42 ms |

Training itself is not timed: there is no fixed corpus to compare against yet.

## MoE routed path in BF16 (A7 W1, 2026-10-07)

Default model (55.2M total, 35.7M active, 6 routed layers of 8 experts, top
2), `bench_train_step 4 1024 cuda`, RTX 3060 with only gnome-clocks (41 MiB)
on it. Kernel times are `nsys` totals over the four steps the benchmark runs,
divided by four.

| | Before | After |
|---|---:|---:|
| Kernel time per step | 82.0 ms | 80.1 ms |
| FP32-operand GEMMs (`s1688bf16gemm`, the expert GEMMs) | 18.4 ms | 6.1 ms |
| All GEMMs | 45.7 ms | 44.6 ms |
| `swiglu_fwd` + `swiglu_bwd` | 3.5 ms | 3.2 ms |
| `cast_act` | 1.6 ms | 1.4 ms |
| The two gathers | 1.5 ms | 1.3 ms |
| Best of five, tokens/s | 49,802 | 51,631 |

Most of the expert GEMMs' time moved into the BF16 GEMM rows rather than
disappearing: at this shape (512 wide, about 1,000 rows per expert) the
operand conversion was the smaller part of their cost.

Loss check: the default model at vocabulary 4096 (a byte-level BPE trained
with `Bpe::train_json` on the repository's markdown, 180k tokens), 16 x 256
tokens a step, Adam 3e-4, 500 steps, same seed, mixed precision on, before
and after this change.

| Step | FP32 routed | BF16 routed |
|---:|---:|---:|
| 0 | 8.3616 | 8.3616 |
| 100 | 6.2913 | 6.2807 |
| 200 | 5.0341 | 5.0238 |
| 300 | 4.3433 | 4.3658 |
| 400 | 3.8325 | 3.7833 |
| 499 | 3.1830 | 3.1844 |
| Held-out | 4.9438 | 4.9239 |

## MoE router and shared expert in BF16 (A7, 2026-10-07)

`bench_train_step 4 1024 cuda`, default model, mixed precision. The MoE
block takes the BF16 normed input, so the router GEMM, the gather and the
shared expert read BF16 operands. The router logits stay FP32.

nsys kernel sum over 4 steps: 80.1 to 76.7 ms a step. The FP32-operand
`s1688bf16` GEMM category is gone; the added `cast_act` costs 1.30 ms.

Throughput (GPU quiet at the time): 49,193, 52,872, 47,815, 52,823 and
52,914 tok/s. Best of five 52,914 against 51,631 after W1 (+2.5%). The
regression floor was not re-recorded: `--record` also re-records CPU decode,
which needs a quiet machine. `strata_head` took the GPU right after the
profile, so there are no further runs.

Loss, 500 steps of 16 x 256 on the repository's markdown with a 4096-entry
BPE, Adam 3e-4, seed 7:

| Step | This change | W1 run |
|---:|---:|---:|
| 0 | 8.3501 | 8.3616 |
| 100 | 6.2603 | 6.2807 |
| 200 | 4.8686 | 5.0238 |
| 300 | 4.1520 | 4.3658 |
| 400 | 3.6789 | 3.7833 |
| 499 | 2.9449 | 3.1844 |
| Held-out | 4.6321 | 4.9239 |

The W1 column was taken on an earlier copy of the corpus (the roadmap and
baseline files have changed since), so it is a sanity bound, not a paired
comparison. A same-corpus rerun of the previous commit ran out of memory
because `strata` held 10.5 GB of the GPU.

## CPU MoE pieces (D1, D2, 2026-10-07)

Release build on the Ryzen 5 7600X, one `MoeLayer` with `d_model` 512, 8
experts, top 2, expert width 640, and 2048 random token rows. Timed once in
a temporary test, which was not committed. The forward and backward times
spread by about 1.5x between runs; the two small pieces do not depend on
that spread.

| Piece | Time |
|---|---:|
| `forward_train` | 26-35 ms |
| `backward` | 41-67 ms |
| `softmax_rows` (mean of 100) | 91-95 µs |
| `auxiliary_losses` (mean of 100) | 11 µs |

Both D1 and D2 are rejected. Each touches only `[tokens, experts]`, which
is about 1,000 times smaller than the expert GEMMs.

## Loss-spike guard (B7, 2026-10-07)

No measurement: `SpikeGuard` is a scalar check per step on the host, off the
GPU path, and adds no work to the training step itself.

## Tiled VAE decode (C7, 2026-10-07)

No measurement: no Stable Diffusion VAE checkpoint run was made, and
`strata` held 10.5 GB of the GPU, so peak memory against a whole decode
could not be compared. The tiled path calls `decode` per tile, so its peak
is that of one tile's decode plus the FP32 output and weight buffers.

## MoE permutation passes re-measured (A7 W2, 2026-10-07)

`nsys` over `bench_train_step 4 1024 cuda` at `6f48659`, idle GPU, grouped
per step:

| Kernel | Calls | ms/step |
|---|---:|---:|
| `gather_scale_rows` (forward input, backward gradient) | 12 | 1.29 |
| `scatter_add_scaled` (forward) | 6 | 0.93 |
| `scatter_add_rows` (backward) | 6 | 0.92 |
| `row_dot_scatter` (backward) | 6 | 0.63 |
| `moe_stats` | 6 | 0.36 |
| Total kernel time | | 78.1 |

The five kernels are 4.2 ms (5.4%) against 17.8 ms when W2 was written,
since W1 halved their bytes and the router work moved. The only redundant
read left is the gathered output-gradient rows, read by both
`row_dot_scatter` and the backward gather: about 0.5 ms (0.6%), below the
run-to-run spread, so W2 parts 1 and 2 are rejected.

## Adam re-measured (A9, 2026-10-07)

Same `nsys` report as "MoE permutation passes re-measured" (`6f48659`, idle
GPU, 4 steps). `adam`: 828 launches, 20.43 ms in all, so 207 launches and
5.11 ms a step. Median launch 17.3 µs, largest 1.42 ms. The default model has
55,181,824 parameters (`parameter_counts().total`). Adam reads the value,
gradient and both moments and writes the value and both moments: 28 bytes a
parameter, 32 with the weight-decay re-read. That is 1.55 to 1.77 GB a step,
302 to 345 GB/s against the RTX 3060's 360 GB/s. A fused multi-tensor kernel
can only close the gap to the roofline, at most about 0.8 ms (1%), so A9 is
rejected.

## Routed expert GEMMs attributed (A7 W4 step 1, 2026-10-07)

Same idle-GPU `nsys` report at `6f48659` (`bench_train_step 4 1024 cuda`,
4 steps), exported with `nsys stats -r cuda_gpu_trace`. Each MoE layer's
kernels run in a fixed order: the forward gather, the per-expert gate/up and
down GEMMs, then `scatter_add_scaled`; in the backward pass
`row_dot_scatter`, the gradient gather, the per-expert GEMMs, then
`scatter_add_rows`. The GEMMs inside those brackets are the routed experts.

| | ms/step | Launches/step | FLOPs/step | TFLOPS |
|---|---:|---:|---:|---:|
| Routed forward | 3.92 | 96 | 53.2 G | 13.6 |
| Routed backward | 6.75 | 192 | 106.3 G | 15.8 |
| All other GEMMs | 32.50 | | | |
| Total kernel time | 78.1 | | | |

FLOPs: 6 MoE layers, 8192 routed rows each (4096 tokens, top 2), `d_model`
512, expert width 352, `2 * rows * 3 * 512 * 352` forward and twice that
backward. Each per-expert GEMM is about 1024 rows, and the grids are 24 to 48
blocks on the RTX 3060's 28 SMs: one partial wave, so the split costs
occupancy. At the dense GEMMs' 22 TFLOPS the routed set would take about
7.3 ms, so about 3.4 ms (4.4%) is recoverable, under the 10 ms bar the plan
set for writing a kernel. `bench` routes uniformly random tokens almost
evenly, so real text (step 3) may differ.

## Expert loads on real text (A7 W4 step 3, 2026-10-07)

Default model on the GPU (default mixed precision), Adam 3e-4,
seed 7, 16 x 256 windows of the repository's markdown with a 4096-entry BPE.
Range of the per-expert share across the 8 experts, for each of the 6 MoE
layers, read with `TransformerCache::expert_loads` on the training batch.
An even split is 0.125.

| Step | Loss | Min-max share per layer |
|---:|---:|---|
| 0 | 8.355 | 0.058-0.242, 0.011-0.331, 0.035-0.304, 0.016-0.221, 0.019-0.406, 0.005-0.304 |
| 100 | 6.294 | 0.027-0.243, 0.050-0.207, 0.049-0.219, 0.026-0.194, 0.060-0.206, 0.010-0.234 |
| 200 | 5.004 | 0.082-0.148, 0.067-0.157, 0.092-0.153, 0.101-0.158, 0.095-0.194, 0.093-0.169 |
| 300 | 4.290 | 0.090-0.168, 0.093-0.160, 0.082-0.152, 0.055-0.178, 0.095-0.165, 0.103-0.146 |
| 499 | 3.151 | 0.111-0.145, 0.097-0.155, 0.089-0.152, 0.094-0.151, 0.085-0.148, 0.100-0.169 |

Held-out batch at step 499: every share is between 0.089 and 0.158. The
load-balancing loss evens the routing out within about 200 steps; after that
the busiest expert carries at most 1.35 times an even share, so the uniform
benchmark is a fair stand-in for the expert GEMM estimate. The GPU had a
21 MiB test process on it; these are counts, not timings.

## Expert weight packs re-measured (A7 W3, 2026-10-07)

Same idle-GPU trace at `6f48659`. The `cast_act` launches between each MoE
layer's forward gather and its scatter are the per-step weight packs: 144
launches (6 layers x 8 experts x gate, up and down) and 0.69 ms a step, out
of 1.31 ms for all 218 `cast_act` launches and 78.1 ms of kernel time. The
backward pass reuses the packed copies from the cache. Deferred: the fix
changes `SwiGlu`'s public fields.

## Device VAE overflow guard (C6, 2026-10-07)

No measurement. The guard is one pass over the downloaded image to look for
non-finite values, which is a host-side scan of the output only; it adds no
kernel and does not change the decode itself.

## Flash attention at head width 128 (A3, 2026-10-07)

Idle GPU (gnome-clocks only), `bench_train_step 4 1024 cuda <head_dim>`, the
default model with its 512-wide attention cut into heads of the given width
(2 KV heads), best of three:

| Head width | Fused | Three-kernel path (`RUSTING_BRAIN_NO_FLASH`) |
|---|---|---|
| 64  | 53,315 tok/s | 33,066 tok/s |
| 128 | 48,456 tok/s | 39,374 tok/s |

Kernel time per step at 128 (nsys, four steps): `dkv` 8.66 ms, `dq` 7.21 ms,
`fwd` 3.45 ms, `delta` 0.24 ms. At 64 the same build gives `dkv` 6.72, `dq`
3.60 and `fwd` 2.59 ms, unchanged from `6f48659`. A first version that
divided a signed index by the width raised `fwd_64` from 126 to 146
registers and doubled its time; the unsigned divide restores the old counts.
ptxas at 128: `fwd` 165 registers, `dq` 254, `dkv` 255 with 128 bytes spilled.

## LM head re-measured (A6, 2026-10-07)

Idle GPU at `96ffcc5`, `bench_train_step 4 1024 cuda`, nsys kernel summary over
four steps (77.5 ms of kernel time a step):

| Kernel | ms/step | Rate |
|---|---|---|
| `ampere_bf16_s16816gemm_bf16_256x128_..._tn` (logits) | 5.83 | 23.0 TFLOP/s |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x128_..._nn` (grad of hidden) | 5.21 | 25.8 TFLOP/s |
| `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_128x256_..._nt` (grad of weight) | 5.06 | 26.5 TFLOP/s |
| `ce_loss_grad_bf16` | 2.73 | ~288 GB/s |

Each GEMM is 4096 x 512 x 32000 (134 GFLOP). Rejected for speed: see
roadmap A6.

## Device decode before a device KV cache (C2 step 1, 2026-10-07)

Idle GPU at `53ce3b1`, `bench_decode 512 128 cuda` (the default 55.2M model,
FP32 weights on the device, host `KvCache`): 5.08, 5.16 and 5.53 ms/token,
best 197 tokens/s. The host row on the small model measured 2346 tokens/s in
the same session.

nsys over the whole run (prefill plus 128 tokens), per decoded token:

| Item | Per token |
|---|---|
| Kernel time | ~1.1 ms (LM-head GEMV 0.20 ms) |
| Copy time | ~0.8 ms |
| Host-to-device copies | ~194 |
| Device-to-host copies | ~174 |
| Kernel launches | ~245 |

Kernels and copies explain under 2 ms of the 5.1 ms; the rest is the host
waiting on each device-to-host copy (`cuMemcpyDtoHAsync` totals 270 ms) and
per-op event, allocation and launch calls. Each layer runs attention on the
host cache, so every linear layer round-trips its activations. This is what
C2's device-resident cache removes.

## Device decode with a device KV cache (C2 step 2, 2026-10-07)

Idle GPU, `bench_decode 512 128 cuda`, which now drives `Decoder`: 2.80, 2.60
and 2.61 ms/token, best 384 tokens/s, up from 197 (1.95x). The host decode
row was not re-measured: four `rustc` processes from another project held the
CPU (1651-1813 tokens/s, load average 5-6), and the host path is the same
`forward_cached` call as before.

nsys over the whole run, per decoded token: about 2.0 ms of kernels, of which
`cast_act` is 0.97 ms. That is `Gpu::pack` re-narrowing every weight to BF16
on every call, the one note left in `decode_block`, `forward_swiglu` and
`forward_moe`. Device-to-host copies fell from about 174 a token to about 19.

## Device decode with weight packs kept (C2 step 3, 2026-10-07)

Idle GPU (CPU load average about 5), `bench_decode 512 128 cuda`: 1.82, 1.60,
1.63, 1.62 and 1.61 ms/token, best 624 tokens/s, up from 384. nsys:
`cast_act` no longer appears; kernel time is 1.01 ms/token, of which the
LM-head GEMV is 0.20 ms. The remaining 0.6 ms of the 1.6 ms step is launch
and host overhead, which C1 (CUDA graphs) targets.

## Batched device decode (C2 step 4, 2026-10-07)

Idle GPU checked before and after every run (CPU load average about 2),
`bench_decode 512 128 cuda <sequences>`, default model. Three runs each,
ms per step and tokens/s over all sequences:

| sequences | ms/step | tokens/s |
|-----------|---------|----------|
| 1  | 1.62, 1.62 (first run 2.42, warm-up) | 618 |
| 4  | 2.01, 2.01, 2.02 | 1994 |
| 16 | 3.08, 3.09, 3.09 | 5194 |
| 64 | 7.04, 7.04, 7.05 | 9093 |

One sequence is unchanged from step 3 (624). A step costs 0.39 ms more at 4
sequences than at 1 and 5.4 ms more at 64, so the single-sequence step is
almost all fixed launch overhead and weight reads.

## LCM sampling (C7, 2026-10-07)

No measurement: `sample_consistency` costs one model call per step like the
other solvers, and its own arithmetic is elementwise. The GPU was held by
another process (RustingStrata, 10.5 GB) this iteration.

## UniPC sampling (C7, 2026-10-07)

No measurement: one model call per step like the other solvers, plus
elementwise arithmetic. The GPU was held by another process (RustingStrata)
this iteration.

## Speculative decoding (C4, 2026-10-07)

No measurement: the GPU was held by another process (RustingStrata, 10.5 GB)
this iteration, and the speedup depends on how often a real draft agrees
with its target, which the random test models do not show. A round costs one
target pass over `lookahead + 1` rows plus `lookahead` draft passes.

## BF16 LM head in device decode (C2 step 5, 2026-10-07)

Profile first, idle GPU at `0f28b3e`, nsys over `bench_decode 512 128 cuda`:
the LM-head GEMV (`gemv2T_kernel_val` on FP32 weights) took 30.8 ms over 129
calls, median 200 us, a fifth of the decode's kernel time; every other
projection already read BF16 packs. After narrowing the head once per
decoder, the same kernel on BF16 weights has a median of 120 us.

Same-session A/B, idle GPU (only gnome-clocks) before and after, CPU load
average about 6, `bench_decode 512 128 cuda`, warm runs:

| Build | ms/token | tokens/s |
|---|---|---|
| `0f28b3e` (FP32 head) | 1.60, 1.60 | 626 |
| BF16 head | 1.50, 1.50 | 666 |

A background run during a `cargo build` on the same machine gave 155-488
tokens/s and was discarded.

## GGUF export (F3, 2026-10-07)

No measurement: export is a one-off file write, and correctness was checked
against the `gguf` Python reader instead. The GPU was held by another process
(RustingEngine tests) this iteration.

## Event tracking off (C1 probe, 2026-10-07)

nsys over 129 decode calls: 228,466 `cuStreamWaitEvent` (65 ms), 124,805
`cuEventRecord` (38.6 ms), 24,274 `cudaLaunchKernel` (50.5 ms). The event
calls come from cudarc's per-`CudaSlice` tracking in multi-stream mode.

Same-session A/B, `bench_decode 512 128 cuda`, idle GPU (gnome-clocks only)
before and after, second (warm) round:

| Build | 1 sequence | 16 sequences |
|---|---|---|
| `74fd609`, tracking on | 1.53 ms, 655 tokens/s | 3.03 ms, 5280 tokens/s |
| tracking off | 1.52 ms, 656 tokens/s | 2.93 ms, 5455 tokens/s |

Not adopted: no gain at batch 1 and +3% at batch 16 do not pay for a global
unsafe switch.

## Activation checkpointing (A10, 2026-10-07)

`examples/checkpoint_memory.rs <batch> <seq_len>`: default model, every block
checkpointed or none, peak of the device's default memory pool over one
training step (`CU_MEMPOOL_ATTR_USED_MEM_HIGH`, weights and optimizer state
included), then the mean of three timed steps. GPU idle (gnome-clocks only)
before and after; host load average about 11 from other work.

| Shape | Peak kept | Peak checkpointed | ms/step kept | ms/step checkpointed |
|---|---|---|---|---|
| 4 x 512 | 1508 MiB | 1078 MiB | 47.9, 43.6 | 53.3, 53.8 |
| 8 x 512 | 2037 MiB | 1251 MiB | - | - |
| 4 x 1024 | 2037 MiB | 1251 MiB | - | - |
| 16 x 512 | 2843 MiB | 1353 MiB | 131.7, 131.8 | 162.1, 160.9 |

Each further 8 x 512 tokens costs about 102 MiB checkpointed against 806 MiB
kept. The recompute adds 23% at 16 x 512, under the 30% the item estimated.
An earlier timing run overlapped a `cargo clippy` build and was discarded.

## Fused decode attention (C8, 2026-10-07)

`nsys` of `bench_decode 512 128 cuda 1` at `1c3dd1f`: per layer and token,
eight FP32 score GEMVs (about 3.8 us each), `attention_softmax_lse` (4.9 us)
and eight FP32 value GEMVs (about 4.3 us), about 70 us, a third of kernel
time. A first fused kernel with one block per (head, sequence) took 105 us a
call at batch 1 (8 blocks, each walking 640 positions serially) and was slower
end to end; splitting the history into 64-position chunks fixed that:
`decode_attention` 16.7 us and `decode_attention_merge` 2.7 us a layer. Total
kernel time over the 129 decode calls is 107 ms, against about 195 ms before.

Same-session A/B, `bench_decode 512 128 cuda <sequences>`, idle GPU
(gnome-clocks only) before and after, second (warm) round, CPU load average
about 5:

| Build | 1 sequence | 16 sequences |
|---|---|---|
| `74fd609` | 1.51 ms, 663 tokens/s | 3.00 ms, 5335 tokens/s |
| fused | 1.16 ms, 864 tokens/s | 2.35 ms, 6819 tokens/s |

The first round's baseline batch-1 run (2.10 ms) was a cold outlier.

## BF16 decode attention output (C9, 2026-10-07)

Profile at `b014141` (`nsys`, `bench_decode 512 128 cuda 1`): the attention
output projection was the last FP32 GEMV in a mixed-precision decode step,
1024 calls at a median 6.0 us (5.7% of kernel time), against 5.5 us for the
BF16 weight GEMVs of four times the size. The merge kernel now writes BF16.

Same-session A/B, `bench_decode 512 128 cuda <sequences>`, three rounds, run
only once the GPU was idle (gnome-clocks only) and the CPU load average was
under 2. A first attempt ran beside another session's `rustc` build and a
`forever_bear` process at load average 9; batch-1 numbers then varied from 349
to 663 tokens/s and were discarded. Warm rounds (the first `b014141` batch-1
run, 653 tokens/s, was cold):

| Build | 1 sequence | 16 sequences |
|---|---|---|
| `b014141` | 1.11-1.12 ms, 893-898 tokens/s | 2.25-2.27 ms, 7035-7100 tokens/s |
| BF16 output | 1.07-1.09 ms, 919-931 tokens/s | 2.25-2.26 ms, 7089-7123 tokens/s |

## Int8 device decode (C5 slice 1, 2026-10-07)

Profile at `0f20131` before the change: BF16 weight GEMVs 31.9% and the BF16
head 12.2% of a batch-1 decode step's kernel time, the bandwidth-bound part
int8 halves. `bench_decode 512 128 cuda <sequences> [int8]`, three rounds
interleaved, started once the GPU was idle (gnome-clocks only) and the CPU
load average under 2; `forever_bear` reached the GPU near the end of the
third round, whose numbers match the first two.

| Build | 1 sequence | 16 sequences |
|---|---|---|
| `0f20131` | 891-909 tokens/s | 6985-7034 tokens/s |
| new, int8 off | 904-910 tokens/s | 7049-7071 tokens/s |
| new, int8 on | 998-1006 tokens/s | 6081-6105 tokens/s |

(One `0f20131` batch-1 run read 717 tokens/s, a cold outlier.)

`nsys` kernel sums over the 129 decode calls, idle GPU:

| Run | `gemv_q8` | Other |
|---|---|---|
| int8, 1 sequence | 4225 calls, 5.2 us avg, 28.5% | BF16 expert GEMVs 3.8 us avg, 19.2%; `decode_attention` 18.3% |
| int8, 16 sequences | 4225 calls, 21.8 us avg, 92 ms | `decode_attention` 99 ms |
| BF16, 16 sequences | - | cuBLAS BF16 GEMV kernels about 78 ms; `decode_attention` 107 ms |

At 16 rows the int8 kernel re-reads each weight four times (four rows a
pass) and spends most of its time on per-row warp reductions and input
loads, so it loses to cuBLAS.

### Rejected: shared-memory input rows for the int8 GEMV (2026-10-07)

A `gemv_q8_rows` kernel for more than four rows: each block of eight units
staged a 512-wide slice of every input row in shared memory as FP32, and each
lane read four consecutive weights in four 128-wide groups, so all loads were
contiguous across lanes. It passed the int8 parity test, now with a six-token
turn, and dropping one weight in four failed it. `bench_decode 512 128 cuda
<sequences> int8`, three rounds interleaved, started on an idle GPU with the
load average under 2 (`RustingStrata` reached the GPU near the end; rounds 2
and 3 match round 1):

| Int8 decode | `9561323` | Shared-memory kernel |
|---|---|---|
| 1 sequence | 953-1000 tokens/s | 984-1002 tokens/s (first run cold, 654) |
| 8 sequences | 4325-4360 tokens/s | 3517-3658 tokens/s |
| 16 sequences | 5978-6093 tokens/s | 4874-4900 tokens/s |

The 16-row input is 16 KB in BF16, so the old kernel's per-warp re-reads
most likely hit L1, and the estimate of L2 traffic behind the change was
wrong. Staging adds two block barriers and 8192 scalar loads a slice for 256
multiply-adds per lane, and 32 KB of shared memory caps a block at three per
SM. Nsight Compute cannot read the counters on this machine
(`ERR_NVGPUCTRPERM`, needs admin), so the next attempt must start from an
`nsys` sweep of kernel variants, not from a counter profile.

## Int8 routed experts (C5 slice 2, 2026-10-07)

Profile at `9561323` (int8, one sequence): the routed experts' BF16 GEMVs
were 19.2% of kernel time, the largest share left at full precision. With
this change the experts and the router read int8 as well.
`bench_decode 512 128 cuda <sequences> [int8]`, three rounds interleaved,
idle GPU, load average under 2, CLEAN. The machine ran slower than in the
slice-1 session (int8 off 820 against 905 tokens/s at one sequence), so
compare within this table only:

| Build | 1 sequence | 16 sequences |
|---|---|---|
| `9561323`, int8 on | 893-897 tokens/s | 5568-5617 tokens/s |
| new, int8 off | 818-824 tokens/s | 6389-6518 tokens/s |
| new, int8 on | 971-979 tokens/s | 5626-5726 tokens/s |

(The first new int8 run at one sequence read 831, cold.) Int8 is now +19%
over BF16 at one sequence, and still -12% at 16, the multi-row GEMV.

## LoRA on an int8 base (B4 QLoRA slice 1, 2026-10-07)

`TransformerLm::quantize_lora_base` stores every adapted projection's frozen
base as int8 with a scale per row; the device uploads those bytes and
dequantizes them into each weight pack where it used to cast FP32.
`checkpoint_memory 4 512 lora`, default model, rank-16 LoRA, no
checkpointing, two runs, load average 16 (another loop's `rustc`; the A/B
ran in the same process):

| Base | Peak device memory | Step |
|---|---|---|
| FP32 | 889 MiB | 46.1, 46.5 ms |
| int8 | 778 MiB | 46.6, 46.3 ms |

The 111 MiB is the adapted weights going from 4 bytes to 1 (about 38M of
the 101.7M; the embedding, the norms and the routers stay FP32). The step
does not slow: the dequantize kernel replaces a cast of the same length.
The BF16 packs a step keeps for its backward pass are unchanged, so at
this shape activations and packs, not the base, are most of the peak; the
saving grows with the model and is largest with `checkpoint`, which drops
the packs between blocks.
