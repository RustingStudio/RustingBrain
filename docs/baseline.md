# Training throughput baseline

Measured with `examples/sweep_arch.rs`, one configuration per process, on an
otherwise idle RTX 3060 12 GB (desktop applications holding 1258 MiB).

Every earlier measurement in this repository's history was taken while a
second training job held 5754 MiB and 100% of the GPU. Those numbers are
superseded by this table and should not be compared against it.

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
