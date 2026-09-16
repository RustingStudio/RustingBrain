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
