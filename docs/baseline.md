# Training throughput baseline

Measured with `examples/sweep_arch.rs`, one configuration per process, on an
otherwise idle RTX 3060 12 GB (desktop applications holding 1258 MiB).

## Regression check

`scripts/bench_check.sh` compares the best tokens/s of five runs against these
rows and fails below 97%. Measured on the idle RTX 3060
(train step) and the Ryzen 5 7600X (decode, CPU). One run spreads about 3.5% on
both. `bench_decode` is sensitive to anything else on the CPU: with one
parallel `rustc` it measured 1464-2088 tokens/s, so run it on a quiet machine.
Re-record with `scripts/bench_check.sh --record` after a change that is meant
to move these numbers.

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

The dated measurements behind each change are in
[measurements.md](measurements.md).
