# RustingBrain

A deep-learning library in Rust, from a two-layer XOR network up to a
300M-parameter transformer language model trained on one desktop GPU.

No Python in the loop, no C++ build step, no framework runtime. The CUDA
kernels are compiled at startup by NVRTC and the rest is Rust.

## Where it stands against PyTorch and TensorFlow

Same 101.7M-parameter dense transformer (`d_model` 768, 16 layers, GQA 12/4
heads, `head_dim` 64, SwiGLU, `d_ff` 1408), same batch and sequence shapes,
mixed precision on both sides, each framework on its fastest compiled path —
`torch.compile` for PyTorch, `jit_compile=True` (XLA) for TensorFlow. One idle
RTX 3060 12 GB. Training tokens per second, higher is better:

| batch × seq | RustingBrain | TensorFlow | PyTorch |
|---|---|---|---|
| 1 × 512  | 13071 | 15560 | 9607  |
| 1 × 1024 | **16352** | 15652 | 12235 |
| 4 × 512  | 22045 | 22412 | 17993 |
| 4 × 1024 | **21787** | 20063 | 20544 |
| 8 × 512  | 22926 | 23906 | 21366 |

RustingBrain beats or ties PyTorch at every shape and is within 5–8% of
TensorFlow except at 4×1024, where it is ahead. GEMMs are 60% of a step at
78–98% of the card's 25.5 TFLOPS BF16 peak.

These are single-GPU numbers on one architecture family. There is no
multi-GPU, no distributed training, and no CPU-offload path; if your run needs
those, use PyTorch. Per-architecture throughput for other shapes is in
[docs/baseline.md](docs/baseline.md), and the optimization work behind the
table is in [docs/optimize-plan.md](docs/optimize-plan.md).

## What is in the box

**Transformer language models** — decoder-only, with grouped-query attention,
rotary positions, RMSNorm, SwiGLU feed-forwards, tied embeddings, and a fused
causal flash-attention kernel for Ampere tensor cores.

**Mixture of experts** — dropless top-k routing, an optional always-active
shared expert, Switch-style load-balancing loss and ST-MoE router z-loss. Any
subset of layers can be sparse while the rest stay dense.

**Dense feed-forward networks** — regression, binary and multiclass
classification, with SGD and Adam, mini-batch training, and reproducible
shuffling. This is where the tutorials start.

**Training infrastructure** — gradient accumulation, mixed precision (BF16
GEMMs with FP32 master weights), KV-cached decoding, Adam state that survives a
restart, and F32 or int8-quantized binary checkpoints.

**Backends** — CPU (rayon, AVX-512-friendly matmul kernels), CUDA for both
model types, Metal for dense networks on Apple Silicon.

**Interop** — ONNX inference for models trained in TensorFlow or PyTorch.

## Install

```bash
cargo add rusting_brain                    # CPU only
cargo add rusting_brain --features cuda    # NVIDIA GPU
cargo add rusting_brain --features onnx    # ONNX inference
```

CUDA is off by default. The crate builds and every test passes with no driver,
no toolkit, and no device present. See [INSTALL_CUDA.md](INSTALL_CUDA.md) for
the driver and toolkit setup.

Upgrading from 1.x: three unused modules were removed. See
[CHANGELOG.md](CHANGELOG.md).

## A language model in twenty lines

```rust
use rusting_brain::{Optimizer, TransformerLm};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model = TransformerLm::builder()
        .vocab_size(32_000)
        .d_model(512)
        .n_layers(8)
        .heads(8, 2, 64)          // 8 query heads, 2 key/value heads
        .d_ff(1408)
        .experts(8, 2)            // 8 experts, 2 active per token
        .moe_layers(2..8)         // layers 0 and 1 stay dense
        .max_seq_len(1024)
        .optimizer(Optimizer::adam(3e-4))
        .seed(42)
        .build()?;

    println!("{}", model.parameter_counts());   // 55.2M total, 35.7M active

    let batch: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
    let loss = model.train_step(&batch)?;
    println!("loss {:.4}", loss.lm_loss);
    Ok(())
}
```

Tokens go in as `u32` ids. RustingBrain does not ship a tokenizer; the
[`tokenizers`](https://crates.io/crates/tokenizers) crate trains and loads
Hugging Face BPE vocabularies, and chapter 14 of the tutorials walks through
it.

## Generating text

`forward_cached` appends to a per-layer KV cache, so decoding the *n*-th token
costs one row of attention instead of *n*:

```rust
let mut caches = model.new_kv_caches();
let mut logits = model.forward_cached(&prompt_ids, &mut caches)?;   // prefill

for _ in 0..max_new_tokens {
    let next = sample(logits.row(logits.rows - 1), temperature, top_k);
    ids.push(next);
    logits = model.forward_cached(&[next], &mut caches)?;            // decode
}
```

## Training on a GPU

```rust
model.set_mixed_precision(true);   // BF16 GEMM operands, FP32 master weights
model.to_cuda(0, 9_000)?;          // device 0, 9000 MiB budget
```

Every later `train_step`, `forward_batch`, and `backward` runs on the device.
The path is fail-closed: a driver, cuBLAS, kernel, allocation, or numerical
failure returns a `NetworkError` rather than silently continuing on the CPU.
`sync_from_device()` pulls the weights back before a save.

A 12 GiB card should be given a budget around `9000` MiB, leaving room for the
desktop, the driver, and everything else on the machine.

## Gradient accumulation

The effective batch is not bounded by device memory:

```rust
model.zero_grad();
for micro_batch in group {
    model.accumulate_step(&TokenBatch::new(micro_batch)?)?;
}
model.step(1.0 / group.len() as f32);
```

Assigning `model.optimizer` a fresh `Optimizer::adam(rate)` before `step` is
all a warmup or cosine schedule needs — the optimizer is re-read every step.

## Saving and resuming

```rust
model.save_bin("model.rbw", Precision::F32)?;     // 4 bytes per weight
model.save_bin("model.rbw", Precision::Q8)?;      // 1 byte per weight, lossy
model.save_optimizer_state("model.rbw.opt")?;     // Adam moments and step

let mut model = TransformerLm::load_bin("model.rbw")?;
model.load_optimizer_state("model.rbw.opt")?;
```

Save `F32` for anything you intend to resume: int8 rounding costs about 0.4%
per weight, which a resumed Adam run turns into a visible step in the loss
curve. `Q8` is for shipping a model that will only be run.

Dense `Network`s use `save_json` / `load_json`, which is readable and portable
but costs roughly ten bytes per weight.

## Dense networks

The tabular side of the library, unchanged and still the place to start:

```rust
let mut model = Network::builder()
    .input_size(2)
    .dense(8, Activation::Tanh)
    .dense(1, Activation::Sigmoid)
    .loss(Loss::BinaryCrossEntropy)
    .optimizer(Optimizer::adam(0.05))
    .build();

model.fit(&data, TrainConfig { epochs: 2_000, batch_size: 4, shuffle: true, seed: Some(42) })?;
println!("{:?}", model.predict(&[1.0, 0.0])?);
```

```bash
cargo run --example xor
cargo run --example regression
cargo run --example classification
cargo run --example save_load
```

## Examples

```bash
cargo run --release --example language_model -- train            # dense
cargo run --release --example language_model -- train --moe      # sparse
cargo run --release --example language_model -- generate "the borrow checker"
```

`train` writes a checkpoint and a vocabulary; `generate` is a separate process
that reads them back. Chapters 14 to 16 of the tutorials explain the file.

## Importing models

```bash
cargo run --example onnx_inference --features onnx -- model.onnx
```

Some TensorFlow exports leave the input shape dynamic; pass the shape and the
values explicitly:

```bash
cargo run --example onnx_inference --features onnx -- xor.onnx 1,2 0,1
```

ONNX support is inference only. See [IMPORT_MODELS.md](IMPORT_MODELS.md) for
the TensorFlow/Keras export flow.

## Tutorials

Sixteen chapters in [tutorials/](tutorials/README.md), from what a neural
network is to training a language model end to end. Chapters 1–10 need nothing
but Rust; 11 and 14–16 want a GPU.

## Benchmarks

```bash
cargo run --release --example bench_train_step --features cuda -- 4 1024 cuda
cargo run --release --example bench_decode -- 512 128
cargo run --release --example sweep_arch --features cuda
cargo run --release --example cuda_benchmark --features cuda
```

## Scope

Decoder-only transformers and dense feed-forward networks, on one machine.
Convolutions, encoder-decoder models, multi-GPU and distributed training are
not implemented and are not planned.

## License

RustingBrain is released under the RustingBrain License 1.0. You can use,
modify, and distribute it, including in commercial projects, but redistributed
copies must keep the license and credit Vasyl Trefilov as the original author.

See [LICENSE.md](LICENSE.md) for the full terms.
