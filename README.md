# RustingBrain

A deep-learning library in Rust, from a two-layer XOR network up to a
300M-parameter transformer language model trained on one desktop GPU.

No Python in the loop, no C++ build step, no framework runtime. The CUDA
kernels are compiled at startup by NVRTC and the rest is Rust.

## Speed

Training tokens per second for the same 101.7M-parameter transformer, mixed
precision on every side, `torch.compile` for PyTorch and XLA for TensorFlow, on
one idle RTX 3060 12 GB:

| batch × seq | RustingBrain | TensorFlow | PyTorch |
|---|---|---|---|
| 1 × 512  | 13071 | 15560 | 9607  |
| 1 × 1024 | **16352** | 15652 | 12235 |
| 4 × 512  | 22045 | 22412 | 17993 |
| 4 × 1024 | **21787** | 20063 | 20544 |
| 8 × 512  | 22926 | 23906 | 21366 |

Single GPU, one architecture family. Other shapes are in
[docs/baseline.md](docs/baseline.md).

## Features

- **Language models**: decoder-only transformers with grouped-query attention,
  rotary positions, RMSNorm, SwiGLU, flash attention and mixture-of-experts
  layers.
- **Training**: mixed precision, gradient accumulation, clipping, schedules,
  LoRA, resumable checkpoints, corpora larger than memory.
- **Inference**: KV-cached and streaming generation, samplers, int8 weights.
- **Images and video**: FLUX, FLUX.2, Stable Diffusion 1.x to 3 and Wan 2.2,
  loaded from the published safetensors directory.
- **Dense networks**: regression and classification with CSV, `.npy`, IDX and
  image-folder loaders.
- **Backends**: CPU, CUDA, and Metal for dense networks.
- **Interop**: ONNX import and export, Hugging Face tokenizers and checkpoints.

## Install

```bash
cargo add rusting_brain                    # CPU only
cargo add rusting_brain --features cuda    # NVIDIA GPU
```

The crate builds and tests with no GPU present. Driver and toolkit setup is in
[INSTALL_CUDA.md](INSTALL_CUDA.md). Other features: `onnx`, `images`, `metal`.

## A language model

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

    model.set_mixed_precision(true);
    model.to_cuda(0, 9_000)?;     // device 0, 9000 MiB budget; skip for CPU

    let batch: Vec<Vec<u32>> = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
    let loss = model.train_step(&batch)?;
    println!("loss {:.4}", loss.lm_loss);
    Ok(())
}
```

## A dense network

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

## Text to image

```rust
let mut pipeline = ImagePipeline::load("models/FLUX.2-klein-4B")?;
let image = pipeline.generate("a lighthouse in fog", 1024, 1024, Some(7), |_, _| true)?;
rusting_brain::vae::save_png(&image, "lighthouse.png")?;
```

## Learn more

The [tutorials](tutorials/README.md) are an eighteen-chapter course with
runnable code for each chapter.

| Topic | Chapter |
|---|---|
| Loading and preparing data | [4](tutorials/4_data.md) |
| Training loop, evaluation, saving | [7](tutorials/7_training_loop.md), [8](tutorials/8_evaluation.md), [9](tutorials/9_save_load.md) |
| CUDA | [11](tutorials/11_gpu_cuda.md) |
| ONNX import and export | [12](tutorials/12_import_models.md) |
| Tokenizers, generation, sampling | [14](tutorials/14_language_model.md) |
| Mixture of experts | [15](tutorials/15_mixture_of_experts.md) |
| Long runs, resuming, instruction tuning | [16](tutorials/16_training_an_llm.md) |
| LoRA | [17](tutorials/17_lora_fine_tuning.md) |
| Text to image | [18](tutorials/18_text_to_image.md) |

## Examples

```bash
cargo run --example xor
cargo run --release --example language_model -- train --moe
cargo run --release --example language_model -- generate "the borrow checker"
cargo run --release --features images --example text_to_image -- \
    models/FLUX.2-klein-4B "a lighthouse in fog" out.png
cargo run --release --features cuda --example bench_train_step -- 4 1024 cuda
```

The rest are in [examples/](examples).

## Scope

One machine, one GPU. Encoder-decoder models, multi-GPU and distributed
training are not implemented and are not planned.

## License

RustingBrain License 1.1: use, modify and distribute it, and sell projects
built with it, but do not sell RustingBrain itself. See
[LICENSE.md](LICENSE.md).
