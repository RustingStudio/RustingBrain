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

**Text-to-image models** — an MMDiT denoiser, a flow-matching and DDIM
sampler, a VAE decoder, a byte-level BPE tokenizer and a LLaMA-family prompt
encoder, read straight from a published safetensors directory.

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

## Held-out loss

`evaluate` is a forward pass and a loss, with no gradients and no step:

```rust
let validation = model.evaluate(&held_out)?;
println!("val {:.4}  ppl {:.2}", validation.lm_loss, validation.perplexity());
```

Training loss falls whether or not the model is learning anything general; the
gap between the two is what says which.

## Corpora too large for memory

`TokenBatch` takes ids the caller has already materialized, which caps a run at
what fits in RAM. `TokenFile` reads windows out of a flat file of little-endian
`u32` ids instead, so only the batch is resident:

```rust
TokenFile::write("corpus.bin", &ids)?;          // once, after tokenizing

let mut corpus = TokenFile::open("corpus.bin", 42)?;
for step in 0..steps {
    let loss = model.train_step_batch(&corpus.batch(step, 8, 512)?)?;
}
```

Windows start at random offsets, which is how a corpus too large to shuffle
gets shuffled, and the offsets come from the step number alone — a run resumed
at step 40,000 sees the data it would have seen, with no cursor to checkpoint.

A held-out file wants the opposite: every token once, in order, so the number
is comparable between evaluations. That is `chunk`, which returns `None` at the
end of the corpus:

```rust
while let Some(batch) = held_out.chunk(batches, 8, 512)? {
    total += model.evaluate(&batch)?.lm_loss;
    batches += 1;
}
```

Public corpora ship as JSONL, so that conversion is one call. The file is read
a line at a time and ids are written as they come, and `tokenize` is where your
BPE vocabulary goes:

```rust
TokenFile::write_jsonl("corpus.bin", "corpus.jsonl", "text", Some(eos), |text| {
    tokenizer.encode(text, false).unwrap().get_ids().to_vec()
})?;
```

## Generating text

`generate` prefills a KV cache with the prompt and then decodes one token at a
time, which costs one row of attention per token instead of *n*:

```rust
let mut sampler = Sampler::temperature(0.8, Some(42))   // None seeds from the OS
    .top_k(40)
    .top_p(0.95)
    .repetition_penalty(1.1);

let new_ids = model.generate(&prompt_ids, 120, &mut sampler)?;
```

Each token costs a forward pass, so a caller that waits for the whole
continuation waits in silence. `generate_with` hands over each id as it is
decoded, and stops the run when the callback returns `false` — which is where
an end-of-text id or a stop sequence gets noticed:

```rust
let new_ids = model.generate_with(&prompt_ids, 120, &mut sampler, |id| {
    print!("{}", tokenizer.decode(&[id]));
    id != end_of_text
})?;
```

`Sampler::greedy()` takes the argmax instead, which is what an evaluation
wants.

A conversation is several generations over a growing history, and `generate`
throws its caches away each time, so turn *n* re-reads everything before it. A
`Decoder` keeps them:

```rust
let mut decoder = model.decoder();
decoder.feed(&prompt_ids)?;                       // prefill
let id = decoder.next(&mut sampler)?;             // one token
decoder.feed(&next_turn_ids)?;                    // only the new tokens are read
```

The cache underneath is `forward_cached`, and calling it directly is still the
way to drive decoding yourself:

```rust
let mut caches = model.new_kv_caches();
let mut logits = model.forward_cached(&prompt_ids, &mut caches)?;   // prefill
logits = model.forward_cached(&[next], &mut caches)?;               // decode
```

Decoding one token reads every active weight once, so a CPU decode step is
bound by memory bandwidth. `quantize` rounds the weights to one byte each,
which is a quarter of the traffic and roughly twice the tokens per second:

```rust
let mut model = TransformerLm::load_bin("model.rbw")?;
model.quantize();                                 // one-way, inference only
let new_ids = model.generate(&prompt_ids, 120, &mut sampler)?;
```

The `f32` weights are gone afterwards, so training, `save_bin` and `to_cuda`
return an error rather than write out rounded weights.

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

The optimizer is re-read every step, so a schedule is one assignment and there
is nothing to register:

```rust
let schedule = Schedule::warmup_cosine(3e-4, 2_000, 100_000);   // peak, warmup, total
model.optimizer.set_learning_rate(schedule.rate(step));
```

Linear warmup to the peak, then a cosine decay to a tenth of it; `floor` moves
that last figure. `set_learning_rate` rather than a fresh `Optimizer::adam`,
which would drop the betas and the weight decay back to their defaults.

`step_clipped` is `step` with the gradients clipped to a global L2 norm first,
and it returns the pre-clip norm, which is worth logging: it rises a step or
two before a run diverges. The norm is computed on whichever device the model
is on, so clipping on a GPU costs one cuBLAS reduction per parameter and no
transfers.

```rust
let norm = model.step_clipped(1.0 / group.len() as f32, 1.0)?;
```

## Fine-tuning on instructions

Supervised fine-tuning is pre-training with the prompt excluded from the loss:
the instruction conditions the model, the response is what it learns to write.
`TokenBatch::supervised` builds that batch from prompt-and-response pairs, and
the mask travels with it:

```rust
let batch = TokenBatch::supervised(&[
    (tokenizer.encode("<user>what owns a value?<assistant>"), tokenizer.encode("one owner.<end>")),
])?;
model.accumulate_step(&batch)?;      // only the response tokens count
model.step(1.0);
```

`with_loss_mask` takes the flat `batch * seq_len` flags directly, for a layout
this does not cover.

## Fine-tuning with LoRA

`add_lora` attaches a low-rank adapter to every attention and feed-forward
projection and freezes everything else. The base weights keep their values but
give up their gradient and their two Adam moments, so a model that needed four
weight-sized buffers to train needs one plus the adapters:

```rust
let mut model = TransformerLm::load_bin("base.rbw")?;
model.add_lora(LoraConfig::new(16).alpha(32.0))?;

let total = model.parameter_counts().total;
println!("{} of {total} weights train", model.trainable_parameters());

for batch in batches {
    model.zero_grad();
    model.train_step_batch(&batch)?;
    model.step(1.0);
}

model.save_lora("adapter.rbl")?;     // the adapters alone, a few megabytes
```

The adapters start at zero, so the model predicts exactly what it predicted
before the call; the first optimizer step is what moves it. Nothing else in the
training loop changes — the frozen parameters ignore `step` and accumulate no
gradient.

An adapter file loads onto a base model prepared the same way, which is how one
base checkpoint serves several adapters:

```rust
let mut model = TransformerLm::load_bin("base.rbw")?;
model.add_lora(LoraConfig::new(16).alpha(32.0))?;
model.load_lora("adapter.rbl")?;
```

`merge_lora` folds the adapters into the weights they adapt and unfreezes the
model, leaving an ordinary checkpoint that predicts the same thing at no extra
inference cost.

`to_cuda` carries the adapters onto the device with the projections they adapt,
and the whole loop above runs there unchanged, including the experts and the
shared expert of a mixture-of-experts layer. The saving is the same one it is
on the host: a frozen weight never allocates its gradient or its two Adam
moments, so a device that held a model at four buffers per weight holds it at
one. Attach and merge are host operations — call `to_cpu` first, or attach
before `to_cuda`.

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
println!("{:.1}%", 100.0 * model.accuracy(&test)?);   // argmax, or 0.5 for one output
println!("{:?}", model.confusion_matrix(&test)?);     // [actual][predicted] counts
```

Two thousand epochs are two thousand epochs of silence. `fit_with` is `fit`
with a callback per epoch, and returning `false` stops the run early — the
history it returns holds only the epochs that actually ran:

```rust
let history = model.fit_with(&data, config, |epoch, loss| {
    if epoch % 100 == 0 {
        println!("epoch {epoch}  loss {loss:.6}");
    }
    loss > 1e-5                                      // stop once it is small enough
})?;
```

Tabular data can come straight off disk. The last `target_columns` columns are
the target, a non-numeric first row is treated as a header, and quoted fields
are understood:

```rust
let dataset = Dataset::from_csv("iris.csv", 1)?;
let (train, test) = dataset.split(0.8);
```

`split_stratified` is the same split with each class's share held on both
sides, which is what an imbalanced set needs — a plain split can leave a rare
class out of the test half and the accuracy then says nothing about it:

```rust
let (train, test) = dataset.split_stratified(0.8);
```

A classification file off the internet ends in a class name rather than a
number — `5.1,3.5,1.4,0.2,setosa`. `from_csv_labeled` reads that directly, and
hands back one-hot targets with the class names beside them:

```rust
let (dataset, classes) = Dataset::from_csv_labeled("iris.csv")?;
```

Columns on different scales — an age beside an income — cost a network its
first epochs and can stall it outright. `standardize` centers each input column
and returns the statistics, which belong with the model: fit them on the
training split, apply them everywhere else.

```rust
let statistics = train.standardize();
statistics.apply(&mut test);
statistics.apply_row(&mut row);        // before predicting on anything new
```

Arrays exported from Python arrive as `.npy`, which is smaller and exact where
a CSV is neither. `float32`, `float64`, `int32`, `int64` and `uint8` all read;
labels stored as class indices become one-hot rows in one call:

```rust
let mut dataset = Dataset::from_npy("x.npy", "y.npy")?;   // numpy.save wrote both
dataset.one_hot_targets(10)?;
```

MNIST and everything shaped like it ship as IDX instead. `from_idx` reads the
pair, flattens each image into a row and scales the pixels to `0.0..=1.0`
(gunzip the archives first):

```rust
let mut train = Dataset::from_idx("train-images-idx3-ubyte", "train-labels-idx1-ubyte")?;
train.one_hot_targets(10)?;
```

Images load the same way, behind `--features images`: one subdirectory per
class, resized and flattened into one row per image, with a one-hot target and
the class names alongside.

```rust
let (dataset, classes) = Dataset::from_image_folder("images", 16, 16, true)?;
```

That is pixels into a dense network, not a convolutional one — enough for
separable shapes, nothing like enough for photographs. See
`cargo run --release --example image_classification --features images`.

`flip_horizontal` mirrors every image and appends it with its label, which is
twice the training data for one decode. Run it on the training split only,
after `split`:

```rust
let (mut train, test) = dataset.split(0.8);
train.flip_horizontal(16)?;                          // the image width, in pixels
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

## Text to image

```rust
use rusting_brain::ImagePipeline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = ImagePipeline::load("models/FLUX.2-klein-4B")?;
    let image = pipeline.generate("a lighthouse in fog", 1024, 1024, Some(7), |_, _| true)?;
    rusting_brain::vae::save_png(&image, "lighthouse.png")?;
    Ok(())
}
```

The directory is the one the model was published in: a denoiser (a
`transformer` or a `unet`), a `vae`, a `text_encoder` and a `tokenizer` beside
each other, each with its own `config.json`. A model that carries two encoders
is loaded the same way, both towers and both tokenizers: a `text_encoder_2`
holding T5 for the per-token states with a CLIP tower in `text_encoder` for the
pooled vector, which is FLUX.1, or two CLIP towers whose states are laid side
by side, which is Stable Diffusion XL. A directory with three encoders is
Stable Diffusion 3: both CLIP towers padded out to T5's width with the T5
states stacked underneath them in one sequence. Which encoder the
states come from is read from that encoder's own configuration: a LLaMA-family
decoder over byte-level BPE, T5 over sentencepiece, or CLIP. Nothing about the
model is compiled in — the shapes are read from those files, and both published
spellings of a transformer checkpoint are understood, so a repacked model loads
without conversion.

A run can start from a picture instead of from noise. `generate_from_image`
takes one in `[-1, 1]` and a `strength` between zero and one, which says how
far back up the schedule to push it before denoising: low values keep the
composition and repaint the detail, high values keep little but the layout.
It reads the encoder half of the same `vae` file, which `load` attaches
whenever the checkpoint holds it.

`ImagePipeline::load_at(directory, Precision::Q8)` holds the denoiser and the
prompt encoder at one byte per weight, a quarter of the memory, which is what
decides whether a 4B-parameter model fits. The parts are also usable one at a
time — `condition`, `sample_latent` and `decode` — so a caller short of memory
drops each one as it finishes.

```bash
cargo run --release --features images --example text_to_image -- \
    models/FLUX.2-klein-4B "a lighthouse in fog" out.png
```

`PipelineConfig::solver` picks how a step is taken — Euler, Euler ancestral or
DPM++ 2M — and `load` starts it at whichever one the checkpoint's scheduler
names.

Sampling runs on the CPU today. Both denoisers in this field are implemented —
the rectified-flow transformer (`Dit`, which is FLUX, FLUX.2 and Stable
Diffusion 3) and the UNet (`Unet`, which is Stable Diffusion 1.x, 2.x and XL,
sampled with DDIM and classifier-free guidance) — as are the three prompt
encoders: a LLaMA-family decoder (`TextEncoder`), T5 (`T5Encoder`, with the
`Unigram` tokenizer beside it) and CLIP (`ClipTextEncoder`), one tower or two.

## Importing models

```bash
cargo run --example onnx_inference --features onnx -- model.onnx
```

Some TensorFlow exports leave the input shape dynamic; pass the shape and the
values explicitly:

```bash
cargo run --example onnx_inference --features onnx -- xor.onnx 1,2 0,1
```

Reading an ONNX file needs `--features onnx`; writing one does not.
`save_onnx` exports a dense `Network` as a `Gemm`-per-layer graph at opset 13,
with a symbolic batch dimension, which is how a model trained here runs under
ONNX Runtime, TensorRT or a browser:

```rust
model.save_onnx("model.onnx")?;
```

Dense networks only — a transformer's RMSNorm, SwiGLU and MoE routing have no
such short representation, and `TransformerLm::save_bin` stays the way to move
one. See [IMPORT_MODELS.md](IMPORT_MODELS.md) for the TensorFlow/Keras export
flow in the other direction.

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

Decoder-only transformers, dense feed-forward networks and transformer
text-to-image models, on one machine. Convolutions exist for the image
autoencoder's sake and are inference-only, with no backward pass. Encoder-decoder
models, multi-GPU and distributed training are not implemented and are not
planned.

## License

RustingBrain is released under the RustingBrain License 1.0. You can use,
modify, and distribute it, including in commercial projects, but redistributed
copies must keep the license and credit Vasyl Trefilov as the original author.

See [LICENSE.md](LICENSE.md) for the full terms.
