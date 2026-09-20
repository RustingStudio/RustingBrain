# Changelog

## Unreleased

Everything here is additive except the seams around `ImagePipeline`:
`ImagePipeline::new` takes a `PromptEncoder` in place of a tokenizer and an
encoder and an `ImageDenoiser` in place of a `Dit` (a `Dit` converts into one,
so passing one still compiles), the pipeline's `tokenizer` and `encoder` fields
are now the one `encoder` field, its `denoiser` field is an `ImageDenoiser`,
`PipelineConfig` has gained `betas` and `solver` fields, `SamplingConfig` has
gained a `solver` field, and `PromptEncoder::encode` hands back the pooled
vector alongside the states in place of the separate `pool`.

### Fixed

- `examples/text_to_image.rs` ran no classifier-free guidance. It called
  `sample_latent`, which passes no unconditional conditioning, so a model whose
  configuration asked for a guidance above one silently got one denoiser pass a
  step instead of two — a faster run and a weaker image. It now builds the
  unconditional conditioning when the guidance calls for it, from `NEGATIVE` or
  from the empty prompt.

### Changed

- A `DatasetStream` epoch's row order is a function of that epoch's seed alone.
  `Dataset::shuffle` permutes the order it is given, so shuffling once per epoch
  made every epoch's order depend on all the orders before it. Seeded runs are
  still reproducible, but a given seed no longer produces the row order it did
  in 2.0.0.
- `Sampler::pick` no longer sorts the whole vocabulary. With a top-k it
  partitions instead, which is most of a small model's decode step back.
- The CUDA image kernels were rewritten around what a profile said they were
  actually spending time on, which took a Stable Diffusion XL UNet pass at
  1024x1024 from 0.65 s to 0.33 s and the VAE decode of that image from 1.72 s
  to 0.53 s, and a whole guided 1024x1024 image at 30 steps from 37.4 s to
  20.5 s. The convolution lowering writes a whole KxK patch per thread
  rather than one tap, and writes its columns transposed so the matmul lands
  channel-major with no separate pass to reorder it. Group normalization is
  three kernels — partial sums, a double-precision fold, then one pass that
  scales and activates — instead of one, which is also where the activation
  after a normalization now happens. The attention softmax normalizes the
  narrow score matrix in place rather than expanding it into a second buffer, and it
  makes one pass over each row with a running maximum. Every kernel that walks
  a plane takes its row from the grid rather than dividing a flat index, and
  every kernel that streams one moves four values a thread instead of one.

  A residual connection no longer costs a kernel of its own: the projection or
  convolution that feeds it accumulates into the sum, which is what cuBLAS does
  when its output is scaled by one rather than zero.

  A self-attention's query, key and value projections are one weight and one
  matmul rather than three. Three narrow products leave the last wave of tiles
  on the card half empty; one wide product does not. Attention reads the three
  back as column windows on the result, so nothing is copied to split them.

  The attention softmax gives a row to a warp rather than to a whole block,
  which reduces in registers instead of across sixteen barriers, and reads four
  scores per load where the row length allows it. A row too long for one warp
  to walk still gets a block.

  The image path holds its weights and activations as FP16 rather than BF16,
  and its matmuls accumulate in FP16 as well. A consumer Ampere card runs its
  tensor cores at half rate when the accumulator is FP32, so this is the same
  arithmetic at about 1.7 times the throughput; FP16's eleven mantissa bits
  pay back what the narrow accumulator loses, and the worst relative error
  against the CPU reference on a whole UNet pass went down rather than up.
  Training is unaffected and stays on BF16, where the wider exponent range is
  what keeps small gradients from flushing to zero.

### Added

**Datasets**

- `Dataset::from_csv` and `from_csv_labeled` read a CSV directly, the second
  returning the class names alongside a one-hot dataset.
- `Dataset::from_npy` reads a NumPy `.npy` array, and `one_hot_targets`
  expands an integer label column into one-hot rows.
- `Dataset::from_idx` reads the IDX pair the MNIST-shaped datasets ship as.
- `Dataset::from_image_folder` reads a `class/*.png` tree (`--features images`).
- `Dataset::split_stratified` keeps every class's proportion on both sides of
  the split, which a plain `split` does not on an imbalanced dataset.
- `Dataset::standardize` returns a `Standardizer` holding the mean and
  deviation it used, with `apply` and `apply_row` for the test split and for
  inference. Serializing it is how preprocessing survives `save_json`.
- `Dataset::flip_horizontal` mirrors every image row and appends the result,
  doubling a training split.
- `BatchSource`, a trait yielding one batch at a time, and `Dataset::stream`,
  the in-memory implementation of it. `Network::fit_stream` and
  `fit_stream_with` train from any implementation, so a corpus too large to hold
  in memory no longer has to become a `Dataset` first. A batch borrows its
  source, so a streaming reader can hand out slices of a buffer it reuses.
  `fit` and `fit_with` are unchanged and run through the trait.
- `JsonlStream` reads a JSONL file of numeric rows as a `BatchSource`, one
  batch in memory at a time, and `Dataset::from_jsonl` reads the same format
  whole. Both take the names of the input and target fields, each of which may
  hold an array of numbers or a single number. No new dependency.
- `BatchCursor` and `Network::fit_stream_resuming`: a run that stops at batch
  40,000 saves the cursor beside its weights and resumes on the same row of the
  same epoch rather than at the top of it. The per-batch callback also ends a
  run on a step count instead of an epoch count.
- `TokenStream` tokenizes a text or JSONL corpus as it reads it and hands back
  the same `TokenBatch` a `TokenFile` does, so a language model can train
  straight off the corpus with no pre-tokenizing pass and no second copy on
  disk. Tokenization is the caller's closure, as it is for
  `TokenFile::write_jsonl`; windows come out in file order rather than from
  random offsets.

**Training**

- `Network::fit_with` calls a closure with each epoch's index and mean loss and
  stops when it returns `false`: progress reporting and early stopping without
  writing the loop out.
- `Network::accuracy` and `Network::confusion_matrix` score a dataset directly.
- `Optimizer::Lion`, with `Optimizer::lion` and `lion_with_weight_decay`: the
  update is the sign of a momentum-smoothed gradient, so every parameter moves
  by exactly the learning rate and the rate wants to be several times smaller
  than an Adam rate. CPU only; the CUDA and Metal paths report that they have no
  kernel for it rather than taking an Adam step.
- `Optimizer::set_learning_rate` and `learning_rate`, and `Schedule`, a
  warmup-then-cosine learning-rate schedule read as a function of the step.
- `TransformerLm::grad_norm` and `step_clipped`, gradient clipping to a global
  L2 norm. `step_clipped` returns the pre-clip norm, which rises ahead of a
  loss spike.
- `TransformerLm::evaluate`: a forward pass and a loss with no backward pass.
- `TotalLoss::perplexity`.

**Language models**

- `Sampler`: temperature, top-k, top-p and a repetition penalty over one row of
  logits, seeded or not.
- `TransformerLm::generate` and `generate_with` run the whole prefill-and-decode
  loop; `generate_with` hands over each token as it arrives and stops when the
  callback returns `false`.
- `TransformerLm::decoder` returns a `Decoder` that keeps its KV caches between
  turns, so a chat loop does not re-read the conversation every turn.
- `TokenFile`: a pre-tokenized corpus on disk, read as random windows for
  training (`batch`) or consecutive ones for evaluation (`chunk`). Offsets come
  from the step number, so a resumed run sees the data it would have seen.
  `TokenFile::write_jsonl` tokenizes a JSONL corpus into that format without
  holding it in memory.
- `TokenBatch::supervised` builds a batch from prompt-and-response pairs with
  the prompt masked out of the loss. The mask is honoured on the CUDA path too.
- `TransformerLm::quantization_aware` rounds every weight `quantize` would
  round through the int8 grid in the forward pass, while the stored weights and
  the optimizer stay in full precision. Training then sees the arithmetic the
  int8 checkpoint does, so what `quantize` and a `Precision::Q8` save cost in
  accuracy shrinks. Reversible, and CPU only: a model on a device reports that
  it cannot train quantization-aware rather than ignoring the flag.
- `TransformerLm::quantize` rounds every weight matrix to one byte per value
  for inference, which about halves the time a CPU decode step takes. It is
  one-way: training, `save_bin` and `to_cuda` return an error afterwards.

- `TransformerBuilder::bidirectional` and `MultiHeadAttention::set_causal` drop
  the causal mask, so every position reads the whole sequence. This is the
  encoder shape a masked-language model or a vision transformer wants. Such a
  model cannot generate — the KV cache, `generate` and `decoder` report that,
  since an appended token changes the tokens before it — and `to_cuda` refuses
  it, because the flash-attention kernel is causal. `save_onnx` writes it
  without a mask node. A checkpoint written before this loads as the decoder it
  was.

- `MaskedBatch` and `masked_lm_loss`, with `TransformerLm::train_step_masked`
  and `evaluate_masked`: the encoder-only objective. `MaskedBatch::corrupt`
  applies BERT's 80/10/10 corruption at a chosen probability and keeps the
  original tokens as targets, and the loss scores the corrupted positions only.
  It needs a bidirectional model and says so on a causal one, where the
  objective would be the next-token loss with holes in it.

**Vision**

- `VisionTransformer` and `VitConfig`: an image classifier over the existing
  transformer block stack. A patch embedding is one `Linear` over
  `channels * patch * patch` pixels, attention is bidirectional, the pooled
  representation is the mean over patches rather than a class token, and
  positions come from the rotary embedding the language models already use.
  `fit` trains straight from a `Dataset`, and `predict`, `accuracy`,
  `save_json` and `load_json` round it out.

**Image generation**

- `SafeTensors` and `ShardedSafeTensors` read the `.safetensors` format every
  published model ships, including the sharded layout behind a
  `model.safetensors.index.json`. A tensor is read by seeking to its offset, so
  an eight-gigabyte checkpoint is never held in memory whole. `Dtype` converts
  bf16, fp16, both fp8 encodings and the integer types to `f32` on read,
  subnormals included. No new dependency: the header is JSON and the crate
  already reads JSON.
- `Denoiser`, `Scheduler`, `SamplingConfig` and `sample`: the loop every
  diffusion image model runs. `Scheduler::FlowMatch` covers the rectified-flow
  models — FLUX, FLUX.2, Stable Diffusion 3 — with the resolution shift their
  configurations name, and `Scheduler::Ddim` covers the Stable Diffusion 1.x
  and XL line, where the model predicts noise rather than velocity.
  Classifier-free guidance is there for models trained with a dropped
  condition and off for the distilled ones, which would otherwise pay two
  forward passes a step for nothing. The loop reads latents elementwise only,
  so the same code drives a patch-sequence model and a convolutional one.

- `Conv2d`, `GroupNorm`, `FeatureMap`, `upsample_nearest`, `pixel_shuffle` and
  `pixel_unshuffle`: the convolutional layers an image decoder is built from.
  The convolution lowers to the existing matrix multiply through `im2col`, so
  it runs on the same tuned, threaded kernel the rest of the crate does. These
  are inference only — no gradients, no device path.
- `VaeDecoder` and `VaeConfig` turn a diffusion latent into pixels.
  `VaeDecoder::load` reads the `AutoencoderKL` decoder every latent diffusion
  checkpoint ships, straight from `.safetensors` under the `diffusers` key
  names and without a conversion step, sharded or not. `VaeConfig::flux`
  carries the FLUX and FLUX.2 latent shape and scaling. `to_rgb8` converts the
  result to eight-bit rows, and `save_png` writes it out with
  `--features images`.

- `Dit`, `DitConfig` and `Conditioning`: the MMDiT denoiser FLUX and Stable
  Diffusion 3 are built from — double-stream blocks with separate image and
  text weights around one joint attention, single-stream blocks over the
  concatenated sequence, adaptive layer norm driven by the timestep, per-head
  query and key normalization, and two-dimensional rotary positions over the
  patch grid. `Dit::load` reads a checkpoint under the key names Black Forest
  Labs publishes FLUX with. It implements `Denoiser`, so `sample` drives it.
- `Bpe`: byte-level byte-pair encoding read from a `tokenizer.json`, which is
  how a prompt becomes the token ids a text encoder was trained on. The GPT-2
  pre-tokenizer is written out as a state machine rather than a regular
  expression, added tokens are matched whole, and CLIP's end-of-word suffix is
  honoured. No new dependency. T5's sentencepiece unigram is a different
  algorithm and is not covered.
- `TextEncoder` and `TextEncoderConfig` run a LLaMA-family decoder — Qwen2,
  Qwen3, LLaMA, Mistral — as a prompt encoder, read straight from a Hugging
  Face checkpoint and its `config.json`. Grouped-query attention, rotary
  positions in the half-split arrangement those checkpoints use, SwiGLU, and
  Qwen3's per-head query and key normalization, which is picked up from the
  file when it is present.
- `ImagePipeline` and `PipelineConfig` wire the four parts together: prompt to
  ids, ids to hidden states, noise to latent, latent to pixels, with the patch
  packing in between. `generate` runs the lot; `condition`, `sample_latent`
  and `decode` are the same run in three pieces, so a caller short of video
  memory can drop each part as it finishes.
- `Dense`, an inference-only linear layer with a bias, shared by the decoder,
  the denoiser and the text encoder.
- `ImagePipeline::load` reads a published model directory — a `transformer`,
  `vae`, `text_encoder` and `tokenizer` beside each other, each with its own
  `config.json` and its own weights, sharded or not — so hosting a new model is
  a download rather than a code change. `DitConfig::from_file` and
  `VaeConfig::from_file` read those configurations on their own where a caller
  wants one part.
- `Dit` reads both published spellings of the same weights: the reference
  implementation's names and the ones diffusers repacks them under, with the
  separate query, key and value projections read as the one fused projection
  the model runs. Which spelling a file uses is read from the file.
- `examples/text_to_image.rs`: a model directory, a prompt and a PNG.
- `Precision::Q8` reaches the image models: `ImagePipeline::load_at`,
  `Dit::load_at` and `TextEncoder::load_at` hold every projection at one byte
  per weight, quantizing each tensor as it is read so the peak is one `f32`
  tensor rather than the whole model. Measured on a 283M-parameter denoiser
  over 1536 tokens on twelve cores: 1291 MiB peak and 4.0 s per step in `f32`,
  430 MiB and 4.4 s at one byte. The decoder stays in `f32`, being small and
  being what the eye sees.
- `DynamicShift`: the schedule bends with the image's size for the models that
  record `use_dynamic_shifting`, which is how the published flow-matching
  schedulers behave and what a fixed shift gets wrong at anything but one
  resolution. `ImagePipeline::load` reads it from the scheduler's own file.
- `ClipTextEncoder` and `ClipTextConfig` run the CLIP text tower, which is the
  prompt encoder Stable Diffusion and FLUX.1 read their pooled vector from:
  learned positions, LayerNorm, quick-GELU and causal attention, read from a
  Hugging Face checkpoint and its `config.json`, with `Precision::Q8` available
  as everywhere else. The pooled vector is taken at the end-of-text token
  rather than the last position, which is what the padded prompts those models
  feed require, and an optional `text_projection` is applied when the
  checkpoint carries one.
- `ImagePipeline` takes a second encoder for the pooled vector alone
  (`with_pooled`, and the `pooled` field). `ImagePipeline::load` picks it up
  from a published directory that has a `text_encoder_2`: the per-token states
  come from that one, and the CLIP tower in `text_encoder` pools, each with its
  own tokenizer.
- `T5Encoder` and `T5Config` run the T5 encoder FLUX.1 and Stable Diffusion 3
  read their per-token prompt states from: bidirectional attention with the
  learned relative-position bias the first block owns and every block after it
  reuses, root-mean-square norms, and the gated feed-forward of the 1.1 line as
  well as the original's single projection. T5 does not scale its queries, and
  this does not either.
- `Unigram`, the sentencepiece tokenizer T5 reads its prompts with, from the
  same `tokenizer.json`: every piece carries a score, and tokenizing is one
  Viterbi pass for the highest-scoring way to cut the text into pieces. Still
  no new dependency.
- `PromptEncoder`, the encoder an `ImagePipeline` conditions with, holding the
  tokenizer it was trained with: a LLaMA-family decoder over byte-level
  byte-pair encoding, or T5 over sentencepiece. `ImagePipeline::load` reads
  which one a directory holds from that encoder's own configuration, and a T5
  prompt is padded to the length its `tokenizer_config.json` states.
  `ImagePipeline::new` takes one of these in place of the tokenizer and encoder
  pair, and the pipeline's `tokenizer` and `encoder` fields are the one
  `encoder` field now.
- `Unet` and `UnetConfig` run the UNet denoiser, which is what Stable Diffusion
  1.x, 2.x and XL are: a ladder of residual convolution blocks with
  cross-attention to the prompt on the rungs that have it, the noise level
  added to every block, and every rung on the way up handed the matching rung
  from the way down. Almost nothing about the shape is read from the
  configuration — how many blocks there are, which of them attend, and whether
  the projections into attention are linear layers or one-by-one convolutions
  are all read from what the checkpoint holds. The sampler's continuous noise
  level is turned back into the step index the network counts in, by inverting
  the training schedule that `set_schedule` names.
- `ImageDenoiser`, the pipeline's denoiser: a transformer over patches, or a
  UNet over the latent itself. `ImagePipeline::load` reads a directory with a
  `unet` beside one with a `transformer`, packs for the one and not the other,
  and picks the DDIM schedule from `scheduler_config.json` for it.
- Classifier-free guidance runs end to end: `PipelineConfig::guidance` above
  one makes `generate` encode the empty prompt as well and take each step along
  the difference, which is what every Stable Diffusion release was trained for.
  `ImagePipeline::sample_latent_guided` is the same thing for a caller holding
  its own conditionings, and `Dit::set_unconditional` and
  `Unet::set_unconditional` are where the empty prompt goes.
- `Solver` picks how a step is taken: Euler, Euler ancestral, or DPM++ 2M, a
  second-order multistep solver that reuses the previous step's answer and so
  costs no extra forward pass. `SamplingConfig` and `PipelineConfig` carry it,
  and `ImagePipeline::load` reads which one a checkpoint's
  `scheduler_config.json` names.
- `VaeEncoder` reads the other half of the same `AutoencoderKL` file, which is
  what image-to-image needs: a picture becomes the latent the sampler works on.
  `ImagePipeline::load` attaches one whenever the checkpoint holds it, and
  `ImagePipeline::generate_from_image` runs the whole thing — encode the
  picture, noise it as far up the schedule as `strength` asks for, and denoise
  from there. `Scheduler::add_noise` and `sample_from` are the same two pieces
  for a caller driving the loop itself.
- `PromptEncoder::Clip` and `PromptEncoder::ClipPair` read a prompt the way the
  Stable Diffusion line does: one CLIP tower for 1.x, and for XL two towers
  whose states are laid side by side with the pooled vector taken from the
  second alone. `ClipTextEncoder::forward_skipping` hands back the states as
  they were a given number of layers from the end alongside the normalized
  ones, in one pass, because XL reads its prompt from the second-to-last layer
  and its pooled vector from the end of the tower.
- `PromptEncoder::ClipPairAndT5` reads the three-encoder prompt Stable
  Diffusion 3 publishes: two CLIP towers laid side by side and padded out to
  T5's width, the T5 states stacked underneath them in one sequence, and the
  pooled vector taken from both towers together. `ImagePipeline::load` picks it
  when the directory holds a `text_encoder_3`.
- The quantized matmul now forks across tokens when it is given a sequence,
  which is what a diffusion step is. It kept forking across the weight's rows,
  the right axis for the one token of a decode step and the wrong one here:
  5.0 s to 4.4 s per step on the model above.

**Fine-tuning**

- `TransformerLm::add_lora` attaches a low-rank adapter to every attention and
  feed-forward projection and freezes the rest of the model. A frozen
  parameter releases its gradient and both Adam moments, so the base costs one
  weight-sized buffer to train instead of four. `LoraConfig` sets the rank,
  the `alpha / rank` scale and the initialization seed.
- `TransformerLm::merge_lora` folds the adapters back into the weights and
  unfreezes the model; `save_lora` and `load_lora` carry the adapters alone,
  so one base checkpoint serves several of them.
- `TransformerLm::trainable_params_mut` and `trainable_parameters` report what
  a step will actually move, and `has_lora` whether an adapter is attached.
- `Param::freeze`, `unfreeze` and `is_frozen`, underneath all of the above.
- CUDA runs adapted models: `to_cuda` carries the adapters onto the device with
  the projections they adapt, and the frozen base gives up its device gradient
  and moments there too. Every adapted site is covered, including the experts
  and the shared expert of an MoE layer. Attaching and merging stay host
  operations.

**Export**

- `Network::save_onnx` writes a dense network as an ONNX graph — one `Gemm` and
  one activation node per layer. Writing needs no feature flag; reading still
  needs `--features onnx`.
- `TransformerLm::save_onnx` writes a language model as an ONNX graph at one
  fixed sequence length: ids in as `int64`, logits out, with RMSNorm, rotary
  positions, grouped-query attention, the causal mask and SwiGLU written as
  opset-13 nodes. Verified against the forward pass by running the file back
  through `tract`. Mixture-of-experts layers, an unmerged LoRA adapter and a
  quantized or device-resident model are refused rather than exported as
  something else, and the graph carries no KV cache.

**Image generation on CUDA**

- `ImagePipeline::to_cuda` and `try_cuda` (`--features cuda`) move the UNet, the
  VAE decoder and the CLIP towers onto a device, where they stay for the whole
  run. Weights are uploaded once as bf16 — `Precision::Q8` is widened on the way
  up — and every accumulation is still fp32. A Stable Diffusion XL UNet pass at
  512x512 goes from 7.4 s to 0.11 s and at 1024x1024 costs 0.42 s, and the VAE
  decode of a 1024x1024 image costs 0.70 s, in 6.2 GB of device memory. A whole
  1024x1024 image at 30 steps with guidance, which is sixty denoiser passes and
  a decode, takes 26.6 s.
  `Unet::attach_device`, `VaeDecoder::attach_device` and
  `ClipTextEncoder::attach_device` do one part at a time.
- The CPU path is unchanged and is still the default. A build without the
  feature, a machine with no device, and a model that does not fit all behave
  exactly as they did; `try_cuda` answers `false` rather than failing.
- A transformer denoiser and a T5 encoder have no device path yet and stay on
  the host, so a FLUX or Stable Diffusion 3 pipeline is unaffected.
- Reading a checkpoint and narrowing it for a device both run on every core.
  Widening a stored dtype to `f32` is the whole cost of loading a model, and
  one core doing it took longer than the disk took to hand the bytes over: a
  Stable Diffusion XL checkpoint loads in 6.6 s rather than 13.0 s.

**Examples**

- `examples/image_classification.rs` (`--features images`): an image folder into
  a dense network, with a stratified split, mirrored training data and a
  confusion matrix.
- `examples/cuda_image_bench.rs` (`--features cuda`): times one UNet pass on the
  host and on the device at each of several resolutions, compares the two
  outputs, and reports what the device path held. `REPEAT` times several passes
  and reports the fastest, which is steadier than one pass on a card whose
  clock sags as it heats.
- `examples/text_to_image.rs` takes `CUDA=<device index>`, and `STEPS` and
  `GUIDANCE` to override what the model asked for, which is what lets a run be
  lined up against another implementation's.

### Documentation

- Tutorial chapters 4, 7, 8, 9, 12, 14 and 16 cover all of the above, and
  chapters 14 and 16's programs were rewritten onto `Sampler` and `Schedule`.
- `IMPORT_MODELS.md` documents the export direction.

## 2.0.0

The first release aimed at production use. The breaking changes are removals of
API that had no implementation behind it.

### Breaking

- **Removed `tensor::Tensor`.** The trait had one implementor, `Matrix`, and
  was never used as a bound anywhere in the crate or its examples. Call the
  inherent methods on `Matrix` instead; every signature is unchanged.
- **Removed the `layers` module.** It re-exported `Dense` and `DenseLayer` from
  `network` and nothing else. Import them from `rusting_brain` directly.
- **Removed the `rusting_brain` binary.** `src/main.rs` duplicated
  `examples/xor.rs`. Run `cargo run --example xor` instead.
- **`gpu_matrix` is now private.** `GpuMatrix` and `gpu_dot` existed only for
  the matmul benchmark in `gpu_test`; the module documentation claimed the
  dense CUDA path used them, which was never true. The unused `to_cpu` and the
  unused `context` field are gone, and the remaining calls return `Result`
  instead of unwrapping a driver failure, which the rest of the device code
  has never done.

### Added

- Crate-level documentation with worked examples for training, KV-cached
  generation, and moving a model to a device.
- Module-level documentation on every source file.
- Tutorial chapters 11–16: CUDA training, importing ONNX models, a
  troubleshooting reference, transformer language models, mixture of experts,
  and an end-to-end training run. Chapters 14–16 are new material; the course
  previously stopped at dense networks.
- `tutorials/README.md`, an index over all sixteen chapters.
- `examples/language_model.rs`: trains a character-level transformer language
  model and generates from the checkpoint in a separate process. `--moe`
  switches the same model to sparse layers. This was the one model type the
  examples did not cover.
- An MSRV job in CI pinned to 1.85, so the `rust-version` in `Cargo.toml` is
  checked rather than asserted.
- `gemm_probe` and `flash_probe` declare `required-features = ["cuda"]`, so
  `cargo check --all-targets` no longer fails without the feature.

### Fixed

- Pinned `kstring` to 2.0.2. 2.0.4 requires rustc 1.96.0, which broke
  `--features onnx` against the declared MSRV of 1.85.
- Clippy is clean under `--all-targets` with no features, with `cuda`, and with
  `onnx`.

### Documentation

- `README.md` rewritten around measured throughput against PyTorch and
  TensorFlow on the same model and hardware, rather than unqualified claims.
- Planning documents moved under `docs/`.
- `Network::forward` documents that it panics on a wrong-length input, next to
  the `predict` that returns an error instead.
- `.cargo/config.toml` says what `target-cpu=native` does to a binary that is
  copied to another machine.
