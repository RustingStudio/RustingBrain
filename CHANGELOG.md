# Changelog

## Unreleased

- FLUX.2-klein's transformer can store its block linears as FP8 E4M3, with one
  scale per row: `Precision::Fp8` (opt-in; Q8 stays the default). This adds a
  variant to `Precision`, so an exhaustive `match` on it outside the crate
  breaks, and RustingFrame needs its own "fp8" value to reach it. Only the
  FLUX-family `Dit` uses it. Modulations stay int8 per row, embedders and
  proj_out stay F32, other models load it as Q8 or F32, and `save_bin`
  refuses it. On the device a `dequantize_fp8` kernel widens the bytes to
  BF16 for the existing GEMM (sm_86; the native FP8 GEMM on sm_89+ is
  UNVERIFIED and not used). Parity is worse than Q8 and misses the image gate:
  lighthouse seed 0 on the device, step-0 3.10e-2 against 8.73e-3, whole-chain
  image 6.93e-2 against 3.03e-2 (gate 0.034). The host gives the same numbers.
  3060, 1024x1024, 4 steps, 3 interleaved runs, load 1.7-5.8: denoise 6.77 s
  against 6.82 s, upload 7.0 s against 12.5 s, process peak 9186 against
  10626 MiB.
- Wan 2.2 TI2V-5B image-to-video at 1280x704x121 now fits on a 12 GB card.
  Above 12288 tokens the device DiT runs the QKV projection and the
  feed-forward 4096 tokens at a time, so their FP32 `[tokens, 3 * dim]` and
  `[tokens, 14336]` buffers no longer scale with the clip (121 frames failed
  allocating 746 MiB with 94 MiB free). 1280x704x49 and below run the same
  code as before and print the same parity numbers. 3060, UniPC 20 steps,
  CFG 5, release test build without LTO: 121 frames sample in 1146.7 s, peak
  10748 MiB (desktop included); 81 frames peak 10994 to 10164 MiB with the
  same decoded frames.
- Wan 2.2 TI2V-5B image-to-video at 1280x704x49 now fits on a 12 GB card.
  The device DiT's per-block temporaries were shadowed, not dropped, so they
  all sat under the feed-forward's two FP32 `[tokens, 14336]` buffers (626 MiB
  each at 49 frames; one of them was the failed allocation). They now drop when
  their stage ends, the attention scratch is allocated once per call, and the
  feed-forward's bias and GELU run in one kernel. Activation peak at 49 frames
  goes from about 2.7 GB to 1.1 GB. Output is unchanged: the parity tests print
  the same numbers. 3060, 1280x704x33, UniPC 20 steps, CFG 5, release test
  build without LTO, 2 interleaved runs each, load 1.2-2.4: whole-process peak
  (desktop included) 11061/10877 to 9885/9723 MiB, sampling 227.1/226.5 s to
  221.0/215.1 s. 1280x704x49 went from out of memory before step 0 to 335.7 s
  of sampling plus a working decode, peak 10086 MiB.
- Wan 2.2 TI2V-5B image-to-video on the UniPC + CFG path can skip steps:
  `VideoConfig::skip = Some(StepSkip { full: 4, every: 2 })` runs the first 4
  steps in full, then skips both DiT passes on every second step but the last
  and gives UniPC the previous guided velocity; default `None`, bit-identical.
  1280x704x17, 20 steps, seed 7, guidance 5, 3060: 40 to 26 DiT calls, 111.6 s
  (3 runs, spread 0.2 s) to 74.7 s (3 runs, spread 4.6 s), 33 % less; latent
  relative L2 to the 20-step run 0.224, against 0.426 for 13 plain steps (26
  calls, 73.3 s). At 49 frames 329.5 s to 215.8 s, 0.176. Test
  `wan22_image_to_video_step_skip_sweep` (ignored, `RB_WAN_CUDA=1`,
  `RB_WS3_IMAGE`).
- Q8 denoiser projections carry one int8 scale per 64 columns instead of one
  per row (`RB_Q8_GROUP`, 0 restores the old layout; the modulations stay per
  row), and the input/time embeddings and output projection stay F32. FLUX.2-klein
  step-0 against diffusers: 0.0127 to 0.0068 host, 0.0141 to 0.0087 device
  (0.0226 to 0.0139 with a reference image); 1024x1024 denoise 6.24 s to 6.37 s,
  peak VRAM 10434 to 10626 MiB.
- FLUX.2-klein reference-image conditioning (image editing):
  `ImagePipeline::set_references(&[FeatureMap])` and
  `generate_with_references(prompt, width, height, seed, &references, on_step)`
  encode each reference with the pipeline's VAE encoder and append its tokens
  (ids `10 + 10 i` on the time axis) to the image tokens, host and device;
  without references nothing changes. 512x512 edit, 4 steps, Q8 on the 3060:
  3.92 s warm; image 0.0112 against diffusers. Example `image_edit`.
- Wan VAE device decode gathers each frame's convolution columns once for all
  three temporal taps (the layers keep partial sums of the next outputs
  instead of the last two input frames) and skips the zero-bias epilogue on
  the older taps: 1280x704x17 decodes 7.4 % faster (11.04 s to 10.22 s in a
  debug test build) with a lower peak (5.4 against 5.9 GB). Relative L2 to
  diffusers 1.13e-3 (was 1.29e-3), to the previous decode 1.6e-3.
- Wan 2.2 TI2V-5B gets FastWan's 3-step DMD sampler (`VideoConfig::dmd =
  Some(Dmd { timesteps: vec![1000, 757, 522], seed })`, `dmd_sigmas`; no
  guidance, one DiT call a step) and `VideoPipeline::set_dit_weights` for a
  single-file DiT. 1280x704x49 samples in about 21 s, against a 450-500 s
  30-step CFG run; default sampling unchanged.
- The Wan 2.2 device VAE decoder takes the blocks after the last one that adds
  frames one frame at a time instead of four, output bit-identical: the
  704x1280x17 decode peaks at 5856 MiB of device memory instead of 6784 MiB, in
  the same 9.5 s. `make_room` uses the new fit (650 MiB + 5.0e-3 MiB a pixel);
  the DiT still goes at 12 GB (3.5 GB free beside it against 5.7 GB needed)
  and is estimated to stay on a 16 GB card (not measured).
- Wan 2.2's video pipeline: `video_pipeline::VideoPipeline` turns a prompt,
  and optionally a first frame, into RGB video with umT5, the Wan DiT under
  classifier-free guidance and the Wan VAE, sampled with `UniPc`, diffusers'
  UniPC scheduler as Wan configures it (exact schedule, a bit-exact replay of
  the reference trajectory). Requests are checked (frames 4k+1, sides a
  multiple of 32). On an RTX 3060: a 256x256x5 clip in 6 steps takes 2.2 s,
  a 704x1280x17 step 4.5-5.4 s; umT5 stays on the host and the DiT is
  dropped before a decode that does not fit beside it (`docs/baseline.md`,
  W4).
- Wan 2.2's video transformer: `wan_dit::WanDit` runs one denoiser call of
  the TI2V-5B DiT (a timestep per frame, so the first frame of an
  image-conditioned clip can sit at t = 0), matching diffusers'
  `WanTransformer3DModel` in F32 (output relative L2 2.3e-5) and 3.3e-2 at
  Q8. `WanDit::attach_device` runs it on the GPU (int8 weights, BF16
  operands, FP32 residual): 4.4e-2 against diffusers, 704x1280x17 in
  2.3-2.6 s a step and about 7.7 GB on an RTX 3060 (`docs/baseline.md`, W3).
  A malformed checkpoint is refused with `InvalidConfig` before any upload.
- Wan 2.2's video autoencoder: `wan_vae::WanVaeDecoder` decodes a latent
  (one map per latent frame) into RGB frames and `WanVaeEncoder` encodes a
  `1 + 4k`-frame clip, both matching diffusers' `AutoencoderKLWan` in F32
  (relative L2 under 1e-6). `WanVaeDecoder::attach_device` decodes on the GPU
  in FP16 (1.3e-3): 704x1280x17 in 9.5 s and 7.8 GB on an RTX 3060, untiled
  (`docs/baseline.md`, W2).
- umT5, Wan 2.2's prompt encoder, runs through `T5Encoder`: a checkpoint with
  a position table in every block is read that way (T5's single shared table
  is unchanged), on the host and the device. `T5Encoder::encode_real` gives
  the states Wan hands its denoiser: the real tokens only, zero rows after
  them. umT5-XXL at Q8 against transformers: token ids exact, prompt
  embeddings 0.0203 / 0.0245 on the host, 0.0204 / 0.0245 on an RTX 3060
  (95 / 194 ms for 9 / 39 tokens; `docs/baseline.md`, W1).
- `ImagePipeline::encoder_on_device` says whether `to_cuda` (or `try_cuda`)
  moved a sequence prompt encoder such as FLUX.2-klein's Qwen3. That upload
  may fail when VRAM is short and the encoder then stays on the host (prompt
  encode about 12 s instead of 0.7 s on an RTX 3060); before, nothing said so.
- FLUX.2-klein-4B denoises faster on the device: 1024x1024 Q8 on an RTX 3060
  9.96 s to 6.61 s for 4 steps (2.49 to 1.65 s a step), parity against
  diffusers unchanged (`docs/baseline.md`, P2). The per-head norm and
  rotation runs one warp per head, flash attention streams K and V with
  `cp.async` (also faster for the LM shapes: 0.557 to 0.398 ms at 4x1024,
  head 64), and the MLP activation writes BF16 straight from the fused GEMM.
  `examples/flash_probe` now compares kernel source files.
- `vae::png_bytes` (`images` feature) encodes a decoded image to PNG in
  memory, the same bytes `save_png` writes.
- `ImagePipeline::to_cuda` (and `try_cuda`) moves a sequence prompt encoder
  such as FLUX.2-klein's Qwen3 to the device as well: 1024x1024 Q8 prompt
  encode 12.4 s to 0.7 s on an RTX 3060, embeddings 0.0249 against diffusers
  (host Q8 0.0251), peak VRAM 6754 to 10402 MiB (`docs/baseline.md`, P1).
  New: `TextEncoder::attach_device`, `on_device`, `detach_device` and the
  `cuda_text_encoder` module.
- FLUX.2-klein loads and runs through `ImagePipeline`, checked stage by stage
  against diffusers (`docs/baseline.md`, K14). New public items:
  `ChatPrompt`, `ImagePipeline::sample_latent_from_noise`, `TextEncoder::forward_layers`,
  `VaeDecoder::patch_norm`. A diffusers-layout DiT loaded at Q8 no longer
  panics. Breaking for code that names these: `DynamicShift` gains
  `empirical` and `DynamicShift::shift` takes the step count,
  `PromptEncoder::Sequence` gains `chat` (`None` keeps the old behaviour), and
  `VaeConfig` gains `batch_norm_eps`.

## 4.0.0

A major version because 3.0.1 broke the 3.0.0 API in a patch release; 3.0.1 is
yanked and its changes are released here. Code written for 3.0.0 needs these
changes:

- `AgentConfig::causal` is a new field (`None` keeps the old MLP trunk). Build
  from `AgentConfig::new` and set fields, or name the field in a literal.
- `ClipTextConfig::argmax_pool` is a new field (`ClipTextConfig::default()`
  sets `true`). Name it, or use `..Default::default()`.
- `QNetwork::trunk` is now a `Trunk` (`Trunk::Mlp` holds the old
  `MarketTrunk`), and `ForwardCache::trunk` is a `TrunkState`
  (`TrunkState::Mlp` holds the old `TrunkCache`). Read the trunk's output
  through `TrunkState::h()`.
- The minimum Rust version is 1.88, up from 1.85. The `cuda` feature already
  needed 1.88 through `cudarc`'s `libloading` dependency.

### Changed

- `Dataset::shuffle` moves rows instead of cloning them: 200,000 rows of 64 features shuffle in about 5 ms instead of 31 ms, with no second copy of the data held (R4, RustingTrade P08). Same order for the same seed.
- `Param::step` (and so every `TransformerLm`, flow and image model step, on CPU and CUDA) no longer applies weight decay to one-row parameters: biases and norm gains keep their value, and matrices decay as before (R2, asked by RustingLLM). `Network` already skipped its biases. A rank-1 LoRA `down` matrix is one row too and is no longer decayed.
- `marching_tetrahedra` and `marching_tetrahedra_sparse` now wind every face outward, from the field's inside (below the isovalue) to its outside (K11). Before, the winding followed the tetrahedron table and could point either way.
- `TrainableConv2d::to_cuda_on` moves the bias to the device as well; the host-input `forward`/`backward` now run through the device-resident pair, so the bias add and its gradient no longer happen on the host. `bias.value`/`bias.grad` are stale until `to_cpu`, as for the weight.
- `Param::step` (and so `apply_step` and `step_clipped`) panics on step 0 instead of turning every parameter NaN through Adam's bias correction; steps count from 1 (asked by RustingChop). An assert rather than an error, as the consumer allowed: the signatures stay the same.
- A device decoder under mixed precision keeps its cache as BF16 keys and
  values, 512 bytes a position a layer on the default model instead of
  3072, and later turns read it in place through the flash kernel. 16
  sequences after a 2000-token prompt decode 7040 to 7360 tokens/s, and
  over 8 steps 2.89 to 2.30 ms a step; one sequence is unchanged.
- A device decoder's first prompt runs the fused flash attention kernel
  under mixed precision at head widths 64 and 128, instead of a full score
  matrix a head: 16 prompts of 2000 tokens prefill in 149 ms instead of
  364 ms, and with every other prompt half as long in 125 ms instead of
  438 ms. `bench_decode` prints the prefill time.
- A device decoder's later turns of 128 tokens or more take the same flash
  kernel: 16 sequences feeding 512 tokens after 2000 run 140 to 60 ms.
  `bench_decode` takes a sixth argument, the length of a timed second turn.
- Device decode attention runs one block per key-value head, scoring every
  query head of its group against each cached key and value it loads,
  instead of one block per query head. On the default model (8 heads over
  2 key-value heads) 16 sequences after a 2000-token prompt decode 5016 to
  6906 tokens/s, and one int8 sequence after 4000 tokens 1205 to 1430.
- A decoder's device KV cache grows with the session, doubling up to
  `max_seq_len`, instead of taking `max_seq_len` positions per sequence when
  it is made. Forks copy only what the session holds room for.
- Stride-1 convolutions on the image path gather only their horizontal taps
  and run one accumulating GEMM per kernel row, instead of an im2col nine
  times the input: the SDXL VAE decode at 1024x1024 runs 0.52 to 0.49 s.
- Device cross-attention (the shape VAE and the flow transformer) runs in
  BF16 through the fused flash kernels under mixed precision with heads 64 or
  128 wide, instead of FP32 around a score matrix. The shape VAE's training
  step runs 0.084-0.087 to 0.062-0.065 s (-25%).
- The device memory pool keeps the memory a training step frees instead of
  returning it to the driver at every synchronization. The shape VAE's
  training step runs 0.130 to 0.085 s (-35%); the language-model step is
  unchanged.
- The fused attention query-gradient kernel at head width 128 uses 34 KiB
  of shared memory instead of 52 KiB, so two blocks fit per SM: a training
  step with heads of 128 runs about 2.6% faster.
- `GpuContext` runs on the CUDA default stream, so cudarc no longer records
  and waits on an event for every kernel argument. Batch-1 int8 decode gains
  about 2% (1676-1712 to 1711-1747 tokens/s); training is unchanged.
- Device decode attention issues its cache loads in parallel: 16 sequences
  after a 512-token prompt decode 7780-7980 to 8731-8835 tokens/s.
- Int8 device decode runs the routed experts with an int8 routed GEMV instead
  of downloading the routing each MoE layer: one sequence of the default model
  decodes 1014-1049 to 1535-1613 tokens/s.
- `ImagePipeline::to_cuda` also moves a transformer denoiser and a gated T5
  encoder, so FLUX and Stable Diffusion 3 pipelines run on the device. Both
  use the BF16 kernels; the original ungated T5 stays on the host.
- Device decode of up to 16 sequences routes its mixture-of-experts layers
  on the device instead of copying the routing back to the host each layer:
  one sequence of the default model decodes about 1.5x faster (about 900 to
  1360 tokens per second), four 1.4x, sixteen 1.14x.
- The fused attention backward pass's key/value kernel reads its query and
  output-gradient tiles transposed with `ldmatrix` instead of keeping a
  second copy, which halves its shared memory and fits three blocks on an
  SM: 25% less time in that kernel, about 1% more training tokens per
  second on the default model.
- Int8 device decode of three or more sequences multiplies on the tensor
  cores (Ampere and later, mixed precision): 16 sequences on the default
  model went from 13% slower than BF16 to about 12% faster.
- Int8 device decode (`TransformerLm::set_int8_decode`) now covers the
  routed MoE experts and the router as well, matching the host
  `Precision::Q8` scheme: about 19% more tokens per second than BF16 for one
  sequence on the default model, up from 10%.
- Under mixed precision the device decoder's attention output projection
  reads BF16, about 3% more tokens per second for one sequence.
- A device decode step with one new token per sequence runs attention as
  two fused kernels per layer instead of seventeen cuBLAS and softmax
  launches: about 30% more tokens per second on the default model, for one
  sequence or sixteen.
- Under mixed precision the device decoder reads a BF16 copy of the LM head,
  about 6% more tokens per second for one sequence on the default model.
- Fused flash attention also serves a head width of 128, the width most
  LLaMA-family checkpoints use, in training and in the forward pass. It used
  to fall back to the three-kernel path; a 4 x 1024 training step at that
  width runs 23% faster. `bench_train_step` takes the head width as an
  optional fourth argument.
- A `Decoder`, and so `generate`, on a model moved to the GPU keeps its
  key/value cache on the device and runs every layer there, instead of
  copying activations to the host for each projection. Cached decode of the
  default model runs about three times as fast (197 to 624 tokens/s).
- `bench_decode` takes `cuda` as an optional third argument to time cached
  decode of the default model on the GPU.
- Mixed-precision training runs the routed experts of a MoE layer in BF16, as
  the dense feed-forward already did. The router, the expert outputs and the
  scatter back into the residual stay FP32. About 4% more tokens per second
  on the default model.
- Under mixed precision the router and the shared expert of a MoE layer also
  run in BF16, with FP32 router logits. About 2.5% more tokens per second on
  the default model.

### Added

- `Bpe::train_json_split_digits`: `train_json` with Qwen2's split, so every digit is its own word and no token spans two digits (R3, for RustingLLM). The file carries the rule as a `Split` pre-tokenizer, which `Bpe::from_json` already reads.
- `Schedule::warmup_stable_decay(peak, warmup, steps, decay_fraction)`: warmup, the peak held, then a linear decay to the floor over the last fraction of the run (R1, for RustingLLM). Rates before the decay do not depend on `steps`, so a run can be extended.
- YOLO training throughput (K13g): the depthwise weight gradient reads channels coalesced, 1x1 stride-1 convolutions skip `im2col` and `col2im`, `detection_loss` runs each image on its own rayon task, several device buffers that a kernel overwrites whole skip their memset, and `examples/yolo_detect` trains in mixed precision (BF16 GEMMs, FP32 accumulate). SiLU runs inside the device batch norm kernels, and on a mixed-precision context a training convolution keeps bf16 columns and runs its three GEMMs and its `col2im` on bf16, four channels a thread. The batch norm's column sums come from one coalesced kernel instead of GEMVs, and its backward pass reads the incoming gradient through the SiLU instead of writing it out first. YOLO's convolutions no longer copy their input into their caches. The host detection loss reads anchors from the output maps in place, decodes them in parallel, and writes its gradients straight into the map buffers, and `TaskAlignedAssigner` picks each box's top candidates with a select instead of a full sort, with the boxes in parallel. `yolo_detect --warmup N` times the steady state. The device batch norm keeps its input for the backward pass instead of writing its normalized values out. YOLO's device backward pass accumulates gradients in place, a slice's straight into its channels. `yolo_detect --batch 4` at 1280 goes from 5.3 to 12.8 samples/s inside the warmup and runs at about 15.75 samples/s in the steady state.
- `interoceptive::CausalTrunk`: a causal transformer over the observation window, one token per candle, chosen with `AgentConfig::causal: Some(CausalConfig { lookback, d_model, layers, heads })` (G8). `QNetwork::trunk` is now the `Trunk` enum and `AgentConfig` has a new `causal` field; code that builds `AgentConfig` with a struct literal and no `..AgentConfig::new(..)` needs `causal: None`.
- `train_unet_lora` example: trains a LoRA on the attention projections of a Stable Diffusion 1.x UNet on the CUDA device and writes a diffusers PEFT file. New `UnetLora::new`, `UnetLora::write_peft` and `DeviceUnet::noise_step`. diffusers' `load_lora_adapter` loads the written file to within 8.5e-7 of `UnetLora::read_peft` (G10 c).
- An ignored timing test, `cuda_image::lora::tests::an_sd15_step_at_512_is_timed`: one adapted forward and backward of Stable Diffusion 1.5's UNet at 512x512 on the device takes 191.0 ms on an RTX 3060 (65.1 forward, 125.9 backward; random weights of the 1.5 shapes), recorded in `docs/baseline.md` (G10 b-iii-c).
- YOLO11 inference at 1280 under 15 ms, Gate 3 of the detection track (K13f). `Yolo11::fuse()` folds every batch norm into its convolution as ultralytics' `fuse()` does. On a `CudaDevice::with_mixed_precision` device, the fused 3x3 and strided convolutions build their `im2col` columns in bf16 and run bf16 GEMMs with an FP32 accumulator. `TrainableConv2d::infer_device(input, silu)` is a cache-free forward that adds the bias and SiLU in one kernel and reads a 1x1 convolution's input as its columns. `Yolo11::save_ultralytics(path)` writes an ultralytics state dict that `load_ultralytics` and ultralytics read back. `Validation::stages_ms` gives the time of each stage. On the 278 player val images (rect letterboxed, e.g. 1280x736), batch 1 on the 3060: upload 0.56 ms, forward 13.56 ms and download 0.47 ms, 14.6 ms for the model (37.7 ms unfused before), with mAP50 0.7390 (fp32: 0.7394). Decode on the host takes another 3.34 ms and NMS 0.02 ms. Other speedups: im2col reads and writes float4 with 32-bit indices, bias plus activation runs on float4, outputs written in full are left uninitialised, and a `Slice` node copies only its own channels.

  The API a consumer crate uses, with features `cuda,images`:

  ```rust
  use rusting_brain::detection::{DetectionImage, non_max_suppression, read_yolo_split};
  use rusting_brain::param::CudaDevice;
  use rusting_brain::safetensors::SafeTensors;
  use rusting_brain::yolo::{Yolo11, Yolo11Scale, decode, detection_loss, input_batch, validate};

  // Load: weights exported from an ultralytics .pt as safetensors.
  let mut model = Yolo11::new(Yolo11Scale::S, classes, &mut rng)?;
  model.load_ultralytics(&mut SafeTensors::open("best.safetensors")?)?;
  // or, to fine-tune from a checkpoint with other classes:
  let skipped = model.transfer_ultralytics(&mut SafeTensors::open("yolo11s.safetensors")?)?;

  // Train: see examples/yolo_detect.rs for ultralytics' full recipe.
  let device = CudaDevice::new(0, 11000)?;
  model.set_recompute_columns(true); // batch 4 at 1280 fits a 12 GB card
  model.to_cuda_on(&device)?;
  let images: Vec<DetectionImage> = /* DetectionImage::open(path, labels, 1280), augmented */;
  let tape = model.forward_device(&device.upload(&input_batch(&refs))?, true)?;
  let maps: Vec<_> = model.outputs_device(&tape).map(|m| m.download()).into_iter().collect::<Result<_, _>>()?;
  let (losses, grads) = detection_loss([&maps[0], &maps[1], &maps[2]], classes, &targets);
  let grads: Vec<_> = grads.iter().map(|g| device.upload(g)).collect::<Result<_, _>>()?;
  model.backward_device(tape, [&grads[0], &grads[1], &grads[2]])?;
  // then rusting_brain::optimizers::apply_step over model.params_mut(), and an
  // rusting_brain::ema::Ema over model.state_mut()
  model.to_cpu()?;
  model.save_ultralytics("trained.safetensors")?;

  // Run: fuse on the host, then infer one letterboxed image at a time.
  let device = CudaDevice::with_mixed_precision(0, 8192)?;
  let mut model = model.fuse();
  model.to_cuda_on(&device)?;
  let (image, _, _) = DetectionImage::open(path, &[], 1280)?.letterbox(736, 1280, false);
  let tape = model.forward_device(&device.upload(&input_batch(&[&image]))?, false)?;
  let maps: Vec<_> = model.outputs_device(&tape).map(|m| m.download()).into_iter().collect::<Result<_, _>>()?;
  let (boxes, scores) = decode([&maps[0], &maps[1], &maps[2]], classes).remove(0);
  let detections = non_max_suppression(&boxes, &scores, classes, 0.25, 0.7, 300);
  // or validate(&mut model, &device, &val_images, 1280) for mAP50 and mAP50-95
  ```
- YOLO11 fine-tuning, Gate 2 of the detection track (K13e): `examples/yolo_detect.rs` reproduces ultralytics' training recipe and reaches val mAP50 0.7336 (mAP50-95 0.5467) on the player dataset in 25 epochs at batch 4 (1.57 h, 4.9 samples/s). New: `DetectionImage::affine` and `hsv` augmentations, `Yolo11::transfer_ultralytics` (loads a checkpoint with other classes, as `intersect_dicts`), `Yolo11::set_recompute_columns`, the Detect head's bias initialisation, `yolo::input_batch`, and `yolo::validate` returning `Validation`; `examples/yolo_val.rs` now calls `validate`.
- GRU cell with backward through time (K9): `recurrent::Gru::new(input, hidden, rng)`, `forward_seq(x, batch, h0) -> (Matrix, GruCache)`, `backward_seq(&cache, grad_out, grad_h_last) -> (dx, dh0)`, `step` for a frame at a time, `params_mut`, `to_cuda_on` for the gate GEMMs. PyTorch's equations; every gradient within 1.7e-4 of finite differences over 16 steps.
- `examples/yolo_val.rs` (K13d, Gate 1): validates YOLO11 weights on a YOLO dataset with ultralytics' rect batches, NMS and mAP. The trained player model scores mAP50 0.7394 and mAP50-95 0.5803 on its 278 val images, against ultralytics' 0.739 and 0.576.
- YOLO11 graph (K13a-iii): `yolo::Yolo11` builds any YOLO11 scale, loads an ultralytics state dict (`load_ultralytics`), and runs forward and backward on the host or the device (`forward_device`, `backward_device`). `yolo::decode` turns the raw maps into boxes and scores for `detection::non_max_suppression`; `yolo::detection_loss` is ultralytics' `v8DetectionLoss` with its gradient. Matches ultralytics 8.4.174 within 1e-4.
- DINOv2 towers in `vit_encoder` (K12): `VitEncoder::load_dinov2(path, image_size, precision)` loads `facebook/dinov2-small` and `-base` safetensors; patch tokens are rows 1.. of `encode`, normalised with `IMAGENET_MEAN`/`IMAGENET_STD`. Within 4.2e-5 relative of HF `transformers` on two images. `ffn::gelu_erf` is the exact GELU, also on the device.
- `DepthwiseConv2d` (no bias, padding k/2) and `pixel_attention`, the middle of ultralytics' `Attention` from its qkv map to the attended map and values, each with backward on host and device. The attention takes the qkv map rather than owning its convolutions, so C2PSA composes it from `TrainableConv2d`, `TrainableBatchNorm2d` and `DepthwiseConv2d` (K13a-ii).
- Detection layers (K13a-i) in `conv`: `TrainableBatchNorm2d` with training and evaluating modes, PyTorch's running statistics and `fuse_into` a convolution; `upsample_nearest_batch`, `concat_channels`, `split_channels`, `max_pool_same` and their backwards; each with a device version, plus `add_device`.
- Detection ops (K13b) in `detection`: `box_iou` with `Overlap::{Iou, Giou, Ciou}` and its gradient, `distribution_focal_loss`, `make_anchors`, `box_to_ltrb`/`ltrb_to_box`, `TaskAlignedAssigner`, `non_max_suppression` and `mean_average_precision`, all matching ultralytics 8.4.174 on exported fixtures. Host only; CUDA NMS waits for the Gate 3 profile.
- `detection` module (K13c): `read_yolo_split` and `parse_yolo_labels` for YOLO-format datasets, `rect_shape` for ultralytics' validation batch shape, and `DetectionImage` with `open` (decode and resize the long side), `resized` (OpenCV-style bilinear), `letterbox` and `flip_horizontal`, plus `mosaic` for the four-image mosaic. Boxes match ultralytics' pipeline on the CS2 player dataset.
- `examples/chop_block_step.rs` times RustingChop's seven convolution blocks on the per-layer and the device-resident paths (K8d). At batch 128 on the RTX 3060 a forward and backward step drops from 1002 ms to 99 ms.
- Device spatial softmax and global average pooling (K8c): `conv::spatial_softmax_device` and `spatial_softmax_backward_device` with `DeviceSpatialSoftmaxCache`, `conv::global_avg_pool_device` and `global_avg_pool_backward_device`. Outputs are `batch x 1 x 1 x C` device batches (2C for the softmax keypoints) and match the host functions without leaving the card.
- `Glb::base_color_image(mesh, primitive) -> Option<(&[u8], &str)>`, `Glb::json()` and `Glb::binary()` (asked by RustingMedia). The request left the primitive's type open; it is the pair of indices into `meshes` and that mesh's `primitives`. Images held by URI return `None`.
- Device group norm and SiLU for `DeviceImageBatch`: `TrainableGroupNorm::to_cuda_on`/`to_cpu`/`forward_device`/`backward_device` (with `DeviceGroupNormCache`), `conv::silu_device` and `conv::silu_backward_device` (K8b). The request named the layer `GroupNormTrainable`; it is `TrainableGroupNorm`, with the device pass as methods.
- `conv::DeviceImageBatch` with `CudaDevice::upload`/`download`, and `TrainableConv2d::forward_device`/`backward_device`: a convolution chain whose activations and gradients stay on the device (K8a, asked by RustingChop and RustingCS2). The wanted `ConvCache` return is kept; the conv bias and its gradient now run on the device too.
- `conv::TrainableGroupNorm`: group norm over an `ImageBatch` with `gamma`/`beta` parameters, `forward -> (ImageBatch, GroupNormCache)`, `backward` and `params_mut` (asked by RustingChop and RustingCS2). Its forward is bit-equal to `conv::GroupNorm`. `new` and `forward` return a `Result` (bad group counts and channel mismatches), and `eps` is an argument where the Chop stopgap fixed Keras' 1e-3.
- `param::Dropout`: inverted dropout with `forward(&input, &mut rng, training) -> (Matrix, DropoutMask)` and `backward(&mask, &grad)` (asked by RustingChop and RustingCS2). `Dropout::new` returns a `Result`, refusing rates outside [0, 1), where the request had a plain constructor.
- `conv::spatial_softmax` and `conv::global_avg_pool` over an `ImageBatch`, each with its backward (`spatial_softmax_backward`, `global_avg_pool_backward`) (asked by RustingChop and RustingCS2).
- `TrainableConv2d::without_bias()`: a convolution that adds, gathers and trains no bias, for one before a norm (asked by RustingChop). A builder rather than `bias: Option<Param>`, so the public `bias` field keeps its type.
- `losses::bce_with_logits`, `losses::soft_cross_entropy` and `losses::huber` take logits (or raw predictions) and optional per-sample weights, reduce as `sum(w * loss) / batch`, and return the gradient with the loss (asked by RustingChop and RustingCS2).
- `ffn::silu_derivative` is public, for hand-written backward passes through SiLU (asked by RustingChop and RustingCS2).
- FP16 device backward kernels for UNet adapter training in `cuda_image`: convolution input gradients, group and layer norm, the gated GELU, SiLU and nearest upsampling, each matched to the host `unet_lora` functions.
- `DeviceUnet::forward_adapted` runs a UNet LoRA's attention adapters on the device and keeps a tape for the backward pass, matched to `UnetLora::forward`.
- A device attention backward for any head width in `cuda_image`, a chunk of query rows at a time with key and value gradients summed in FP32.
- `DeviceUnet::backward_adapted` takes a `DeviceTape` and `dL/d(prediction)` back through the whole UNet to every attention adapter's gradients on the device; `DeviceAdapters::add_gradients` adds them to the host `UnetLora`. Within 0.3% of `UnetLora::backward` overall.
- `unet_lora::UnetLora`: LoRA adapters on a UNet's attention projections,
  read from a diffusers PEFT file, with a host forward and backward pass
  whose adapter gradients match PyTorch autograd within 5e-5. It is the
  reference for UNet LoRA training on the GPU.
- `examples/train_dit_lora.rs` trains a LoRA on a FLUX-style DiT from a
  folder of captioned images on the CUDA device (`DitLora::new`,
  `DeviceDit::flow_step`) and writes a diffusers PEFT file. diffusers loads
  it to within 1.1e-6 of this crate's own prediction with it.
- `DeviceDit` runs DiT LoRA adapters on CUDA and returns their gradients
  from a device backward, within 2e-2 of the host reference. One step at
  1024 image tokens on a 768-wide, six-block DiT takes 45-56 ms.
- `dit_lora::DitLora`: LoRA adapters on a DiT's block projections, read from
  a diffusers-FLUX PEFT file, with a host forward and backward pass whose
  adapter gradients match PyTorch autograd within 6e-5. It is the reference
  for DiT LoRA training on the GPU.
- Textual inversion: `ImagePipeline::add_embedding` adds a learned prompt
  token from a diffusers, A1111 or SDXL (`clip_l`/`clip_g`) safetensors file
  to every CLIP tower, and a multi-vector token stands for all its vectors,
  as diffusers expands it. Hidden states and pooled vectors match diffusers
  on a tiny tower with the SDXL tokenizers within 1e-5. Legacy CLIP configs
  (`eos_token_id = 2`) now pool at the highest id, as Transformers does.
- Inpainting: `Inpaint` wraps any `Denoiser` with a picture latent and a
  keep weight a latent row, and `ImagePipeline::sample_latent_inpaint`
  repaints only where the weight is zero, under every solver and schedule.
- `ImagePipeline::generate_inpaint` repaints a picture under a pixel mask.
  Inpainting UNets (nine input channels) load and read the mask and the
  masked picture as diffusers feeds them, matching diffusers within 6e-7;
  other models are steered through `Inpaint`.
- `ControlNet` loads diffusers ControlNets for UNet models, and
  `Unet::set_controls` lets one or more of them steer the layout from an edge,
  depth or pose picture, on the CPU or a CUDA device; matches diffusers
  within 9e-7.
- `ImagePipeline::load_with_loras` and `ShardedSafeTensors::add_lora` merge
  published LoRA files (PEFT/diffusers and kohya spellings, including
  kohya's SDXL and FLUX files that use the original models' module names,
  whose fused FLUX projections are split onto diffusers' parts) into a UNet
  or DiT, and into the CLIP text encoders, as they load; matches PEFT's
  `fuse_lora` within 1.1e-6.
- `TransformerLm::log_probabilities` gives each token's log-probability
  after the ones before it, for scoring and ranking texts.
- `TransformerLm::generate_beam` runs beam search, with an optional end
  token.
- `Decoder::fork` copies a session, so a shared prefix is prefilled once
  and each request continues its own fork.
- `Decoder::logits` and `Decoder::push`, and the same on `BatchDecoder`:
  read the next-token logits and append an id of the caller's choosing, for
  constrained decoding, logit biases and log-probabilities.
- `BatchDecoder::feed` accepts prompts of different lengths. On a device the
  shorter ones are padded on the left and the padding is masked out of
  attention; later feeds still give every sequence the same number of
  tokens.
- `Sampler::min_p`, `Sampler::presence_penalty` and
  `Sampler::frequency_penalty`.
- `TransformerLm::generate_speculative_with` runs speculative decoding with
  any `Sampler`, accepting draft proposals by rejection sampling so the
  output is distributed as sampling the target model alone.
- CLIP zero-shot classification and retrieval: `VitEncoder::embed` projects
  the image tower's class token, `clip::logit_scale` reads the learned
  temperature, and `clip::zero_shot` turns an image and captions into
  probabilities. Matches `transformers`' `CLIPModel` within 2.2e-7.

- `TransformerLm::set_8bit_optimizer_state` holds the device Adam moments at
  one byte each (block-wise, bitsandbytes' dynamic map). The default model
  trains in 1723 MiB instead of 2037 at the same speed, and its loss over
  2000 steps tracks the FP32 state.

- `VaeEncoder::attach_device`, and `ImagePipeline::to_cuda` moves the VAE
  encoder too: image-to-image encodes a 1024x1024 picture in 0.25 s instead
  of about two minutes on the CPU.
- Tutorial chapter 18, generating images from text with a published
  diffusion model, and a new Part V in the course index.
- CI runs `cargo semver-checks` against the latest release tag.
- `cuda_mmdit::DeviceDit` runs the FLUX-style MMDiT denoiser on the device
  with BF16 weights, FP32 residual streams and the fused attention kernel.
  A small FLUX-shaped model over a 512x512 image takes about 16 ms a pass
  against 1.27 s on the host, at the same speed with int8 weights.
- `cuda_t5::DeviceT5` runs a gated T5 encoder (T5 1.1, the one FLUX and
  Stable Diffusion 3 read) on the device with BF16 weights and an FP32
  residual stream. A T5-base-sized encoder over 512 tokens takes about 29 ms
  against 528 ms on the host.
- `TransformerLm::quantize_lora_base_4bit(Codebook::Nf4)`: QLoRA on an NF4
  base, on the host and the device. A rank-16 run on the default model
  peaks at 762 MiB against 778 on an int8 base and 889 on FP32, at the same
  step time. `has_int8_base` is now `has_quantized_base`.
- 4-bit host weights: `TransformerLm::quantize_4bit(Codebook::Nf4 | Int4)`
  stores every block linear as two 4-bit codebook indices a byte with an
  absmax scale per block of 64 (0.5625 bytes a weight), keeping the
  embedding and an untied head at int8. `bench_decode` takes `nf4` as its
  quantization argument. Host decode of a 100M-parameter shape runs about 17%
  faster than int8.
- `TransformerLm::quantize_lora_base`: holds a LoRA run's frozen base
  projections as int8 with a scale per row, on the host and the GPU, so
  fine-tuning keeps the base in a quarter of the memory at the same step
  time. The adapters train in full precision and save with `save_lora`.
  `examples/checkpoint_memory` takes `lora` as a third argument to compare.
- `TransformerLm::set_int8_decode` makes device decoders read the attention,
  dense and shared-expert feed-forward and output-head weights as int8 with a
  scale per row: about 10% more tokens per second for one sequence on the
  default model, 13% fewer for 16. `bench_decode` takes `int8` as a fifth
  argument.
- `TransformerBuilder::checkpoint` and `TransformerLm::set_checkpoint` pick
  blocks whose activations are recomputed in the backward pass on the device
  instead of kept. With every block checkpointed, a 16 x 512 step on the
  default model peaks at 1353 MiB instead of 2843 MiB and takes 23% longer.
  `examples/checkpoint_memory.rs` measures both.
- `TransformerLm::save_gguf`: writes a LLaMA-layout model as a GGUF file in
  F32 or Q8_0 for llama.cpp and Ollama, with query and key rows reordered for
  llama.cpp's rotary convention.
- `TransformerLm::generate_speculative`: greedy speculative decoding with a
  smaller draft model, one target pass per round of drafted ids, giving the
  same continuation as greedy `generate`. `KvCache::truncate` drops cached
  positions.
- `sample_unipc`: the UniPC multistep predictor-corrector solver at
  diffusers' defaults, matching `UniPCMultistepScheduler` step for step.
- `Scheduler::lcm_sigmas` and `sample_consistency`: the timesteps and the
  multistep consistency loop a latent consistency model (LCM, LCM-LoRA) runs
  in two to eight steps, matching diffusers' `LCMScheduler` timesteps.
- `TransformerLm::batch_decoder(n)` returns a `BatchDecoder`, which decodes n
  sequences in step and samples one token for each per `next`. On a device
  model the n sequences share every weight read: the default model decodes
  1994 tokens/s over 4 sequences and 9093 over 64, against 618 for one.
  `bench_decode` takes the sequence count as an optional fourth argument.
- `TransformerCache::expert_loads`: each MoE layer's share of routed tokens
  per expert, from a forward pass on the host or the GPU, for checking that
  routing on real data stays balanced.
- `VaeDecoder::decode_tiled`: decodes a latent in overlapping tiles blended
  with linear ramps, so images too large to decode at once fit in memory.
- `SpikeGuard`: flags a training loss that is not finite or jumps above a
  multiple of its moving mean, so a loop can skip that step and, after
  several in a row, roll back to its last checkpoint.
- `Bpe::train_json`: trains a byte-level BPE tokenizer on a text corpus and
  writes a `tokenizer.json` that `Bpe::from_json` loads.
- `TransformerLm::step_with_update_ratios`: one optimizer step that returns
  each parameter's update-to-weight ratio, for spotting a layer that steps
  too far or has stopped learning. Copies the weights to the host, so call
  it occasionally.
- Karras noise levels for DDIM-schedule models: `Scheduler::karras_sigmas`,
  run through the new `sample_with_sigmas`. With `Solver::DpmPlusPlus2m` this
  is "DPM++ 2M Karras".
- z-loss on the LM head: `TransformerBuilder::z_loss(weight)` or
  `TransformerLm::set_z_loss` adds `weight * logsumexp(logits)^2` to the
  training loss, reported in `TotalLoss::auxiliary_loss`, on the host and in
  the fused CUDA head. Off by default.
- Tutorial chapter 17, fine-tuning with LoRA: attach an adapter, train it,
  save it alone, load it onto the base, merge it, and measure what the base
  forgets.

- `webdataset::WebDataset` reads WebDataset tar shards one sample at a
  time: the members that share a key, such as `000017.jpg` and
  `000017.txt`, come back together. No new dependency.
- Prioritized replay for the interoceptive agent: build the trainer's buffer
  with `ReplayBuffer::prioritized` and it samples by TD error, weights the
  loss by importance (`InteroceptiveAgent::train_step_weighted`,
  `weighted_td_huber_loss`) and updates the priorities after each step.
- `scripts/bench_check.sh`: fails when `bench_train_step` or `bench_decode`
  is more than 3% below its row in `docs/baseline.md`. Run it locally before a
  release; CI has no GPU.
- `rusting_brain::BUILD`: the library version and the git commit it was built
  from. Print it at startup to spot a binary built from an older checkout.
- `Bpe` tokenizes LLaMA 3 and Qwen2 files the way the reference does: their
  split rule (digits in runs of up to three, or one for Qwen2,
  case-insensitive contractions) and `ignore_merges`.
- `Bpe` reads sentencepiece-style BPE files (LLaMA 2, Mistral, TinyLlama):
  `▁` for spaces and `<0xNN>` byte fallback, with the `Metaspace`
  pre-tokenizer or the older `Prepend` normalizer. Ids match the Hugging Face
  `tokenizers` library.
- `interoceptive::backtest` scores the greedy policy over real candles, with
  costs and without shocks: total return, per-step Sharpe, maximum drawdown
  and liquidation, beside a buy-and-hold baseline. `walk_forward` makes the
  train and test windows.

### Changed

- Tutorial chapter 16 trains from `TokenFile` shards, and chapters 14 and 16
  and the README load tokenizers with `Bpe` rather than the `tokenizers` crate.

- `Bpe` merges in `n log n` instead of `n^2`. A sentencepiece-style file
  (LLaMA 2, Mistral, TinyLlama) merges the whole text as one piece, so a
  50 KB document went from 43 s to 13 ms. Ids are unchanged.
- CUDA tests that find no device print `skipped: no CUDA device`, and fail
  instead when `RUSTING_BRAIN_REQUIRE_CUDA` is set, so a GPU run cannot pass
  without running them. CI builds the docs with `-D warnings`.

- `InteroceptiveAgent::train_step` uses Double DQN targets: the online
  network picks the next action and the target network scores it, which
  removes the overestimation of a plain max. `double_bellman_targets` is
  public; `bellman_targets` is unchanged.

### Fixed

- The CUDA test suite no longer fails at random in tests that need the fused attention kernel (J11). Two parity tests turned that kernel off through a process-wide switch while they ran, so a test running alongside them on another thread lost it too. The tests' switch is now per thread; `RUSTING_BRAIN_NO_FLASH` still turns the kernel off for the whole process. A full `cargo test --features cuda,images --lib` failed in 5 of 6 runs before and in none of 6 after.
- `Mesh::simplify` and `Mesh::simplify_within` no longer depend on which of several equally cheap edges the queue hands out first (J10). An edge refused because it would fold a face used to be dropped for good; now, once nothing cheap is left, the refused edges are weighed again for as long as a round still collapses something. A flat 6x6 grid within 1e-6 failed to reach its two corner faces in 3 of 200 runs before and in none of 400 after.
- A device convolution asked to add into a map with a different channel
  count is an error, instead of writing past the end of that map.
- A CLIP text or image tower saved by Transformers 5, whose keys leave off
  the `text_model.` or `vision_model.` prefix, loads instead of failing on a
  missing tensor.
- A `.safetensors` header whose shape product overflows, and a LoRA file
  with a scalar, empty or mismatched rank, are now errors. Before, release
  builds could wrap the product and read past a buffer.
- `ClipTextConfig::from_file` read the legacy `bos 0, eos 2` of older CLIP
  configurations (OpenAI CLIP, Stable Diffusion 1.x, FLUX.1's CLIP-L) as
  real tokens, so prompts started with `!` and pooled at the wrong row. It
  now uses the vocabulary's last two ids, as `transformers` does.

- Stable Diffusion XL, 2.x and 3 prompts are padded with the pad token the
  tokenizer names (`!` for SDXL's second tower) instead of the end token. The
  UNet reads the padded positions, so the wrong token gave every SDXL image
  a cross-hatched texture and made image-to-image repaint far more than its
  strength asked. New `ClipTextEncoder::set_pad_token`.
- The device VAE decoder returns an error when an FP16 overflow leaves
  non-finite values in the image, instead of a black image.
- Byte-level BPE encoding no longer slows down quadratically on long text:
  the GPT-2 pre-tokenizer copied the rest of the text at every character
  (78,000 characters went from 617 ms to 1.4 ms).
- LoRA training on CUDA works under mixed precision, the default. Folding the
  adapter into a BF16 weight asked cuBLAS for an FP32-to-BF16 GEMM it does
  not have, so every step failed with `CUBLAS_STATUS_NOT_SUPPORTED`; the test
  covered FP32 only.

- `TransformerLm::save_bin` and `save_lora` copy the weights back from the
  device first, as `save_hf` already did. A model trained on CUDA used to
  save the weights it had before its first device step.

- Corrupt files no longer panic or allocate without limit: a huge token id in
  `tokenizer.json`, a truncated or overflowing `.npy` or IDX header, and a
  weight, LoRA or weight-average file whose header claims more than the file
  holds now return an error. `load_bin` also refuses a LoRA rank larger than
  the smallest weight it adapts. A seeded corruption test (`fuzz::*`) covers the
  file readers.
- `step_clipped` skips the update when the gradient norm is NaN or infinite
  and clears the gradients, instead of writing NaN into the weights and the
  optimizer moments. The returned norm shows the skip.
- `Bpe` no longer reads the empty `end_of_word_suffix` that Qwen files write
  as a CLIP-style suffix, which had made every Qwen tokenization wrong.

## 3.0.0

A major version because three public configuration structs gained fields, so
a struct literal that names every field no longer compiles. Add the field, or
build from `Default` with `..Default::default()` where the struct has one:

- `TransformerConfig::rope_scaling` (`None` keeps the old behaviour).
- `RopeSpec::scaling` (`None` keeps the old behaviour). `RopeSpec` has no
  `Default`, so name the field.
- `ShapeVaeConfig::free_bits` (`0.0` keeps the old behaviour).

Nothing was removed or renamed. Checkpoints written by 2.x load unchanged:
every new field defaults when it is missing from the file.

### Added

- `TransformerLm::save_hf` and `TransformerLm::load_hf` (`src/hf.rs`) write and
  read the Hugging Face LLaMA layout, a `config.json` beside a
  `model.safetensors`. `transformers` reads the export as `LlamaForCausalLM`.
  The loader takes LLaMA and Mistral checkpoints, single-file or sharded, and
  refuses what the layout cannot hold (biases, per-head q/k norms, MoE or GELU
  layers, attached LoRA, quantized weights, YaRN or dynamic rope scaling, a
  sliding window shorter than `max_seq_len`) instead of approximating it.
- `RopeScaling`, with linear and Llama 3.1 (`llama3`) scaling, set through
  `TransformerBuilder::rope_scaling`. It changes how the rotary tables are
  built, so the CPU path, the CUDA upload and the ONNX export all use it.
- `TokenMix` draws each batch from several `TokenFile`s by weight. Every draw
  comes from the step number and the mix seed, so a resumed run sees the same
  batches.
- The `interoceptive` module: a DQN trading agent whose market trunk is gated
  by a model of the account's own state, with a homeostatic reward. A test
  holds its simulation and replay loop to zero heap allocations per step.
- `DeviceVitEncoder` runs the CLIP vision tower on CUDA in batches.
- `ShapeVaeConfig::free_bits`, a per-unit KL floor for
  `losses::kl_divergence_free_bits`. The shape decoder's norms run in BF16,
  and the device decode goes through `decode_train`, so chunked decoding works
  on CUDA.

### Fixed

- `TrainableConv2d` trained its bias with a zero gradient on CUDA: the bias
  lived on the device while its gradient landed on the host copy.

### Changed

- The cross-attention backward is split across sequences, and short softmax
  rows stay in registers.
- `save_bin` and `save_optimizer_state` stream each tensor through a fixed
  16 KiB buffer instead of building a full byte copy of it.

### Documentation

- The CUDA out-of-memory error explains fragmentation.
- `roadmap.md` plans the next releases. The README scope now covers
  `Conv2d` and `TrainableConv2d` and lists the model types.
- `Matrix::random` is documented as unseeded test filler.

## 2.0.1 and 2.0.2

These two releases shipped without their own changelog headings. Everything
below up to 2.0.0 went out in one of them. The `ImagePipeline` changes
described next were breaking and went out in 2.0.1, a patch release.

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
  1024x1024 from 0.65 s to 0.25 s and the VAE decode of that image from 1.72 s
  to 0.53 s, and a whole guided 1024x1024 image at 30 steps from 37.4 s to
  16.1 s. The convolution lowering writes a whole KxK patch per thread
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

  Self-attention in the image path never writes its score matrix. The fused
  kernel `src/cuda_flash.rs` already ran for the text models; it gained an
  FP16 variant that reads its keys and values as windows on the fused QKV
  projection, and the image path uses it wherever a head is 64 wide and no
  mask is in play. The scores stay in registers between the two matmuls and
  the softmax runs online against a running maximum, so the largest buffer in
  the pass is neither written nor read back three times: at 1024x1024 that is
  5.8 GB of traffic a pass that no longer happens. A device older than Ampere,
  a head of another width, or `RUSTING_BRAIN_NO_FLASH` falls back to the
  three-kernel path. The scores themselves accumulate in FP16 inside the
  kernel, at twice the tensor-core rate, while the weighted values keep an
  FP32 accumulator: a score is a dot product over one head width, where
  eleven mantissa bits are finer than the softmax downstream can resolve,
  and the values are a sum over the whole context, where they are not.

  The gated feed-forward applies its projection's bias itself. That
  projection writes the widest tensor a transformer block holds -- four
  times the model width, twice over for the gate -- and a bias pass of its
  own cost a full read and write of it.

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
