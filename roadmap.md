# RustingBrain Roadmap

Where the library goes after 2.0.2. This file is the index: it names every
piece of work worth doing, sizes it, orders it, and points at the document that
holds the detail when one already exists. It does not repeat those documents.

| Detail lives in | For |
|---|---|
| `featuresPlan.md` | Feature candidates with costs, and the "not on this list" rationale |
| `docs/improvement-plan.md` | MoE training throughput, work items W0-W9 |
| `docs/optimize-plan.md` | Dense training throughput, Tasks 0-9 |
| `docs/baseline.md` | Measured numbers and the current per-kernel profile |
| `2Dto3DPlan.md` | Image-to-3D stages A-H and the deferred texture work |
| `docs/interoceptive_spec.md` | The interoceptive trading agent and its acceptance criteria |

## How to read this

**Effort** is for one person: **S** is days, **M** is one to two weeks, **L**
is a month or more.

**Priority:**

- **P0**: blocks a release, or fixes something wrong today.
- **P1**: the next things to build. Most users will notice them.
- **P2**: worth doing once P1 is done, or when someone asks for it.
- **P3**: parked. Written down so nobody has to argue it again.

**Ground rules that apply to every item:**

1. **Profile first.** No performance item starts without a fresh profile
   (`nsys` or `examples/profile_step.rs`) that shows the target is where the
   time goes. The estimates below come from earlier profiles. Re-measure
   before trusting them.
2. **Training runs on CUDA.** Every training-throughput number and every
   end-to-end training check runs on the GPU path. The CPU path is the
   reference for parity tests, not a training target.
3. **Parity before speed.** A new kernel lands only with a test that compares
   it against the CPU reference or the kernel it replaces.
4. **Read the reject lists first.** `docs/improvement-plan.md` has two lists,
   "Already landed — do not redo" and "Already measured and rejected — do not
   redo". Reopen an item from them only with a new measurement.

## Where things stand

- Dense transformer training on one RTX 3060: 21.8k tokens/s at 4 x 1024 for
  a 101.7M-parameter model. That ties or beats PyTorch with `torch.compile`
  at every measured shape. GEMMs run at 78-98% of the card's BF16 peak.
- Kernel time per step: GEMMs 60%, the three flash-attention kernels 19%,
  everything else 20%. The flash kernels run at about half of peak, so they
  are the largest target left.
- MoE costs 1.37x a dense step for 1.14x the arithmetic. The routed path still
  runs in FP32.
- The library has MoE, LoRA (CPU and CUDA), quantization-aware training,
  int8 inference, ViT, a masked-LM (bidirectional) mode, Lion and Adam,
  streaming datasets, the text-to-image pipeline (MMDiT, UNet, VAE, CLIP, T5,
  LLaMA-family prompt encoders), image-to-3D, ONNX import and export, and the
  interoceptive RL agent.
- Single GPU only. No multi-GPU, no distributed training, no CPU offload.
- Structural tax: no autograd and no N-D tensor type. CPU and CUDA paths are
  written twice. Every new layer is two forward passes, two hand-written
  backward passes and a finite-difference test.

---

## Milestone 2.1 — finish what is open (P0)

Release the work that is already in the tree, and close what is half done.

| # | Item | Effort | Notes |
|---|---|---|---|
| 2.1.1 | Land the uncommitted diff | S | `attention.rs`, `cuda_image.rs`, `gpu_shape.rs`, `vit_encoder.rs`, `losses.rs` and others carry about 780 changed lines. Split them into commits by topic. Run `cargo test` and `cargo test --features cuda`. |
| 2.1.2 | ~~Interoceptive acceptance criteria~~ **done** | S | Clippy is clean with and without `cuda`. The stress-reflex test already existed (`agent.rs:246`). `tests/interoceptive_no_alloc.rs` counts allocations over 1000 steps, resets and replay samples, and a mutation check confirmed that it fails when one allocation is added. Original scope: `docs/interoceptive_spec.md` §5: zero warnings, the stress-reflex logit test (`margin_stress` 0.05 against 0.95, `EmergencyFlatten` logit rises above the threshold), and no heap allocation per step in the replay loop. Prove the last one with a counting allocator in a test, not by reading the code. |
| 2.1.3 | ~~Fix `Matrix::random`~~ **done, smaller than written** | S | `Network` already seeds its own He init, and no production code calls `Matrix::random`. It is now documented as unseeded test filler. Original note: `src/matrix.rs:98` draws from `thread_rng` over `0..1`, ignores the seed, and makes every weight positive. Make it take an RNG and use a zero-centred range, or deprecate it in favour of `Param::he_uniform`. The legacy `Network` path uses it. |
| 2.1.4 | ~~Stream the checkpoint writer~~ **done** | S | `write_f32s` in `src/transformer.rs`, covered by the existing round-trip tests. Original note: `save_optimizer_state` and `save_bin` (`src/transformer.rs`) build a full `Vec<u8>` copy of each tensor before they write it. Write little-endian chunks into the `BufWriter` instead. This matters on long runs that save every 200 steps. |
| 2.1.5 | ~~Update the README Scope section~~ **done for the facts** | S | The convolution claim and the model list are fixed. "Not planned" for encoder-decoder and multi-GPU is still decision 1. Original note: The section says convolutions have no backward pass and encoder-decoder is "not planned". Check both claims against the tree and against the decisions in this file. |
| 2.1.6 | Changelog and release | S | Move "Unreleased" to 2.1.0. Run `cargo semver-checks`, because `ImagePipeline::new` changed shape. |

---

## Track A — Training performance (CUDA)

Every item here is gated on ground rule 1.

### A1. `flash_attention_dkv` fragment packing — P1, M

This is the largest single kernel left, at 15.45 ms a step. The ablations in
`docs/baseline.md` ("What is left") show that the kernel is bound by mma work
and not by bandwidth. The second mma loop costs 0.44 ms against 0.17 ms for the
first, with identical mma and `LDS.32` counts. Start with the fragment packing
in that second loop. Use `examples/flash_probe.rs` against its 0.995 ms
baseline. Target: the same mma throughput as the first loop, about 5 ms a step.

### A2. `flash_attention_dq` and `flash_attention_fwd` — P2, M

These cost 10.45 ms and 7.24 ms. Apply whatever A1 learns about the second loop.
Do not start before A1 lands.

### A3. Flash attention for other head widths — P1, M

`cuda_flash::eligible` requires `head_dim == 64` and mixed precision. Every
other width falls back to the three-kernel path that used to be 21% of a step.
Add `head_dim` 128, which most published LLaMA-family checkpoints use and which
B5 needs. Add 32 only if a model needs it.

### A4. Non-causal flash in BF16 for training — P1, M

The FP16 non-causal variant exists for inference in the image path. Training a
bidirectional model (masked LM, ViT, the flow transformer) has no fused path.
`gpu_cross.rs` and `gpu_flow.rs` both say "FP32 throughout". This item
unblocks A5.

### A5. BF16 for `gpu_cross` and `gpu_flow` — P2, M

The 3D training path runs in FP32, so it gets half the tensor-core rate or
less. Do this after A4 and after a profile of `examples/train_shape.rs`.

### A6. Fused LM head and cross-entropy — P1, M

The LM head GEMMs are 22.72 ms a step. `src/transformer.rs:1367` materializes
the full `[rows, vocab]` logits on the device. A chunked fused
GEMM-plus-softmax-plus-loss that never writes the logits saves both time and
the largest activation buffer. At vocab 32k and 4 x 1024 tokens, that buffer
is 512 MB in FP32. This also helps if W8 (halve the vocabulary) is rejected.

### A7. MoE work items W1-W4 — P1, S to M each

These are already scoped in `docs/improvement-plan.md`:

- **W1:** run the routed path in BF16. Estimate: 10-20 ms a step.
- **W2:** fuse the permutation passes. Up to 6 ms.
- **W3:** fold `pack_gate_up` into the parameter layout. 2.5 ms, and it lowers
  the VRAM ceiling.
- **W4:** measure the expert GEMM split, and only then optimize it.

Also move the router GEMM onto the tensor cores. With 8 experts, cuBLAS picks an
FP32 SGEMM. Pad the output to 16, or compute the router in BF16.

### A8. MoE capacity factor — P2, M

This is the TODO at `src/moe.rs:200`. A fixed-capacity variant gives
statically shaped expert buffers. It is the precondition for the third part of
W2, for a batched-pointer expert GEMM, and for C1 (CUDA graphs). Keep dropless
as the default.

### A9. Fused multi-tensor optimizer step — P2, S

Adam is 113 launches and 8.96 ms a step. One kernel over a pointer table of all
parameters removes 112 launches. Measure first: Adam is bandwidth-bound, so the
gain may only be the launch overhead. Do the same for Lion.

### A10. Activation checkpointing — P1, M

Recompute each block's forward pass during the backward pass instead of
storing it. This trades about 30% more compute for most of the activation
memory. It is the cheapest way to train longer sequences or a larger model on
a 12 GB card. Make it per-layer and opt-in on the builder.

### A11. 8-bit optimizer state — P2, M

Block-wise quantized Adam moments cut optimizer memory by 75%. Gate it on the
loss-curve check from `docs/optimize-plan.md` Task 4 Step 7. Note W6 in
`docs/improvement-plan.md`: optimizer-state traffic "can hurt".

### A12. Overlap host round trips — P3

W5 measured the ceiling at 5%. Leave it parked until everything above lands.

---

## Track B — Training features

### B1. Data-parallel training on two GPUs — P1, L

This is the item the README apologizes for. Plan:

1. One process with one context per device, and one model replica each.
2. All-reduce the gradients with NCCL through cudarc. Bucket the gradients and
   overlap the all-reduce with the backward pass.
3. Each rank draws from `TokenFile` with its rank folded into the window
   offsets, so the "resume at step N, same data" property still holds.
4. Acceptance: on two cards, the loss curve matches a single card at the same
   effective batch, and throughput is 1.8x or better.

Do not build tensor parallelism, pipeline parallelism or multi-node until two
cards on one host works. See `featuresPlan.md` "Multi-GPU".

### B2. ZeRO-1 optimizer-state sharding — P2, M

Shard the optimizer state across the data-parallel ranks. Do this after B1.

### B3. More optimizers — P2, S each

- **AdamW with decoupled weight decay**, if `adam_with_weight_decay` turns out
  to be coupled. Check the code first.
- **Muon** for the 2D weight matrices, with AdamW for the rest. Measure it
  against Adam on the 101.7M benchmark model.
- **Schedule-free AdamW.** It removes the schedule argument from long runs.
- **Adafactor.** It has factored second moments, and is cheap in memory.

Ship each one only if it beats Adam or Lion on loss per token in a real run.

### B4. LoRA completion and QLoRA — P1, M

LoRA works on CPU and CUDA. What is left:

- `src/param.rs:810`: a device-resident adapter accumulates its gradient on
  the host. Keep it on the device.
- `src/gpu_model.rs:1390`: the adapter is folded into the weight once per
  forward pass. Use two thin GEMMs instead, which also lets the base weight
  stay frozen in int8.
- **QLoRA**: an int8 or NF4 frozen base with BF16 adapters. This depends on
  D3.
- Save and load an adapter on its own, without the base model.

### B5. Load published LLaMA-family weights into `TransformerLm` — P1, M

`TextEncoder` already reads Qwen2 and LLaMA safetensors for inference. Map the
same tensors into a trainable `TransformerLm`, so a user can fine-tune a
published model with B4. This needs A3 (`head_dim` 128 is the common case),
RoPE scaling (B6) and a tokenizer round trip through `Bpe`.

### B6. Long context — P2, M

- RoPE scaling: linear, NTK-aware and YaRN. B5 needs these to read published
  configurations correctly.
- Sliding-window attention in the flash kernel (a mask on the tile loop).
- Sequence packing with document masks in `TokenBatch`, so packed documents
  do not attend to each other.

### B7. Training stability tools — P2, S each

- Track the gradient norm and the per-layer update-to-weight ratio, and log
  them through `fit_with`-style callbacks.
- Skip a step when the loss is NaN or a spike, and roll back to the last good
  state.
- z-loss on the LM head (the router already has one).

### B8. Preference and RL fine-tuning — P3, revisit

`featuresPlan.md` excludes RLHF and DPO. The interoceptive module now ships a
DQN loop inside the crate, which weakens the "different library" argument.
DPO needs no reward model and no sampling loop: it is a loss over pairs of
sequences. Reopen this if B5 lands and someone wants to align a fine-tuned
model.

---

## Track C — Inference performance

### C1. CUDA graphs for decode — P1, M

A decode step is many small kernels, so launch overhead dominates. Capture one
decode step per shape and replay it. This needs a device-resident KV cache
(C2) and static buffers. `cuda_image.rs` says "no graph capture" too. The
UNet loop is also a candidate.

### C2. Device-resident KV cache and batched decode — P1, M

This is W9 in `docs/improvement-plan.md`, still not measured. Keep the KV
cache on the device across `Decoder` calls, and decode several sequences in
one batch. Measure tokens/s with `examples/bench_decode.rs` before and after.

### C3. Paged KV cache — P2, M

Allocate the KV cache in fixed blocks so sequences of different lengths share
a pool. Do this after C2, and only if batched decode with uneven lengths is a
real use case.

### C4. Speculative decoding — P2, M

Use a small draft model and verify with the large one. This needs C2. It is
pure library code once batched verification exists.

### C5. Weight-only int8 and int4 GEMM on CUDA — P1, M

`Precision::Q8` is CPU only for the language model. A dequantize-in-register
GEMV kernel makes decode memory-bound on 1 byte (or half a byte) per weight
instead of 2. Decode on a 3060 is bandwidth-bound, so this is close to a 2x or
4x gain for batch 1.

### C6. Image models on CUDA, the rest — P1, M

- The transformer denoiser (`Dit`) and the T5 encoder still run on the CPU
  ("no device path yet" in the changelog). They need A4.
- Measure BF16 weights against the byte path, per `featuresPlan.md` Stage 6.
- `src/cuda_image.rs:438`: replace the naive transpose with a tiled
  shared-memory transpose, if the profile shows it.
- FP16 overflow guard on the SDXL decoder (`cuda_image.rs:25`): make sure that
  it fails loudly and does not produce a black image.

### C7. Diffusion sampling — P2, S each

- More solvers: DPM++ 2M Karras sigmas, UniPC, LCM.
- Batch the guided and unguided passes into one forward pass of batch 2.
  Check whether this is already done first.
- Tiled VAE decode, so large images fit in 12 GB.

---

## Track D — CPU path

The CPU path is the parity reference and the no-GPU fallback. Keep it correct
and reasonably fast, and do not chase peak throughput on it.

| # | Item | Effort | Notes |
|---|---|---|---|
| D1 | MoE loop order | S | `MoeLayer::backward` auxiliary term and `auxiliary_losses` walk `[tokens, experts]` down the columns. Swap the loops. |
| D2 | One `exp` per element in `softmax_rows` | S | `src/moe.rs:631` computes `exp` twice. |
| D3 | int4 / NF4 weight storage | M | Shared by B4 (QLoRA) and C5. Block-wise scales, same layout on CPU and CUDA. |
| D4 | Drop the cloned inputs in caches | S | Measured at 0.4% and rejected. It only becomes worth it with many experts. P3. |
| D5 | ~~Conv2d backward on CUDA~~ | — | Already exists: `TrainableConv2d` has a backward pass and a device path. The uncommitted diff fixes its bias, which never trained on the device. |

---

## Track E — Data

`BatchSource`, `DatasetStream`, `JsonlStream`, `TokenStream` and `TokenFile`
exist already. What is missing:

| # | Item | Priority | Effort | Notes |
|---|---|---|---|---|
| E1 | Weighted dataset mixing | P1 | S | Draw from N `TokenFile`s by weight, determined by the step number so that a resume is exact. Every real pretraining mix needs this. |
| E2 | Parquet reader | P2 | M | Most public corpora on the Hub ship as Parquet. Put it behind a feature flag so the default build stays small. Read one column of text, a row group at a time. |
| E3 | WebDataset (tar shards) | P2 | S | Image and text pairs for C6 training and for ViT. `tar` is a small crate, or about 100 lines by hand. |
| E4 | Shuffled `chunk` windows | P3 | S | `src/token_file.rs:288` returns windows in file order. Evaluation wants that, so add it only if something else needs a shuffled full pass. |
| E5 | Multi-line CSV fields | P3 | S | `src/dataset.rs:924` reads single-line rows only. |
| E6 | Memory-mapped shards | P3 | S | `src/shards.rs:32` seeks instead of mapping. Change it only if a profile shows that the reads cost time. |
| E7 | Tokenizer training | P2 | M | `Bpe` reads a `tokenizer.json` but cannot train one. `optimize-plan.md` Task 8 Step 3 needs a retrained tokenizer. Today this needs Python or the `tokenizers` crate. |

---

## Track F — Interop and formats

| # | Item | Priority | Effort | Notes |
|---|---|---|---|---|
| F1 | Write safetensors | P1 | S | `src/safetensors.rs` only reads. Writing lets a trained `TransformerLm` or LoRA adapter load in PyTorch, llama.cpp converters and the Hub. The format is a JSON header plus raw bytes. |
| F2 | Hugging Face config export | P1 | S | Write a LLaMA-compatible `config.json` next to F1's output, so that `transformers` loads the result directly. This is the reverse of B5. |
| F3 | GGUF export | P2 | M | Run trained models in llama.cpp and Ollama. Q8_0 first, because it maps to `Precision::Q8`. |
| F4 | ONNX export for MoE | P3 | M | `save_onnx` covers dense networks and the dense LM. MoE needs dynamic routing in the graph, which ONNX handles poorly. Park it until someone asks. |
| F5 | Checkpoint format version | P1 | S | Put a version field and a model-shape header in binary checkpoints, so that an old checkpoint fails with a clear error and not with garbage weights. Check first whether one already exists. |

---

## Track G — Model families and modalities

| # | Item | Priority | Effort | Notes |
|---|---|---|---|---|
| G1 | CLIP image tower | P2 | M | `src/clip.rs:13` has no image tower. With it, the library can do zero-shot classification and image-text retrieval, and it reuses `VisionTransformer`. |
| G2 | ControlNet and inpainting | P2 | L | Left out in `featuresPlan.md` Stage 7. Both condition on a second image. |
| G3 | Diffusion fine-tuning (LoRA on a UNet or DiT) | P2 | L | Training for the image path. It needs D5, A4 and B4 on the image blocks. |
| G4 | 3D texture | P2 | L | Deferred in `2Dto3DPlan.md` §6. The rasterizer route is the open decision. |
| G5 | Eikonal regularization for the shape VAE | P3 | M | `src/shape_vae.rs:34` needs `dL/dquery_point`. |
| G6 | T5 attention mask | P3 | S | `src/t5.rs:13` has no mask. Padded prompts in a batch need one. |
| G7 | Encoder-decoder (seq2seq) training | P3 | L | The README says "not planned". T5 inference exists, and cross-attention exists in `gpu_cross`. Revisit if translation or summarization becomes a goal. |
| G8 | Time-series models for the interoceptive agent | P2 | M | The `MarketTrunk` is an MLP. A small causal transformer over a window of ticks reuses `TransformerLm` blocks. |

---

## Track H — Interoceptive agent

After 2.1.2:

| # | Item | Priority | Effort | Notes |
|---|---|---|---|---|
| H1 | Real-data backtest harness | P1 | M | `ticks_from_csv` exists. Add a walk-forward split (train on one window, evaluate on the next), costs and slippage, and a report (Sharpe, maximum drawdown, liquidation count) against a buy-and-hold baseline. Without this, a training curve says nothing about the agent. |
| H2 | Double DQN and a target network | P1 | S | Check what `trainer.rs` does today. If it is plain DQN, the Q-values overestimate. |
| H3 | Prioritized replay | P2 | S | A change to `ReplayBuffer`. |
| H4 | Actor-critic (PPO) | P2 | M | Continuous position sizing instead of discrete actions. |
| H5 | Ablation: interoception on against off | P1 | S | The spec's whole claim is that steering the trunk from outside beats concatenating the state onto the input. Train both with the same seeds and the same shocks, and report the difference. |
| H6 | CUDA trunk | P3 | M | Only if training time becomes the bottleneck. The networks are small, so the CPU and rayon may win. |

---

## Track I — Metal

`metal_training` covers dense networks only.

| # | Item | Priority | Effort |
|---|---|---|---|
| I1 | Transformer forward and inference on Metal | P3 | L |
| I2 | Transformer training on Metal | P3 | L |

Park both unless a Mac user asks. Every CUDA kernel would need a Metal twin,
and that is the structural tax at its worst.

---

## Track J — Engineering and quality

| # | Item | Priority | Effort | Notes |
|---|---|---|---|---|
| J1 | Throughput regression check | P1 | S | Write a script that runs `bench_train_step` and `bench_decode` and compares the result against the numbers in `docs/baseline.md`. Fail if a result is more than 3% slower. Run it before every release. CI has no GPU, so run it locally. |
| J2 | Parity tests in CI | P1 | S | Check that `ci.yml` runs `cargo test`, `clippy -D warnings`, the MSRV build (1.85) and `cargo doc`. The CUDA tests skip without a device. Make sure they skip and do not pass silently. |
| J3 | `cargo semver-checks` in CI | P2 | S | 2.0 broke the API once. Make the next break a choice. |
| J4 | Fuzz the file readers | P2 | S | `safetensors`, `gltf`, `npy`, `idx`, `tokenizer.json` and the binary checkpoints all read untrusted bytes. Use `cargo fuzz` on each header parser. A malformed file must return an error and must not panic or allocate without a limit. |
| J5 | Stale-binary guard | P1 | S | W0 in `improvement-plan.md`. A trainer that was nine hours stale cost 1.63x. Check whether the guard landed. |
| J6 | Kernel source of truth | P3 | L | `featuresPlan.md` asks whether a shared kernel description pays for itself. Decide when the fourth new layer type needs two hand-written backward passes. Until then, accept the tax. |
| J7 | Tutorials | P2 | S each | One chapter each for LoRA fine-tuning, `TokenFile` pretraining end to end, the image pipeline, and the interoceptive agent. |
| J8 | `ponytail:` debt review | P2 | S | The tree has about 40 `ponytail:` comments. Each one names a known limit. Review them once per release, and move the ones that now matter into this file. |

---

## Suggested order

```
2.1  (now)        2.1.1-2.1.6, H5, J1, J5
2.2  (fine-tune)  F1, F2, F5, A3, B5, B4, B6 (RoPE scaling), E1
2.3  (speed)      A1, A6, A7, A10, C2, C5, C1
2.4  (scale)      B1, B2, A8, A11, E2, E7
2.5  (images)     A4, A5, C6, G1, G3, D5
later             everything at P2/P3, re-ranked by what users ask for
```

Why this order:

- **2.1** ships what already exists and fixes correctness problems.
- **2.2** is the largest gain for a one-card user: take a published model,
  fine-tune it with LoRA, and export it to the Hub. This uses code that
  already exists (`TextEncoder`, LoRA, `SafeTensors`).
- **2.3** goes after the measured remainder of the profile: the flash kernels
  are 19% of a step and the LM head is 13%. It also makes inference worth
  using.
- **2.4** is the multi-GPU work. It is large, and it pays off only for users
  with two cards.
- **2.5** gives the image path the same treatment the text path got.

## Decisions that need an owner

1. **Encoder-decoder (G7):** keep "not planned" in the README, or reopen it?
2. **DPO and RL fine-tuning (B8):** the interoceptive module changes the
   argument in `featuresPlan.md`. Decide whether alignment belongs in the crate.
3. **Vocabulary size (W8):** a 15% gain, but it is a model decision and not a
   kernel decision.
4. **N-D tensor type:** a 3.0 breaking change, or never? It is the
   precondition for J6 and for a cleaner convolution path.
5. **Parquet (E2):** add the `arrow`/`parquet` dependency tree behind a
   feature flag, or write a minimal reader?

## Not doing

These are the same as `featuresPlan.md` "Not on this list", and they still
hold:

- Serving, batching servers and HTTP APIs. ONNX, GGUF and safetensors export
  are the seam.
- A Python binding.
- An autograd engine. This is a rewrite and not a feature (see J6).
- Tensor parallelism, pipeline parallelism and multi-node, until B1 works on
  two cards.
