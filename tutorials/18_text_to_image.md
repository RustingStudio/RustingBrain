# 18. Generating Images From Text

**You need:** chapters 1–10, chapter 11 if you have a GPU, and a published
diffusion model on disk.

**Time:** 45 minutes, plus the download.

**Full code:** [`code/18_text_to_image.rs`](code/18_text_to_image.rs)

Every chapter so far trained its own model. This one runs somebody else's. A
text-to-image model is three networks trained together: a text encoder that
reads the prompt, a denoiser that turns noise into a picture a step at a time,
and an autoencoder (the VAE) whose decoder turns the denoiser's small latent
into pixels. The library loads all three from the directory a Hugging Face
diffusers repository ships, and nothing about the model is written in the
code.

---

## 18.1 Get a model

Any diffusers-format directory works: Stable Diffusion 1.x, 2.x and XL (a
UNet denoiser), Stable Diffusion 3 and FLUX (a transformer denoiser). The
directory holds one folder per part:

```text
scheduler/  text_encoder/  text_encoder_2/  tokenizer/  tokenizer_2/  unet/  vae/
```

Download one with the Hugging Face CLI, for example:

```bash
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
    --include "*.json" "*.txt" "*.safetensors" --exclude "*fp16*" \
    --local-dir models/stable-diffusion-xl-base-1.0
```

Build with the `images` feature, which writes PNGs, and with `cuda` if you
have an NVIDIA card:

```toml
rusting_brain = { version = "3", features = ["images", "cuda"] }
```

## 18.2 Load

```rust
let mut pipeline = ImagePipeline::load_at(&directory, Precision::Q8)?;
pipeline.try_cuda(0);
```

`load_at` reads every configuration file, decides which encoders and which
denoiser the model has, and loads the weights. `Precision::Q8` keeps the
encoders and the denoiser at one byte per weight, a quarter of what `f32`
takes. That is what fits SDXL's 2.6 billion UNet weights, or a 4B FLUX
transformer, on a 12 GB card. The VAE decoder stays in `f32`: it is small and
it is what the eye sees.

`try_cuda` moves the UNet, both halves of the VAE, the CLIP towers and, on a
FLUX or SD3 model, the transformer and the T5 encoder onto the device. It returns
`false` and leaves everything on the CPU if there is no device or the model
does not fit, so the same program runs on both. On the CPU a 1024x1024 SDXL
image takes many minutes; on the GPU it takes seconds.

## 18.3 Steps, solver and guidance

`pipeline.config` holds what the model's own `scheduler_config.json` asked
for:

| Field | What it does | SDXL default |
|---|---|---|
| `steps` | How many denoising steps | 50 |
| `solver` | How each step is taken | `Euler` |
| `guidance` | Classifier-free guidance scale | 7.5 |

Change them before sampling:

```rust
pipeline.config.solver = Solver::DpmPlusPlus2m;
pipeline.config.steps = 25;
```

DPM++ 2M is a second-order solver that reuses the previous step's prediction,
so a step costs what an Euler step costs and 20 to 30 steps look like 50
Euler steps. `Solver::EulerAncestral` adds fresh noise each step: more
variety between seeds, more steps to settle.

A distilled model (FLUX schnell or klein, an LCM) is built for few steps and
no guidance. Leave its `guidance` at 1.0 and its `steps` where the model put
them.

## 18.4 Guidance and the negative prompt

```rust
let conditioning = pipeline.condition(&prompt, 1024, 1024)?;
let unconditional = Some(pipeline.condition("blurry, low quality", 1024, 1024)?);
let latent = pipeline.sample_latent_guided(conditioning, unconditional, Some(7), |step, _| {
    println!("step {step}");
    true
})?;
```

Classifier-free guidance runs the denoiser twice a step, once with the prompt
and once with a second conditioning, and steps along the difference, scaled
by `guidance`. The second conditioning is usually the empty prompt. Put words
in it and the image is pushed away from them: that is a negative prompt.
Guidance above one therefore costs two denoiser passes a step.

The seed decides the starting noise. The same seed, prompt, size and settings
give the same image. The callback sees every step; return `false` from it to
stop early, which is what a cancel button needs.

The image's sides must be multiples of the decoder's upscale times the patch
size (eight for SD and SDXL, sixteen for FLUX). `patch_grid` checks it and
says so.

Finally decode and write:

```rust
let (columns, rows) = pipeline.patch_grid(1024, 1024)?;
let image = pipeline.decode(&latent, columns, rows)?;
save_png(&image, "image.png")?;
```

`decode` returns a `FeatureMap` of three channels in `[-1, 1]`;
`vae::to_rgb8` gives the bytes if you want to write them yourself.

`ImagePipeline::generate(prompt, width, height, seed, callback)` does all of
18.4 in one call, with the empty prompt as the negative one.

## 18.5 Image-to-image

```rust
let repainted = pipeline.generate_from_image("..., oil painting", &image, 0.5, Some(7), |_, _| true)?;
```

The VAE's encoder turns the picture into a latent, noise is added up to a
point on the schedule, and only the rest of the schedule is run. `strength`
picks that point: 1.0 throws the picture away and starts from pure noise, 0.0
returns it untouched, and 0.3 to 0.8 keeps the composition and repaints the
detail. It costs `strength` times the steps, plus one pass of the encoder.

## 18.6 The script

[`code/18_text_to_image.rs`](code/18_text_to_image.rs) does 18.2 to 18.5 at
1024x1024. On an RTX 3060 with a Stable Diffusion XL model (Pony Diffusion V6
XL, int8 weights), it printed:

```text
loaded in 7.4s
running on the GPU
model asks for 50 steps, guidance 7.5, Euler
step 25/25
1024x1024 in 13.2s
wrote image.png
repainted in 6.9s
wrote repainted.png
```

The prompt was `"score_9, a lighthouse on a rocky coast in fog, dramatic
light"` (`score_9` is a tag Pony was trained to read). The 13.2 seconds are
25 steps at two UNet passes each, plus the decode. Q8 weights and a card
shared with another job both cost some of it. Euler at its own 50 steps took
about 45 seconds for a picture no better.

The repaint redoes half the schedule, so it costs about half the first run.
The VAE encoder takes a quarter of a second of it on the device; on the CPU
the same 1024x1024 encode takes close to two minutes.

## 18.7 When it goes wrong

- **Out of device memory.** Use `Precision::Q8`. Close whatever else holds the
  card (`nvidia-smi` lists it). A smaller image needs less.
- **A black or NaN image from SDXL's VAE on the device.** The device decoder
  runs in FP16, and the original SDXL VAE can overflow it. `decode` returns an
  error instead of a black picture; use the fp16-fix VAE, or run the decoder
  on the host.
- **A very large image does not fit in the decoder.**
  `pipeline.decoder.decode_tiled(&latent_map, 64, 8)` decodes overlapping
  latent tiles and blends the seams.
- **A distilled model looks burnt or noisy.** Its guidance should be 1.0 and
  its step count low; check what you changed in `config`.

---

## Exercises

1. Run the same prompt with seeds 1 to 4. Then run seed 1 four times. What
   changes and what does not?
2. Set `steps` to 10, 25 and 50 with DPM++ 2M, then with Euler. At what step
   count does each stop improving?
3. Set `guidance` to 1.0, 4.0 and 12.0. Describe what happens at each end.
   Time a run at 1.0 against one at 7.5.
4. Repaint with `strength` 0.2, 0.5 and 0.8. Which keeps the layout?
5. Put the thing you see most often in the first image into the negative
   prompt and run again with the same seed.

---

## Recap

- A text-to-image model is a text encoder, a denoiser and a VAE decoder;
  `ImagePipeline::load_at` reads all three from a diffusers directory.
- `Precision::Q8` makes large models fit; `try_cuda` moves them to the GPU and
  falls back to the CPU on its own.
- `config.steps`, `config.solver` and `config.guidance` come from the model
  and can be overridden. DPM++ 2M needs about half the steps of Euler.
- Guidance runs the denoiser twice a step and steps away from the negative
  prompt.
- `generate_from_image` with a `strength` between 0.3 and 0.8 repaints a
  picture and keeps its layout.

**Back to:** [the index](README.md)
