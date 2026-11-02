"""Writes what diffusers' Flux2KleinPipeline makes of one prompt at 512x512,
stage by stage, for `pipeline::tests::flux2_klein_matches_diffusers`.

The checkpoint is BF16. Every weight is upcast to F32 (exact) and every
operation runs in F32 on the CPU. The upcast happens one block at a time,
so the run needs about 13 GB, not the 32 GB a whole F32 model would need.
The stages run one model at a time with the pipeline's own helpers, which
is what `Flux2KleinPipeline.__call__` does, minus the dtype casts to the
models' storage type.

With `REF_IMAGE=<png>` the run is an edit: the image is a reference, as in
`Flux2KleinPipeline(image=...)`. It is resized to at most a megapixel and
cropped to a multiple of 32, encoded by the VAE (distribution mode),
patchified and batch-normalised, packed, and given time offset 10 in its ids;
its tokens follow the image's own through every step and their output is
dropped. Extra dumps: `ref_pixels` (the preprocessed reference, 3x512x512 in
[-1, 1]), `ref_latents` (its packed tokens), `ref_ids`, and `ids` (the ids of
text, image and reference tokens in the transformer's order). The output size is the reference's, and the prompt defaults to a watercolor
edit.

Run with an environment that has torch, diffusers and transformers:
`python scripts/flux2_reference.py <model dir> <out dir>`, then
`RB_FLUX2_REF=<out dir> cargo test --release --lib flux2_klein -- --ignored`.
"""
import gc
import json
import os
import sys

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2KleinPipeline as P, compute_empirical_mu, retrieve_timesteps)
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3ForCausalLM

model = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "."
ref_path = os.environ.get("REF_IMAGE")
prompt = os.environ.get("PROMPT", "turn this into a watercolor painting" if ref_path
                        else "a lighthouse on a rocky coast in fog")
size, seed, steps = 512, 0, 4
os.makedirs(out, exist_ok=True)
torch.set_grad_enabled(False)
dumps = {}


def f32_by_block(module, blocks):
    """Everything outside `blocks` in F32 for good; each block in F32 only
    while it runs, then back to BF16, which is lossless for BF16 weights."""
    inside = {id(p) for b in blocks for p in b.parameters()}
    for p in module.parameters():
        if id(p) not in inside:
            p.data = p.data.float()
    for b in blocks:
        b.register_forward_pre_hook(lambda m, a: (m.float(), None)[1])
        b.register_forward_hook(lambda m, a, o: (m.to(torch.bfloat16), None)[1])


# Text encoder: the pipeline's own prompt path, chat template and padding
# to 512 included.
tokenizer = AutoTokenizer.from_pretrained(f"{model}/tokenizer")
encoder = Qwen3ForCausalLM.from_pretrained(f"{model}/text_encoder", torch_dtype=torch.bfloat16).eval()
f32_by_block(encoder, encoder.model.layers)
text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False)
ids = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
embeds = P._get_qwen3_prompt_embeds(encoder, tokenizer, prompt, dtype=torch.float32,
                                    device=torch.device("cpu"))
dumps["input_ids"] = ids["input_ids"][0].to(torch.int32)
dumps["attention_mask"] = ids["attention_mask"][0].to(torch.int32)
dumps["prompt_embeds"] = embeds[0]
print("embeds", list(embeds.shape), "real tokens", int(ids["attention_mask"].sum()), flush=True)
del encoder
gc.collect()

# Denoiser and scheduler.
transformer = Flux2Transformer2DModel.from_pretrained(f"{model}/transformer",
                                                      torch_dtype=torch.bfloat16).eval()
f32_by_block(transformer, list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks))
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{model}/scheduler")
generator = torch.Generator("cpu").manual_seed(seed)
noise = torch.randn((1, 128, size // 16, size // 16), generator=generator, dtype=torch.float32)
latent_ids = P._prepare_latent_ids(noise)
latents = P._pack_latents(noise)
text_ids = P._prepare_text_ids(embeds)
model_ids = [text_ids[0], latent_ids[0]]
if ref_path:
    from PIL import Image
    from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
    processor = Flux2ImageProcessor(vae_scale_factor=16)
    img = Image.open(ref_path).convert("RGB")
    if img.size[0] * img.size[1] > 1024 * 1024:
        img = processor._resize_to_target_area(img, 1024 * 1024)
    w, h = (img.size[0] // 32) * 32, (img.size[1] // 32) * 32
    assert (w, h) == (size, size), f"the dumps assume a {size}x{size} reference, got {w}x{h}"
    pixels = processor.preprocess(img, height=h, width=w, resize_mode="crop")
    ref_vae = AutoencoderKLFlux2.from_pretrained(f"{model}/vae", torch_dtype=torch.float32).eval()
    ref_latent = P._encode_vae_image(type("S", (), {"vae": ref_vae, "_patchify_latents": staticmethod(P._patchify_latents)})(),
                                     pixels, generator)
    ref_ids = P._prepare_image_ids([ref_latent])
    ref_tokens = P._pack_latents(ref_latent)
    del ref_vae
    dumps["ref_pixels"] = pixels[0]
    dumps["ref_latents"] = ref_tokens[0]
    dumps["ref_ids"] = ref_ids[0].to(torch.int32)
    model_ids.append(ref_ids[0])
    img_ids = torch.cat([latent_ids, ref_ids], dim=1)
dumps["ids"] = torch.cat(model_ids, dim=0).to(torch.int32)
sigmas = np.linspace(1.0, 1 / steps, steps)
mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=steps)
timesteps, _ = retrieve_timesteps(scheduler, steps, "cpu", sigmas=sigmas, mu=mu)
scheduler.set_begin_index(0)
dumps["noise"] = noise[0]
dumps["timesteps"] = timesteps.float()
dumps["sigmas"] = scheduler.sigmas.float()
for i, t in enumerate(timesteps):
    inputs, input_ids = latents, latent_ids
    if ref_path:
        inputs, input_ids = torch.cat([latents, ref_tokens], dim=1), img_ids
    pred = transformer(hidden_states=inputs, timestep=t.expand(1) / 1000, guidance=None,
                       encoder_hidden_states=embeds, txt_ids=text_ids, img_ids=input_ids,
                       return_dict=False)[0][:, :latents.size(1)]
    if i == 0:
        dumps["step0_pred"] = pred[0]
    latents = scheduler.step(pred, t, latents, return_dict=False)[0]
    print("step", i, float(t), flush=True)
dumps["final_latents"] = latents[0]
del transformer
gc.collect()

# Decoder: batch-norm statistics undone, patches unpacked, then the VAE.
vae = AutoencoderKLFlux2.from_pretrained(f"{model}/vae", torch_dtype=torch.float32).eval()
grid = P._unpack_latents_with_ids(latents, latent_ids, size // 16, size // 16)
mean = vae.bn.running_mean.view(1, -1, 1, 1)
std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
vae_in = P._unpatchify_latents(grid * std + mean)
image = vae.decode(vae_in, return_dict=False)[0]
dumps["vae_latent"] = vae_in[0]
dumps["image"] = image[0]
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
Flux2ImageProcessor(vae_scale_factor=16).postprocess(image, output_type="pil")[0].save(f"{out}/diffusers.png")

save_file({k: v.contiguous() for k, v in dumps.items()}, f"{out}/flux2.safetensors")
json.dump({"model": model, "prompt": prompt, "size": size, "seed": seed, "steps": steps, "mu": mu},
          open(f"{out}/reference.json", "w"))
print("wrote", out, flush=True)
os._exit(0)
