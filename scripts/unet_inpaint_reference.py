"""Writes a tiny random diffusers inpainting UNet and one forward pass of it.

Run with an environment that has torch and diffusers, then
`RB_UNET_REF=<out> cargo test --lib an_inpainting_model_matches_diffusers -- --ignored`.
"""
import json
import sys

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel

out = sys.argv[1] if len(sys.argv) > 1 else "."
torch.manual_seed(0)
model = UNet2DConditionModel(
    sample_size=8,
    in_channels=9,
    out_channels=4,
    block_out_channels=(32, 64),
    layers_per_block=2,
    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    cross_attention_dim=16,
    attention_head_dim=8,
    norm_num_groups=8,
)
with torch.no_grad():
    for param in model.parameters():
        param.copy_(torch.randn_like(param) * 0.1)
model.save_pretrained(f"{out}/unet", safe_serialization=True)

latent, extra, text = torch.randn(1, 4, 8, 8), torch.randn(1, 5, 8, 8), torch.randn(1, 5, 16)
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)
step = 500
alpha = scheduler.alphas_cumprod[step]
sigma = float(((1 - alpha) / alpha).sqrt())
# The scheduler scales the noisy latent alone; the mask and masked picture
# go in as they are.
scale = (sigma**2 + 1) ** -0.5
with torch.no_grad():
    sample = model(torch.cat([latent * scale, extra], 1), step, encoder_hidden_states=text).sample
json.dump(
    {
        "latent": latent.flatten().tolist(),
        "extra": extra.flatten().tolist(),
        "text": text.flatten().tolist(),
        "sigma": sigma,
        "out": sample.flatten().tolist(),
    },
    open(f"{out}/ref.json", "w"),
)
