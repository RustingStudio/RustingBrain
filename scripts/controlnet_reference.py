"""Writes a tiny random diffusers UNet and ControlNet and one guided forward pass.

Run with an environment that has torch and diffusers, then
`RB_CONTROLNET_REF=<out> cargo test --lib a_controlnet_matches_diffusers -- --ignored`.
"""
import json
import sys

import torch
from diffusers import ControlNetModel, DDIMScheduler, UNet2DConditionModel

out = sys.argv[1] if len(sys.argv) > 1 else "."
torch.manual_seed(0)
unet = UNet2DConditionModel(
    sample_size=8,
    in_channels=4,
    out_channels=4,
    block_out_channels=(32, 64),
    layers_per_block=2,
    down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
    cross_attention_dim=16,
    attention_head_dim=8,
    norm_num_groups=8,
)
controlnet = ControlNetModel.from_unet(unet, conditioning_embedding_out_channels=(8, 8, 16, 16))
with torch.no_grad():
    for model in (unet, controlnet):
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.1)
unet.save_pretrained(f"{out}/unet", safe_serialization=True)
controlnet.save_pretrained(f"{out}/controlnet", safe_serialization=True)

latent, text = torch.randn(1, 4, 8, 8), torch.randn(1, 5, 16)
image = torch.rand(1, 3, 64, 64)
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)
step, strength = 500, 0.7
alpha = scheduler.alphas_cumprod[step]
sigma = float(((1 - alpha) / alpha).sqrt())
scaled = latent * (sigma**2 + 1) ** -0.5
with torch.no_grad():
    down, middle = controlnet(
        scaled, step, encoder_hidden_states=text, controlnet_cond=image,
        conditioning_scale=strength, return_dict=False,
    )
    sample = unet(
        scaled, step, encoder_hidden_states=text,
        down_block_additional_residuals=down, mid_block_additional_residual=middle,
    ).sample
json.dump(
    {
        "latent": latent.flatten().tolist(),
        "image": image.flatten().tolist(),
        "text": text.flatten().tolist(),
        "sigma": sigma,
        "scale": strength,
        "out": sample.flatten().tolist(),
    },
    open(f"{out}/ref.json", "w"),
)
