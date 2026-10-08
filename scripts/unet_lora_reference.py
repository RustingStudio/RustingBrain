"""Writes a tiny random diffusers UNet, a LoRA for it in the diffusers and the
kohya spellings, and one forward pass of the UNet with the LoRA merged by PEFT.

Run with an environment that has torch, diffusers and peft, then
`RB_LORA_REF=<out> cargo test --lib a_lora_file_merges_as_peft_does -- --ignored`.
"""
import json
import sys

import torch
from diffusers import DDIMScheduler, UNet2DConditionModel
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import save_file

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
with torch.no_grad():
    for param in unet.parameters():
        param.copy_(torch.randn_like(param) * 0.1)
unet.save_pretrained(f"{out}/unet", safe_serialization=True)

rank, alpha = 4, 8
unet.add_adapter(
    LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "conv1", "ff.net.2"],
    )
)
with torch.no_grad():
    for name, param in unet.named_parameters():
        if "lora_" in name:
            param.copy_(torch.randn_like(param) * 0.1)
peft = {k: v.contiguous() for k, v in get_peft_model_state_dict(unet).items()}
# Diffusers' own spelling: the PEFT names under `unet.`, with the strength
# left at alpha / rank by writing alpha next to each pair.
diffusers = {f"unet.{k}": v for k, v in peft.items()}
kohya = {}
for name, value in peft.items():
    stem, kind = name.rsplit(".lora_", 1)
    flat = "lora_unet_" + stem.replace(".", "_")
    kohya[f"{flat}.lora_{'down' if kind.startswith('A') else 'up'}.weight"] = value
    kohya[f"{flat}.alpha"] = torch.tensor(float(alpha))
    diffusers[f"unet.{stem}.alpha"] = torch.tensor(float(alpha))
# A text encoder adapter, which a UNet load skips.
kohya["lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"] = torch.randn(rank, 8)
kohya["lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight"] = torch.randn(8, rank)
# kohya's SDXL spelling: the original UNet's block names, here laid out for
# this UNet's two down and two up blocks of two layers.
blocks = {
    "down_blocks_0_attentions_0": "input_blocks_1_1",
    "down_blocks_0_attentions_1": "input_blocks_2_1",
    "down_blocks_0_resnets_0_conv1": "input_blocks_1_0_in_layers_2",
    "down_blocks_0_resnets_1_conv1": "input_blocks_2_0_in_layers_2",
    "down_blocks_1_resnets_0_conv1": "input_blocks_4_0_in_layers_2",
    "down_blocks_1_resnets_1_conv1": "input_blocks_5_0_in_layers_2",
    "mid_block_attentions_0": "middle_block_1",
    "mid_block_resnets_0_conv1": "middle_block_0_in_layers_2",
    "mid_block_resnets_1_conv1": "middle_block_2_in_layers_2",
}
for layer in range(3):
    blocks[f"up_blocks_0_resnets_{layer}_conv1"] = f"output_blocks_{layer}_0_in_layers_2"
    blocks[f"up_blocks_1_resnets_{layer}_conv1"] = f"output_blocks_{layer + 3}_0_in_layers_2"
    blocks[f"up_blocks_1_attentions_{layer}"] = f"output_blocks_{layer + 3}_1"
original = {}
for name, value in kohya.items():
    for diffusers_name, ldm in blocks.items():
        name = name.replace(f"_{diffusers_name}_", f"_{ldm}_").replace(f"_{diffusers_name}.", f"_{ldm}.")
    original[name] = value
# Diffusers' own reading of the original names must land on the same modules.
from diffusers.loaders.lora_conversion_utils import (
    _convert_unet_lora_key,
    _maybe_map_sgm_blocks_to_diffusers,
)


def modules(names):
    return sorted({_convert_unet_lora_key(n.split(".")[0]) for n in names if "unet" in n})


assert modules(_maybe_map_sgm_blocks_to_diffusers(dict(original), unet.config)) == modules(kohya)
save_file(original, f"{out}/kohya_sdxl_lora.safetensors")
save_file(diffusers, f"{out}/diffusers_lora.safetensors")
save_file(kohya, f"{out}/kohya_lora.safetensors")

latent, text = torch.randn(1, 4, 8, 8), torch.randn(1, 5, 16)
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)
step, strength = 500, 0.6
alpha_bar = scheduler.alphas_cumprod[step]
sigma = float(((1 - alpha_bar) / alpha_bar).sqrt())
scaled = latent * (sigma**2 + 1) ** -0.5
unet.fuse_lora(lora_scale=strength)
with torch.no_grad():
    sample = unet(scaled, step, encoder_hidden_states=text).sample
json.dump(
    {
        "latent": latent.flatten().tolist(),
        "text": text.flatten().tolist(),
        "sigma": sigma,
        "scale": strength,
        "out": sample.flatten().tolist(),
    },
    open(f"{out}/ref.json", "w"),
)
