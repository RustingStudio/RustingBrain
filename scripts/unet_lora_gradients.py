"""Writes a tiny random diffusers UNet, PEFT adapters on every attention
projection, and the gradients PyTorch takes for those adapters through one
forward pass and a fixed loss.

The loss is `sum(output * weights)` with fixed random weights, so the
gradient of the output is `weights`. GEGLU's gate runs through the tanh
GELU, which is the one `unet.rs` computes, so the comparison is of the
backward pass alone. Run with an environment that has torch, diffusers and
peft, then
`RB_UNET_LORA_REF=<out> cargo test --lib unet_lora_gradients_match_pytorch -- --ignored`.
"""
import json
import sys

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DConditionModel
from diffusers.models.activations import GEGLU
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import save_file

GEGLU.gelu = lambda self, gate: F.gelu(gate, approximate="tanh")

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
).double()
with torch.no_grad():
    for param in unet.parameters():
        param.copy_(torch.randn_like(param) * 0.2)
unet.float().save_pretrained(f"{out}/unet", safe_serialization=True)
unet = unet.double()

rank, alpha = 4, 8
unet.add_adapter(
    LoraConfig(r=rank, lora_alpha=alpha, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
)
# PEFT starts `lora_B` at zero, which would leave every `lora_A` gradient at
# zero too, so both factors are drawn at random.
with torch.no_grad():
    for name, param in unet.named_parameters():
        if "lora_" in name:
            param.copy_(torch.randn_like(param) * 0.3)
adapters = {k: v.float().contiguous() for k, v in get_peft_model_state_dict(unet).items()}
save_file(adapters, f"{out}/adapters.safetensors")

latent, text = torch.randn(1, 4, 8, 8, dtype=torch.float64), torch.randn(1, 5, 16, dtype=torch.float64)
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)
step = 500
alpha_bar = scheduler.alphas_cumprod[step].double()
sigma = float(((1 - alpha_bar) / alpha_bar).sqrt())
scaled = latent * (sigma**2 + 1) ** -0.5
output = unet(scaled, step, encoder_hidden_states=text).sample
weights = torch.randn_like(output)
(output * weights).sum().backward()
gradients = {
    name.replace(".default", ""): param.grad.flatten().tolist()
    for name, param in unet.named_parameters()
    if "lora_" in name
}
json.dump(
    {
        "latent": latent.flatten().tolist(),
        "text": text.flatten().tolist(),
        "sigma": sigma,
        "scale": alpha / rank,
        "output": output.flatten().tolist(),
        "weights": weights.flatten().tolist(),
        "gradients": gradients,
    },
    open(f"{out}/reference.json", "w"),
)
print(len(gradients), "adapter gradients written")
