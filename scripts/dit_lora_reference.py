"""Writes a tiny random diffusers FLUX transformer, PEFT adapters on every
block projection, and the gradients PyTorch takes for those adapters through
one forward pass and a fixed loss.

The loss is `sum(output * weights)` with fixed random weights, so the
gradient of the output is `weights` and nothing depends on a training
recipe. Run with an environment that has torch, diffusers and peft, then
`RB_DIT_LORA_REF=<out> cargo test --lib dit_lora_gradients_match_pytorch -- --ignored`.
"""
import json
import sys

import torch
from diffusers import FluxTransformer2DModel
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import save_file

out = sys.argv[1] if len(sys.argv) > 1 else "."
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
model = FluxTransformer2DModel(
    patch_size=1,
    in_channels=4,
    num_layers=2,
    num_single_layers=2,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=12,
    pooled_projection_dim=6,
    guidance_embeds=False,
    axes_dims_rope=(2, 2, 4),
).double()
with torch.no_grad():
    for param in model.parameters():
        param.copy_(torch.randn_like(param) * 0.2)
model.float().save_pretrained(f"{out}/transformer", safe_serialization=True)
model = model.double()

suffixes = (
    "to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj",
    "to_add_out", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj",
    "ff_context.net.2", "proj_mlp", "proj_out",
)
# Full names, so the model's last `proj_out`, which is not in a block, is left
# alone.
targets = [
    name
    for name, _ in model.named_modules()
    if "blocks." in name and name.endswith(suffixes)
]
model.add_adapter(LoraConfig(r=4, lora_alpha=8, target_modules=targets))
# PEFT starts `lora_B` at zero, which would leave every `lora_A` gradient at
# zero too, so both factors are drawn at random.
with torch.no_grad():
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.copy_(torch.randn_like(param) * 0.3)
adapters = {
    key: value.float().contiguous() for key, value in get_peft_model_state_dict(model).items()
}
save_file(adapters, f"{out}/adapters.safetensors")

height, width, text = 3, 4, 5
patches = torch.randn(1, height * width, 4, dtype=torch.float64)
states = torch.randn(1, text, 12, dtype=torch.float64)
pooled = torch.randn(1, 6, dtype=torch.float64)
timestep = torch.tensor([0.7], dtype=torch.float64)
image_ids = torch.zeros(height * width, 3, dtype=torch.float64)
image_ids[:, 1] = torch.arange(height).repeat_interleave(width)
image_ids[:, 2] = torch.arange(width).repeat(height)
output = model(
    hidden_states=patches,
    encoder_hidden_states=states,
    pooled_projections=pooled,
    timestep=timestep,
    img_ids=image_ids,
    txt_ids=torch.zeros(text, 3, dtype=torch.float64),
    return_dict=False,
)[0]
weights = torch.randn_like(output)
(output * weights).sum().backward()

gradients = {}
for name, param in model.named_parameters():
    if "lora_" in name:
        key = name.replace(".default", "")
        gradients[key] = param.grad.flatten().tolist()
assert set(gradients) == set(adapters), set(gradients) ^ set(adapters)

json.dump(
    {
        "height": height,
        "width": width,
        "timestep": timestep.item(),
        "scale": 8 / 4,
        "patches": patches.flatten().tolist(),
        "text": states.flatten().tolist(),
        "pooled": pooled.flatten().tolist(),
        "output": output.flatten().tolist(),
        "weights": weights.flatten().tolist(),
        "gradients": gradients,
    },
    open(f"{out}/reference.json", "w"),
)
print(len(gradients), "adapter tensors;", sorted(gradients)[:4])
