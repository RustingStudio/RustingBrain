"""Checks that diffusers loads a LoRA `train_dit_lora` wrote.

    python dit_lora_trained_check.py prepare <model dir> <pictures dir>
        swaps the model's transformer for a random diffusers one of the same
        shape, and writes a few captioned 64x64 pictures to train on.
    python dit_lora_trained_check.py check <model dir> <lora file> <out dir>
        loads the model's transformer and the LoRA with diffusers, and writes
        the prediction with and without it on fixed inputs to
        `<out dir>/reference.json`, for
        `RB_DIT_LORA_TRAINED=<out dir> cargo test --lib a_trained_lora_loads_as_diffusers_loads_it -- --ignored`.

The model is `pipeline::tests::publish_a_tiny_flux`'s. Needs torch, diffusers,
peft and Pillow.
"""
import json
import os
import sys

import torch
from diffusers import FluxTransformer2DModel

if sys.argv[1] == "prepare":
    from PIL import Image

    _, _, model_dir, out = sys.argv
    config = json.load(open(f"{model_dir}/transformer/config.json"))
    torch.manual_seed(0)
    model = FluxTransformer2DModel(
        patch_size=1,
        in_channels=config["in_channels"],
        num_layers=config["num_layers"],
        num_single_layers=config["num_single_layers"],
        attention_head_dim=config["attention_head_dim"],
        num_attention_heads=config["num_attention_heads"],
        joint_attention_dim=config["joint_attention_dim"],
        pooled_projection_dim=config["pooled_projection_dim"],
        guidance_embeds=False,
        axes_dims_rope=tuple(config["axes_dims_rope"]),
    )
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.05)
    for name in ["config.json", "diffusion_pytorch_model.safetensors"]:
        os.remove(f"{model_dir}/transformer/{name}")
    model.save_pretrained(f"{model_dir}/transformer", safe_serialization=True)
    for index, (colour, caption) in enumerate(
        [((200, 40, 40), "a"), ((40, 200, 40), "b"), ((40, 40, 200), "c")]
    ):
        Image.new("RGB", (64, 64), colour).save(f"{out}/{index}.png")
        open(f"{out}/{index}.txt", "w").write(caption)
    sys.exit()

_, _, model_dir, lora, out = sys.argv
model = FluxTransformer2DModel.from_pretrained(f"{model_dir}/transformer")
config = model.config
model = model.double()

torch.manual_seed(0)
height, width, text = 3, 4, 5
patches = torch.randn(1, height * width, config["in_channels"], dtype=torch.float64)
states = torch.randn(1, text, config["joint_attention_dim"], dtype=torch.float64)
pooled = torch.randn(1, config["pooled_projection_dim"], dtype=torch.float64)
timestep = torch.tensor([0.6], dtype=torch.float64)
image_ids = torch.zeros(height * width, 3, dtype=torch.float64)
image_ids[:, 1] = torch.arange(height).repeat_interleave(width)
image_ids[:, 2] = torch.arange(width).repeat(height)


def predict():
    with torch.no_grad():
        return model(
            hidden_states=patches,
            encoder_hidden_states=states,
            pooled_projections=pooled,
            timestep=timestep,
            img_ids=image_ids,
            txt_ids=torch.zeros(text, 3, dtype=torch.float64),
            return_dict=False,
        )[0]


plain = predict()
model.load_lora_adapter(lora, prefix="transformer", adapter_name="trained")
model = model.double()
loaded = sum(1 for name, _ in model.named_parameters() if "lora_B" in name)
adapted = predict()
print(loaded, "adapters loaded; the LoRA moves the prediction", (adapted - plain).norm().item())
json.dump(
    {
        "height": height,
        "width": width,
        "timestep": timestep.item(),
        "adapters": loaded,
        "model": os.path.abspath(model_dir),
        "lora": os.path.abspath(lora),
        "patches": patches.flatten().tolist(),
        "text": states.flatten().tolist(),
        "pooled": pooled.flatten().tolist(),
        "plain": plain.flatten().tolist(),
        "adapted": adapted.flatten().tolist(),
    },
    open(f"{out}/reference.json", "w"),
)
