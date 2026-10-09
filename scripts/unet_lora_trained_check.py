"""Checks that diffusers loads a LoRA `train_unet_lora` wrote.

    python unet_lora_trained_check.py prepare <model dir> <pictures dir>
        swaps the model's UNet for a random diffusers one of the same
        two-level shape, and writes a few captioned 64x64 pictures to train on.
    python unet_lora_trained_check.py check <model dir> <lora file> <out dir>
        loads the model's UNet and the LoRA with diffusers, and writes the
        prediction with and without it on fixed inputs to
        `<out dir>/reference.json`, for
        `RB_UNET_LORA_TRAINED=<out dir> cargo test --lib a_trained_unet_lora_loads_as_diffusers_loads_it -- --ignored`.

The model is `pipeline::tests::publish_a_tiny_sd`'s. GEGLU's gate runs
through the tanh GELU, the one `unet.rs` computes, so the comparison is of
the adapters alone. Needs torch, diffusers, peft and Pillow.
"""
import json
import os
import sys

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, UNet2DConditionModel
from diffusers.models.activations import GEGLU

GEGLU.gelu = lambda self, gate: F.gelu(gate, approximate="tanh")

if sys.argv[1] == "prepare":
    from PIL import Image

    _, _, model_dir, out = sys.argv
    text = json.load(open(f"{model_dir}/text_encoder/config.json"))
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        block_out_channels=(32, 64),
        layers_per_block=2,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=text["hidden_size"],
        attention_head_dim=8,
        norm_num_groups=8,
    )
    with torch.no_grad():
        for param in unet.parameters():
            param.copy_(torch.randn_like(param) * 0.05)
    for name in ["config.json", "diffusion_pytorch_model.safetensors"]:
        os.remove(f"{model_dir}/unet/{name}")
    unet.save_pretrained(f"{model_dir}/unet", safe_serialization=True)
    os.makedirs(out, exist_ok=True)
    for index, (colour, caption) in enumerate(
        [((200, 40, 40), "a"), ((40, 200, 40), "b"), ((40, 40, 200), "c")]
    ):
        Image.new("RGB", (64, 64), colour).save(f"{out}/{index}.png")
        open(f"{out}/{index}.txt", "w").write(caption)
    sys.exit()

_, _, model_dir, lora, out = sys.argv
unet = UNet2DConditionModel.from_pretrained(f"{model_dir}/unet")

torch.manual_seed(0)
latent = torch.randn(1, 4, 8, 8, dtype=torch.float64)
text = torch.randn(1, 5, unet.config.cross_attention_dim, dtype=torch.float64)
scheduler = DDIMScheduler(
    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
)
step = 500
alpha_bar = scheduler.alphas_cumprod[step].double()
sigma = float(((1 - alpha_bar) / alpha_bar).sqrt())
scaled = latent * (sigma**2 + 1) ** -0.5


def predict():
    with torch.no_grad():
        return unet(scaled, step, encoder_hidden_states=text).sample


unet = unet.double()
plain = predict()
unet.load_lora_adapter(lora, prefix="unet", adapter_name="trained")
unet = unet.double()
loaded = sum(1 for name, _ in unet.named_parameters() if "lora_B" in name)
adapted = predict()
print(loaded, "adapters loaded; the LoRA moves the prediction", (adapted - plain).norm().item())
json.dump(
    {
        "sigma": sigma,
        "adapters": loaded,
        "model": os.path.abspath(model_dir),
        "lora": os.path.abspath(lora),
        "latent": latent.flatten().tolist(),
        "text": text.flatten().tolist(),
        "plain": plain.flatten().tolist(),
        "adapted": adapted.flatten().tolist(),
    },
    open(f"{out}/reference.json", "w"),
)
