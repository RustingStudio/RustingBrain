"""Writes what transformers' Dinov2Model makes of two 224x224 images, for
`vit_encoder::tests::dinov2_matches_transformers`.

Needs facebook/dinov2-small and facebook/dinov2-base in the Hugging Face cache.
Run with an environment that has torch and transformers, then
`RB_DINOV2_REF=<out> cargo test --lib dinov2_matches_transformers -- --ignored`.
"""
import json
import os
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import Dinov2Model

out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
# Smooth images rather than noise, so attention has structure to find.
grid = torch.linspace(0, 1, 224)
base = torch.stack([grid[None, :].expand(224, 224), grid[:, None].expand(224, 224),
                    (grid[None, :] * grid[:, None])])
pixels = torch.stack([base, base.flip(-1) * 0.5 + torch.rand(3, 224, 224) * 0.5])
mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
pixels = (pixels - mean) / std
paths = {}
for name in ["small", "base"]:
    path = snapshot_download(f"facebook/dinov2-{name}", local_files_only=True,
                             allow_patterns=["*.json", "*.safetensors"])
    model = Dinov2Model.from_pretrained(path).eval()
    with torch.no_grad():
        tokens = model(pixel_values=pixels).last_hidden_state
    save_file({"pixels": pixels.contiguous(), "tokens": tokens.contiguous()},
              f"{out}/{name}.safetensors")
    paths[name] = f"{path}/model.safetensors"
    print(name, list(tokens.shape), flush=True)
json.dump(paths, open(f"{out}/reference.json", "w"))
os._exit(0)
