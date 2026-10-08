"""Writes two tiny random CLIP text towers, textual inversion files in the
diffusers, A1111 and SDXL layouts, and what the towers make of a prompt that
uses them once diffusers has loaded them.

The tokenizers are Stable Diffusion XL's, so the script needs that model in the
Hugging Face cache. Run with an environment that has torch, transformers and
diffusers, then
`RB_TI_REF=<out> cargo test --lib textual_inversion_matches_diffusers -- --ignored`.
"""
import json
import sys

import torch
from diffusers.loaders import TextualInversionLoaderMixin
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer, CLIPTokenizerFast

out = sys.argv[1] if len(sys.argv) > 1 else "."
sdxl = snapshot_download(
    "stabilityai/stable-diffusion-xl-base-1.0",
    allow_patterns=["tokenizer/*", "tokenizer_2/*"],
    local_files_only=True,
)
torch.manual_seed(0)


def tower(name, width, eos):
    # `eos_token_id = 2` is the legacy configuration, which pools at the
    # highest id rather than at the end token, so the two towers between them
    # cover both rules.
    config = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=width,
        intermediate_size=2 * width,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        bos_token_id=0 if eos == 2 else 49406,
        eos_token_id=eos,
    )
    model = CLIPTextModel(config).eval()
    # Saved by hand with the `text_model.` prefix the published checkpoints
    # carry, which Transformers 5 leaves off.
    config.save_pretrained(f"{out}/{name}")
    state = {f"text_model.{key}": value.contiguous() for key, value in model.state_dict().items()}
    save_file(state, f"{out}/{name}/model.safetensors")
    tokenizer = CLIPTokenizer.from_pretrained(f"{sdxl}/{name}")
    CLIPTokenizerFast.from_pretrained(f"{sdxl}/{name}").backend_tokenizer.save(
        f"{out}/{name}/tokenizer.json"
    )
    return tokenizer, model


first, first_model = tower("tokenizer", 32, 2)
second, second_model = tower("tokenizer_2", 48, 49407)

files = {
    "diffusers.safetensors": {"<cat-toy>": torch.randn(32)},
    "a1111.safetensors": {"emb_params": torch.randn(3, 32)},
    "sdxl.safetensors": {"clip_l": torch.randn(2, 32), "clip_g": torch.randn(2, 48)},
}
for name, tensors in files.items():
    save_file(tensors, f"{out}/{name}")


class Loader(TextualInversionLoaderMixin):
    hf_device_map = None
    components = {}


loader = Loader()
loader.load_textual_inversion(
    f"{out}/diffusers.safetensors",
    tokenizer=first,
    text_encoder=first_model,
)
loader.load_textual_inversion(
    f"{out}/a1111.safetensors", token="<style>", tokenizer=first, text_encoder=first_model
)
for key, tokenizer, model in [("clip_l", first, first_model), ("clip_g", second, second_model)]:
    loader.load_textual_inversion(
        files["sdxl.safetensors"][key], token="<sdxl>", tokenizer=tokenizer, text_encoder=model
    )

prompt = "a <cat-toy> painted in <style>, <sdxl> lighting"
reference = {"prompt": prompt}
for name, tokenizer, model in [("tokenizer", first, first_model), ("tokenizer_2", second, second_model)]:
    ids = tokenizer(
        loader.maybe_convert_prompt(prompt, tokenizer),
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids
    with torch.no_grad():
        output = model(ids)
    reference[name] = {
        "ids": ids[0].tolist(),
        "hidden": output.last_hidden_state[0].flatten().tolist(),
        "pooled": output.pooler_output[0].tolist(),
    }
json.dump(reference, open(f"{out}/reference.json", "w"))
print({name: reference[name]["ids"][:12] for name in ["tokenizer", "tokenizer_2"]})
