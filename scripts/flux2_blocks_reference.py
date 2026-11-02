"""Step-0 prediction of FLUX.2-klein-4B's transformer cut down to its first
two double and first two single blocks, in F32, for
`pipeline::tests::flux2_klein_blocks_match_diffusers`.

The whole transformer in F32 needs about 16 GB on our side, more than the
machine can spare, and Q8 alone does not say whether a gap is rounding or a
bug. The cut-down model runs both kinds of block, the shared modulations and
every embedder on real weights, in about 3 GB.

Reads `flux2.safetensors` from `scripts/flux2_reference.py` for the prompt
embeddings and the noise. Run with
`python scripts/flux2_blocks_reference.py <model dir> <ref dir>`.
"""
import os
import sys

import torch
from diffusers import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline as P
from safetensors.torch import load_file, save_file

model, out = sys.argv[1], sys.argv[2]
torch.set_grad_enabled(False)
ref = load_file(f"{out}/flux2.safetensors")
transformer = Flux2Transformer2DModel.from_pretrained(
    f"{model}/transformer", num_layers=2, num_single_layers=2,
    torch_dtype=torch.bfloat16).float().eval()
noise = ref["noise"][None]
embeds = ref["prompt_embeds"][None]
pred = transformer(hidden_states=P._pack_latents(noise), timestep=torch.ones(1),
                   guidance=None, encoder_hidden_states=embeds,
                   txt_ids=P._prepare_text_ids(embeds), img_ids=P._prepare_latent_ids(noise),
                   return_dict=False)[0]
dumps = {"step0_pred_blocks": pred[0].contiguous()}
if "ref_latents" in ref:
    # An edit run (scripts/flux2_reference.py with REF_IMAGE): the same cut-down
    # model with the reference's tokens and ids after the image's.
    tokens, ids = ref["ref_latents"][None], ref["ref_ids"][None]
    pred = transformer(hidden_states=torch.cat([P._pack_latents(noise), tokens], dim=1),
                       timestep=torch.ones(1), guidance=None, encoder_hidden_states=embeds,
                       txt_ids=P._prepare_text_ids(embeds),
                       img_ids=torch.cat([P._prepare_latent_ids(noise), ids], dim=1),
                       return_dict=False)[0][:, :1024]
    dumps["step0_pred_blocks_ref"] = pred[0].contiguous()
save_file(dumps, f"{out}/flux2_blocks.safetensors")
print("wrote", out, flush=True)
os._exit(0)
