# maskvar

## Environment Setup

```
conda env create -f environment.yml
```

## Code Structure

| important file |  |
|----------------|----------|
| maskseg_build_everything.py | configuration builder/manager |
| maskvar_train.py | train script (adpated from VAR) |
| mskvar_trainer.py | trainer (adapted from VAR) |
| models/flex_maskvar.py | maskvar backbone (w/ flex attn) |
| models/maskvar.py | maskvar backbone (w/o flex attn) |
| models/vqvae_single.py | single channel VQVAE implementation (adapted from VAR) |
| models/image_encoder.py | image feature adaptor/projector |

Dataset implementation: `datasets/mask_level_dataset.py`

From SAM:

| file |  |
|-----|---|
| models/sam_image_encoder.py | image encoder (from SAM) |
| models/prompt_encoder.py | prompt encoder (from SAM) |
| models/positional_embedding_random.py | PE (from SAM) |
