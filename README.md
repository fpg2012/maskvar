# maskvar

## Environment Setup

Change directory to the project root and then

```
conda env create -f environment.yml
```

and then

```
pip install -e .
```

## Code Structure

| important scripts | |
|-------------------|---|
| `train_scripts/maskvar_train.py` | train script (adpated from VAR) |
| `train_scripts/maskvar_trainer.py` | trainer (adapted from VAR) |
| `train_scripts/train_vqvae_example.py` | train VQVAE  |

under `maskvar` directory:

| important file |  |
|----------------|----------|
| `maskseg_build_everything.py` | configuration builder/manager |
| `models/flex_maskvar.py` | maskvar backbone (w/ flex attn) |
| `models/maskvar.py` | maskvar backbone (w/o flex attn) |
| `models/vqvae_single.py` | single channel VQVAE implementation (adapted from VAR) |
| `models/image_encoder.py` | image feature adaptor/projector |

Dataset implementation: `datasets/mask_level_dataset.py`

From SAM: `models/sam`

## Dataset Placement

```
data
├── coco_lvis
│   ├── cocolvis_annotation
│   ├── train
│   └── val
└── sam-hq
    ├── cascade_psp
    ├── DIS5K
    ├── hqseg44k_ignore_prefix.json
    └── thin_object_detection
```