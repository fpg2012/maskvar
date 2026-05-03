# Codex Project Notes

This repository is `maskseg`, a segmentation research codebase. Prefer reading
this file first when returning to the project.

## Project Shape

- Builders live in `maskvar/maskseg_build_everything.py`.
- Main SimpleMaskVqvae training entry is `train_scripts/train_simple_mask_vqvae.py`.
- SimpleMaskVqvae model family lives in `maskvar/models/simple_mask_vqvae/`.
- Training shell scripts live under `train_scripts/simple_mask_vqvae/` and
  `train_scripts/simple_mask_ar/`.
- Experiment / analysis scripts are currently placed under `notebooks/`, even
  when they are normal Python scripts rather than `.ipynb` notebooks.
- Documentation for design decisions lives under `docs/`.

## SimpleMaskVqvae Baseline

The baseline `SimpleMaskVqvae` data flow is:

```text
mask_normalized -> mask_encoder -> BLC mask tokens
image -> image_encoder -> BHWC image tokens
mask tokens -> SimpleVectorQuantize -> BHWC mask tokens
SimpleMaskDecoder(mask_tokens, image_tokens) -> mask logits
```

The important design constraint is that new VQVAE variants should preserve the
baseline encoder/decoder interface whenever possible, so existing checkpoints
can initialize encoder, decoder, and codebook.

Default useful pretrained checkpoint:

```text
out/ddp_simple_mask_vqvae_coconut_ep5_19bugfix_retrain/checkpoints/latest.pth
```

## Multiscale VQVAE History

V1 multiscale used parallel pooled features from scales
`(1, 2, 4, 8, 16, 32, 64)`, upsampled all to `64x64`, and fused them. Ablation
showed it mostly ignored coarse scales and relied on `64x64`.

V2 should be interpreted as:

- Keep the outer `SimpleMaskVqvae` flow.
- Replace `SimpleVectorQuantize` with `MultiscaleVectorQuantize`.
- Implement VAR-style residual multiscale quantization inside the quantizer.
- Do not use decoder-level cumulative auxiliary loss.

V2 builder name:

```text
simple_mask_vqvae_multiscale_v2_dim384
```

V2 training script:

```text
train_scripts/simple_mask_vqvae/ddp_train_multiscale_v2_coconut_hf_dino.sh
```

V2 design doc:

```text
docs/simple_mask_vqvae_multiscale_v2.md
```

V2 visualizer:

```text
notebooks/visualize_simple_mask_vqvae_multiscale_v2.py
```

## MultiscaleVectorQuantize Conventions

`MultiscaleVectorQuantize` lives in:

```text
maskvar/models/simple_mask_vqvae/quant.py
```

It subclasses `SimpleVectorQuantize` and keeps codebook state names compatible:

```text
quant.embedding.weight
quant.ema_vocab_hit
quant.record_hit
```

New V2 parameters include:

```text
quant.phi.*
quant.scale_gates
```

Stability conventions borrowed from `maskvar/models/quant.py::VectorQuantizer2`:

- Quantization path uses fp32 and disables autocast.
- Residual starts from detached full-resolution features.
- VQ loss is cumulative full-feature loss, not per-scale residual-token loss.
- Downsample uses `area`; upsample uses `bicubic`.
- `phi_k` is residual blend: `h * (1 - quant_resi) + conv(h) * quant_resi`.
- For compatibility with the baseline checkpoint, initialize coarse gates to 0
  and final `64x64` gate to 1, with final `phi` as identity.

If VQ loss explodes, do not resume the bad checkpoint. Restart from the baseline
checkpoint and consider lowering `--vq_loss_weight`.

## Current V2 Visualization Interpretation

Use only the V2 visualizer for current V2 checkpoints:

```bash
python -m notebooks.visualize_simple_mask_vqvae_multiscale_v2 \
  --out_dir out/ddp_simple_mask_vqvae_multiscale_v2_coconut_ep10 \
  --num_samples 4 \
  --split val
```

The visualizer reads scale behavior directly from `model.quant`, not from old
outer-model multiscale helper methods.

Important metrics:

- `full_iou`: final reconstruction quality.
- `cumulative_iou_to_s*`: how much coarse-to-fine information is usable up to a
  scale.
- `drop_one_delta_s*`: how much full IoU drops if a scale contribution is
  removed.
- `contribution_norm_mean_s*`: magnitude of each projected scale contribution.
- `scale_gate_s*`: learned gate per scale.
- `token_unique_s*`: codebook diversity per scale.

The V2 visualization uses red-blue logits (`RdBu_r`) for full logits,
single-scale decoded contribution, cumulative coarse-to-fine, and drop-one
logit deltas.

## Training Script Conventions

For DDP shell scripts:

- `N_NODE` is usually first arg.
- `OUTDIR` is usually second arg.
- `MASTER_PORT` is usually third arg.
- For V2, fourth arg may override init checkpoint.

V2 script defaults:

```text
--config simple_mask_vqvae_multiscale_v2_dim384
--enable_vq
--vq_loss_weight 0.25
--freeze_image_encoder
--disable_find_unused_parameters
```

## Coding Preferences

- Prefer `rg` / `rg --files` for searching.
- Use `apply_patch` for manual edits.
- Keep new model variants close to existing builder/trainer patterns.
- Avoid unrelated refactors while experiments are in flight.
- Be careful with dirty worktrees; do not revert user or previous experiment
  changes unless explicitly asked.
