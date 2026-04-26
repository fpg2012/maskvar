# SimpleMaskAR Summary

This document summarizes the current `SimpleMaskAR` pipeline in this repo, based on the implementation in:

- [maskvar/models/simple_mask_ar/simple_mask_ar.py](/home/clc/workspace/maskseg/maskvar/models/simple_mask_ar/simple_mask_ar.py)
- [maskvar/models/simple_mask_ar/basic.py](/home/clc/workspace/maskseg/maskvar/models/simple_mask_ar/basic.py)
- [train_scripts/train_simple_mask_ar.py](/home/clc/workspace/maskseg/train_scripts/train_simple_mask_ar.py)

## Overview

`SimpleMaskAR` is an autoregressive model over VQ token ids of segmentation masks.

The high-level pipeline is:

1. Use a frozen `SimpleMaskVqvae` model to encode `(image, mask)` into:
   - `token_ids`: discrete mask tokens
   - `image_tokens`: image features used as cross-attention condition
2. Train `SimpleMaskAR` to predict mask token ids autoregressively.
3. For visualization or evaluation, decode predicted token ids back to mask logits through the frozen VQ-VAE decoder.

## Token Prediction Setup

Training uses full teacher forcing.

Given ground-truth token ids `x` with spatial shape `(B, H, W)`:

1. Flatten in row-major order.
2. Drop the last token.
3. Prepend `sos`.
4. Reshape back to `(B, H, W, C)` and run the transformer.

This means the logits at each spatial position are aligned with the original token ids directly. There is no extra target shift in the loss code.

In practice:

- model input sequence: `[sos, t0, t1, ..., t(L-2)]`
- training target sequence: `[t0, t1, ..., t(L-1)]`

So the training loss is:

- `cross_entropy(logits_flat, token_ids_flat)`

## Architecture

Each `SimpleMaskARBlock` contains:

1. `SimpleCrossBlock`
   - query: mask token embeddings
   - key/value: frozen image tokens
2. `SimpleSelfBlock`
   - causal self-attention over the mask token sequence

Current builder config for `simple_mask_ar` is:

- `dim=384`
- `depth=2`
- `vocab_size=4096`
- `h=64`
- `w=64`
- `num_heads=4`

## Attention Implementation

The current implementation does not use `flex_attention`.

Both training and inference now use `torch.nn.functional.scaled_dot_product_attention`:

- training self-attention: `is_causal=True`
- inference self-attention: `is_causal=True`
- cross-attention: standard non-causal SDPA

This was done to simplify the code path and avoid the heavy backward memory cost observed with `flex_attention`.

## RoPE Precision

RoPE internally still uses float32-style coordinate/frequency computation.

The practical rule in the current code is:

1. Compute RoPE in higher precision.
2. Cast `q/k` back to `v.dtype` before attention.

This keeps attention dtype consistent while avoiding unnecessary numeric instability in the sinusoidal part.

## Autoregressive Inference

`autoregressive_infer()` now uses KV cache.

### Previous behavior

The old inference path recomputed the entire prefix on every decoding step, which was very slow for `64 x 64 = 4096` tokens.

### Current behavior

The current implementation performs:

1. Per-block precomputation of static cross-attention image `k/v`.
2. Per-block caching of self-attention token `k/v`.
3. Step-by-step decoding using only the current token embedding plus caches.

This significantly reduces inference cost, especially for validation-time visualization.

`autoregressive_infer()` also supports `num_samples`.

- `num_samples == 1`: returns `(B, H, W)`
- `num_samples > 1`: returns `(B, num_samples, H, W)`

The implementation reuses the same conditioning image tokens and only duplicates the decoding batch/caches across samples.

## Training Script Behavior

The trainer lives in [train_scripts/train_simple_mask_ar.py](/home/clc/workspace/maskseg/train_scripts/train_simple_mask_ar.py).

### VQ-VAE loading

The AR script no longer hardcodes the VQ-VAE builder config.

It supports:

- `--vqvae_config`
- `--vqvae_image_encoder_checkpoint`
- `--vqvae_image_encoder_config`

It also tries to inherit VQ-VAE metadata from the checkpoint output directory's `config.json`.

### Mixed precision

Current precision policy:

- AR forward/backward: autocast with the configured dtype, typically `bfloat16`
- VQ-VAE encoder path in `encode_mask_to_tokens()`: also under autocast
- VQ nearest-neighbor lookup `x_to_idx`: explicitly cast to float32 before lookup for stability

### Gradient accumulation

Gradient accumulation was fixed so optimizer stepping is controlled by the current iteration count rather than stale `global_step`.

## Validation Policy

Current validation is intentionally asymmetric:

- teacher metrics: full validation set
- infer metrics: optional, small subset only

Specifically:

1. `loss`, `accuracy`, `teacher_iou`
   - computed on all validation batches
2. `infer_iou`
   - disabled by default
   - enabled only with `--enable_infer_iou`
   - if enabled, only computed on the first `--val_infer_batches` validation batches

Default:

- `--enable_infer_iou` is off
- `--val_infer_batches 4`

This keeps validation usable while still allowing occasional pure autoregressive inspection.

## Visualization Policy

### Training visualization

Training visualization no longer computes autoregressive inference by default.

It only shows:

1. original image
2. ground-truth binary mask
3. GT token reconstruction
4. GT token reconstruction overlay
5. teacher-forcing prediction
6. teacher-forcing overlay

### Validation visualization

Validation visualization always includes:

1. original image
2. ground-truth binary mask
3. GT token reconstruction
4. GT token reconstruction overlay
5. teacher-forcing prediction
6. teacher-forcing overlay

If `--enable_infer_iou` is enabled, it also includes:

7. autoregressive prediction
8. autoregressive overlay

## Why GT Token Reconstruction Is Important

The visualization now includes the mask reconstructed from the ground-truth token ids through the frozen VQ-VAE decoder.

This is a useful upper-bound reference:

- if GT token reconstruction IoU is already low, the limitation is in the tokenizer/decoder
- if GT token reconstruction IoU is high but teacher/infer IoU is low, the limitation is in AR prediction

This is often the fastest way to distinguish "bad AR" from "bad discrete mask codec".

## Timing and Profiling Support

The training script includes two lightweight analysis tools.

### `--timing`

Prints timing breakdown on log steps:

- `data`
- `encode`
- `train`
- `vis`

This is useful to distinguish:

- data loading bottleneck
- VQ-VAE encoding bottleneck
- AR training bottleneck
- visualization/inference bottleneck

### `--profile`

Enables short-window `torch.profiler` tracing and writes traces to:

- `<out_dir>/profiler`

This can be inspected with TensorBoard Profile view.

## Current Performance Takeaways

From recent debugging on this code path:

1. Data loading was not the main bottleneck when GPU utilization was saturated.
2. The dominant steady-state training cost was usually the frozen VQ-VAE encoding path.
3. The dominant debug-mode slowdown was pure autoregressive inference inside visualization.
4. KV cache helped reduce inference cost, but full `4096`-step AR decoding is still expensive enough that it should be used sparingly.

## Recommended Usage

### For normal training

- keep `--enable_infer_iou` off
- use teacher IoU for frequent monitoring
- use low-frequency train visualization
- use small `val_infer_batches` only when explicitly inspecting inference quality

### For quick chain validation

Use `--debug_smoketest`, which forces:

- very short run
- frequent logging
- fast validation entry
- timing/profiler support

This is for functionality checking, not throughput benchmarking.

## Known Practical Constraints

1. Pure autoregressive validation is still expensive even with KV cache.
2. VQ-VAE encoding is still a major runtime cost because AR training currently tokenizes on the fly.
3. If further speedup is needed, the next likely optimization target is cached or offline-precomputed token/image features rather than the AR forward itself.
