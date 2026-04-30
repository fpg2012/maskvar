# SimpleMaskAR Click Conditioning

This document describes the current click-conditioned `SimpleMaskAR` implementation.

Main code entry points:

- [maskvar/utils/clicker_v2.py](/home/clc/workspace/maskseg/maskvar/utils/clicker_v2.py)
- [maskvar/models/simple_mask_ar/simple_mask_ar.py](/home/clc/workspace/maskseg/maskvar/models/simple_mask_ar/simple_mask_ar.py)
- [maskvar/models/simple_mask_ar/basic.py](/home/clc/workspace/maskseg/maskvar/models/simple_mask_ar/basic.py)
- [train_scripts/train_simple_mask_ar.py](/home/clc/workspace/maskseg/train_scripts/train_simple_mask_ar.py)

## Enable Flag

Click conditioning is optional and is enabled by:

```bash
python train_scripts/train_simple_mask_ar.py \
  ... \
  --enable_click
```

When `--enable_click` is not set, the model keeps the old image-only behavior:

1. mask token embedding
2. image cross-attention
3. causal self-attention

When `--enable_click` is set, each AR block becomes:

1. click cross-attention
2. image cross-attention
3. causal self-attention

## Click Sampling

Training and validation currently sample clicks in a DataLoader dataset wrapper:

- `ClickConditionDataset` for map-style datasets
- `ClickConditionIterableDataset` for iterable datasets

For each mask in the batch:

1. Convert `single_mask[0]` to a CPU numpy binary mask in the DataLoader worker.
2. Randomly choose `num_clicks` from `{1, 2}`.
3. Call `clicker_v2.init_clicks(mask, num_random_clicks=num_clicks, random_sample=True)`.
4. Pad to exactly 2 click slots with `to_sam_format(..., pad_size=2)`.

The training loop receives `click_coords` and `click_labels` as part of the batch and only moves them to the training device. This keeps OpenCV click sampling out of the main GPU training loop and lets `num_workers`/`prefetch_factor` provide parallelism.

The returned click format remains compatible with the older clicker:

```python
click_list = [(y, x, label), ...]
```

For current training, all real clicks are positive:

- `label == 1`: positive click
- `label == -1`: padding click

Negative clicks are not sampled by this AR training path.

## clicker_v2 Sampling Rule

`clicker_v2` keeps the same external interface as `maskvar.utils.clicker.init_clicks`.

The current policy is intentionally simple:

1. Empty mask: return an empty click list plus zero dummy arrays.
2. Very small available component, area `<= 9`: sample uniformly from foreground pixels.
3. Normal mask/component:
   - compute distance transform inside the component
   - use `distance ** 2` as the sampling weight
   - multiply by `not_clicked_map` to avoid duplicates
4. Multiple connected components:
   - sort components by area, descending
   - the first clicks try to cover the largest components one by one
   - if component-specific sampling fails, fall back to the full mask

This means center/interior pixels have higher probability than boundary pixels, and disconnected large foreground regions are more likely to each receive a click.

## Coordinate Conversion

`to_sam_format()` returns click coordinates as `(x, y)` in the original mask pixel grid.

The AR model expects click coordinates as `(row, col)` in the AR token grid, so training converts:

```python
click_coords[..., 0] = y * (ar_h / mask_h)
click_coords[..., 1] = x * (ar_w / mask_w)
```

For the default builder:

- input mask size is usually `1024 x 1024`
- AR token grid is `64 x 64`
- one AR token roughly corresponds to a `16 x 16` image/mask patch

The resulting `click_coords` tensor has shape:

```python
(B, 2, 2)
```

where the last dimension is:

```python
(row, col)
```

The `click_labels` tensor has shape:

```python
(B, 2)
```

## Click Encoding Flow

The click encoding happens in `SimpleMaskAR.encode_clicks()`.

Inputs:

```python
click_coords: (B, N, 2)  # row/col in AR token-grid units
click_labels: (B, N)     # 1 for positive, -1 for padding
```

The model owns two trainable click embeddings:

```python
self.positive_click: (C,)
self.padding_click:  (C,)
```

Encoding proceeds as:

1. Expand both embeddings to `(B, N, C)`.
2. Select `positive_click` where `label == 1`.
3. Select `padding_click` otherwise.
4. Apply 2D RoPE to the positive-click tokens using `click_coords`.
5. Keep padding tokens unrotated.

In formula form:

```python
base_token = positive_click if label == 1 else padding_click
click_token = RoPE(base_token, click_coord) if label == 1 else base_token
```

The output is:

```python
click_tokens: (B, N, C)
click_coords: (B, N, 2)
click_labels: (B, N)
```

Important detail: the positive click embedding itself is rotated before entering the AR blocks. The click cross-attention block also applies RoPE to click keys. This means the current implementation injects click position both at the click-token construction stage and again in click-attention key space.

## RoPE With Batched Coordinates

The original RoPE helpers supported:

- spatial tensors: `(B, H, W, C)`
- sequence tensors with shared coordinates: `(B, L, C)` plus `(L, 2)`

Click prompts need per-sample coordinates, so `RotaryPositionEmbedding.apply_2d_rope_with_batched_coords()` was added.

It accepts:

```python
x:      (B, L, C)
coords: (B, L, 2)
```

Coordinates can be floating point. This is useful because a click from a `1024 x 1024` mask may map to a fractional position in the `64 x 64` AR grid.

## Click Cross-Attention

`SimpleClickCrossBlock` is structurally similar to `SimpleCrossBlock`, but its key/value source is the small click-token sequence instead of image tokens.

Training path:

```python
x = click_cross_block(x, click_tokens, click_coords, click_labels)
x = image_cross_block(x, image_tokens)
x = self_block(x)
```

Shapes:

```python
x:            (B, H, W, C)
click_tokens: (B, N, C)
click_coords: (B, N, 2)
click_labels: (B, N)
```

Inside click cross-attention:

1. Query comes from AR mask tokens.
2. Key/value come from click tokens.
3. Query receives standard spatial RoPE for the AR grid.
4. Click key receives batched-coordinate RoPE only for real positive clicks.
5. Padding keys remain unrotated and use the trainable padding embedding.

The attention itself is non-causal because clicks are external conditioning tokens.

## Autoregressive Inference With Clicks

`autoregressive_infer()` accepts the same click tensors:

```python
generated = model.autoregressive_infer(
    image_tokens,
    click_coords=click_coords,
    click_labels=click_labels,
)
```

At inference time:

1. image cross-attention K/V are precomputed per block
2. click cross-attention K/V are precomputed per block
3. self-attention K/V are cached step by step

For `num_samples > 1`, both image and click K/V caches are repeated across samples.

## Training Data Flow

In one training step:

1. Dataloader returns:
   - `image`
   - `single_mask_normalized`
   - `single_mask`
   - `click_coords`, if `--enable_click`
   - `click_labels`, if `--enable_click`
2. Frozen VQ-VAE encodes:
   - `token_ids`: `(B, L)`
   - `image_tokens`: `(B, L, C)`
3. If `--enable_click`:
   - DataLoader workers have already sampled 1-2 positive clicks
   - click tensors are moved to the training device
4. Forward:

```python
logits = model(
    token_ids,
    image_tokens,
    click_coords=click_coords,
    click_labels=click_labels,
)
```

5. Loss remains unchanged:

```python
loss = cross_entropy(logits.reshape(B * L, vocab), token_ids.reshape(B * L))
```

The click condition changes the model input, but it does not change the target alignment.

## Validation Behavior

Validation samples clicks in the same way as training.

Teacher-forcing metrics use the sampled clicks. If `--enable_infer_iou` is enabled, pure autoregressive inference also uses the same sampled click tensors for that validation batch.

Because clicks are sampled randomly, validation metrics with `--enable_click` are not fully deterministic unless the numpy random seed and dataloader behavior are controlled.

## Checkpoint Compatibility

Old non-click checkpoints do not contain:

- `positive_click`
- `padding_click`
- click cross-attention block weights

When building with `enable_click=True`, the builder loads checkpoints with `strict=False`, so older AR weights can initialize the shared image/self-attention parts while new click parameters are randomly initialized.

When resuming training with `--enable_click` from a non-click optimizer checkpoint, optimizer state loading may be skipped because the parameter groups no longer match.

## Current Limitations

1. Only positive clicks are generated.
2. The number of click slots is fixed at 2 in the trainer.
3. Click sampling happens on CPU with numpy/OpenCV.
4. Validation click randomness can add metric noise.
5. Padding clicks are visible as learned padding tokens, not masked out of attention.

The padding behavior is deliberate for now: the model receives a consistent fixed-length click prompt sequence and learns how to treat missing clicks through `padding_click`.
