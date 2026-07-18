# Interaction Click Sampling Acceleration

## Context

`maskvar/interaction` adds C++/CUDA click sampling backends for interactive
segmentation. The main training use case is batch click sampling from mask-level
targets:

- Input masks are usually `1024x1024`, occasionally `2048x2048`.
- Decoder feature grids are usually `64x64`.
- Batch size is usually `1..8`, and should not exceed `32`.
- The number of clicks is usually <= 20. Current SimpleMaskAR/MaskGIT click
  conditioning uses 1..2 clicks; interactive SAM/RopeSAM paths can use more.

The old `maskvar.utils.clicker` path runs OpenCV distance transforms on CPU and
does per-sample Python loops. This is expensive at `1024x1024`.

## Implementation

The compatibility entry remains `maskvar.utils.clicker_v2`:

- `init_clicks(...)`
- `predict_next_click(...)`
- `to_sam_format(...)`

`clicker_v2` now delegates to `maskvar.interaction.ClickSampler` while
preserving the old `(y, x, label)` click format.

For training, use:

```python
from maskvar.utils.clicker_v2 import sample_batched_click_conditions

click_coords, click_labels = sample_batched_click_conditions(
    masks,          # (B, 1, H, W) or (B, H, W)
    out_h=64,
    out_w=64,
    max_clicks=10,
    backend="cuda",
    random_click_counts=False,
)
```

This returns:

- `click_coords`: `(B, max_clicks, 2)` in row/col decoder-grid coordinates.
- `click_labels`: `(B, max_clicks)`, with `1` for positive click and `-1` for
  padding.

The SimpleMaskAR, SimpleMaskMaskGIT, and RopeSAM training scripts now sample
initial click conditions in the trainer loop after the batch is moved to GPU.
DataLoader workers no longer sample clicks.

## Extension Loading

The backend first tries to import prebuilt extensions:

- `maskvar.interaction._cpu_ext`
- `maskvar.interaction._cuda_ext`

If they are unavailable, it falls back to JIT compilation with
`torch.utils.cpp_extension.load`.

Ahead-of-time build:

```bash
conda run -n var_v2 python -m maskvar.interaction.build_extensions \
  --backend cuda \
  --cuda_arch_list 8.0
```

Set `--cuda_arch_list` for the target GPU to avoid compiling for every visible
architecture. Examples:

- A100: `8.0`
- RTX 3090 / A6000: `8.6`
- RTX 4090 / L40: `8.9`
- H100: `9.0`

## Benchmarks

Benchmark command for the default training-style setting:

```bash
conda run -n var_v2 python -m notebooks.benchmark_interaction_clicker \
  --backend cuda \
  --num_masks 32 \
  --iters 64 \
  --warmup 8 \
  --height 1024 \
  --width 1024 \
  --batch_size 8 \
  --max_clicks 10 \
  --fixed_clicks
```

Fixed 10-click sampling, `1024x1024` input masks, output coordinates scaled to
`64x64`:

| Batch | Old CPU OpenCV us/click | CUDA low-level us/click | Training batch us/click | Training speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 21928.69 | 573.42 | 1125.46 | 19.48x |
| 2 | 23096.28 | 574.08 | 862.81 | 26.77x |
| 4 | 22441.27 | 567.01 | 723.57 | 31.01x |
| 8 | 22093.44 | 581.78 | 661.83 | 33.38x |

Approximate per-step click time for fixed 10-click sampling:

| Batch | Old CPU click time / step | New CUDA training click time / step |
| ---: | ---: | ---: |
| 1 | 219 ms | 11 ms |
| 2 | 462 ms | 17 ms |
| 4 | 898 ms | 29 ms |
| 8 | 1767 ms | 53 ms |

## SAM Decoder Training-Script Reference

SAM decoder timing was measured with a benchmark that mirrors
`train_scripts/train_sam_decoder.py::predict_batch`:

- `image_embeddings` are already available as `(B, 256, 64, 64)`, matching the
  training script when image feature cache is used.
- Each sample is processed in a Python loop, as in the current trainer.
- `prompt_encoder` is called from click lists.
- `mask_decoder` is called with `image_embeddings[i:i+1]`.
- Low-resolution decoder masks are interpolated back to the `1024x1024` target
  mask size.

Benchmark command:

```bash
conda run -n var_v2 python -m notebooks.benchmark_sam_decoder_training_style \
  --batch_sizes 1,2,4,8 \
  --num_clicks 10 \
  --height 1024 \
  --width 1024
```

Random ViT-B decoder weights and random embeddings were used. Checkpoint values
do not affect the compute graph. Inputs were:

- `image_embeddings`: `(B, 256, 64, 64)`
- click lists with 10 positive clicks per sample
- output masks interpolated to `(B, 1, 1024, 1024)`

The current local SAM decoder implementation is effectively single-sample in
the trainer: `train_sam_decoder.py::predict_batch` loops over samples. Direct
B>1 decoder forward is not currently valid because `MaskDecoder.predict_masks`
uses `repeat_interleave(image_embeddings, tokens.shape[0])`, which expands the
batch as `B^2`.

Measured current training-style cost:

| Batch | `predict_batch` ms / click round | ms / sample | 10 rounds ms |
| ---: | ---: | ---: | ---: |
| 1 | 4.495 | 4.495 | 44.952 |
| 2 | 8.941 | 4.470 | 89.407 |
| 4 | 17.794 | 4.449 | 177.944 |
| 8 | 35.628 | 4.453 | 356.277 |

For SAM decoder training with 10 click rounds, B=8 decoder/prompt/interpolation
compute is roughly `356 ms` per step. New CUDA click sampling is about `53 ms`
for the same B=8, 10-click, 1024x1024 setting. Old CPU click sampling would be
about `1767 ms`.

## Bottleneck Analysis

Before acceleration, click sampling can dominate training at 1024 resolution.
For B=8 and 10 clicks:

- SAM decoder train-script `predict_batch` loop: ~356 ms / step.
- Old CPU click sampling: ~1767 ms / step.
- New CUDA batch click sampling: ~53 ms / step.

After acceleration, the bottleneck generally moves back to model forward /
decoder work. Click sampling is still visible for light models and many click
rounds, but it is no longer the dominant cost in the SAM decoder example.

The current batched CUDA API removes Python per-sample click sampling from the
training path, but internally it still reuses the single-mask CUDA sampler per
sample. Remaining click-side costs are:

- per-sample distance computation on `1024x1024` masks;
- bbox / max / candidate selection steps;
- host synchronization inside the current low-level sampler;
- repeated recomputation for each click round because clicked pixels update the
  ignore mask.

Further speedups require a deeper batched CUDA kernel that keeps bbox, max
distance, candidate counting, and sampling on GPU for the whole batch.
