# Interactive Evaluation Archive

This folder collects interactive segmentation evaluation outputs that were
previously scattered under `out/`.

## Files

- `archive/`: copied raw JSON result files.
- `summary/interactive_eval_summary.csv`: compact table for quick comparison.
- `summary/interactive_eval_summary.json`: same summary in JSON form.
- `loop_rope_sam_current_20_cpu/`: fresh quick evaluation for the current
  LoopRopeSAM checkpoint.

## Current LoopRopeSAM Result

Checkpoint:

```text
out/ddp_rope_sam_loop_coconut_hf_dino_click_ep2_resume/checkpoints/latest.pth
```

Evaluation command used CPU and online DINO features because the local
`data/cache/dino_v3_vits/coconut_hf_val_metadata.json` file is missing:

```bash
/home/clc/miniconda3/envs/var_v2/bin/python misc_scripts/eval_interactive_seg.py \
  --model rope_sam \
  --device cpu \
  --outdir eval/loop_rope_sam_current_20_cpu \
  --dataset coconut_hf \
  --dataset_split val \
  --rope_sam_checkpoint out/ddp_rope_sam_loop_coconut_hf_dino_click_ep2_resume/checkpoints/latest.pth \
  --rope_sam_config rope_sam_loop_dim384 \
  --image_encoder_checkpoint ckpt/dino_v3_vits.safetensors \
  --image_encoder_config dino_v3_vits \
  --max_clicks 10 \
  --batch_size 1 \
  --num_workers 0 \
  --val_iters 20 \
  --dtype float32
```

This is a quick 20-sample sanity check, not a full apples-to-apples benchmark.
For a fair comparison to the 200-sample cached SAM runs, regenerate the missing
DINO v3 val cache metadata/cache and rerun with `--val_iters 200`.

## 200-Sample Comparison

The following 200-sample evaluations were added on 2026-06-17:

```text
eval/archive/rope_sam_current_200_20260617.json
eval/archive/loop_rope_sam_current_200_20260617.json
```

Summary:

```text
rope_sam_current_200:
  NoC@80 = 4.93
  NoC@85 = 5.925
  IoU@10 = 0.8483

loop_rope_sam_current_200:
  NoC@80 = 4.94
  NoC@85 = 5.89
  IoU@10 = 0.8457
```

The 200-sample LoopRopeSAM result is very close to the current RoPE SAM result:
slightly better at NoC@85, slightly worse at IoU@10.
