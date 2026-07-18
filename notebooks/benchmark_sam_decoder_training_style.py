"""Benchmark SAM decoder in the same shape as train_sam_decoder.py.

This intentionally mirrors SAMDecoderTrainer.predict_batch:

- image embeddings are already available as (B, 256, 64, 64)
- each sample is processed in a Python loop
- prompt_encoder is called from click lists
- mask_decoder is called with image_embeddings[i:i+1]
- low-res masks are interpolated back to the target mask size

Run:

    conda run -n var_v2 python -m notebooks.benchmark_sam_decoder_training_style \
      --batch_sizes 1,2,4,8 --num_clicks 10 --height 1024 --width 1024
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from maskvar.build_sam import sam_model_registry
from maskvar.utils.clicker import to_sam_format


def make_click_lists(batch_size: int, num_clicks: int, height: int, width: int):
    click_lists = []
    for batch_idx in range(batch_size):
        clicks = []
        for click_idx in range(num_clicks):
            y = int((click_idx + 1) * height / (num_clicks + 1))
            x = int(((click_idx * 37 + batch_idx * 53) % max(width, 1)))
            clicks.append((y, x, 1))
        click_lists.append(clicks)
    return click_lists


@torch.no_grad()
def predict_batch_like_train_script(
    sam,
    mask_decoder,
    image_pe,
    image_embeddings,
    click_lists,
    output_size,
    prev_logits=None,
    multimask_output=False,
):
    masks = []
    scale_x = sam.prompt_encoder.input_image_size[1] / float(output_size[1])
    scale_y = sam.prompt_encoder.input_image_size[0] / float(output_size[0])
    for i, clicks in enumerate(click_lists):
        coords_xy, labels = to_sam_format(clicks, device=image_embeddings.device)
        coords_xy = coords_xy.float()
        coords_xy[:, 0] *= scale_x
        coords_xy[:, 1] *= scale_y
        mask_prompt = None
        if prev_logits is not None:
            mask_prompt = F.interpolate(
                prev_logits[i : i + 1].float(),
                size=sam.prompt_encoder.mask_input_size,
                mode="bilinear",
                align_corners=False,
            )
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(coords_xy.unsqueeze(0), labels.long().unsqueeze(0)),
            boxes=None,
            masks=mask_prompt,
        )
        low_res_masks, iou_predictions = mask_decoder(
            image_embeddings=image_embeddings[i : i + 1],
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        if multimask_output:
            best_idx = int(iou_predictions[0].detach().argmax().item())
            low_res_masks = low_res_masks[:, best_idx : best_idx + 1]
        masks.append(F.interpolate(low_res_masks, size=output_size, mode="bilinear", align_corners=False))
    return torch.cat(masks, dim=0)


def bench_one(sam, batch_size: int, num_clicks: int, height: int, width: int, iters: int, warmup: int):
    image_embeddings = torch.randn(batch_size, 256, 64, 64, device="cuda")
    image_pe = sam.prompt_encoder.get_dense_pe()
    click_lists = make_click_lists(batch_size, num_clicks, height, width)
    output_size = (height, width)

    def one():
        predict_batch_like_train_script(
            sam=sam,
            mask_decoder=sam.mask_decoder,
            image_pe=image_pe,
            image_embeddings=image_embeddings,
            click_lists=click_lists,
            output_size=output_size,
        )

    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        one()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", default="1,2,4,8")
    parser.add_argument("--num_clicks", type=int, default=10)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sam_model_type", default="vit_b", choices=sorted(sam_model_registry.keys()))
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    torch.set_float32_matmul_precision("high")
    sam = sam_model_registry[args.sam_model_type]().cuda().eval()
    batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    print(
        f"sam_model={args.sam_model_type} num_clicks={args.num_clicks} "
        f"output={args.height}x{args.width} iters={args.iters}"
    )
    print(f"{'B':>4s} {'predict_batch_ms':>18s} {'ms/sample':>12s} {'10 rounds ms':>14s}")
    print("-" * 56)
    for batch_size in batch_sizes:
        ms = bench_one(
            sam=sam,
            batch_size=batch_size,
            num_clicks=args.num_clicks,
            height=args.height,
            width=args.width,
            iters=args.iters,
            warmup=args.warmup,
        )
        print(f"{batch_size:4d} {ms:18.3f} {ms / batch_size:12.3f} {ms * args.num_clicks:14.3f}")


if __name__ == "__main__":
    main()
