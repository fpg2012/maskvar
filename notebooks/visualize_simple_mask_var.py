"""
Visualize SimpleMaskVAR teacher-forcing and autoregressive sampling.

This script targets the current VAR-style AR model, not the VQ-VAE.

Example:
    python -m notebooks.visualize_simple_mask_var \
        --out_dir out/ddp_simple_mask_var_v2_overfit8_coconut_ep10 \
        --checkpoint_path out/ddp_simple_mask_var_v2_overfit8_coconut_ep10/checkpoints/latest.pth \
        --num_samples 8 \
        --split val
"""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset, MaskLevelFlatSubsetDataset
from maskvar.maskseg_build_everything import builder_map
from maskvar.utils import restore_normalized_image
from maskvar.utils.metrics import calc_iou


DATASET_PATHS = {
    "hqseg44k": "data/sam-hq",
    "cocolvis": "data/coco_lvis",
    "coconut_hf": "data/coconut_hf",
}


def read_train_config(out_dir: Path) -> dict:
    config_path = out_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return json.load(f)


def resolve_args_from_config(args, cfg: dict):
    if args.checkpoint_path is None and args.out_dir is not None:
        args.checkpoint_path = str(Path(args.out_dir) / "checkpoints" / "best.pth")
    args.config = args.config or cfg.get("config") or "simple_mask_var_v2"
    args.vqvae_config = args.vqvae_config or cfg.get("vqvae_config") or "simple_mask_vqvae_multiscale_v2_dim384"
    args.vqvae_image_encoder_checkpoint = (
        args.vqvae_image_encoder_checkpoint
        or cfg.get("vqvae_image_encoder_checkpoint")
        or cfg.get("image_encoder_checkpoint")
    )
    args.vqvae_image_encoder_config = (
        args.vqvae_image_encoder_config
        or cfg.get("vqvae_image_encoder_config")
        or cfg.get("image_encoder_config")
        or "dino_v3_vits"
    )
    args.dataset = args.dataset or cfg.get("dataset") or "coconut_hf"
    args.dataset_path = args.dataset_path or cfg.get("dataset_path")
    args.enable_vq = cfg.get("enable_vq", args.enable_vq) if args.enable_vq is None else args.enable_vq
    return args


def build_dataset(args):
    dataset_path = args.dataset_path or DATASET_PATHS[args.dataset]
    train_set_base, val_set_base = builder_map["dataset"][args.dataset](dataset_path)
    base_set = train_set_base if args.split == "train" else val_set_base
    index_mapping_path = Path("data/flat") / args.dataset / f"{args.split}_index_mapping.npy"

    if args.train_subset_index or args.val_subset_index:
        subset_path = args.train_subset_index if args.split == "train" else args.val_subset_index
        if subset_path:
            return MaskLevelFlatSubsetDataset(
                subset_list=Path(subset_path),
                index_mapping_path=index_mapping_path,
                dataset=base_set,
                with_image_embed=False,
                image_feature_cache=None,
                mask_filter_thresh=args.mask_filter_thresh,
                dtype=torch.float32,
                image_size_encoder=args.image_size_encoder,
                image_size_mask=args.image_size_mask,
            )

    return MaskLevelFlatDataset(
        index_mapping_path=index_mapping_path,
        dataset=base_set,
        with_image_embed=False,
        image_feature_cache=None,
        mask_filter_thresh=args.mask_filter_thresh,
        dtype=torch.float32,
        image_size_encoder=args.image_size_encoder,
        image_size_mask=args.image_size_mask,
    )


def ensure_var_model(model):
    if not hasattr(model, "autoregressive_infer"):
        raise TypeError("This visualizer expects a SimpleMaskVAR-style model.")
    if not hasattr(model, "_vqvae_model") or model._vqvae_model is None:
        raise TypeError("SimpleMaskVAR must have a VQVAE model attached via set_vqvae_model().")


def decode_token_ids_to_logits(model, token_ids_by_scale, image_tokens, output_size):
    return model._vqvae_model.decode_from_multiscale_token_ids(
        token_ids_by_scale,
        image_tokens=image_tokens,
        output_size=output_size,
    )


def to_numpy_mask(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0, 0]
    elif tensor.dim() == 3:
        tensor = tensor[0]
    return tensor.detach().float().cpu().numpy()


def to_2d_mask(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0, 0]
    elif tensor.dim() == 3:
        tensor = tensor[0]
    return tensor.detach().float().cpu().numpy()


def render_overlay(image_np, mask_np, color):
    base = image_np.astype(np.float32) / 255.0
    mask_bool = mask_np > 0
    overlay = base.copy()
    overlay[mask_bool] = overlay[mask_bool] * 0.55 + np.array(color, dtype=np.float32) * 0.45
    return np.clip(overlay, 0, 1)


def robust_limits(array, center_zero=False):
    finite = np.asarray(array)[np.isfinite(array)]
    if finite.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(finite, [1, 99])
    if abs(hi - lo) < 1e-6:
        lo, hi = float(finite.min()) - 1.0, float(finite.max()) + 1.0
    if center_zero:
        vmax = max(abs(float(lo)), abs(float(hi)), 1e-6)
        return -vmax, vmax
    return float(lo), float(hi)


@torch.no_grad()
def collect_var_outputs(model, image, mask_normalized, temperature=1.0, top_k=None):
    ensure_var_model(model)

    vqvae = model._vqvae_model
    output_size = tuple(mask_normalized.shape[-2:])

    image_tokens = vqvae.image_encoder(image)
    image_tokens = rearrange(image_tokens, "b c h w -> b (h w) c")

    token_ids_by_scale = vqvae.encode_to_multiscale_token_ids(mask_normalized)
    var_input = vqvae.to_var_input(token_ids_by_scale)

    logits_by_scale = model(var_input, image_tokens)
    teacher_token_ids = [logits.argmax(dim=-1) for logits in logits_by_scale]
    teacher_mask_logits = decode_token_ids_to_logits(model, teacher_token_ids, image_tokens, output_size)
    gt_mask_logits = decode_token_ids_to_logits(model, token_ids_by_scale, image_tokens, output_size)

    infer_token_ids_by_scale = model.autoregressive_infer(
        image_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    infer_mask_logits = decode_token_ids_to_logits(model, infer_token_ids_by_scale, image_tokens, output_size)

    teacher_cumulative_logits = []
    infer_cumulative_logits = []
    for scale_idx in range(len(token_ids_by_scale)):
        teacher_cumulative_logits.append(
            decode_token_ids_to_logits(
                model,
                teacher_token_ids[: scale_idx + 1],
                image_tokens,
                output_size,
            )
        )
        infer_cumulative_logits.append(
            decode_token_ids_to_logits(
                model,
                infer_token_ids_by_scale[: scale_idx + 1],
                image_tokens,
                output_size,
            )
        )

    return {
        "token_ids_by_scale": token_ids_by_scale,
        "teacher_token_ids": teacher_token_ids,
        "infer_token_ids_by_scale": infer_token_ids_by_scale,
        "gt_mask_logits": gt_mask_logits,
        "teacher_mask_logits": teacher_mask_logits,
        "infer_mask_logits": infer_mask_logits,
        "teacher_cumulative_logits": teacher_cumulative_logits,
        "infer_cumulative_logits": infer_cumulative_logits,
        "vq_loss": None,
    }


def iou_value(logits, gt_mask):
    return float(calc_iou((logits > 0).float(), gt_mask).item())


def summarize_metrics(outputs, gt_b):
    teacher_iou = iou_value(outputs["teacher_mask_logits"], gt_b)
    infer_iou = iou_value(outputs["infer_mask_logits"], gt_b)
    gt_recon_iou = iou_value(outputs["gt_mask_logits"], gt_b)
    teacher_cumulative_iou = [iou_value(logits, gt_b) for logits in outputs["teacher_cumulative_logits"]]
    infer_cumulative_iou = [iou_value(logits, gt_b) for logits in outputs["infer_cumulative_logits"]]
    return {
        "gt_recon_iou": gt_recon_iou,
        "teacher_iou": teacher_iou,
        "infer_iou": infer_iou,
        "teacher_cumulative_iou": teacher_cumulative_iou,
        "infer_cumulative_iou": infer_cumulative_iou,
    }


def visualize_summary(image, gt_mask, outputs, metrics, scales, save_path):
    image_np = restore_normalized_image(image).cpu().numpy().transpose(1, 2, 0)
    gt_np = to_2d_mask(gt_mask) > 0
    gt_recon_np = to_numpy_mask(outputs["gt_mask_logits"]) > 0
    teacher_np = to_numpy_mask(outputs["teacher_mask_logits"]) > 0
    infer_np = to_numpy_mask(outputs["infer_mask_logits"]) > 0

    teacher_overlay = render_overlay(image_np, gt_np, (0.2, 0.6, 1.0))
    infer_overlay = render_overlay(image_np, gt_np, (1.0, 0.35, 0.35))
    gt_recon_overlay = render_overlay(image_np, gt_np, (0.35, 0.85, 0.35))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Image")
    axes[0, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("GT")
    axes[0, 2].imshow(gt_recon_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"GT Recon IoU={metrics['gt_recon_iou']:.3f}")
    axes[0, 3].imshow(gt_recon_overlay)
    axes[0, 3].set_title("GT Recon Overlay")

    axes[1, 0].imshow(teacher_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Teacher IoU={metrics['teacher_iou']:.3f}")
    axes[1, 1].imshow(teacher_overlay)
    axes[1, 1].set_title("Teacher Overlay")
    axes[1, 2].imshow(infer_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title(f"AR Sample IoU={metrics['infer_iou']:.3f}")
    axes[1, 3].imshow(infer_overlay)
    axes[1, 3].set_title("AR Overlay")

    fig.suptitle("SimpleMaskVAR Summary", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize_progression(image, gt_mask, outputs, metrics, scales, save_path):
    image_np = restore_normalized_image(image).cpu().numpy().transpose(1, 2, 0)
    gt_np = to_2d_mask(gt_mask) > 0

    n_scales = len(scales)
    num_cols = n_scales + 2
    fig, axes = plt.subplots(2, num_cols, figsize=(2.8 * num_cols, 7.5))
    if num_cols == 1:
        axes = axes.reshape(2, 1)

    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Image")
    axes[0, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("GT")
    axes[1, 0].text(0.0, 0.5, "Teacher cumulative", fontsize=12)
    axes[1, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("GT ref")

    for scale_idx, scale in enumerate(scales):
        col = scale_idx + 2
        teacher_np = to_numpy_mask(outputs["teacher_cumulative_logits"][scale_idx]) > 0
        infer_np = to_numpy_mask(outputs["infer_cumulative_logits"][scale_idx]) > 0

        axes[0, col].imshow(teacher_np, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"<=S{scale}\nIoU={metrics['teacher_cumulative_iou'][scale_idx]:.3f}")

        axes[1, col].imshow(infer_np, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"<=S{scale}\nIoU={metrics['infer_cumulative_iou'][scale_idx]:.3f}")

    axes[0, 0].text(0.0, 1.08, "Teacher cumulative", fontsize=12, transform=axes[0, 0].transAxes)
    axes[1, 0].text(0.0, 1.08, "AR cumulative", fontsize=12, transform=axes[1, 0].transAxes)

    fig.suptitle("SimpleMaskVAR Scale Progression", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def add_row(rows, sample_order, dataset_idx, checkpoint_path, scales, metrics):
    row = {
        "sample_order": sample_order,
        "dataset_index": dataset_idx,
        "checkpoint": checkpoint_path,
        "gt_recon_iou": metrics["gt_recon_iou"],
        "teacher_iou": metrics["teacher_iou"],
        "infer_iou": metrics["infer_iou"],
    }
    for scale_idx, scale in enumerate(scales):
        row[f"teacher_cumulative_iou_to_s{scale}"] = metrics["teacher_cumulative_iou"][scale_idx]
        row[f"infer_cumulative_iou_to_s{scale}"] = metrics["infer_cumulative_iou"][scale_idx]
    rows.append(row)


def main():
    parser = argparse.ArgumentParser(description="Visualize SimpleMaskVAR teacher forcing and autoregressive sampling.")
    parser.add_argument("--out_dir", type=str, default="out/ddp_simple_mask_var_v2_overfit8_coconut_ep10")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--vqvae_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--vqvae_config", type=str, default=None)
    parser.add_argument("--vqvae_image_encoder_checkpoint", type=str, default=None)
    parser.add_argument("--vqvae_image_encoder_config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "hqseg44k", "cocolvis", "coconut_hf"])
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--mask_filter_thresh", type=float, default=0.1)
    parser.add_argument("--image_size_encoder", type=int, default=1024)
    parser.add_argument("--image_size_mask", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable_vq", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--train_subset_index", type=str, default=None)
    parser.add_argument("--val_subset_index", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    cfg = read_train_config(out_dir) if out_dir is not None else {}
    args = resolve_args_from_config(args, cfg)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.out_dir) / "simple_mask_var_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SimpleMaskVAR")
    print(f"  config: {args.config}")
    print(f"  checkpoint: {args.checkpoint_path}")
    print(f"  vqvae config: {args.vqvae_config}")
    print(f"  temperature: {args.temperature}, top_k: {args.top_k}")

    vqvae_model = builder_map["simple_mask_vqvae"][args.vqvae_config](
        simple_mask_vqvae_checkpoint_path=args.vqvae_checkpoint or cfg.get("vqvae_checkpoint"),
        image_encoder_checkpoint=args.vqvae_image_encoder_checkpoint,
        image_encoder_config_name=args.vqvae_image_encoder_config,
        device=str(device),
        enable_vq=bool(args.enable_vq) if args.enable_vq is not None else True,
    )
    vqvae_model.eval()

    model = builder_map["simple_mask_ar"][args.config](
        checkpoint_path=args.checkpoint_path,
        device=str(device),
        enable_click=False,
    )
    model.set_vqvae_model(vqvae_model)
    model.eval()
    ensure_var_model(model)

    dataset = build_dataset(args)
    scales = tuple(model.scales)
    print(f"Dataset: {args.dataset}/{args.split}, samples={len(dataset)}")
    print(f"Scales: {scales}")

    max_index = min(len(dataset), args.start_index + args.num_samples * args.sample_stride)
    sample_indices = list(range(args.start_index, max_index, args.sample_stride))[: args.num_samples]
    rows = []

    with torch.no_grad():
        for out_idx, dataset_idx in enumerate(sample_indices):
            image, _, mask_normalized, gt_mask = dataset[dataset_idx]
            image_b = image.unsqueeze(0).to(device)
            mask_b = mask_normalized.unsqueeze(0).to(device)
            gt_b = gt_mask.unsqueeze(0).to(device)

            outputs = collect_var_outputs(
                model,
                image_b,
                mask_b,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            metrics = summarize_metrics(outputs, gt_b)

            summary_path = output_dir / f"sample_{out_idx:03d}_dataset_{dataset_idx}_summary.png"
            progression_path = output_dir / f"sample_{out_idx:03d}_dataset_{dataset_idx}_progression.png"
            visualize_summary(image, gt_mask, outputs, metrics, scales, summary_path)
            visualize_progression(image, gt_mask, outputs, metrics, scales, progression_path)
            add_row(rows, out_idx, dataset_idx, args.checkpoint_path, scales, metrics)

            print(
                f"[{out_idx + 1}/{len(sample_indices)}] dataset_idx={dataset_idx} "
                f"gt_recon={metrics['gt_recon_iou']:.4f} teacher={metrics['teacher_iou']:.4f} "
                f"infer={metrics['infer_iou']:.4f}"
            )

    if rows:
        csv_path = output_dir / "summary.csv"
        json_path = output_dir / "summary.json"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with json_path.open("w") as f:
            json.dump(rows, f, indent=2)

        print(f"Mean teacher IoU: {np.mean([row['teacher_iou'] for row in rows]):.4f}")
        print(f"Mean infer IoU: {np.mean([row['infer_iou'] for row in rows]):.4f}")
        for scale in scales:
            teacher_mean = np.mean([row[f"teacher_cumulative_iou_to_s{scale}"] for row in rows])
            infer_mean = np.mean([row[f"infer_cumulative_iou_to_s{scale}"] for row in rows])
            print(f"  S={scale:<2} teacher={teacher_mean:.4f} infer={infer_mean:.4f}")

    print(f"Saved visualizations and summaries to: {output_dir}")


if __name__ == "__main__":
    main()
