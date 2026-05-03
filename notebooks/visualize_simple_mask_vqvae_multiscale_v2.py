"""
Visualize VAR-style SimpleMaskVqvae multiscale V2.

This script is intentionally separate from the older multiscale visualizer.
V2 stores its multiscale behavior inside model.quant
(MultiscaleVectorQuantize), while the outer model keeps the same forward shape
as SimpleMaskVqvae.

Example:
    python -m notebooks.visualize_simple_mask_vqvae_multiscale_v2 \
        --out_dir out/ddp_simple_mask_vqvae_multiscale_v2_coconut_ep10 \
        --num_samples 4 \
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

from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset
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
    args.config = args.config or cfg.get("config") or "simple_mask_vqvae_multiscale_v2_dim384"
    args.image_encoder_checkpoint = args.image_encoder_checkpoint or cfg.get("image_encoder_checkpoint")
    args.image_encoder_config = args.image_encoder_config or cfg.get("image_encoder_config") or "dino_v3_vits"
    args.dataset = args.dataset or cfg.get("dataset") or "coconut_hf"
    args.dataset_path = args.dataset_path or cfg.get("dataset_path")
    args.enable_vq = cfg.get("enable_vq", args.enable_vq) if args.enable_vq is None else args.enable_vq
    return args


def build_dataset(args):
    dataset_path = args.dataset_path or DATASET_PATHS[args.dataset]
    train_set_base, val_set_base = builder_map["dataset"][args.dataset](dataset_path)
    base_set = train_set_base if args.split == "train" else val_set_base
    index_mapping_path = Path("data/flat") / args.dataset / f"{args.split}_index_mapping.npy"
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


def ensure_v2_model(model):
    if not hasattr(model, "quant") or not hasattr(model.quant, "idxBl_to_full_tokens"):
        raise TypeError(
            "This visualizer expects SimpleMaskVqvae multiscale V2 with "
            "MultiscaleVectorQuantize."
        )


def decode_full_tokens(model, full_tokens_blc, image_tokens, output_size):
    mask_tokens = rearrange(full_tokens_blc, "b (h w) c -> b h w c", h=model.h, w=model.w)
    logits = model.mask_decoder(mask_tokens, image_tokens)
    if logits.shape[-2:] != output_size:
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
    return logits


@torch.no_grad()
def collect_v2_outputs(model, image, mask_normalized):
    ensure_v2_model(model)

    quant = model.quant
    output_size = tuple(mask_normalized.shape[-2:])

    mask_feature = model.mask_encoder(mask_normalized)
    image_tokens = model.image_encoder(image)
    image_tokens = rearrange(image_tokens, "b c h w -> b h w c")

    B, C, h, w = mask_feature.shape
    if (h, w) != (quant.h, quant.w):
        raise ValueError(f"Expected quant grid {quant.h}x{quant.w}, got {h}x{w}")

    z = rearrange(mask_feature, "b c h w -> b (h w) c").float()
    z_map = quant._tokens_to_map(z, h=quant.h, w=quant.w)
    residual = quant._tokens_to_map(z.detach(), h=quant.h, w=quant.w).clone()
    full_map = torch.zeros_like(residual)

    projected_by_scale = []
    token_ids_by_scale = []
    contribution_norms = []
    vq_loss = z.new_tensor(0.0)

    device_type = "cuda" if z.is_cuda else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        for scale_idx, scale in enumerate(quant.scales):
            residual_at_scale = F.interpolate(residual, size=(scale, scale), mode="area")
            tokens = quant._map_to_tokens(residual_at_scale)
            indices = quant._nearest_code_indices(tokens)
            token_ids = indices.view(B, scale * scale)
            tokens_q = quant.embedding(indices.view(B, scale, scale)).view(B, scale * scale, C)
            projected = quant._project_tokens_to_full(tokens_q, scale_idx)

            full_map = full_map + projected
            residual = residual - projected
            vq_loss = vq_loss + (
                F.mse_loss(full_map.data, z_map).mul(quant.beta)
                + F.mse_loss(full_map, z_map.detach())
            )

            projected_by_scale.append(projected)
            token_ids_by_scale.append(token_ids)

            norm_map = projected.float().pow(2).mean(dim=1, keepdim=True).sqrt()
            if norm_map.shape[-2:] != output_size:
                norm_map = F.interpolate(norm_map, size=output_size, mode="bilinear", align_corners=False)
            contribution_norms.append(norm_map)

    vq_loss = vq_loss / len(quant.scales)

    full_tokens = quant._map_to_tokens(full_map)
    full_logits = decode_full_tokens(model, full_tokens, image_tokens, output_size)

    single_logits = []
    cumulative_logits = []
    drop_one_logits = []
    for scale_idx, projected in enumerate(projected_by_scale):
        single_tokens = quant._map_to_tokens(projected)
        single_logits.append(decode_full_tokens(model, single_tokens, image_tokens, output_size))

        cumulative_map = torch.stack(projected_by_scale[: scale_idx + 1], dim=0).sum(dim=0)
        cumulative_tokens = quant._map_to_tokens(cumulative_map)
        cumulative_logits.append(decode_full_tokens(model, cumulative_tokens, image_tokens, output_size))

        drop_map = full_map - projected
        drop_tokens = quant._map_to_tokens(drop_map)
        drop_one_logits.append(decode_full_tokens(model, drop_tokens, image_tokens, output_size))

    return {
        "full": full_logits,
        "single": single_logits,
        "cumulative": cumulative_logits,
        "drop_one": drop_one_logits,
        "contribution_norms": contribution_norms,
        "token_ids_by_scale": token_ids_by_scale,
        "vq_loss": vq_loss,
        "scale_gates": quant.scale_gates.detach().float().cpu().tolist()
        if hasattr(quant, "scale_gates")
        else None,
    }


def iou_value(logits, gt_mask):
    return float(calc_iou((logits > 0).float(), gt_mask).item())


def prob_stats(logits):
    prob = torch.sigmoid(logits.float())
    return {
        "mean": float(prob.mean().item()),
        "max": float(prob.max().item()),
        "area": float((prob > 0.5).float().mean().item()),
    }


def to_numpy_mask(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0, 0]
    elif tensor.dim() == 3:
        tensor = tensor[0]
    return tensor.detach().float().cpu().numpy()


def sigmoid_np(logits):
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))


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


def render_overlay(image_np, mask_np, color):
    base = image_np.astype(np.float32) / 255.0
    mask_bool = mask_np > 0
    overlay = base.copy()
    overlay[mask_bool] = overlay[mask_bool] * 0.55 + np.array(color, dtype=np.float32) * 0.45
    return np.clip(overlay, 0, 1)


def visualize_sample(image, gt_mask, outputs, scales, metrics, save_path):
    image_np = restore_normalized_image(image).cpu().numpy().transpose(1, 2, 0)
    gt_np = gt_mask[0].cpu().numpy() > 0
    full_np = to_numpy_mask(outputs["full"])
    full_pred = full_np > 0

    n_scales = len(scales)
    fig, axes = plt.subplots(4, n_scales + 2, figsize=(3.0 * (n_scales + 2), 12))
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Image")
    axes[0, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("GT")
    axes[0, 2].imshow(full_pred, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title(f"Full IoU {metrics['full_iou']:.3f}")

    err = np.zeros((*gt_np.shape, 3), dtype=np.float32)
    err[gt_np & full_pred] = (0.2, 0.6, 1.0)
    err[(~gt_np) & full_pred] = (1.0, 0.25, 0.25)
    err[gt_np & (~full_pred)] = (0.25, 0.9, 0.25)
    axes[0, 3].imshow(err)
    axes[0, 3].set_title("Full Error")

    axes[0, 4].imshow(render_overlay(image_np, full_pred, (0.2, 0.6, 1.0)))
    axes[0, 4].set_title("Full Overlay")

    vmax_abs = max(abs(float(full_np.min())), abs(float(full_np.max())), 1e-6)
    axes[0, 5].imshow(full_np, cmap="RdBu_r", vmin=-vmax_abs, vmax=vmax_abs)
    axes[0, 5].set_title("Full Logits")

    for scale_idx, scale in enumerate(scales):
        col = scale_idx + 2
        single_np = to_numpy_mask(outputs["single"][scale_idx])
        cumulative_np = to_numpy_mask(outputs["cumulative"][scale_idx])
        drop_np = to_numpy_mask(outputs["drop_one"][scale_idx])
        contribution_np = to_numpy_mask(outputs["contribution_norms"][scale_idx])
        logit_delta_np = full_np - drop_np

        gate = metrics["scale_gates"][scale_idx] if metrics["scale_gates"] is not None else float("nan")
        vmin, vmax = robust_limits(single_np, center_zero=True)
        axes[1, col].imshow(single_np, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[1, col].set_title(
            f"S={scale} logits\nIoU {metrics['single_iou'][scale_idx]:.3f} gate {gate:.3f}"
        )

        vmin, vmax = robust_limits(cumulative_np, center_zero=True)
        axes[2, col].imshow(cumulative_np, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[2, col].set_title(
            f"<=S logits\nIoU {metrics['cumulative_iou'][scale_idx]:.3f} "
            f"max {metrics['cumulative_prob_max'][scale_idx]:.2f}"
        )

        vmin, vmax = robust_limits(logit_delta_np, center_zero=True)
        axes[3, col].imshow(logit_delta_np, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        axes[3, col].contour(contribution_np, levels=3, colors="black", linewidths=0.35, alpha=0.55)
        axes[3, col].set_title(
            f"full - drop S\n"
            f"dIoU {metrics['drop_one_delta'][scale_idx]:+.3f} "
            f"norm {metrics['contribution_norm_mean'][scale_idx]:.2f}"
        )

    axes[1, 0].text(0.0, 0.5, "Single-scale decoded contribution", fontsize=13)
    axes[2, 0].text(0.0, 0.5, "Cumulative coarse-to-fine", fontsize=13)
    axes[3, 0].text(0.0, 0.5, "Drop-one logit delta", fontsize=13)
    for row in (1, 2, 3):
        axes[row, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("GT ref")

    fig.suptitle("SimpleMaskVqvae Multiscale V2 Scale Analysis", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_metrics(outputs, gt_b):
    full_iou = iou_value(outputs["full"], gt_b)
    single_iou = [iou_value(logits, gt_b) for logits in outputs["single"]]
    cumulative_iou = [iou_value(logits, gt_b) for logits in outputs["cumulative"]]
    drop_one_iou = [iou_value(logits, gt_b) for logits in outputs["drop_one"]]
    single_stats = [prob_stats(logits) for logits in outputs["single"]]
    cumulative_stats = [prob_stats(logits) for logits in outputs["cumulative"]]
    contribution_norm_mean = [
        float(norm_map.float().mean().item()) for norm_map in outputs["contribution_norms"]
    ]
    return {
        "full_iou": full_iou,
        "single_iou": single_iou,
        "cumulative_iou": cumulative_iou,
        "drop_one_iou": drop_one_iou,
        "drop_one_delta": [full_iou - value for value in drop_one_iou],
        "single_prob_mean": [item["mean"] for item in single_stats],
        "single_prob_max": [item["max"] for item in single_stats],
        "single_prob_area": [item["area"] for item in single_stats],
        "cumulative_prob_mean": [item["mean"] for item in cumulative_stats],
        "cumulative_prob_max": [item["max"] for item in cumulative_stats],
        "cumulative_prob_area": [item["area"] for item in cumulative_stats],
        "contribution_norm_mean": contribution_norm_mean,
        "scale_gates": outputs["scale_gates"],
    }


def add_row_for_sample(rows, sample_order, dataset_idx, checkpoint_path, scales, outputs, metrics):
    row = {
        "sample_order": sample_order,
        "dataset_index": dataset_idx,
        "checkpoint": checkpoint_path,
        "full_iou": metrics["full_iou"],
        "vq_loss": float(outputs["vq_loss"].item()),
    }
    for scale_idx, scale in enumerate(scales):
        token_ids = outputs["token_ids_by_scale"][scale_idx].detach().cpu().view(-1).numpy()
        row[f"single_iou_s{scale}"] = metrics["single_iou"][scale_idx]
        row[f"single_prob_mean_s{scale}"] = metrics["single_prob_mean"][scale_idx]
        row[f"single_prob_max_s{scale}"] = metrics["single_prob_max"][scale_idx]
        row[f"single_prob_area_s{scale}"] = metrics["single_prob_area"][scale_idx]
        row[f"cumulative_iou_to_s{scale}"] = metrics["cumulative_iou"][scale_idx]
        row[f"cumulative_prob_mean_to_s{scale}"] = metrics["cumulative_prob_mean"][scale_idx]
        row[f"cumulative_prob_max_to_s{scale}"] = metrics["cumulative_prob_max"][scale_idx]
        row[f"cumulative_prob_area_to_s{scale}"] = metrics["cumulative_prob_area"][scale_idx]
        row[f"drop_one_iou_without_s{scale}"] = metrics["drop_one_iou"][scale_idx]
        row[f"drop_one_delta_s{scale}"] = metrics["drop_one_delta"][scale_idx]
        row[f"contribution_norm_mean_s{scale}"] = metrics["contribution_norm_mean"][scale_idx]
        row[f"token_unique_s{scale}"] = int(np.unique(token_ids).size)
        if metrics["scale_gates"] is not None:
            row[f"scale_gate_s{scale}"] = metrics["scale_gates"][scale_idx]
    rows.append(row)


def main():
    parser = argparse.ArgumentParser(description="Visualize SimpleMaskVqvae multiscale V2 scale behavior.")
    parser.add_argument("--out_dir", type=str, default="out/ddp_simple_mask_vqvae_multiscale_v2_coconut_ep10")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--image_encoder_checkpoint", type=str, default=None)
    parser.add_argument("--image_encoder_config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=[None, "hqseg44k", "cocolvis", "coconut_hf"])
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--mask_filter_thresh", type=float, default=0.1)
    parser.add_argument("--image_size_encoder", type=int, default=1024)
    parser.add_argument("--image_size_mask", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable_vq", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir is not None else None
    cfg = read_train_config(out_dir) if out_dir is not None else {}
    args = resolve_args_from_config(args, cfg)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.out_dir) / "multiscale_v2_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SimpleMaskVqvae multiscale V2")
    print(f"  config: {args.config}")
    print(f"  checkpoint: {args.checkpoint_path}")
    print(f"  image encoder: {args.image_encoder_config} ({args.image_encoder_checkpoint})")
    print(f"  enable_vq: {args.enable_vq}")

    model = builder_map["simple_mask_vqvae"][args.config](
        simple_mask_vqvae_checkpoint_path=args.checkpoint_path,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        image_encoder_config_name=args.image_encoder_config,
        device=str(device),
        enable_vq=bool(args.enable_vq),
    )
    model.eval()
    ensure_v2_model(model)

    dataset = build_dataset(args)
    scales = tuple(model.quant.scales)
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

            outputs = collect_v2_outputs(model, image_b, mask_b)
            metrics = summarize_metrics(outputs, gt_b)

            save_path = output_dir / f"sample_{out_idx:03d}_dataset_{dataset_idx}_v2_scales.png"
            visualize_sample(image, gt_mask, outputs, scales, metrics, save_path)
            add_row_for_sample(rows, out_idx, dataset_idx, args.checkpoint_path, scales, outputs, metrics)

            print(
                f"[{out_idx + 1}/{len(sample_indices)}] dataset_idx={dataset_idx} "
                f"full_iou={metrics['full_iou']:.4f} vq={outputs['vq_loss'].item():.4f} "
                f"saved={save_path}"
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

        full_mean = np.mean([row["full_iou"] for row in rows])
        print(f"Mean full IoU: {full_mean:.4f}")
        for scale in scales:
            single_mean = np.mean([row[f"single_iou_s{scale}"] for row in rows])
            cum_mean = np.mean([row[f"cumulative_iou_to_s{scale}"] for row in rows])
            delta_mean = np.mean([row[f"drop_one_delta_s{scale}"] for row in rows])
            norm_mean = np.mean([row[f"contribution_norm_mean_s{scale}"] for row in rows])
            gate_mean = rows[0].get(f"scale_gate_s{scale}", float("nan"))
            print(
                f"  S={scale:<2} single={single_mean:.4f} cumulative={cum_mean:.4f} "
                f"drop_delta={delta_mean:+.4f} norm={norm_mean:.4f} gate={gate_mean:.4f}"
            )

    print(f"Saved visualizations and summaries to: {output_dir}")


if __name__ == "__main__":
    main()
