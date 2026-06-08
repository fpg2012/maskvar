"""
Visualize PointRopeSAM predictions and sampled image points.

Examples:
    python -m notebooks.visualize_rope_sam_point \
        --checkpoint_path out/ddp_rope_sam_coconut_hf_dino_click/checkpoints/latest.pth \
        --config rope_sam_point_dim384 \
        --num_samples 4 \
        --split val

    python -m notebooks.visualize_rope_sam_point \
        --out_dir out/ddp_rope_sam_point_coconut_hf_dino_click \
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

from maskvar.maskseg_build_everything import builder_map
from maskvar.utils import restore_normalized_image
from maskvar.utils.clicker_v2 import init_clicks, to_sam_format
from train_scripts.train_rope_sam import build_datasets


def read_train_config(out_dir: Path) -> dict:
    config_path = out_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return json.load(f)


def resolve_args(args, cfg: dict):
    if args.checkpoint_path is None and args.out_dir is not None:
        args.checkpoint_path = str(Path(args.out_dir) / "checkpoints" / "latest.pth")
    args.config = args.config or cfg.get("config") or "rope_sam_point_dim384"
    args.image_encoder_config = args.image_encoder_config or cfg.get("image_encoder_config") or "dino_v3_vits"
    args.image_encoder_checkpoint = args.image_encoder_checkpoint or cfg.get("image_encoder_checkpoint")
    args.dataset = args.dataset or cfg.get("dataset") or "coconut_hf"
    args.dataset_path = args.dataset_path or cfg.get("dataset_path")
    args.train_subset_index = args.train_subset_index or cfg.get("train_subset_index")
    args.val_subset_index = args.val_subset_index or cfg.get("val_subset_index")
    args.image_feature_cache_dir = args.image_feature_cache_dir or cfg.get("image_feature_cache_dir") or ""
    args.image_feature_cache_max_shard = args.image_feature_cache_max_shard or cfg.get("image_feature_cache_max_shard") or 2
    args.debug = getattr(args, "debug", False)
    args.point_sampling_space = args.point_sampling_space or cfg.get("point_sampling_space") or "feature"
    args.num_points = args.num_points if args.num_points is not None else cfg.get("num_points")
    if args.point_rend_coarse_size == 16 and cfg.get("point_rend_coarse_size") is not None:
        args.point_rend_coarse_size = cfg["point_rend_coarse_size"]
    if args.point_rend_max_size == 256 and cfg.get("point_rend_max_size") is not None:
        args.point_rend_max_size = cfg["point_rend_max_size"]
    if args.sampling_strategy == "model" and cfg.get("point_sampling_strategy"):
        args.sampling_strategy = cfg["point_sampling_strategy"]
    return args


def build_dataset(args):
    train_set, val_set = build_datasets(args, device=args.device, rank=0)
    return train_set if args.split == "train" else val_set


def build_model(args):
    model = builder_map["rope_sam"][args.config](
        checkpoint_path=args.checkpoint_path,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        image_encoder_config_name=args.image_encoder_config,
        max_clicks=args.max_clicks,
        num_points=args.num_points,
        point_rend_coarse_size=args.point_rend_coarse_size,
        point_rend_max_size=args.point_rend_max_size,
        point_sampling_space=args.point_sampling_space,
        device=args.device,
    )
    if args.sampling_strategy != "model":
        if not hasattr(model, "sampling_strategy"):
            raise ValueError(f"Model config {args.config} does not expose sampling_strategy")
        model.sampling_strategy = args.sampling_strategy
        print(f"Override point sampling strategy: {args.sampling_strategy}")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def sample_click_condition(single_mask: torch.Tensor, ar_h: int, ar_w: int, max_clicks: int):
    mask_np = single_mask[0].detach().cpu().numpy() > 0
    click_list, _, _ = init_clicks(mask_np, num_random_clicks=1, random_sample=True)
    coords_xy, labels = to_sam_format(click_list, pad_size=max_clicks)

    h, w = single_mask.shape[-2:]
    click_coords = torch.empty_like(coords_xy, dtype=torch.float32)
    click_coords[..., 0] = coords_xy[..., 1] * (ar_h / h)
    click_coords[..., 1] = coords_xy[..., 0] * (ar_w / w)
    click_coords = click_coords.clamp_min(0)
    click_coords[..., 0].clamp_(max=ar_h - 1)
    click_coords[..., 1].clamp_(max=ar_w - 1)
    return click_coords.unsqueeze(0), labels.long().unsqueeze(0)


def parse_clicks(clicks: str | None) -> list[tuple[int, int, int]] | None:
    if not clicks:
        return None
    parsed = []
    for item in clicks.replace(";", " ").split():
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid click '{item}'. Expected y,x,label")
        y, x, label = (int(part) for part in parts)
        if label not in (0, 1):
            raise ValueError(f"Invalid click label {label}; use 1 for positive and 0 for negative")
        parsed.append((y, x, label))
    return parsed


def clicks_to_tensors(
    click_list: list[tuple[int, int, int]],
    mask_shape: tuple[int, int],
    ar_h: int,
    ar_w: int,
    max_clicks: int,
):
    coords_xy, labels = to_sam_format(click_list[:max_clicks], pad_size=max_clicks)
    mask_h, mask_w = mask_shape
    click_coords = torch.empty_like(coords_xy, dtype=torch.float32)
    click_coords[..., 0] = coords_xy[..., 1] * (ar_h / mask_h)
    click_coords[..., 1] = coords_xy[..., 0] * (ar_w / mask_w)
    click_coords = click_coords.clamp_min(0)
    click_coords[..., 0].clamp_(max=ar_h - 1)
    click_coords[..., 1].clamp_(max=ar_w - 1)
    return click_coords.unsqueeze(0), labels.long().unsqueeze(0)


def draw_clicks(ax, click_coords, click_labels, image_shape):
    img_h, img_w = image_shape
    coords = click_coords.detach().float().cpu()
    labels = click_labels.detach().cpu()
    for coord, label in zip(coords, labels):
        if int(label) < 0:
            continue
        y = float(coord[0]) * img_h / 64.0
        x = float(coord[1]) * img_w / 64.0
        color = "lime" if int(label) == 1 else "red"
        marker = "+" if int(label) == 1 else "x"
        ax.scatter([x], [y], c=color, marker=marker, s=80, linewidths=2)


def draw_points(ax, point_coords, image_shape, coord_size: tuple[int, int], max_points: int, color_by_order: bool = True):
    img_h, img_w = image_shape
    coord_h, coord_w = coord_size
    coords = point_coords.detach().float().cpu()
    order = torch.arange(coords.shape[0], dtype=torch.float32)
    if coords.shape[0] > max_points:
        indices = torch.randperm(coords.shape[0])[:max_points]
        coords = coords[indices]
        order = order[indices]
    ys = (coords[:, 0] + 0.5) * img_h / coord_h
    xs = (coords[:, 1] + 0.5) * img_w / coord_w
    if color_by_order:
        ax.scatter(xs, ys, s=3, c=order, cmap="viridis", alpha=0.55, linewidths=0)
    else:
        ax.scatter(xs, ys, s=2, c="cyan", alpha=0.35, linewidths=0)


def valid_mask_from_image(image: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    valid = image.float().abs().sum(dim=1, keepdim=True) > 1e-6
    valid = F.interpolate(valid.float(), size=output_size, mode="nearest") > 0
    return valid


def masked_iou(logits: torch.Tensor, gt_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    pred = (logits > 0) & valid_mask
    target = (gt_mask > 0.5) & valid_mask
    intersection = (pred & target).sum(dim=(1, 2, 3))
    union = (pred | target).sum(dim=(1, 2, 3))
    return torch.where(union > 0, intersection.float() / union.float(), torch.ones_like(union, dtype=torch.float32))


def build_error_overlay(image_np: np.ndarray, gt_np: np.ndarray, pred_np: np.ndarray, valid_np: np.ndarray) -> np.ndarray:
    overlay = image_np.copy().astype(np.float32)
    overlay[~valid_np] = 0.0

    tp = gt_np & pred_np & valid_np
    fp = (~gt_np) & pred_np & valid_np
    fn = gt_np & (~pred_np) & valid_np
    alpha = 0.55
    colors = [
        (tp, np.array([0.2, 0.6, 1.0], dtype=np.float32)),
        (fp, np.array([1.0, 0.25, 0.25], dtype=np.float32)),
        (fn, np.array([0.25, 0.9, 0.25], dtype=np.float32)),
    ]
    for mask, color in colors:
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
    return np.clip(overlay, 0.0, 1.0)


def resize_image_np(image_np: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
    image_t = F.interpolate(image_t, size=output_size, mode="bilinear", align_corners=False)
    return image_t[0].permute(1, 2, 0).numpy()


@torch.no_grad()
def predict_and_sample_points(model, image, click_coords, click_labels, output_size):
    image_features = model.encode_image(image)
    query_tokens = model.encode_clicks(click_coords, click_labels)
    point_coord_size = output_size if model.point_sampling_space == "output" else (model.h, model.w)
    prev_mask_features = None

    logits = model(
        image=image,
        image_embedding=image_features,
        click_coords=click_coords,
        click_labels=click_labels,
        output_size=output_size,
    )

    if getattr(model, "sampling_strategy", None) == "pointrend":
        valid_mask = torch.ones(
            image.shape[0],
            point_coord_size[0],
            point_coord_size[1],
            device=image.device,
            dtype=torch.bool,
        )
        _, point_coords = model.pointrend_refine_logits(
            image_features=image_features,
            query_tokens=query_tokens,
            valid_mask=valid_mask,
            coord_size=point_coord_size,
            output_size=output_size,
            prev_mask_features=prev_mask_features,
            return_point_coords=True,
        )
    else:
        point_coords = model.sample_point_coords(
            image=image,
            image_features=image_features,
            query_tokens=query_tokens,
            prev_mask_features=prev_mask_features,
            click_coords=click_coords,
            click_labels=click_labels,
            coord_size=point_coord_size,
        )
    return logits, point_coords, point_coord_size


def main():
    parser = argparse.ArgumentParser(description="Visualize PointRopeSAM sampled points and masks.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, choices=sorted(builder_map["rope_sam"].keys()))
    parser.add_argument("--image_encoder_config", type=str, default=None, choices=sorted(builder_map["image_encoder"].keys()))
    parser.add_argument("--image_encoder_checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=sorted(builder_map["dataset"].keys()))
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--train_subset_index", type=str, default=None)
    parser.add_argument("--val_subset_index", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--max_clicks", type=int, default=10)
    parser.add_argument("--mask_filter_thresh", type=float, default=0.1)
    parser.add_argument("--image_size_encoder", type=int, default=1024)
    parser.add_argument("--image_size_mask", type=int, default=256)
    parser.add_argument("--image_feature_cache_dir", type=str, default="")
    parser.add_argument("--image_feature_cache_max_shard", type=int, default=2)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_points_draw", type=int, default=4096)
    parser.add_argument("--num_points", type=int, default=None)
    parser.add_argument("--point_rend_coarse_size", type=int, default=16)
    parser.add_argument("--point_rend_max_size", type=int, default=256)
    parser.add_argument("--point_sampling_space", type=str, default=None, choices=["feature", "output"])
    parser.add_argument("--sampling_strategy", type=str, default="model", choices=["model", "uniform", "edge", "pointrend"])
    parser.add_argument(
        "--clicks",
        type=str,
        default=None,
        help="Manual clicks as 'y,x,label y,x,label'. label: 1 positive, 0 negative. Reused for every sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = read_train_config(Path(args.out_dir)) if args.out_dir else {}
    args = resolve_args(args, cfg)
    if args.checkpoint_path is None:
        raise ValueError("--checkpoint_path or --out_dir is required")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.out_dir) / "point_visualizations" if args.out_dir else Path("out/point_rope_sam_visualizations")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    dataset = build_dataset(args)
    model = build_model(args)

    sample_indices = list(
        range(args.start_index, min(len(dataset), args.start_index + args.num_samples * args.sample_stride), args.sample_stride)
    )[: args.num_samples]
    manual_clicks = parse_clicks(args.clicks)
    rows = []
    for out_idx, dataset_idx in enumerate(sample_indices):
        sample = dataset[dataset_idx]
        if len(sample) == 6:
            image, _, _, single_mask, dataset_click_coords, dataset_click_labels = sample
        else:
            image, _, _, single_mask = sample
            dataset_click_coords = None
            dataset_click_labels = None
        image = image.unsqueeze(0).to(args.device)
        gt_mask = single_mask.unsqueeze(0).to(args.device)
        if manual_clicks is None:
            if dataset_click_coords is not None and dataset_click_labels is not None:
                click_coords = dataset_click_coords.unsqueeze(0)
                click_labels = dataset_click_labels.long().unsqueeze(0)
            else:
                click_coords, click_labels = sample_click_condition(single_mask, 64, 64, args.max_clicks)
        else:
            click_coords, click_labels = clicks_to_tensors(manual_clicks, single_mask.shape[-2:], 64, 64, args.max_clicks)
        click_coords = click_coords.to(args.device)
        click_labels = click_labels.to(args.device)

        torch.manual_seed(args.seed + dataset_idx)
        with torch.no_grad():
            logits, point_coords, point_coord_size = predict_and_sample_points(
                model=model,
                image=image,
                click_coords=click_coords,
                click_labels=click_labels,
                output_size=gt_mask.shape[-2:],
            )
            valid_mask = valid_mask_from_image(image, gt_mask.shape[-2:])
            logits = logits.masked_fill(~valid_mask, 0.0)
            iou = masked_iou(logits, gt_mask, valid_mask)[0]

        img_np = restore_normalized_image(image[0]).detach().float().cpu().numpy().transpose(1, 2, 0)
        img_np = img_np / 255.0 if img_np.max() > 1 else img_np
        valid_np = valid_mask[0, 0].detach().cpu().numpy()
        gt_np = (gt_mask[0, 0].detach().float().cpu().numpy() > 0) & valid_np
        logits_np = logits[0, 0].detach().float().cpu().numpy()
        logits_np = np.where(valid_np, logits_np, 0.0)
        pred_np = (logits_np > 0) & valid_np
        overlay_image_np = resize_image_np(img_np, gt_np.shape)
        overlay_np = build_error_overlay(overlay_image_np, gt_np, pred_np, valid_np)

        fig, axes = plt.subplots(1, 7, figsize=(28, 4))
        axes[0].imshow(img_np)
        draw_clicks(axes[0], click_coords[0], click_labels[0], img_np.shape[:2])
        axes[0].set_title("Image + clicks")
        axes[0].axis("off")

        axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("GT")
        axes[1].axis("off")

        axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[2].set_title(f"Pred IoU={iou.item():.3f}")
        axes[2].axis("off")

        vmax_abs = max(abs(float(logits_np.min())), abs(float(logits_np.max())), 1e-6)
        axes[3].imshow(logits_np, cmap="RdBu_r", vmin=-vmax_abs, vmax=vmax_abs)
        axes[3].set_title("Logits")
        axes[3].axis("off")

        axes[4].imshow(overlay_np)
        axes[4].set_title("Overlay B=TP G=FN R=FP")
        axes[4].axis("off")

        axes[5].imshow(img_np)
        draw_points(axes[5], point_coords[0], img_np.shape[:2], point_coord_size, args.max_points_draw)
        axes[5].set_title("Sampled points")
        axes[5].axis("off")

        axes[6].imshow(np.ones_like(img_np))
        draw_points(axes[6], point_coords[0], img_np.shape[:2], point_coord_size, args.max_points_draw)
        axes[6].set_title("Points only")
        axes[6].axis("off")

        fig.tight_layout()
        fig_path = output_dir / f"sample_{out_idx:03d}_dataset_{dataset_idx}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        rows.append({
            "sample_order": out_idx,
            "dataset_idx": dataset_idx,
            "iou": float(iou.item()),
            "figure": str(fig_path),
        })
        print(f"[{out_idx + 1}/{len(sample_indices)}] dataset_idx={dataset_idx} IoU={iou.item():.4f} -> {fig_path}")

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_order", "dataset_idx", "iou", "figure"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote metrics to {csv_path}")


if __name__ == "__main__":
    main()
