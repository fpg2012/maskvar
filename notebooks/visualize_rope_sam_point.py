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
import torch
import torch.nn.functional as F

from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset, MaskLevelFlatSubsetDataset
from maskvar.maskseg_build_everything import builder_map
from maskvar.utils import restore_normalized_image
from maskvar.utils.clicker_v2 import init_clicks, to_sam_format
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
    return args


def build_dataset(args):
    dataset_path = args.dataset_path or DATASET_PATHS[args.dataset]
    train_set_base, val_set_base = builder_map["dataset"][args.dataset](dataset_path)
    base_set = train_set_base if args.split == "train" else val_set_base
    subset_path = args.train_subset_index if args.split == "train" else args.val_subset_index
    index_mapping_path = Path("data/flat") / args.dataset / f"{args.split}_index_mapping.npy"
    common_kwargs = {
        "index_mapping_path": index_mapping_path,
        "dataset": base_set,
        "with_image_embed": False,
        "image_feature_cache": None,
        "mask_filter_thresh": args.mask_filter_thresh,
        "dtype": torch.float32,
        "image_size_encoder": args.image_size_encoder,
        "image_size_mask": args.image_size_mask,
    }
    if subset_path:
        return MaskLevelFlatSubsetDataset(subset_list=Path(subset_path), **common_kwargs)
    return MaskLevelFlatDataset(**common_kwargs)


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


def draw_points(ax, point_coords, image_shape, max_points: int):
    img_h, img_w = image_shape
    coords = point_coords.detach().float().cpu()
    if coords.shape[0] > max_points:
        indices = torch.randperm(coords.shape[0])[:max_points]
        coords = coords[indices]
    ys = (coords[:, 0] + 0.5) * img_h / 64.0
    xs = (coords[:, 1] + 0.5) * img_w / 64.0
    ax.scatter(xs, ys, s=2, c="cyan", alpha=0.35, linewidths=0)


def main():
    parser = argparse.ArgumentParser(description="Visualize PointRopeSAM sampled points and masks.")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None, choices=sorted(builder_map["rope_sam"].keys()))
    parser.add_argument("--image_encoder_config", type=str, default=None, choices=sorted(builder_map["image_encoder"].keys()))
    parser.add_argument("--image_encoder_checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, choices=sorted(DATASET_PATHS.keys()))
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
    parser.add_argument("--max_points_draw", type=int, default=4096)
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
    model = builder_map["rope_sam"][args.config](
        checkpoint_path=args.checkpoint_path,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        image_encoder_config_name=args.image_encoder_config,
        max_clicks=args.max_clicks,
        device=args.device,
    )
    model.eval()

    sample_indices = list(
        range(args.start_index, min(len(dataset), args.start_index + args.num_samples * args.sample_stride), args.sample_stride)
    )[: args.num_samples]
    rows = []
    for out_idx, dataset_idx in enumerate(sample_indices):
        image, _, _, single_mask = dataset[dataset_idx]
        image = image.unsqueeze(0).to(args.device)
        gt_mask = single_mask.unsqueeze(0).to(args.device)
        click_coords, click_labels = sample_click_condition(single_mask, 64, 64, args.max_clicks)
        click_coords = click_coords.to(args.device)
        click_labels = click_labels.to(args.device)

        torch.manual_seed(args.seed + dataset_idx)
        with torch.no_grad():
            point_coords = model.sample_point_coords(image)
            density = model.edge_density(image)
            logits = model(
                image=image,
                click_coords=click_coords,
                click_labels=click_labels,
                output_size=gt_mask.shape[-2:],
                point_coords=point_coords,
            )
            iou = calc_iou(logits, gt_mask)[0]

        img_np = restore_normalized_image(image[0]).detach().float().cpu().numpy().transpose(1, 2, 0)
        img_np = img_np / 255.0 if img_np.max() > 1 else img_np
        gt_np = gt_mask[0, 0].detach().float().cpu().numpy() > 0
        logits_np = logits[0, 0].detach().float().cpu().numpy()
        pred_np = logits_np > 0
        density_np = F.interpolate(density, size=gt_mask.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        density_np = density_np.detach().float().cpu().numpy()

        fig, axes = plt.subplots(1, 6, figsize=(24, 4))
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

        axes[4].imshow(density_np, cmap="magma")
        axes[4].set_title("Edge density")
        axes[4].axis("off")

        axes[5].imshow(img_np)
        draw_points(axes[5], point_coords[0], img_np.shape[:2], args.max_points_draw)
        axes[5].set_title("Sampled points")
        axes[5].axis("off")

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
