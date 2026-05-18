"""
Visualize click-conditioned SimpleMaskMaskGIT teacher forcing and MaskGIT sampling.

Example:
    python -m notebooks.visualize_simple_mask_maskgit_click \
        --out_dir out/ddp_simple_mask_maskgit_coconut_click_ep10 \
        --num_samples_per_split 4 \
        --num_infer_samples 4 \
        --maskgit_num_steps 12

The script samples from both train and val splits, saves one teacher summary and
one step-by-step sampling figure per inference sample, and writes summary.csv/json.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset, MaskLevelFlatSubsetDataset
from maskvar.maskseg_build_everything import builder_map
from maskvar.models.simple_mask_ar.simple_mask_ar import make_uncond_click_labels, sample_from_logits
from maskvar.utils import restore_normalized_image
from maskvar.utils.clicker_v2 import init_clicks, to_sam_format
from maskvar.utils.metrics import calc_iou


COLOR_TP = np.array([0.2, 0.6, 1.0], dtype=np.float32)
COLOR_FP = np.array([1.0, 0.3, 0.3], dtype=np.float32)
COLOR_FN = np.array([0.3, 0.9, 0.3], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SimpleMaskMaskGIT click-conditioned outputs.")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / "out/ddp_simple_mask_maskgit_coconut_click_ep10",
        help="MaskGIT training output directory containing config.json and checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=None,
        help="MaskGIT checkpoint path. Defaults to out_dir/checkpoints/latest.pth.",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for saved visualizations.")
    parser.add_argument("--num_samples_per_split", type=int, default=4)
    parser.add_argument(
        "--num_infer_samples",
        type=int,
        default=1,
        help="Number of stochastic MaskGIT samples to draw per selected dataset item.",
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--sample_stride", type=int, default=20)
    parser.add_argument(
        "--train_indices",
        type=str,
        default=None,
        help="Comma-separated train dataset indices. Overrides start/stride for train.",
    )
    parser.add_argument(
        "--val_indices",
        type=str,
        default=None,
        help="Comma-separated val dataset indices. Overrides start/stride for val.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default=None, choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--num_clicks", type=int, default=0, help="0 samples 1..max_clicks clicks like training.")
    parser.add_argument("--max_clicks", type=int, default=2)
    parser.add_argument("--teacher_mask_ratio", type=float, default=1.0)
    parser.add_argument("--maskgit_num_steps", type=int, default=None)
    parser.add_argument("--maskgit_sampling_mode", type=str, default=None, choices=["confidence", "click_expand"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0, help="0 disables top-k.")
    parser.add_argument("--min_p", type=float, default=0.0, help="0 disables min-p.")
    parser.add_argument("--cfg_guidance_scale", type=float, default=None)
    parser.add_argument("--cfg_keep_click", action="store_true")
    parser.add_argument("--cfg_drop_image", action="store_true")
    parser.add_argument("--overlay_alpha", type=float, default=0.25)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def resolve_repo_path(path_like):
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(out_dir):
    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing training config: {cfg_path}")
    with cfg_path.open("r") as f:
        return json.load(f)


def parse_index_list(value):
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def select_indices(dataset, explicit, start_index, count, stride):
    if explicit is not None:
        return [idx for idx in explicit if 0 <= idx < len(dataset)]
    max_index = min(len(dataset), start_index + count * stride)
    return list(range(start_index, max_index, stride))[:count]


def build_split_dataset(cfg, split):
    dataset_path_map = {
        "hqseg44k": REPO_ROOT / "data/sam-hq",
        "cocolvis": REPO_ROOT / "data/coco_lvis",
        "coconut_hf": REPO_ROOT / "data/coconut_hf",
    }
    dataset_name = cfg["dataset"]
    dataset_path = resolve_repo_path(cfg["dataset_path"]) if cfg.get("dataset_path") else dataset_path_map[dataset_name]
    train_base, val_base = builder_map["dataset"][dataset_name](str(dataset_path))
    dataset_base = train_base if split == "train" else val_base
    index_mapping_path = REPO_ROOT / f"data/flat/{dataset_name}/{split}_index_mapping.npy"

    dataset_kwargs = dict(
        index_mapping_path=index_mapping_path,
        dataset=dataset_base,
        with_image_embed=False,
        image_feature_cache=None,
        mask_filter_thresh=0.1,
        dtype=torch.float32,
        image_size_encoder=1024,
        image_size_mask=1024,
    )
    subset_path = cfg.get(f"{split}_subset_index")
    if subset_path:
        return MaskLevelFlatSubsetDataset(subset_list=resolve_repo_path(subset_path), **dataset_kwargs)
    return MaskLevelFlatDataset(**dataset_kwargs)


def load_models(cfg, checkpoint_path, device):
    vqvae_builder_kwargs = {
        "simple_mask_vqvae_checkpoint_path": str(resolve_repo_path(cfg["vqvae_checkpoint"])),
        "device": device,
    }
    if cfg.get("vqvae_image_encoder_checkpoint"):
        vqvae_builder_kwargs["image_encoder_checkpoint"] = str(resolve_repo_path(cfg["vqvae_image_encoder_checkpoint"]))
    if cfg.get("vqvae_image_encoder_config"):
        vqvae_builder_kwargs["image_encoder_config_name"] = cfg["vqvae_image_encoder_config"]

    vqvae = builder_map["simple_mask_vqvae"][cfg["vqvae_config"]](**vqvae_builder_kwargs).eval().to(device)
    model = builder_map["simple_mask_maskgit"][cfg["config"]](
        checkpoint_path=str(checkpoint_path),
        device=device,
        enable_click=bool(cfg.get("enable_click", True)),
    ).eval().to(device)

    for module in (vqvae, model):
        for param in module.parameters():
            param.requires_grad = False
    return vqvae, model


def autocast_context(device, dtype):
    enabled = device.startswith("cuda") and dtype != torch.float32
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)


@torch.no_grad()
def encode_mask_to_tokens(model, vqvae, image, mask_normalized, device, dtype):
    with autocast_context(device, dtype):
        token_ids, image_tokens = model.encode_mask_to_token_ids(vqvae, mask_normalized, image)
    return token_ids, image_tokens


@torch.no_grad()
def decode_token_ids_to_mask_logits(model, vqvae, token_ids, image_tokens, output_size):
    original_shape = token_ids.shape
    if token_ids.ndim == 3:
        b, n, l = token_ids.shape
        token_ids_flat = token_ids.reshape(b * n, l)
        image_tokens_flat = image_tokens[:, None].expand(b, n, *image_tokens.shape[1:]).reshape(
            b * n, *image_tokens.shape[1:]
        )
    else:
        token_ids_flat = token_ids
        image_tokens_flat = image_tokens

    logits = model.decode_token_ids_to_mask_logits(vqvae, token_ids_flat, image_tokens_flat, output_size)
    if len(original_shape) == 3:
        logits = logits.view(original_shape[0], original_shape[1], *logits.shape[1:])
    return logits


def sample_click_condition(single_mask, ar_h, ar_w, max_clicks, num_clicks):
    mask_np = single_mask[0].detach().cpu().numpy() > 0
    if num_clicks <= 0:
        num_clicks = int(np.random.randint(1, max_clicks + 1))
    num_clicks = min(num_clicks, max_clicks)
    click_list, _, _ = init_clicks(mask_np, num_random_clicks=num_clicks, random_sample=True)
    coords_xy, labels = to_sam_format(click_list, pad_size=max_clicks)

    mask_h, mask_w = single_mask.shape[-2:]
    click_coords = torch.empty_like(coords_xy, dtype=torch.float32)
    click_coords[..., 0] = coords_xy[..., 1] * (ar_h / mask_h)
    click_coords[..., 1] = coords_xy[..., 0] * (ar_w / mask_w)
    click_coords = click_coords.clamp_min(0)
    click_coords[..., 0].clamp_(max=ar_h - 1)
    click_coords[..., 1].clamp_(max=ar_w - 1)
    return click_coords, labels.long(), click_list


def sample_mask_positions(token_ids, mask_ratio):
    b, l = token_ids.shape
    count = int(round(float(mask_ratio) * l))
    count = max(1, min(count, l))
    order = torch.rand(b, l, device=token_ids.device).argsort(dim=1)
    rank = order.argsort(dim=1)
    mask_positions = rank < count
    ratio = torch.full((b,), count / l, dtype=torch.float32, device=token_ids.device)
    return mask_positions, ratio


@torch.no_grad()
def teacher_force_outputs(model, token_ids, image_tokens, click_coords, click_labels, teacher_mask_ratio, device, dtype):
    mask_positions, mask_ratio = sample_mask_positions(token_ids, teacher_mask_ratio)
    with autocast_context(device, dtype):
        logits = model(
            token_ids,
            image_tokens,
            mask_positions=mask_positions,
            mask_ratio=mask_ratio,
            click_coords=click_coords,
            click_labels=click_labels,
            cfg_drop_click_prob=0.0,
            cfg_drop_image_prob=0.0,
        )
    teacher_ids = torch.where(mask_positions, logits.argmax(dim=-1), token_ids)
    return teacher_ids, mask_positions, mask_ratio


def guided_logits(
    model,
    token_ids,
    image_tokens,
    masked,
    mask_ratio,
    click_coords,
    click_labels,
    cfg_guidance_scale,
    cfg_drop_click,
    cfg_drop_image,
):
    cond_logits = model(
        token_ids,
        image_tokens,
        mask_positions=masked,
        mask_ratio=mask_ratio,
        click_coords=click_coords,
        click_labels=click_labels,
    )
    if cfg_guidance_scale == 1.0 or not (cfg_drop_click or cfg_drop_image):
        return cond_logits

    uncond_image_tokens = torch.zeros_like(image_tokens) if cfg_drop_image else image_tokens
    uncond_click_labels = make_uncond_click_labels(click_labels) if cfg_drop_click else click_labels
    uncond_logits = model(
        token_ids,
        uncond_image_tokens,
        mask_positions=masked,
        mask_ratio=mask_ratio,
        click_coords=click_coords,
        click_labels=uncond_click_labels,
    )
    return uncond_logits + cfg_guidance_scale * (cond_logits - uncond_logits)


@torch.no_grad()
def maskgit_infer_with_history(
    model,
    image_tokens,
    click_coords,
    click_labels,
    num_steps,
    temperature,
    top_k,
    min_p,
    cfg_guidance_scale,
    cfg_drop_click,
    cfg_drop_image,
    sampling_mode,
    device,
    dtype,
):
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if sampling_mode not in {"confidence", "click_expand"}:
        raise ValueError(f"Unknown MaskGIT sampling_mode: {sampling_mode}")

    b = image_tokens.shape[0]
    token_ids = torch.full((b, model.max_len), model.mask_token_id, dtype=torch.long, device=image_tokens.device)
    masked = torch.ones((b, model.max_len), dtype=torch.bool, device=image_tokens.device)
    history = []

    for step in range(num_steps):
        mask_ratio = masked.float().mean(dim=1)
        with autocast_context(device, dtype):
            logits = guided_logits(
                model,
                token_ids,
                image_tokens,
                masked,
                mask_ratio,
                click_coords,
                click_labels,
                cfg_guidance_scale,
                cfg_drop_click,
                cfg_drop_image,
            )
        masked_logits = logits[masked]
        sampled = sample_from_logits(masked_logits, temperature=temperature, top_k=top_k, min_p=min_p)
        probs = torch.softmax(masked_logits.float() / max(temperature, 1e-6), dim=-1)
        confidence = probs.gather(1, sampled[:, None]).squeeze(1).float()
        token_ids[masked] = sampled

        visible_ids = token_ids.clamp_max(model.vocab_size - 1).clone()
        visible_masked = masked.clone()
        history.append(
            {
                "step": step + 1,
                "token_ids": visible_ids,
                "masked_before": visible_masked,
                "mask_ratio": mask_ratio.detach().clone(),
            }
        )

        if step == num_steps - 1:
            break

        keep_ratio = math.cos(0.5 * math.pi * (step + 1) / num_steps)
        next_mask_count_scalar = int(model.max_len * keep_ratio)
        conf_full = torch.full((b, model.max_len), float("inf"), device=image_tokens.device)
        conf_full[masked] = confidence
        current_mask_count = masked.sum(dim=1)
        next_mask_count = torch.full((b,), next_mask_count_scalar, device=image_tokens.device, dtype=torch.long)
        next_mask_count = torch.minimum(next_mask_count, current_mask_count - 1).clamp(min=0)

        next_masked = None
        if sampling_mode == "click_expand":
            next_masked = model._click_expand_keep_mask(
                conf_full,
                masked,
                next_mask_count,
                click_coords,
                click_labels,
                step,
                num_steps,
            )

        if next_masked is not None:
            masked = next_masked
            token_ids[masked] = model.mask_token_id
        elif next_mask_count.max().item() == 0:
            masked = torch.zeros_like(masked)
        else:
            order = conf_full.argsort(dim=1)
            rank = order.argsort(dim=1)
            masked = rank < next_mask_count[:, None]
            token_ids[masked] = model.mask_token_id

    return history


def image_to_numpy(image_tensor):
    restored = restore_normalized_image(image_tensor.detach().cpu()).float()
    return restored.numpy().transpose(1, 2, 0) / 255.0


def logits_to_bool(mask_logits):
    if mask_logits.ndim == 4:
        mask_logits = mask_logits[0]
    if mask_logits.ndim == 3:
        mask_logits = mask_logits[0]
    return mask_logits.detach().cpu().float().numpy() > 0


def mask_to_bool(mask_tensor):
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor[0]
    return mask_tensor.detach().cpu().float().numpy() > 0


def build_errormap(gt_mask, pred_mask):
    errormap = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    tp = gt_mask & pred_mask
    fp = (~gt_mask) & pred_mask
    fn = gt_mask & (~pred_mask)
    errormap[tp] = COLOR_TP
    errormap[fp] = COLOR_FP
    errormap[fn] = COLOR_FN
    return errormap


def build_error_overlay(image_np, gt_mask, pred_mask, alpha):
    overlay = image_np.copy()
    errormap = build_errormap(gt_mask, pred_mask)
    active = (gt_mask | pred_mask)[..., None]
    overlay = np.where(active, overlay * (1.0 - alpha) + errormap * alpha, overlay)
    return np.clip(overlay, 0.0, 1.0)


def draw_clicks(ax, click_list, mask_shape, image_shape):
    if not click_list:
        return
    mask_h, mask_w = mask_shape
    img_h, img_w = image_shape[:2]
    xs = [x * img_w / mask_w for y, x, label in click_list if label == 1]
    ys = [y * img_h / mask_h for y, x, label in click_list if label == 1]
    if xs:
        ax.scatter(xs, ys, s=90, c="yellow", marker="*", edgecolors="black", linewidths=1.2)


def set_axis(ax, title):
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def iou_float(mask_logits, gt_mask):
    return float(calc_iou(mask_logits, gt_mask).detach().cpu().view(-1)[0].item())


def visualize_teacher(
    image_np,
    gt_np,
    gt_recon_np,
    teacher_np,
    click_list,
    mask_shape,
    gt_recon_iou,
    teacher_iou,
    teacher_mask_ratio,
    overlay_alpha,
    save_path,
    dpi,
):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(image_np)
    draw_clicks(axes[0, 0], click_list, mask_shape, image_np.shape)
    set_axis(axes[0, 0], "Image + Clicks")

    axes[0, 1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    draw_clicks(axes[0, 1], click_list, mask_shape, gt_np.shape)
    set_axis(axes[0, 1], "GT Mask")

    axes[0, 2].imshow(gt_recon_np, cmap="gray", vmin=0, vmax=1)
    set_axis(axes[0, 2], f"GT Token Recon\nIoU={gt_recon_iou:.4f}")

    axes[0, 3].imshow(build_error_overlay(image_np, gt_np, gt_recon_np, overlay_alpha))
    draw_clicks(axes[0, 3], click_list, mask_shape, image_np.shape)
    set_axis(axes[0, 3], "GT Recon Error Overlay")

    axes[1, 0].imshow(teacher_np, cmap="gray", vmin=0, vmax=1)
    set_axis(axes[1, 0], f"Teacher\nmask={teacher_mask_ratio:.2f} IoU={teacher_iou:.4f}")

    axes[1, 1].imshow(build_errormap(gt_np, teacher_np))
    set_axis(axes[1, 1], "Teacher Error Map\nTP blue FP red FN green")

    axes[1, 2].imshow(build_error_overlay(image_np, gt_np, teacher_np, overlay_alpha))
    draw_clicks(axes[1, 2], click_list, mask_shape, image_np.shape)
    set_axis(axes[1, 2], f"Teacher Error Overlay\nalpha={overlay_alpha:.2f}")

    axes[1, 3].imshow(build_errormap(gt_np, gt_recon_np))
    set_axis(axes[1, 3], "GT Recon Error Map")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def visualize_sampling_steps(
    image_np,
    gt_np,
    step_masks,
    step_ious,
    step_ratios,
    click_list,
    mask_shape,
    overlay_alpha,
    save_path,
    dpi,
):
    num_steps = len(step_masks)
    cols = num_steps + 2
    fig, axes = plt.subplots(3, cols, figsize=(3.2 * cols, 9.2))
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(image_np)
    draw_clicks(axes[0, 0], click_list, mask_shape, image_np.shape)
    set_axis(axes[0, 0], "Image + Clicks")
    axes[1, 0].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    set_axis(axes[1, 0], "GT Mask")
    axes[2, 0].imshow(build_error_overlay(image_np, gt_np, np.zeros_like(gt_np), overlay_alpha))
    set_axis(axes[2, 0], "GT vs Empty")

    final_mask = step_masks[-1]
    axes[0, 1].imshow(final_mask, cmap="gray", vmin=0, vmax=1)
    set_axis(axes[0, 1], f"Final Sample\nIoU={step_ious[-1]:.4f}")
    axes[1, 1].imshow(build_errormap(gt_np, final_mask))
    set_axis(axes[1, 1], "Final Error Map")
    axes[2, 1].imshow(build_error_overlay(image_np, gt_np, final_mask, overlay_alpha))
    draw_clicks(axes[2, 1], click_list, mask_shape, image_np.shape)
    set_axis(axes[2, 1], f"Final Overlay\nalpha={overlay_alpha:.2f}")

    for idx, (mask_np, iou, ratio) in enumerate(zip(step_masks, step_ious, step_ratios), start=2):
        axes[0, idx].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        set_axis(axes[0, idx], f"Step {idx - 1}\nmasked={ratio:.3f} IoU={iou:.3f}")
        axes[1, idx].imshow(build_errormap(gt_np, mask_np))
        set_axis(axes[1, idx], "Error Map")
        axes[2, idx].imshow(build_error_overlay(image_np, gt_np, mask_np, overlay_alpha))
        draw_clicks(axes[2, idx], click_list, mask_shape, image_np.shape)
        set_axis(axes[2, idx], "Error Overlay")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = load_config(resolve_repo_path(args.out_dir))
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    dtype_name = args.dtype or cfg.get("dtype", "float32")
    dtype = getattr(torch, dtype_name)
    checkpoint_path = resolve_repo_path(args.checkpoint_path) if args.checkpoint_path else resolve_repo_path(args.out_dir) / "checkpoints/latest.pth"
    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else resolve_repo_path(args.out_dir) / "maskgit_click_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    maskgit_num_steps = args.maskgit_num_steps or int(cfg.get("maskgit_num_steps", 12))
    maskgit_sampling_mode = args.maskgit_sampling_mode or cfg.get("maskgit_sampling_mode", "confidence")
    cfg_guidance_scale = args.cfg_guidance_scale
    if cfg_guidance_scale is None:
        cfg_guidance_scale = float(cfg.get("cfg_guidance_scale", 1.0))
    cfg_drop_click = not args.cfg_keep_click
    cfg_drop_image = bool(args.cfg_drop_image or cfg.get("cfg_drop_image", False))
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    min_p = args.min_p if args.min_p and args.min_p > 0 else None
    if args.num_infer_samples < 1:
        raise ValueError(f"num_infer_samples must be >= 1, got {args.num_infer_samples}")

    print("Loading click-conditioned SimpleMaskMaskGIT")
    print(f"  out_dir: {resolve_repo_path(args.out_dir)}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  dtype/device: {dtype_name}/{device}")
    print(
        f"  steps: {maskgit_num_steps}, infer_samples={args.num_infer_samples}, "
        f"sampling_mode={maskgit_sampling_mode}, temperature={args.temperature}, top_k={top_k}, min_p={min_p}"
    )

    vqvae, model = load_models(cfg, checkpoint_path, device)
    datasets = {
        "train": build_split_dataset(cfg, "train"),
        "val": build_split_dataset(cfg, "val"),
    }
    explicit_indices = {
        "train": parse_index_list(args.train_indices),
        "val": parse_index_list(args.val_indices),
    }

    rows = []
    with torch.no_grad():
        for split, dataset in datasets.items():
            indices = select_indices(
                dataset,
                explicit_indices[split],
                args.start_index,
                args.num_samples_per_split,
                args.sample_stride,
            )
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"{split}: dataset_size={len(dataset)}, selected={indices}")

            for sample_order, dataset_idx in enumerate(indices):
                image, _, mask_normalized, gt_mask = dataset[dataset_idx]
                click_coords, click_labels, click_list = sample_click_condition(
                    gt_mask,
                    model.h,
                    model.w,
                    max_clicks=args.max_clicks,
                    num_clicks=args.num_clicks,
                )

                image_b = image.unsqueeze(0).to(device)
                mask_b = mask_normalized.unsqueeze(0).to(device)
                gt_b = gt_mask.unsqueeze(0).to(device)
                click_coords_b = click_coords.unsqueeze(0).to(device)
                click_labels_b = click_labels.unsqueeze(0).to(device)

                token_ids, image_tokens = encode_mask_to_tokens(model, vqvae, image_b, mask_b, device, dtype)
                gt_recon_logits = decode_token_ids_to_mask_logits(model, vqvae, token_ids, image_tokens, gt_b.shape[-2:])
                teacher_ids, _, _ = teacher_force_outputs(
                    model,
                    token_ids,
                    image_tokens,
                    click_coords_b,
                    click_labels_b,
                    args.teacher_mask_ratio,
                    device,
                    dtype,
                )
                teacher_logits = decode_token_ids_to_mask_logits(model, vqvae, teacher_ids, image_tokens, gt_b.shape[-2:])

                image_tokens_samples = image_tokens.repeat_interleave(args.num_infer_samples, dim=0)
                click_coords_samples = click_coords_b.repeat_interleave(args.num_infer_samples, dim=0)
                click_labels_samples = click_labels_b.repeat_interleave(args.num_infer_samples, dim=0)

                history = maskgit_infer_with_history(
                    model,
                    image_tokens_samples,
                    click_coords_samples,
                    click_labels_samples,
                    num_steps=maskgit_num_steps,
                    temperature=args.temperature,
                    top_k=top_k,
                    min_p=min_p,
                    cfg_guidance_scale=cfg_guidance_scale,
                    cfg_drop_click=cfg_drop_click,
                    cfg_drop_image=cfg_drop_image,
                    sampling_mode=maskgit_sampling_mode,
                    device=device,
                    dtype=dtype,
                )
                step_token_ids = torch.stack([item["token_ids"] for item in history], dim=1)
                step_logits = decode_token_ids_to_mask_logits(
                    model,
                    vqvae,
                    step_token_ids,
                    image_tokens_samples,
                    gt_b.shape[-2:],
                )

                image_np = image_to_numpy(image)
                gt_np = mask_to_bool(gt_mask)
                gt_recon_np = logits_to_bool(gt_recon_logits)
                teacher_np = logits_to_bool(teacher_logits)
                step_masks_by_sample = [
                    [logits_to_bool(step_logits[infer_sample_idx, step_idx]) for step_idx in range(step_logits.shape[1])]
                    for infer_sample_idx in range(args.num_infer_samples)
                ]
                step_ious_by_sample = []
                step_ratios_by_sample = []
                for infer_sample_idx in range(args.num_infer_samples):
                    sample_gt_b = gt_b.expand(1, *gt_b.shape[1:])
                    step_ious_by_sample.append(
                        [
                            iou_float(step_logits[infer_sample_idx : infer_sample_idx + 1, step_idx], sample_gt_b)
                            for step_idx in range(step_logits.shape[1])
                        ]
                    )
                    step_ratios_by_sample.append(
                        [
                            float(item["mask_ratio"][infer_sample_idx].detach().cpu().item())
                            for item in history
                        ]
                    )
                gt_recon_iou = iou_float(gt_recon_logits, gt_b)
                teacher_iou = iou_float(teacher_logits, gt_b)

                prefix = f"{split}_sample_{sample_order:03d}_dataset_{dataset_idx}"
                teacher_path = split_dir / f"{prefix}_teacher.png"
                visualize_teacher(
                    image_np,
                    gt_np,
                    gt_recon_np,
                    teacher_np,
                    click_list,
                    gt_np.shape,
                    gt_recon_iou,
                    teacher_iou,
                    args.teacher_mask_ratio,
                    args.overlay_alpha,
                    teacher_path,
                    args.dpi,
                )

                final_ious = []
                for infer_sample_idx, (step_masks, step_ious, step_ratios) in enumerate(
                    zip(step_masks_by_sample, step_ious_by_sample, step_ratios_by_sample)
                ):
                    final_ious.append(step_ious[-1])
                    if args.num_infer_samples == 1:
                        steps_path = split_dir / f"{prefix}_sample_steps.png"
                    else:
                        steps_path = split_dir / f"{prefix}_infer_{infer_sample_idx:02d}_sample_steps.png"
                    visualize_sampling_steps(
                        image_np,
                        gt_np,
                        step_masks,
                        step_ious,
                        step_ratios,
                        click_list,
                        gt_np.shape,
                        args.overlay_alpha,
                        steps_path,
                        args.dpi,
                    )

                    row = {
                        "split": split,
                        "sample_order": sample_order,
                        "dataset_idx": dataset_idx,
                        "infer_sample_idx": infer_sample_idx,
                        "checkpoint_path": str(checkpoint_path),
                        "num_clicks": sum(1 for _, _, label in click_list if label == 1),
                        "gt_recon_iou": gt_recon_iou,
                        "teacher_iou": teacher_iou,
                        "sample_final_iou": step_ious[-1],
                        "teacher_path": str(teacher_path),
                        "sample_steps_path": str(steps_path),
                    }
                    for step_idx, (iou, ratio) in enumerate(zip(step_ious, step_ratios), start=1):
                        row[f"sample_iou_step_{step_idx:02d}"] = iou
                        row[f"sample_masked_ratio_step_{step_idx:02d}"] = ratio
                    rows.append(row)

                print(
                    f"[{split} {sample_order + 1}/{len(indices)}] dataset_idx={dataset_idx} "
                    f"gt_recon={gt_recon_iou:.4f} teacher={teacher_iou:.4f} "
                    f"sample_final_mean={np.mean(final_ious):.4f} sample_final_best={np.max(final_ious):.4f}"
                )

    if rows:
        csv_path = output_dir / "summary.csv"
        json_path = output_dir / "summary.json"
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        with json_path.open("w") as f:
            json.dump(rows, f, indent=2)

        for split in ("train", "val"):
            split_rows = [row for row in rows if row["split"] == split]
            if split_rows:
                print(
                    f"{split} mean: teacher={np.mean([r['teacher_iou'] for r in split_rows]):.4f}, "
                    f"sample_final={np.mean([r['sample_final_iou'] for r in split_rows]):.4f}"
                )
        print(f"Saved summary: {csv_path}")
    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
