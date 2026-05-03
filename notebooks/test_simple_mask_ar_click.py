import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset, MaskLevelFlatSubsetDataset
from maskvar.maskseg_build_everything import builder_map
from maskvar.utils import restore_normalized_image
from maskvar.utils.clicker_v2 import init_clicks, to_sam_format
from maskvar.utils.metrics import calc_iou


def parse_args():
    parser = argparse.ArgumentParser(description="Test click-conditioned SimpleMaskAR sampling.")
    parser.add_argument(
        "--ar_out_dir",
        type=Path,
        default=REPO_ROOT / "out/ddp_simple_mask_ar_coconut_click_ep5",
        help="Training output directory containing config.json and checkpoints.",
    )
    parser.add_argument("--ar_ckpt", type=Path, default=None, help="AR checkpoint path. Defaults to ar_out_dir/checkpoints/latest.pth.")
    parser.add_argument("--sample_index", type=int, default=0, help="Starting dataset sample index to visualize.")
    parser.add_argument(
        "--split",
        type=str,
        default="mixed",
        choices=["train", "val", "mixed"],
        help="Dataset split to visualize. mixed defaults to 2 train samples and 3 val samples.",
    )
    parser.add_argument("--num_images", type=int, default=5, help="Number of spaced samples to visualize for train/val split mode.")
    parser.add_argument("--index_stride", type=int, default=10, help="Stride between selected mask-level samples.")
    parser.add_argument(
        "--sample_indices",
        type=str,
        default=None,
        help="Comma-separated dataset indices for the selected split. Overrides sample_index/num_images when set.",
    )
    parser.add_argument("--train_count", type=int, default=2, help="Number of train samples in mixed mode.")
    parser.add_argument("--val_count", type=int, default=3, help="Number of val samples in mixed mode.")
    parser.add_argument("--num_clicks", type=int, default=2, help="Number of positive clicks. Use 0 to sample 1..max_clicks.")
    parser.add_argument("--max_clicks", type=int, default=2, help="Number of click slots expected by the model.")
    parser.add_argument("--sample_count", type=int, default=6, help="Number of stochastic samples to draw.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for stochastic samples.")
    parser.add_argument("--top_k", type=int, default=8, help="Top-k for stochastic samples. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for click and token sampling.")
    parser.add_argument("--print_every", type=int, default=512, help="Progress interval for AR decoding.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "notebooks/test_outputs/simple_mask_ar_click.png",
        help="Path to save the visualization.",
    )
    return parser.parse_args()


def resolve_repo_path(path_like):
    path = Path(path_like)
    return path if path.is_absolute() else REPO_ROOT / path


def build_split_dataset(cfg, split):
    dataset_path_map = {
        "hqseg44k": REPO_ROOT / "data/sam-hq",
        "cocolvis": REPO_ROOT / "data/coco_lvis",
        "coconut_hf": REPO_ROOT / "data/coconut_hf",
    }
    dataset_name = cfg["dataset"]
    dataset_path = resolve_repo_path(cfg["dataset_path"]) if cfg.get("dataset_path") else dataset_path_map[dataset_name]
    train_set_base, val_set_base = builder_map["dataset"][dataset_name](str(dataset_path))
    dataset_base = train_set_base if split == "train" else val_set_base
    subset_key = f"{split}_subset_index"
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
    if cfg.get(subset_key):
        return MaskLevelFlatSubsetDataset(
            subset_list=resolve_repo_path(cfg[subset_key]),
            **dataset_kwargs,
        )
    return MaskLevelFlatDataset(**dataset_kwargs)


def load_models(cfg, ar_ckpt, device):
    vqvae_builder_kwargs = {
        "simple_mask_vqvae_checkpoint_path": str(resolve_repo_path(cfg["vqvae_checkpoint"])),
        "device": device,
    }
    if cfg.get("vqvae_image_encoder_checkpoint"):
        vqvae_builder_kwargs["image_encoder_checkpoint"] = str(resolve_repo_path(cfg["vqvae_image_encoder_checkpoint"]))
    if cfg.get("vqvae_image_encoder_config"):
        vqvae_builder_kwargs["image_encoder_config_name"] = cfg["vqvae_image_encoder_config"]

    vqvae = builder_map["simple_mask_vqvae"][cfg["vqvae_config"]](**vqvae_builder_kwargs).eval().to(device)
    ar_model = builder_map["simple_mask_ar"][cfg["config"]](
        checkpoint_path=str(ar_ckpt),
        device=device,
        enable_click=True,
    ).eval().to(device)
    checkpoint = torch.load(ar_ckpt, map_location="cpu", weights_only=True)

    for module in (vqvae, ar_model):
        for param in module.parameters():
            param.requires_grad = False
    return vqvae, ar_model, checkpoint


def encode_mask_to_tokens(vqvae, image, mask_normalized, device):
    autocast_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.startswith("cuda")):
            mask_tokens = vqvae.mask_encoder(mask_normalized)
            image_tokens = vqvae.image_encoder(image)
        mask_tokens_blc = rearrange(mask_tokens, "b c h w -> b (h w) c")
        token_ids = vqvae.quant.x_to_idx(mask_tokens_blc.float()).view(mask_tokens.shape[0], *mask_tokens.shape[-2:])
        image_tokens = rearrange(image_tokens, "b c h w -> b h w c")
    return token_ids, image_tokens


def decode_token_ids_to_mask_logits(vqvae, token_ids, image_tokens, output_size, device):
    original_shape = token_ids.shape
    if token_ids.ndim == 4:
        b, n, h, w = token_ids.shape
        flat_ids = token_ids.view(b * n, h, w)
        flat_image_tokens = image_tokens.unsqueeze(1).expand(b, n, *image_tokens.shape[1:]).contiguous()
        flat_image_tokens = flat_image_tokens.view(b * n, *image_tokens.shape[1:])
    else:
        flat_ids = token_ids
        flat_image_tokens = image_tokens

    flat_ids_bl = rearrange(flat_ids, "b h w -> b (h w)")
    mask_tokens = vqvae.quant.idx_to_x(flat_ids_bl)
    mask_tokens = rearrange(mask_tokens, "b (h w) c -> b h w c", h=flat_ids.shape[-2], w=flat_ids.shape[-1])

    autocast_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.startswith("cuda")):
        mask_logits = vqvae.mask_decoder(mask_tokens, flat_image_tokens)

    if mask_logits.shape[-2:] != output_size:
        mask_logits = F.interpolate(mask_logits.float(), size=output_size, mode="bilinear", align_corners=False)
    if len(original_shape) == 4:
        mask_logits = mask_logits.view(original_shape[0], original_shape[1], *mask_logits.shape[1:])
    return mask_logits


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


@torch.no_grad()
def autoregressive_infer_verbose(
    model,
    image_tokens,
    click_coords,
    click_labels,
    temperature=1.0,
    top_k=None,
    num_samples=1,
    print_every=512,
    tag="infer",
):
    b, h, w, _ = image_tokens.shape
    device = image_tokens.device
    batch_size = b * num_samples
    total_steps = h * w
    coords = torch.tensor([[row, col] for row in range(h) for col in range(w)], device=device, dtype=torch.long)

    print(f"[{tag}] preparing caches for batch={b}, num_samples={num_samples}, steps={total_steps}")
    x_step = model.sos.view(1, 1, model.dim).expand(batch_size, 1, -1)
    click_tokens, click_coords, click_labels = model.encode_clicks(click_coords, click_labels)
    cross_caches = []
    click_caches = []
    for block_idx, block in enumerate(model.blocks):
        cached_k, cached_v, full_h, full_w = block.precompute_cross_kv(image_tokens)
        if num_samples > 1:
            cached_k = cached_k.repeat_interleave(num_samples, dim=0)
            cached_v = cached_v.repeat_interleave(num_samples, dim=0)
        cross_caches.append((cached_k, cached_v, full_h, full_w))

        click_cache = block.precompute_click_kv(click_tokens, click_coords, click_labels, h, w)
        if click_cache is not None and num_samples > 1:
            click_k, click_v, click_h, click_w = click_cache
            click_cache = (
                click_k.repeat_interleave(num_samples, dim=0),
                click_v.repeat_interleave(num_samples, dim=0),
                click_h,
                click_w,
            )
        click_caches.append(click_cache)
        print(f"[{tag}] caches ready for block {block_idx}")

    self_caches = [None for _ in model.blocks]
    generated_ids = []
    for i in range(total_steps):
        if i == 0 or (i + 1) % print_every == 0 or i + 1 == total_steps:
            print(f"[{tag}] step {i + 1}/{total_steps}")

        x_out = x_step
        curr_coord = coords[i:i + 1]
        for block_idx, block in enumerate(model.blocks):
            x_out, self_caches[block_idx] = block.forward_step(
                x_out,
                cross_caches[block_idx],
                self_caches[block_idx],
                curr_coord,
                click_cache=click_caches[block_idx],
            )

        logits = model.cls(x_out[:, 0, :])
        if temperature == 0:
            next_token = logits.argmax(dim=-1)
        else:
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.clone()
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        generated_ids.append(next_token)
        if i < total_steps - 1:
            x_step = model.embed(next_token).unsqueeze(1)

    generated = torch.stack(generated_ids, dim=1)
    if num_samples == 1:
        return generated.view(b, h, w)
    return generated.view(b, num_samples, h, w)


def mask_iou_from_logits(mask_logits, gt_mask):
    pred_mask = (mask_logits > 0).float()
    return float(calc_iou(pred_mask, gt_mask).item())


@torch.no_grad()
def teacher_force_argmax(model, token_ids, image_tokens, click_coords, click_labels, device):
    token_ids_bl = rearrange(token_ids, "b h w -> b (h w)")
    image_tokens_blc = rearrange(image_tokens, "b h w c -> b (h w) c")
    autocast_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=device.startswith("cuda")):
        logits = model(
            token_ids_bl,
            image_tokens_blc,
            click_coords=click_coords,
            click_labels=click_labels,
        )
    return logits.argmax(dim=-1).view(token_ids.shape)


def to_numpy_image(image_tensor):
    restored = restore_normalized_image(image_tensor[0].detach().cpu())
    return restored.permute(1, 2, 0).numpy()


def to_numpy_mask(mask_logits):
    if mask_logits.ndim == 4:
        mask_logits = mask_logits[0]
    if mask_logits.ndim == 3:
        mask_logits = mask_logits[0]
    return (mask_logits.detach().cpu().float().numpy() > 0).astype(np.float32)


def overlay_mask(image_np, mask_np, color=(0.0, 1.0, 1.0), alpha=0.8):
    canvas = image_np.astype(np.float32) / 255.0
    mask_np = mask_np[..., None]
    color = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(canvas * (1.0 - alpha * mask_np) + color * (alpha * mask_np), 0.0, 1.0)


def draw_clicks(ax, click_list, mask_shape, image_shape):
    if not click_list:
        return
    mask_h, mask_w = mask_shape
    img_h, img_w = image_shape[:2]
    xs = [x * img_w / mask_w for y, x, label in click_list if label == 1]
    ys = [y * img_h / mask_h for y, x, label in click_list if label == 1]
    ax.scatter(xs, ys, s=120, c="yellow", marker="*", edgecolors="black", linewidths=1.5)


def click_mask_stats(mask_np, click_list):
    pos_clicks = [(y, x) for y, x, label in click_list if label == 1]
    if not pos_clicks:
        hit_count = 0
        hit_rate = 0.0
    else:
        h, w = mask_np.shape[-2:]
        hit_count = 0
        for y, x in pos_clicks:
            yy = int(np.clip(round(y), 0, h - 1))
            xx = int(np.clip(round(x), 0, w - 1))
            hit_count += int(mask_np[yy, xx] > 0)
        hit_rate = hit_count / len(pos_clicks)

    area_frac = float(mask_np.mean())
    # Prefer masks that cover positive clicks, with a small bias against huge blobs.
    click_score = hit_rate - 0.05 * area_frac
    return {
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "area_frac": area_frac,
        "click_score": click_score,
    }


def set_overlay_axis(ax, image_np, mask_np, click_list, mask_shape, title, color=(0.0, 1.0, 1.0)):
    ax.imshow(overlay_mask(image_np, mask_np, color=color))
    draw_clicks(ax, click_list, mask_shape, image_np.shape)
    ax.set_title(title)


def visualize(
    output,
    image_np,
    gt_mask_np,
    gt_recon_np,
    teacher_np,
    greedy_np,
    sample_masks_np,
    click_list,
    mask_shape,
    gt_recon_iou,
    teacher_iou,
    greedy_iou,
    sample_ious,
    sample_stats,
    best_click_idx,
    best_iou_idx,
):
    sample_count = len(sample_masks_np)
    rows = 3 if sample_count > 0 else 1
    cols = max(6, sample_count, 1)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")

    axes[0, 0].imshow(image_np)
    draw_clicks(axes[0, 0], click_list, mask_shape, image_np.shape)
    axes[0, 0].set_title("Image + Clicks")

    axes[0, 1].imshow(gt_mask_np, cmap="gray")
    draw_clicks(axes[0, 1], click_list, mask_shape, gt_mask_np.shape)
    axes[0, 1].set_title("GT Mask + Clicks")

    set_overlay_axis(
        axes[0, 2],
        image_np,
        gt_recon_np,
        click_list,
        mask_shape,
        f"GT Token Recon\nIoU={gt_recon_iou:.4f}",
    )

    set_overlay_axis(
        axes[0, 3],
        image_np,
        teacher_np,
        click_list,
        mask_shape,
        f"Teacher Argmax\nIoU={teacher_iou:.4f}",
        color=(0.3, 1.0, 0.2),
    )

    greedy_stats = click_mask_stats(greedy_np, click_list)
    set_overlay_axis(
        axes[0, 4],
        image_np,
        greedy_np,
        click_list,
        mask_shape,
        f"Greedy\nIoU={greedy_iou:.4f} hit={greedy_stats['hit_count']} area={greedy_stats['area_frac']:.3f}",
        color=(1.0, 0.2, 0.2),
    )

    if best_click_idx is not None:
        best_stats = sample_stats[best_click_idx]
        set_overlay_axis(
            axes[0, 5],
            image_np,
            sample_masks_np[best_click_idx],
            click_list,
            mask_shape,
            (
                f"Best Click Sample {best_click_idx}\n"
                f"IoU={sample_ious[best_click_idx]:.4f} hit={best_stats['hit_count']} area={best_stats['area_frac']:.3f}"
            ),
            color=(1.0, 0.85, 0.0),
        )

    for i in range(sample_count):
        stats = sample_stats[i]
        set_overlay_axis(
            axes[1, i],
            image_np,
            sample_masks_np[i],
            click_list,
            mask_shape,
            f"Sample {i}\nIoU={sample_ious[i]:.4f} hit={stats['hit_count']} area={stats['area_frac']:.3f}",
        )

    if sample_count > 0:
        sorted_indices = sorted(
            range(sample_count),
            key=lambda idx: (sample_stats[idx]["click_score"], sample_ious[idx]),
            reverse=True,
        )
        for rank, idx in enumerate(sorted_indices[:cols]):
            stats = sample_stats[idx]
            title = (
                f"Click Rank {rank}: S{idx}\n"
                f"score={stats['click_score']:.3f} IoU={sample_ious[idx]:.4f} hit={stats['hit_count']}"
            )
            if idx == best_iou_idx:
                title += " oracle"
            set_overlay_axis(
                axes[2, rank],
                image_np,
                sample_masks_np[idx],
                click_list,
                mask_shape,
                title,
                color=(1.0, 0.85, 0.0) if rank == 0 else (0.0, 1.0, 1.0),
            )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def parse_sample_indices(args, dataset_len, count=None, start_index=None):
    if args.sample_indices:
        indices = [int(item.strip()) for item in args.sample_indices.split(",") if item.strip()]
    else:
        n = args.num_images if count is None else count
        start = args.sample_index if start_index is None else start_index
        indices = [start + i * args.index_stride for i in range(n)]
    bad_indices = [idx for idx in indices if idx < 0 or idx >= dataset_len]
    if bad_indices:
        raise IndexError(f"Sample indices out of range for dataset size {dataset_len}: {bad_indices}")
    return indices


def output_path_for_sample(output, split, sample_index, multi_image):
    if output.suffix == "":
        return output / f"simple_mask_ar_click_{split}_idx{sample_index:06d}.png"
    if not multi_image:
        return output
    return output.with_name(f"{output.stem}_{split}_idx{sample_index:06d}{output.suffix}")


def run_one_sample(args, dataset, split, vqvae, ar_model, device, sample_index, output_path):
    torch.manual_seed(args.seed + sample_index)
    np.random.seed(args.seed + sample_index)

    print("=" * 80)
    print(f"Sample: {split}[{sample_index}]")

    image, _, mask_normalized, gt_mask = dataset[sample_index]
    image = image.unsqueeze(0).to(device)
    mask_normalized = mask_normalized.unsqueeze(0).to(device)
    gt_mask = gt_mask.unsqueeze(0).to(device)

    click_coords, click_labels, click_list = sample_click_condition(
        gt_mask[0],
        ar_h=ar_model.h,
        ar_w=ar_model.w,
        max_clicks=args.max_clicks,
        num_clicks=args.num_clicks,
    )
    click_coords = click_coords.unsqueeze(0).to(device)
    click_labels = click_labels.unsqueeze(0).to(device)
    print("Clicks (y, x, label):", click_list)
    print("Click coords in AR grid:", click_coords.detach().cpu().tolist())
    print("Click labels:", click_labels.detach().cpu().tolist())

    token_ids, image_tokens = encode_mask_to_tokens(vqvae, image, mask_normalized, device)
    output_size = tuple(gt_mask.shape[-2:])
    gt_recon_logits = decode_token_ids_to_mask_logits(vqvae, token_ids, image_tokens, output_size, device)
    gt_recon_iou = mask_iou_from_logits(gt_recon_logits, gt_mask)

    print("token_ids shape:", tuple(token_ids.shape))
    print("image_tokens shape:", tuple(image_tokens.shape))
    print("gt token recon IoU:", gt_recon_iou)

    print("[main] compute click-conditioned teacher-forcing argmax")
    teacher_ids = teacher_force_argmax(
        ar_model,
        token_ids,
        image_tokens,
        click_coords=click_coords,
        click_labels=click_labels,
        device=device,
    )
    teacher_logits = decode_token_ids_to_mask_logits(vqvae, teacher_ids, image_tokens, output_size, device)
    teacher_iou = mask_iou_from_logits(teacher_logits, gt_mask)

    print("[main] start click-conditioned greedy decode")
    greedy_ids = autoregressive_infer_verbose(
        ar_model,
        image_tokens,
        click_coords=click_coords,
        click_labels=click_labels,
        temperature=0.0,
        print_every=args.print_every,
        tag=f"greedy-click-{sample_index}",
    )
    greedy_logits = decode_token_ids_to_mask_logits(vqvae, greedy_ids, image_tokens, output_size, device)
    greedy_iou = mask_iou_from_logits(greedy_logits, gt_mask)

    sample_ious = []
    sample_masks_np = []
    sample_stats = []
    best_click_idx = None
    best_iou_idx = None
    if args.sample_count > 0:
        print("[main] start click-conditioned stochastic multi-sample decode")
        sample_ids = autoregressive_infer_verbose(
            ar_model,
            image_tokens,
            click_coords=click_coords,
            click_labels=click_labels,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            num_samples=args.sample_count,
            print_every=args.print_every,
            tag=f"sample-click-{sample_index}",
        )
        sample_logits = decode_token_ids_to_mask_logits(vqvae, sample_ids, image_tokens, output_size, device)
        sample_ious = [mask_iou_from_logits(sample_logits[:, i], gt_mask) for i in range(args.sample_count)]
        sample_masks_np = [to_numpy_mask(sample_logits[:, i]) for i in range(args.sample_count)]
        sample_stats = [click_mask_stats(mask_np, click_list) for mask_np in sample_masks_np]
        best_click_idx = max(
            range(args.sample_count),
            key=lambda idx: (sample_stats[idx]["click_score"], sample_ious[idx]),
        )
        best_iou_idx = max(range(args.sample_count), key=lambda idx: sample_ious[idx])

    image_np = to_numpy_image(image)
    gt_mask_np = gt_mask[0, 0].detach().cpu().numpy()
    gt_recon_np = to_numpy_mask(gt_recon_logits)
    teacher_np = to_numpy_mask(teacher_logits)
    greedy_np = to_numpy_mask(greedy_logits)

    greedy_stats = click_mask_stats(greedy_np, click_list)
    print("greedy IoU:", greedy_iou)
    print("teacher IoU:", teacher_iou)
    print("greedy click stats:", greedy_stats)
    print("sample IoUs:", sample_ious)
    if sample_stats:
        for idx, stats in enumerate(sample_stats):
            print(f"sample {idx} click stats:", stats)
        print("best click sample:", best_click_idx)
        print("oracle best IoU sample:", best_iou_idx)

    visualize(
        output_path,
        image_np=image_np,
        gt_mask_np=gt_mask_np,
        gt_recon_np=gt_recon_np,
        teacher_np=teacher_np,
        greedy_np=greedy_np,
        sample_masks_np=sample_masks_np,
        click_list=click_list,
        mask_shape=tuple(gt_mask.shape[-2:]),
        gt_recon_iou=gt_recon_iou,
        teacher_iou=teacher_iou,
        greedy_iou=greedy_iou,
        sample_ious=sample_ious,
        sample_stats=sample_stats,
        best_click_idx=best_click_idx,
        best_iou_idx=best_iou_idx,
    )
    print(f"Saved visualization to: {output_path}")

    return {
        "split": split,
        "sample_index": sample_index,
        "gt_recon_iou": gt_recon_iou,
        "teacher_iou": teacher_iou,
        "greedy_iou": greedy_iou,
        "greedy_hit_rate": greedy_stats["hit_rate"],
        "greedy_area_frac": greedy_stats["area_frac"],
        "best_click_idx": best_click_idx,
        "best_click_iou": sample_ious[best_click_idx] if best_click_idx is not None else None,
        "best_click_hit_rate": sample_stats[best_click_idx]["hit_rate"] if best_click_idx is not None else None,
        "best_iou_idx": best_iou_idx,
        "best_iou": sample_ious[best_iou_idx] if best_iou_idx is not None else None,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ar_ckpt = args.ar_ckpt or args.ar_out_dir / "checkpoints/latest.pth"
    config_path = args.ar_out_dir / "config.json"
    with open(config_path, "r") as f:
        train_cfg = json.load(f)

    print(f"Device: {device}")
    print(f"AR checkpoint: {ar_ckpt}")
    print(f"Config: {config_path}")

    train_set = build_split_dataset(train_cfg, "train")
    val_set = build_split_dataset(train_cfg, "val")
    vqvae, ar_model, ar_checkpoint = load_models(train_cfg, ar_ckpt, device)
    print("Loaded AR step:", ar_checkpoint.get("step"))
    print("Loaded AR global_step:", ar_checkpoint.get("global_step"))
    print("Train dataset size:", len(train_set))
    print("Val dataset size:", len(val_set))

    if args.split == "mixed":
        if args.sample_indices:
            raise ValueError("--sample_indices is only supported with --split train or --split val")
        sample_plan = []
        train_indices = parse_sample_indices(args, len(train_set), count=args.train_count, start_index=args.sample_index)
        val_indices = parse_sample_indices(args, len(val_set), count=args.val_count, start_index=args.sample_index)
        sample_plan.extend(("train", idx) for idx in train_indices)
        sample_plan.extend(("val", idx) for idx in val_indices)
    else:
        dataset_len = len(train_set) if args.split == "train" else len(val_set)
        indices = parse_sample_indices(args, dataset_len)
        sample_plan = [(args.split, idx) for idx in indices]
    print("Sample plan:", sample_plan)

    summaries = []
    multi_image = len(sample_plan) > 1
    for split, sample_index in sample_plan:
        dataset = train_set if split == "train" else val_set
        output_path = output_path_for_sample(args.output, split, sample_index, multi_image)
        summaries.append(run_one_sample(args, dataset, split, vqvae, ar_model, device, sample_index, output_path))

    print("=" * 80)
    print("Summary:")
    for item in summaries:
        print(
            "{split}[{sample_index}] gt_recon={gt_recon_iou:.4f} teacher={teacher_iou:.4f} "
            "greedy={greedy_iou:.4f} greedy_hit={greedy_hit_rate:.2f} greedy_area={greedy_area_frac:.3f} "
            "best_click={best_click_idx} best_click_iou={best_click_iou} best_click_hit={best_click_hit_rate} "
            "oracle={best_iou_idx} oracle_iou={best_iou}".format(**item)
        )


if __name__ == "__main__":
    main()
