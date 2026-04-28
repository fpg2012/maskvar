"""
Visualize SimpleMaskVqvaeV2 query/image attention maps.

Default settings match train_scripts/simple_mask_vqvae/ddp_train_v2_coconut_hf_dino.sh.

Example:
    python notebooks/visualize_simple_mask_vqvae_v2_attention.py \
        --num_samples 2 \
        --sample_stride 200
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset
from maskvar.maskseg_build_everything import builder_map
from maskvar.utils import restore_normalized_image


def _to_numpy_image(image: torch.Tensor) -> np.ndarray:
    image = restore_normalized_image(image.detach().cpu())
    image = image.numpy().transpose(1, 2, 0)
    if image.max() > 1:
        image = image / 255.0
    return np.clip(image, 0, 1)


def _plot_mask_overlay(ax, image: np.ndarray, mask: torch.Tensor, title: str) -> None:
    mask_np = mask.detach().cpu().numpy() > 0.5
    overlay = image.copy()
    color = np.array([1.0, 0.22, 0.18])
    overlay[mask_np] = overlay[mask_np] * 0.55 + color * 0.45
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")


def _prepare_kv(block, kv: torch.Tensor, pe_type: str = "rope", image_pe=None) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, w, _ = kv.shape
    kv = block.linear_kv(kv)
    k, v = kv.chunk(2, dim=-1)
    k = rearrange(k, "b h w (nh c) -> b nh h w c", nh=block.num_heads, c=block.dim_head)
    v = rearrange(v, "b h w (nh c) -> b nh (h w) c", nh=block.num_heads, c=block.dim_head)

    if pe_type == "rope":
        k = rearrange(k, "b nh h w c -> (b nh) h w c")
        k = block.rope.apply_2d_rope(k)
        k = rearrange(k, "(b nh) h w c -> b nh (h w) c", b=b, nh=block.num_heads)
    elif pe_type == "sam":
        k = k + image_pe
    else:
        raise ValueError(f"Unsupported pe_type: {pe_type}")

    return k, v


def cross_block_with_attention(block, q: torch.Tensor, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_input = q
    q_proj = block.linear_q(q)
    q_proj = rearrange(q_proj, "b l (nh c) -> b nh l c", nh=block.num_heads, c=block.dim_head)
    k, v = _prepare_kv(block, kv)

    attn_logits = (q_proj.float() @ k.float().transpose(-2, -1)) / math.sqrt(block.dim_head)
    attn = attn_logits.softmax(dim=-1)
    out = (attn.to(v.dtype) @ v).to(q.dtype)
    out = rearrange(out, "b nh l c -> b l (nh c)")
    out = q_input + block.out_proj(out)
    out = out + block.ffn(block.layernorm(out))
    return out, attn.detach()


def reverse_cross_block_with_attention(block, q: torch.Tensor, kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, w, _ = q.shape
    q_input = q

    q_proj = block.linear_q(q)
    kv_proj = block.linear_kv(kv)
    k, v = kv_proj.chunk(2, dim=-1)

    q_proj = rearrange(q_proj, "b h w (nh c) -> b nh h w c", nh=block.num_heads, c=block.dim_head)
    k = rearrange(k, "b l (nh c) -> b nh l c", nh=block.num_heads, c=block.dim_head)
    v = rearrange(v, "b l (nh c) -> b nh l c", nh=block.num_heads, c=block.dim_head)

    q_proj = rearrange(q_proj, "b nh h w c -> (b nh) h w c")
    q_proj = block.rope.apply_2d_rope(q_proj)
    q_proj = rearrange(q_proj, "(b nh) h w c -> b nh (h w) c", b=b, nh=block.num_heads)

    attn_logits = (q_proj.float() @ k.float().transpose(-2, -1)) / math.sqrt(block.dim_head)
    attn = attn_logits.softmax(dim=-1)
    out = (attn.to(v.dtype) @ v).to(q.dtype)
    out = rearrange(out, "b nh (h w) c -> b h w (nh c)", h=h, w=w)
    out = q_input + block.out_proj(out)
    out = out + block.ffn(block.layernorm(out))
    return out, attn.detach()


@torch.no_grad()
def run_v2_forward_with_decoder_attention(model, mask_normalized: torch.Tensor, image: torch.Tensor):
    mask_tokens = model.mask_encoder(mask_normalized)
    image_tokens = model.image_encoder(image)
    mask_tokens = rearrange(mask_tokens, "b c h w -> b h w c")
    image_tokens = rearrange(image_tokens, "b c h w -> b h w c")

    query_tokens = model.mask_feature_compactor(mask_tokens)
    if model.enable_vq:
        query_tokens, _ = model.quant(query_tokens)

    records = []
    for layer_idx, block in enumerate(model.mask_decoder.two_way_blocks):
        query_tokens, q_to_i = cross_block_with_attention(block.block1, query_tokens, image_tokens)
        image_tokens, i_to_q = reverse_cross_block_with_attention(block.reverse_block1, image_tokens, query_tokens)
        records.append(
            {
                "layer_idx": layer_idx,
                "query_to_image": q_to_i.cpu(),
                "image_to_query": i_to_q.cpu(),
                "image_hw": image_tokens.shape[1:3],
            }
        )

    image_feature_map = rearrange(image_tokens, "b h w c -> b c h w")
    up_query_token = model.mask_decoder.hyper_in(query_tokens[:, 0, :])
    up_image_map = model.mask_decoder.output_upscaling(image_feature_map)
    up_query_token = model.mask_decoder.layer_norm_post_query(up_query_token)
    up_image_map = model.mask_decoder.layer_norm_post_image(rearrange(up_image_map, "b c h w -> b h w c"))
    pred_logits = torch.einsum("bc,bhwc->bhw", up_query_token, up_image_map).unsqueeze(1)
    if pred_logits.shape[-2:] != mask_normalized.shape[-2:]:
        pred_logits = F.interpolate(pred_logits, size=mask_normalized.shape[-2:], mode="bilinear", align_corners=False)

    return pred_logits, records


def save_attention_figure(records, image: np.ndarray, mask: torch.Tensor, pred_logits: torch.Tensor, output_path: Path) -> None:
    selected = [records[0]]
    if records[-1]["layer_idx"] != records[0]["layer_idx"]:
        selected.append(records[-1])

    num_queries = records[0]["query_to_image"].shape[2]
    fig, axes = plt.subplots(len(selected) * 2 + 1, num_queries, figsize=(2.35 * num_queries, 2.6 * (len(selected) * 2 + 1)))
    if axes.ndim == 1:
        axes = axes[None, :]

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")
    _plot_mask_overlay(axes[0, 1], image, mask[0], "GT mask")
    pred = (pred_logits[0, 0].detach().cpu() > 0).float()
    _plot_mask_overlay(axes[0, 2], image, pred, "Prediction")
    for col in range(3, num_queries):
        axes[0, col].axis("off")

    row = 1
    for record in selected:
        h, w = record["image_hw"]
        q_to_i = record["query_to_image"][0].mean(dim=0)
        i_to_q = record["image_to_query"][0].mean(dim=0).transpose(0, 1)

        for q_idx in range(num_queries):
            heat = q_to_i[q_idx].reshape(h, w).numpy()
            axes[row, q_idx].imshow(heat, cmap="magma")
            axes[row, q_idx].set_title(f"L{record['layer_idx']} q{q_idx}->img")
            axes[row, q_idx].axis("off")
        row += 1

        for q_idx in range(num_queries):
            heat = i_to_q[q_idx].reshape(h, w).numpy()
            axes[row, q_idx].imshow(heat, cmap="viridis")
            axes[row, q_idx].set_title(f"L{record['layer_idx']} img->q{q_idx}")
            axes[row, q_idx].axis("off")
        row += 1

    fig.suptitle("SimpleMaskVqvaeV2 decoder two-way attention, head-averaged", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_dataset(args):
    dataset_path_map = {
        "hqseg44k": "data/sam-hq",
        "cocolvis": "data/coco_lvis",
        "coconut_hf": "data/coconut_hf",
    }
    dataset_path = args.dataset_path or dataset_path_map[args.dataset]
    train_set_base, val_set_base = builder_map["dataset"][args.dataset](dataset_path)
    base_set = train_set_base if args.split == "train" else val_set_base
    index_mapping_path = Path(f"data/flat/{args.dataset}") / f"{args.split}_index_mapping.npy"

    return MaskLevelFlatDataset(
        index_mapping_path=index_mapping_path,
        dataset=base_set,
        with_image_embed=False,
        image_feature_cache=None,
        mask_filter_thresh=0.1,
        dtype=torch.float32,
        image_size_encoder=1024,
        image_size_mask=1024,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SimpleMaskVqvaeV2 decoder attention heatmaps.")
    parser.add_argument("--checkpoint_path", type=str, default="out/ddp_simple_mask_vqvae_v2_coconut_ep10_1fix_bugs/checkpoints/latest.pth")
    parser.add_argument("--config", type=str, default="simple_mask_vqvae_v2_dim384")
    parser.add_argument("--image_encoder_checkpoint", type=str, default="ckpt/dino_v3_vits.safetensors")
    parser.add_argument("--image_encoder_config", type=str, default="dino_v3_vits")
    parser.add_argument("--dataset", type=str, default="coconut_hf", choices=["hqseg44k", "cocolvis", "coconut_hf"])
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--sample_stride", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="notebooks/attention_outputs/simple_mask_vqvae_v2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--enable_vq", action="store_true", help="Use quantized query tokens before the decoder.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.checkpoint_path}")
    model = builder_map["simple_mask_vqvae"][args.config](
        simple_mask_vqvae_checkpoint_path=args.checkpoint_path,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        image_encoder_config_name=args.image_encoder_config,
        enable_vq=args.enable_vq,
        device=device,
    ).eval()

    dataset = build_dataset(args)
    print(f"Dataset size: {len(dataset)} ({args.dataset}/{args.split})")

    for out_idx in range(args.num_samples):
        dataset_idx = args.sample_index + out_idx * args.sample_stride
        image, _, mask_normalized, mask = dataset[dataset_idx]
        image_batch = image.unsqueeze(0).to(device)
        mask_batch = mask_normalized.unsqueeze(0).to(device)

        pred_logits, records = run_v2_forward_with_decoder_attention(model, mask_batch, image_batch)
        output_path = output_dir / f"sample_{dataset_idx:06d}_decoder_attention.png"
        save_attention_figure(records, _to_numpy_image(image), mask, pred_logits, output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
