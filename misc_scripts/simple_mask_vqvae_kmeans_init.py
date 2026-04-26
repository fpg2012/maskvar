#!/usr/bin/env python3
"""
Extract latent tokens from SimpleMaskVqvae / SimpleMaskVqvaeV2 and run KMeans
to initialize the VQ codebook.

Example:
    python misc_scripts/simple_mask_vqvae_kmeans_init.py \
        --config simple_mask_vqvae_dim384 \
        --checkpoint-path out/exp/checkpoints/latest.pth \
        --image-encoder-checkpoint ckpt/dino_v3_vits.safetensors \
        --image-encoder-config dino_v3_vits \
        --dataset coconut_hf \
        --subset-index data/subset/coconut_hf_train-25_percent.npy \
        --num-samples 10000 \
        --n-clusters 256 1024 4096 \
        --output-dir out/kmeans_init
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset, MaskLevelFlatSubsetDataset
from maskvar.maskseg_build_everything import builder_map
from maskvar.models.simple_mask_vqvae.simple_mask_vqvae import SimpleMaskVqvae, SimpleMaskVqvaeV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run KMeans init for SimpleMaskVqvae / SimpleMaskVqvaeV2 codebooks."
    )

    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save centroids and metadata.")

    parser.add_argument(
        "--config",
        type=str,
        default="simple_mask_vqvae_dim384",
        choices=sorted(builder_map["simple_mask_vqvae"].keys()),
        help="Model builder config.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Model checkpoint path.")
    parser.add_argument(
        "--image-encoder-checkpoint",
        type=str,
        default=None,
        help="Image encoder checkpoint path passed into the builder.",
    )
    parser.add_argument(
        "--image-encoder-config",
        type=str,
        default="dino_v3_vits",
        choices=sorted(builder_map["image_encoder"].keys()),
        help="Image encoder config passed into the builder.",
    )
    parser.add_argument("--device", type=str, default=None, help="Runtime device, default: cuda if available else cpu.")
    parser.add_argument(
        "--token-source",
        type=str,
        default="auto",
        choices=["auto", "mask_encoder", "compactor"],
        help="Latent source for clustering. V1 usually uses mask_encoder; V2 uses compactor.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="coconut_hf",
        choices=sorted(builder_map["dataset"].keys()),
        help="Dataset name.",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="Dataset path. Defaults follow training script.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split used to build default index mapping path.",
    )
    parser.add_argument(
        "--index-mapping-path",
        type=Path,
        default=None,
        help="Flat dataset index mapping path. Defaults to data/flat/<dataset>/<split>_index_mapping.npy",
    )
    parser.add_argument("--subset-index", type=Path, default=None, help="Optional flat subset index path.")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of flat mask samples to draw.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for mask and token sampling.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor.")
    parser.add_argument("--image-size-encoder", type=int, default=1024, help="Image encoder input size.")
    parser.add_argument("--image-size-mask", type=int, default=1024, help="Mask size fed into the mask encoder.")
    parser.add_argument("--mask-filter-thresh", type=float, default=0.1, help="Mask filter threshold.")

    parser.add_argument(
        "--n-clusters",
        type=int,
        nargs="+",
        default=[64, 256, 1024, 4096],
        help="One or more KMeans cluster counts.",
    )
    parser.add_argument(
        "--max-tokens-for-clustering",
        type=int,
        default=500000,
        help="Upper bound of tokens retained for clustering after extraction. <=0 keeps all extracted tokens.",
    )
    parser.add_argument(
        "--token-collection-mode",
        type=str,
        default="exact",
        choices=["exact", "stream"],
        help="exact matches notebook behavior more closely; stream reduces peak host memory.",
    )
    parser.add_argument(
        "--kmeans-algorithm",
        type=str,
        default="minibatch",
        choices=["minibatch", "full"],
        help="KMeans backend.",
    )
    parser.add_argument("--kmeans-random-state", type=int, default=42, help="Random state for clustering.")
    parser.add_argument("--kmeans-n-init", type=int, default=3, help="Number of KMeans initializations.")
    parser.add_argument("--kmeans-max-iter", type=int, default=100, help="Maximum KMeans iterations.")
    parser.add_argument(
        "--kmeans-batch-size",
        type=int,
        default=None,
        help="MiniBatchKMeans batch size. Defaults to min(10000, len(tokens)//10).",
    )

    parser.add_argument(
        "--save-pca-plot",
        action="store_true",
        help="Save a 2D PCA projection of the sampled tokens.",
    )
    parser.add_argument("--pca-sample-size", type=int, default=20000, help="Token sample size for PCA plotting.")

    parser.add_argument(
        "--save-initialized-checkpoint-template",
        type=str,
        default=None,
        help="Optional output path template for initialized checkpoints. Supports {n_clusters}.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str | None) -> str:
    if device_arg is not None:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dataset_path(dataset_name: str, dataset_path: str | None) -> str:
    if dataset_path is not None:
        return dataset_path
    dataset_path_map = {
        "hqseg44k": "data/sam-hq",
        "cocolvis": "data/coco_lvis",
        "coconut_hf": "data/coconut_hf",
    }
    return dataset_path_map[dataset_name]


def resolve_index_mapping_path(dataset_name: str, split: str, index_mapping_path: Path | None) -> Path:
    if index_mapping_path is not None:
        return index_mapping_path
    return Path(f"data/flat/{dataset_name}/{split}_index_mapping.npy")


def build_model(args: argparse.Namespace, device: str) -> SimpleMaskVqvae | SimpleMaskVqvaeV2:
    model = builder_map["simple_mask_vqvae"][args.config](
        simple_mask_vqvae_checkpoint_path=args.checkpoint_path,
        image_encoder_checkpoint=args.image_encoder_checkpoint,
        image_encoder_config_name=args.image_encoder_config,
        enable_vq=False,
        device=device,
    )
    model.eval()
    return model


def build_dataset(args: argparse.Namespace):
    dataset_path = resolve_dataset_path(args.dataset, args.dataset_path)
    train_set_base, val_set_base = builder_map["dataset"][args.dataset](dataset_path)
    dataset_base = train_set_base if args.split == "train" else val_set_base
    index_mapping_path = resolve_index_mapping_path(args.dataset, args.split, args.index_mapping_path)

    dataset_cls = MaskLevelFlatSubsetDataset if args.subset_index is not None else MaskLevelFlatDataset
    dataset_kwargs = dict(
        index_mapping_path=index_mapping_path,
        dataset=dataset_base,
        with_image_embed=False,
        image_feature_cache=None,
        mask_filter_thresh=args.mask_filter_thresh,
        dtype=torch.float32,
        image_size_encoder=args.image_size_encoder,
        image_size_mask=args.image_size_mask,
    )
    if args.subset_index is not None:
        dataset_kwargs["subset_list"] = args.subset_index

    return dataset_cls(**dataset_kwargs)


def sample_dataset(dataset, num_samples: int, seed: int):
    target_count = min(num_samples, len(dataset))
    rng = np.random.default_rng(seed)
    subset_indices = rng.choice(len(dataset), size=target_count, replace=False)
    return Subset(dataset, subset_indices.tolist()), subset_indices


def infer_token_source(
    model: SimpleMaskVqvae | SimpleMaskVqvaeV2,
    requested_source: str,
) -> str:
    if requested_source != "auto":
        if requested_source == "compactor" and not hasattr(model, "mask_feature_compactor"):
            raise ValueError("Requested --token-source=compactor, but model does not have mask_feature_compactor.")
        return requested_source

    if isinstance(model, SimpleMaskVqvaeV2) or hasattr(model, "mask_feature_compactor"):
        return "compactor"
    return "mask_encoder"


@torch.no_grad()
def extract_latent_tokens(
    model: SimpleMaskVqvae | SimpleMaskVqvaeV2,
    loader: DataLoader,
    device: str,
    token_source: str,
    max_tokens_for_clustering: int,
    rng: np.random.Generator,
    collection_mode: str,
):
    sampled_batches: list[np.ndarray] = []
    total_tokens_seen = 0
    total_masks = 0
    sampled_token_count = 0
    tokens_per_mask: int | None = None
    token_dim: int | None = None
    sampling_prob: float | None = None

    progress = tqdm(loader, desc="Extracting latent tokens")
    for batch in progress:
        _, _, mask_normalized, _ = batch
        mask_normalized = mask_normalized.to(device, non_blocking=True)

        mask_tokens = model.mask_encoder(mask_normalized)
        if token_source == "compactor":
            mask_tokens = rearrange(mask_tokens, "b c h w -> b h w c")
            latent_tokens = model.mask_feature_compactor(mask_tokens)
        else:
            latent_tokens = rearrange(mask_tokens, "b c h w -> b (h w) c")

        latent_tokens = latent_tokens.float().cpu()
        batch_size, batch_tokens_per_mask, batch_token_dim = latent_tokens.shape
        flat_tokens = latent_tokens.reshape(-1, batch_token_dim).numpy()

        if tokens_per_mask is None:
            tokens_per_mask = batch_tokens_per_mask
            token_dim = batch_token_dim
            if collection_mode == "stream" and max_tokens_for_clustering > 0:
                expected_total_tokens = len(loader.dataset) * tokens_per_mask
                sampling_prob = min(1.0, max_tokens_for_clustering / expected_total_tokens)
            else:
                sampling_prob = 1.0

        total_masks += batch_size
        total_tokens_seen += flat_tokens.shape[0]

        if collection_mode == "exact":
            chosen = flat_tokens
        else:
            if sampling_prob is None:
                raise RuntimeError("sampling_prob was not initialized.")
            keep = rng.random(flat_tokens.shape[0]) < sampling_prob
            chosen = flat_tokens[keep]

        if chosen.size > 0:
            sampled_batches.append(chosen)
            sampled_token_count += chosen.shape[0]

        progress.set_postfix(
            masks=total_masks,
            tokens_seen=total_tokens_seen,
            tokens_kept=sampled_token_count,
        )

    if not sampled_batches:
        raise RuntimeError("No latent tokens were collected. Check dataset size and sampling settings.")

    tokens = np.concatenate(sampled_batches, axis=0)
    if max_tokens_for_clustering > 0 and len(tokens) > max_tokens_for_clustering:
        keep_indices = rng.choice(len(tokens), size=max_tokens_for_clustering, replace=False)
        tokens = tokens[keep_indices]

    metadata = {
        "total_masks": total_masks,
        "total_tokens_seen": total_tokens_seen,
        "sampled_token_count_before_trim": sampled_token_count,
        "sampled_token_count_after_trim": int(len(tokens)),
        "tokens_per_mask": tokens_per_mask,
        "token_dim": token_dim,
        "sampling_prob": sampling_prob,
        "collection_mode": collection_mode,
    }
    return tokens, metadata


def save_pca_plot(tokens: np.ndarray, output_path: Path, sample_size: int, rng: np.random.Generator) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(tokens) > sample_size:
        sample_indices = rng.choice(len(tokens), size=sample_size, replace=False)
        tokens_sample = tokens[sample_indices]
    else:
        tokens_sample = tokens

    pca = PCA(n_components=2)
    tokens_pca = pca.fit_transform(tokens_sample)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(tokens_pca[:, 0], tokens_pca[:, 1], alpha=0.5, s=1)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("PCA 2D Projection")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(1, 3), pca.explained_variance_ratio_)
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Explained Variance Ratio")
    axes[1].set_title("PCA Explained Variance")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "sample_size": int(len(tokens_sample)),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
    }


def run_kmeans(tokens: np.ndarray, n_clusters: int, args: argparse.Namespace):
    if n_clusters > len(tokens):
        raise ValueError(
            f"n_clusters={n_clusters} is larger than sampled token count={len(tokens)}. "
            "Increase --max-tokens-for-clustering or reduce --n-clusters."
        )

    if args.kmeans_algorithm == "full":
        estimator = KMeans(
            n_clusters=n_clusters,
            random_state=args.kmeans_random_state,
            n_init=args.kmeans_n_init,
            max_iter=args.kmeans_max_iter,
            verbose=0,
        )
    else:
        batch_size = args.kmeans_batch_size
        if batch_size is None:
            batch_size = min(10000, max(n_clusters, len(tokens) // 10))
        estimator = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=args.kmeans_random_state,
            n_init=args.kmeans_n_init,
            max_iter=args.kmeans_max_iter,
            batch_size=batch_size,
            verbose=0,
            compute_labels=True,
        )

    estimator.fit(tokens)
    return estimator, estimator.cluster_centers_.astype(np.float32), float(estimator.inertia_)


def maybe_load_original_checkpoint(checkpoint_path: str | None):
    if checkpoint_path is None:
        return None
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def save_initialized_checkpoint(
    model: SimpleMaskVqvae | SimpleMaskVqvaeV2,
    centroids: np.ndarray,
    output_path: Path,
    original_checkpoint: Any,
    metadata: dict[str, Any],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        model.quant.embedding.weight.copy_(torch.from_numpy(centroids))

    state_dict = model.state_dict()
    if isinstance(original_checkpoint, dict) and "model_state_dict" in original_checkpoint:
        checkpoint_to_save = dict(original_checkpoint)
        checkpoint_to_save["model_state_dict"] = state_dict
    else:
        checkpoint_to_save = {
            "model_state_dict": state_dict,
        }

    checkpoint_to_save["kmeans_init"] = metadata
    torch.save(checkpoint_to_save, output_path)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    rng = np.random.default_rng(args.sample_seed)

    print(f"Using device: {device}")
    print(f"Building model with config={args.config}")
    model = build_model(args, device)

    print(f"Building dataset={args.dataset}, split={args.split}")
    dataset = build_dataset(args)
    sampled_dataset, sampled_indices = sample_dataset(dataset, args.num_samples, args.sample_seed)
    print(f"Sampled {len(sampled_dataset)} flat masks from dataset of size {len(dataset)}")

    loader = DataLoader(
        sampled_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    token_source = infer_token_source(model, args.token_source)
    print(f"Using token source: {token_source}")

    tokens, token_metadata = extract_latent_tokens(
        model=model,
        loader=loader,
        device=device,
        token_source=token_source,
        max_tokens_for_clustering=args.max_tokens_for_clustering,
        rng=rng,
        collection_mode=args.token_collection_mode,
    )

    print(f"Collected tokens for clustering: {tokens.shape}")

    summary: dict[str, Any] = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "device": device,
        "model_class": type(model).__name__,
        "token_source": token_source,
        "dataset_size": len(dataset),
        "sampled_dataset_size": len(sampled_dataset),
        "sampled_dataset_indices_preview": sampled_indices[: min(10, len(sampled_indices))].tolist(),
        "token_extraction": token_metadata,
        "clusters": {},
    }

    if args.save_pca_plot:
        pca_plot_path = args.output_dir / "tokens_pca.png"
        summary["pca"] = save_pca_plot(tokens, pca_plot_path, args.pca_sample_size, rng)
        print(f"Saved PCA plot to {pca_plot_path}")

    original_checkpoint = maybe_load_original_checkpoint(args.checkpoint_path)

    for n_clusters in args.n_clusters:
        print(f"Running {args.kmeans_algorithm} KMeans with n_clusters={n_clusters}")
        estimator, centroids, inertia = run_kmeans(tokens, n_clusters, args)

        output_npy = args.output_dir / f"kmeans_centroids_n{n_clusters}.npy"
        output_pt = args.output_dir / f"kmeans_centroids_n{n_clusters}.pt"
        np.save(output_npy, centroids)
        torch.save(torch.from_numpy(centroids), output_pt)

        cluster_metadata = {
            "n_clusters": n_clusters,
            "centroids_shape": list(centroids.shape),
            "inertia": inertia,
            "centroids_mean": float(centroids.mean()),
            "centroids_std": float(centroids.std()),
            "centroids_min": float(centroids.min()),
            "centroids_max": float(centroids.max()),
            "n_iter": int(estimator.n_iter_),
            "output_npy": str(output_npy),
            "output_pt": str(output_pt),
        }
        summary["clusters"][str(n_clusters)] = cluster_metadata

        print(
            f"Saved n={n_clusters} centroids to {output_npy} and {output_pt} "
            f"(inertia={inertia:.4f}, std={cluster_metadata['centroids_std']:.4f})"
        )

        if args.save_initialized_checkpoint_template is not None:
            checkpoint_path = Path(args.save_initialized_checkpoint_template.format(n_clusters=n_clusters))
            save_initialized_checkpoint(
                model=model,
                centroids=centroids,
                output_path=checkpoint_path,
                original_checkpoint=original_checkpoint,
                metadata=cluster_metadata,
            )
            cluster_metadata["initialized_checkpoint"] = str(checkpoint_path)
            print(f"Saved initialized checkpoint to {checkpoint_path}")

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
