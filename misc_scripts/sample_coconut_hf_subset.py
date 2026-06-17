"""
随机采样 COCONut HF 数据集的子集。

使用方法:
    python misc_scripts/sample_coconut_hf_subset.py \
        --dataset_path data/coconut_hf \
        --output_path data/coconut_hf_subset_25.npy \
        --seed 42 \
        --split train

输出:
    一个 .npy 文件，包含随机采样的 flat mask 索引数组，可用于 MaskLevelFlatSubsetDataset
"""

import argparse
import numpy as np
from pathlib import Path


def _save_subset(output_path: Path, subset_indices: np.ndarray):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, subset_indices)

    print(f"Subset indices saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")


def sample_subset(
    dataset_path: str,
    index_mapping_dir: str,
    output_path: str,
    percent: float = 0.25,
    seed: int = 42,
    split: str = 'train',
):
    """
    从 COCONut HF 数据集中随机采样 1/4 的子集。

    Args:
        dataset_path: 数据集根目录路径
        output_path: 输出的 .npy 文件路径
        seed: 随机种子
        split: 'train' 或 'val'
    """
    # 构建完整数据集
    print(f"Loading COCONut HF dataset from {dataset_path}...")

    import torch
    from maskvar.maskseg_build_everything import builder_map
    from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset

    if split == 'train':
        dataset, _ = builder_map['dataset']['coconut_hf'](dataset_path=dataset_path)
    else:
        _, dataset = builder_map['dataset']['coconut_hf'](dataset_path=dataset_path)
    
    mask_level_set = MaskLevelFlatDataset(
        index_mapping_path=Path(index_mapping_dir) / f"{split}_index_mapping.npy",
        dataset=dataset,
        with_image_embed=False,  # SimpleMaskVqvae encodes images on-the-fly
        image_feature_cache=None,
        dtype=torch.float32,
        image_size_encoder=1024,
        image_size_mask=1024,
    )

    total_samples = len(mask_level_set)
    print(f"Total samples in {split} split: {total_samples}")

    # 随机采样
    rng = np.random.default_rng(seed)
    subset_size = int(total_samples * percent)
    subset_indices = rng.choice(total_samples, size=subset_size, replace=False)
    subset_indices = np.sort(subset_indices)  # 排序便于查看

    print(f"Sampled subset size: {subset_size} ({subset_size / total_samples * 100:.1f}%)")
    print(f"Subset indices range: [{subset_indices.min()}, {subset_indices.max()}]")

    _save_subset(Path(output_path), subset_indices)

    return subset_indices


def sample_from_existing_subset(
    base_subset_index: str,
    output_path: str | None,
    output_dir: str,
    name_prefix: str,
    fraction: float = 1 / 8,
    seed: int = 42,
):
    base_subset_index = Path(base_subset_index)
    base_indices = np.load(base_subset_index)
    total_samples = len(base_indices)
    subset_size = int(total_samples * fraction)

    if subset_size <= 0:
        raise ValueError(f"fraction={fraction} produced empty subset from {total_samples} samples")

    print(f"Loading base subset from {base_subset_index}...")
    print(f"Total samples in base subset: {total_samples}")

    rng = np.random.default_rng(seed)
    selected_positions = rng.choice(total_samples, size=subset_size, replace=False)
    subset_indices = np.sort(base_indices[selected_positions])

    print(f"Sampled subset size: {subset_size} ({subset_size / total_samples * 100:.1f}% of base subset)")
    print(f"Subset indices range: [{subset_indices.min()}, {subset_indices.max()}]")

    if output_path is None:
        output_path = Path(output_dir) / f"{name_prefix}_{subset_size}.npy"
    else:
        output_path = Path(output_path)

    _save_subset(output_path, subset_indices)
    return subset_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample subset indices from COCONut HF flat mask dataset'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='data/coconut_hf',
        help='Path to COCONut HF dataset directory'
    )
    parser.add_argument(
        '--index_mapping_dir',
        type=str,
        default='data/flat/coconut_hf'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Output path for subset indices (.npy file)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/subset',
        help='Output directory used when output_path is omitted'
    )
    parser.add_argument(
        '--name_prefix',
        type=str,
        default='fast_interation_subset',
        help='Output filename prefix used when sampling from an existing subset'
    )
    parser.add_argument(
        '--base_subset_index',
        type=str,
        default=None,
        help='Optional existing subset index to sample from'
    )
    parser.add_argument(
        '--fraction',
        type=float,
        default=1 / 8,
        help='Fraction to sample from base_subset_index'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--percent',
        type=float,
        default=0.25
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val'],
        help='Which split to sample from'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the subset after creation'
    )

    args = parser.parse_args()

    if args.base_subset_index:
        subset_indices = sample_from_existing_subset(
            base_subset_index=args.base_subset_index,
            output_path=args.output_path,
            output_dir=args.output_dir,
            name_prefix=args.name_prefix,
            fraction=args.fraction,
            seed=args.seed,
        )
    else:
        output_path = args.output_path or f"data/subset/coconut_hf_{args.split}-{int(args.percent * 100)}_percent.npy"
        subset_indices = sample_subset(
            dataset_path=args.dataset_path,
            index_mapping_dir=args.index_mapping_dir,
            output_path=output_path,
            seed=args.seed,
            percent=args.percent,
            split=args.split
        )
