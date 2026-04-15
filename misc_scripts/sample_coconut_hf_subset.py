"""
随机采样 COCONut HF 数据集的 1/4 子集。

使用方法:
    python misc_scripts/sample_coconut_hf_subset.py \
        --dataset_path data/coconut_hf \
        --output_path data/coconut_hf_subset_25.npy \
        --seed 42 \
        --split train

输出:
    一个 .npy 文件，包含随机采样的图像索引数组，可用于 MaskLevelFlatSubsetDataset
"""

import argparse
import numpy as np
from pathlib import Path
import sys
import torch

from maskvar.maskseg_build_everything import builder_map
from maskvar.datasets.mask_level_dataset import (
    MaskLevelFlatDataset,
    MaskLevelFlatSubsetDataset,
)


def sample_subset(dataset_path: str, index_mapping_dir: str, output_path: str, percent: float = 0.25, seed: int = 42, split: str = 'train'):
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

    # 随机采样 1/4
    rng = np.random.default_rng(seed)
    subset_size = int(total_samples * percent)
    subset_indices = rng.choice(total_samples, size=subset_size, replace=False)
    subset_indices = np.sort(subset_indices)  # 排序便于查看

    print(f"Sampled subset size: {subset_size} ({subset_size / total_samples * 100:.1f}%)")
    print(f"Subset indices range: [{subset_indices.min()}, {subset_indices.max()}]")

    # 保存为 npy 文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, subset_indices)

    print(f"Subset indices saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.2f} KB")

    return subset_indices


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample 1/4 subset from COCONut HF dataset'
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
        default='data/subset/coconut_hf_train-25_percent.npy',
        help='Output path for subset indices (.npy file)'
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

    # 采样子集
    subset_indices = sample_subset(
        dataset_path=args.dataset_path,
        index_mapping_dir=args.index_mapping_dir,
        output_path=args.output_path,
        seed=args.seed,
        percent=args.percent,
        split=args.split
    )
