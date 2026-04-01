#!/usr/bin/env python3
"""
Test script for CoconutHFDataset - loads samples from HuggingFace parquet format.

Usage:
    python test_coconut_hf.py \
        --parquet ~/workspace/coconut_cvpr2024/train-00000-of-00002.parquet \
        --image-root datasets/coco/train2017 \
        --num-samples 5
"""
import sys
sys.path.insert(0, '/data/clc/maskseg')

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from maskvar.datasets import CoconutHFDataset


def main():
    parser = argparse.ArgumentParser(description='Test CoconutHFDataset')
    parser.add_argument('--parquet', type=str,
                        default='/home/clc/workspace/coconut_cvpr2024/train-00000-of-00002.parquet',
                        help='Path to parquet file or directory')
    parser.add_argument('--image-root', type=str,
                        default='datasets/coco/train2017',
                        help='Path to COCO images directory')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to load')
    parser.add_argument('--stuff-prob', type=float, default=1.0,
                        help='Probability of keeping background objects')
    args = parser.parse_args()

    print(f"Loading dataset from: {args.parquet}")
    print(f"Image root: {args.image_root}")
    print(f"Stuff probability: {args.stuff_prob}")
    print()

    # Create dataset
    dataset = CoconutHFDataset(
        parquet_path=args.parquet,
        image_root=args.image_root,
        stuff_prob=args.stuff_prob,
    )

    print(f"Dataset size: {len(dataset)}")
    print()

    # Load and inspect samples
    for i in range(min(args.num_samples, len(dataset))):
        print(f"=== Sample {i} ===")

        image, layers, instances_info = dataset[i]

        print(f"  Image shape: {image.shape}")  # (H, W, 3)
        print(f"  Image dtype: {image.dtype}")
        print(f"  Image range: [{image.min()}, {image.max()}]")

        print(f"  Layers shape: {layers.shape}")  # (H, W, L)
        print(f"  Number of instances: {len(instances_info)}")

        # Show instance info
        for inst_id, info in instances_info.items():
            print(f"    Instance {inst_id}: mapping={info.mapping}, level={info.node_level}")

        # Count unique values in each layer
        for j in range(layers.shape[2]):
            unique_vals = np.unique(layers[:, :, j])
            print(f"    Layer {j} unique values: {unique_vals}")

        print()

    print("Test completed successfully!")


if __name__ == '__main__':
    main()
