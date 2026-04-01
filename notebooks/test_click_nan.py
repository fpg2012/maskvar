#!/usr/bin/env python3
"""
Test script to check how many masks in dataset cause NaN probabilities in init_clicks.
Uses MaskLevelFlatDataset to align with training script behavior.
"""
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import torch

from maskvar.maskseg_build_everything import builder_map
from maskvar.utils.clicker import init_clicks
from maskvar.datasets.mask_level_dataset import MaskLevelFlatDataset


def check_mask_for_nan(mask, num_tests=10):
    """
    Check if a mask causes NaN probabilities in init_clicks.
    Returns True if NaN occurs.
    """
    for _ in range(num_tests):
        try:
            click_list, _, _ = init_clicks(mask, num_random_clicks=1, random_sample=True)
            if len(click_list) == 0:
                return True, "empty_click_list"
        except ValueError as e:
            if "probabilities contain NaN" in str(e):
                return True, "nan_probabilities"
            raise
    return False, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hqseg44k')
    parser.add_argument('--output', type=str, default='click_nan_problems.txt')
    args = parser.parse_args()

    print(f"Loading {args.dataset} dataset...")
    train_set, val_set = builder_map["dataset"][args.dataset]()
    print(f"Total images: {len(train_set)}")

    # Setup dataset paths
    dataset_dir_map = {
        "hqseg44k": "data/sam-hq",
        "coco_lvis": "data/coco_lvis",
        "coconut_hf": "data/coconut_hf",
    }
    dataset_dir = dataset_dir_map.get(args.dataset, f"data/{args.dataset}")
    index_mapping_path = f'data/flat/{args.dataset}'

    # Create MaskLevelFlatDataset to match training behavior
    # Note: with_image_embed=False since we only need masks for this test
    print("Creating MaskLevelFlatDataset...")
    train_set_masklevel = MaskLevelFlatDataset(
        index_mapping_path=Path(index_mapping_path) / "train_index_mapping.npy",
        dataset=train_set,
        with_image_embed=False,
        image_feature_cache=None,
        mask_filter_thresh=0.1,
        dtype=torch.float32,
    )
    print(f"Total masks in MaskLevelFlatDataset: {len(train_set_masklevel)}")

    problem_cases = []
    total_masks = 0
    nan_count = 0
    empty_click_count = 0

    for idx in tqdm(range(len(train_set_masklevel)), desc="Checking masks"):
        _, _, _, single_mask = train_set_masklevel[idx]
        total_masks += 1

        # single_mask is a torch tensor (1, H, W), convert to numpy
        single_mask_np = single_mask.squeeze(0).cpu().numpy()

        # Check if mask is valid (after preprocessing in MaskLevelFlatDataset)
        if single_mask_np.sum() == 0:
            problem_cases.append({
                'dataset_idx': idx,
                'reason': 'empty_mask',
                'mask_sum': 0,
                'mask_shape': single_mask_np.shape,
            })
            nan_count += 1
            continue

        # Test init_clicks
        has_nan, reason = check_mask_for_nan(single_mask_np, num_tests=2)

        if has_nan:
            pr_cs = {
                'dataset_idx': idx,
                'reason': reason,
                'mask_sum': float(single_mask_np.sum()),
                'mask_shape': single_mask_np.shape,
                'mask_nonzero': int((single_mask_np > 0).sum()),
                'global_idx': total_masks,
            }
            problem_cases.append(pr_cs)
            print(pr_cs)
            if reason == "nan_probabilities":
                nan_count += 1
            else:
                empty_click_count += 1

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total masks checked: {total_masks}")
    print(f"Problem cases: {len(problem_cases)}")
    print(f"  - NaN probabilities: {nan_count}")
    print(f"  - Empty click list: {empty_click_count}")
    print(f"  - Empty masks: {len([c for c in problem_cases if c['reason'] == 'empty_mask'])}")
    print(f"Problem rate: {len(problem_cases) / total_masks * 100:.2f}%")

    # Save detailed results
    if problem_cases:
        with open(args.output, 'w') as f:
            f.write(f"# Click NaN Problem Report\n")
            f.write(f"# Total masks: {total_masks}\n")
            f.write(f"# Problem cases: {len(problem_cases)}\n")
            f.write(f"# Problem rate: {len(problem_cases) / total_masks * 100:.2f}%\n")
            f.write("\n")
            for case in problem_cases:
                f.write(f"Image {case['image_idx']}, Instance {case['instance_idx']}: {case['reason']}\n")
                f.write(f"  mask_sum={case.get('mask_sum', 'N/A')}, shape={case['mask_shape']}\n")
                if 'mask_nonzero' in case:
                    f.write(f"  nonzero_pixels={case['mask_nonzero']}\n")
                f.write("\n")
        print(f"\nDetailed report saved to: {args.output}")

    # Show some examples
    if problem_cases:
        print("\nSample problem cases:")
        for case in problem_cases[:5]:
            print(f"  Image {case['image_idx']}, Instance {case['instance_idx']}: {case['reason']}")


if __name__ == "__main__":
    main()
