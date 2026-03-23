#!/usr/bin/env python3
"""
Test script to check how many masks in hqseg44k dataset cause NaN probabilities in init_clicks.
"""
import sys
sys.path.insert(0, '/data/clc/maskseg')

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

from maskvar.datasets.hqseg44k import HQSeg44KTrainDataset
from maskvar.utils.clicker import init_clicks


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
    parser.add_argument('--data_root', type=str, default='data/sam-hq')
    parser.add_argument('--output', type=str, default='click_nan_problems.txt')
    args = parser.parse_args()

    print(f"Loading HQSeg-44K dataset from {args.data_root}...")
    dataset = HQSeg44KTrainDataset(data_root=args.data_root)
    print(f"Total images: {len(dataset)}")

    problem_cases = []
    total_masks = 0
    nan_count = 0
    empty_click_count = 0

    for img_idx in tqdm(range(len(dataset)), desc="Checking masks"):
        image, mask, instance_info = dataset[img_idx]

        for instance_idx in instance_info.keys():
            total_masks += 1

            # Extract single mask
            single_mask = mask[:, :, instance_info[instance_idx].mapping[0]] == instance_info[instance_idx].mapping[1]
            single_mask = single_mask.astype(np.float32)

            # Check if mask is valid
            if single_mask.sum() == 0:
                problem_cases.append({
                    'image_idx': img_idx,
                    'instance_idx': instance_idx,
                    'reason': 'empty_mask',
                    'mask_sum': 0,
                    'mask_shape': single_mask.shape
                })
                nan_count += 1
                continue

            # Test init_clicks
            has_nan, reason = check_mask_for_nan(single_mask, num_tests=1)

            if has_nan:
                problem_cases.append({
                    'image_idx': img_idx,
                    'instance_idx': instance_idx,
                    'reason': reason,
                    'mask_sum': float(single_mask.sum()),
                    'mask_shape': single_mask.shape,
                    'mask_nonzero': int((single_mask > 0).sum())
                })
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
