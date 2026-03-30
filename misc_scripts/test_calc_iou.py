"""
Test script for calc_iou function.
Tests various edge cases and scenarios.
"""

import torch
from maskvar.utils.metrics import calc_iou


def test_calc_iou():
    print("Testing calc_iou function...\n")

    # Test 1: Perfect match
    print("Test 1: Perfect match")
    pred = torch.ones(2, 1, 4, 4)
    target = torch.ones(2, 1, 4, 4)
    iou = calc_iou(pred, target)
    expected = torch.tensor([1.0, 1.0])
    assert torch.allclose(iou, expected), f"Expected {expected}, got {iou}"
    print(f"  IoU: {iou.tolist()} ✓")

    # Test 2: No overlap
    print("\nTest 2: No overlap")
    pred = torch.zeros(2, 1, 4, 4)
    pred[0, 0, :2, :2] = 1.0  # Top-left corner
    target = torch.zeros(2, 1, 4, 4)
    target[0, 0, 2:, 2:] = 1.0  # Bottom-right corner (no overlap with top-left)
    # Also set target[1] to have content so union != 0
    target[1, 0, :2, :2] = 1.0  # This overlaps with nothing since pred[1] is empty
    iou = calc_iou(pred, target)
    print(f"  IoU: {iou.tolist()}")
    assert iou[0] == 0.0, f"Sample 0: Expected IoU=0.0, got {iou[0]}"  # No overlap
    assert iou[1] == 0.0, f"Sample 1: Expected IoU=0.0, got {iou[1]}"  # Pred is empty, target is not
    print("  Both samples have IoU=0.0 ✓")

    # Test 3: Partial overlap (50%)
    print("\nTest 3: Partial overlap (50%)")
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, :2, :] = 1.0  # Top half
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 1:3, :] = 1.0  # Middle rows
    # Intersection: 1 row x 4 cols = 4
    # Union: 3 rows x 4 cols = 12
    # IoU = 4/12 = 0.333...
    iou = calc_iou(pred, target)
    expected = torch.tensor([4/12])
    assert torch.allclose(iou, expected), f"Expected {expected}, got {iou}"
    print(f"  IoU: {iou.tolist()} (expected ~0.333) ✓")

    # Test 4: Empty prediction (all zeros)
    print("\nTest 4: Empty prediction (all zeros)")
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.ones(1, 1, 4, 4)
    iou, nan_count = calc_iou(pred, target, return_nan_count=True)
    print(f"  IoU: {iou.tolist()} (empty pred, should be 0.0)")
    print(f"  NaN count: {nan_count.item()}")
    # When both pred and target are empty, union=0, so iou=nan -> converted to 1.0
    # When pred is empty but target is not, iou=0

    # Test 5: Empty target (all zeros)
    print("\nTest 5: Empty target (all zeros)")
    pred = torch.ones(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    iou = calc_iou(pred, target)
    print(f"  IoU: {iou.tolist()} (empty target, should be 0.0)")
    # Intersection=0, Union=16, IoU=0/16=0

    # Test 6: Both empty (should be treated as perfect match after nan_to_num)
    print("\nTest 6: Both empty (pred=0, target=0)")
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    iou, nan_count = calc_iou(pred, target, return_nan_count=True)
    print(f"  IoU: {iou.tolist()} (both empty -> nan -> 1.0)")
    print(f"  NaN count: {nan_count.item()} (should be 1)")
    assert nan_count == 1, f"Expected 1 NaN, got {nan_count}"
    assert iou[0] == 1.0, f"Expected IoU=1.0 after nan_to_num, got {iou[0]}"

    # Test 7: Batch with different scenarios
    print("\nTest 7: Batch with different scenarios")
    pred = torch.zeros(3, 1, 4, 4)
    target = torch.zeros(3, 1, 4, 4)
    # Sample 0: Perfect match
    pred[0] = 1.0
    target[0] = 1.0
    # Sample 1: Half overlap
    pred[1, 0, :2, :] = 1.0
    target[1, 0, 1:3, :] = 1.0
    # Sample 2: No overlap
    pred[2, 0, :2, :2] = 1.0
    target[2, 0, 2:, 2:] = 1.0

    iou = calc_iou(pred, target)
    expected = torch.tensor([1.0, 4/12, 0.0])
    assert torch.allclose(iou, expected, atol=1e-6), f"Expected {expected}, got {iou}"
    print(f"  IoU: {iou.tolist()} ✓")

    # Test 8: Continuous values (before threshold)
    print("\nTest 8: Continuous values (threshold at >0)")
    pred = torch.full((1, 1, 4, 4), 0.5)  # All 0.5 -> after >0, all True
    target = torch.ones(1, 1, 4, 4)
    iou = calc_iou(pred, target)
    expected = torch.tensor([1.0])
    assert torch.allclose(iou, expected), f"Expected {expected}, got {iou}"
    print(f"  IoU: {iou.tolist()} (0.5 > 0, so considered as 1) ✓")

    # Test 9: Negative values
    print("\nTest 9: Negative values")
    pred = torch.full((1, 1, 4, 4), -0.5)  # All -0.5 -> after >0, all False
    target = torch.ones(1, 1, 4, 4)
    iou = calc_iou(pred, target)
    print(f"  IoU: {iou.tolist()} (negative values -> False)")

    # Test 10: Different shapes
    print("\nTest 10: Different spatial sizes")
    pred = torch.ones(2, 1, 256, 256)
    target = torch.ones(2, 1, 256, 256)
    iou = calc_iou(pred, target)
    expected = torch.tensor([1.0, 1.0])
    assert torch.allclose(iou, expected), f"Expected {expected}, got {iou}"
    print(f"  IoU: {iou.tolist()} ✓")

    # Test 11: Typical mask values (0-1 binary)
    print("\nTest 11: Binary masks (0 or 1)")
    pred = torch.randint(0, 2, (4, 1, 64, 64)).float()
    target = torch.randint(0, 2, (4, 1, 64, 64)).float()
    iou = calc_iou(pred, target)
    print(f"  Random binary masks IoU: {iou.tolist()}")
    assert torch.all((iou >= 0) & (iou <= 1)), "IoU should be in [0, 1]"
    print("  All IoU values in valid range [0, 1] ✓")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == "__main__":
    test_calc_iou()
