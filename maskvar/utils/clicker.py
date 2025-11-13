from typing import List, Tuple
import cv2
import numpy as np
import torch
import traceback


def init_clicks(gt_mask, num_random_clicks=2, click_list=[], not_clicked_map=None):
    click_list = []
    if not_clicked_map is None:
        not_clicked_map = np.ones_like(gt_mask, dtype=bool)
    try:
        for _ in range(num_random_clicks):
            # Erode the mask to get points away from edges
            # kernel = np.ones((3, 3), np.uint8)
            # eroded_mask = cv2.erode(gt_mask.astype(np.uint8), kernel, iterations=1)
            eroded_mask = gt_mask.astype(np.uint8)
            # pad eroded_mask with 1 pixel
            eroded_mask = np.pad(eroded_mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)

            # Compute distance transform - points closer to center have higher values
            dt = cv2.distanceTransform(eroded_mask, cv2.DIST_L2, 3)

            # unpad dt
            dt = dt[1:-1, 1:-1]

            # Sample a point based on the probability map
            flat_probs = ((dt*not_clicked_map)**2).flatten()
            flat_probs = flat_probs / flat_probs.sum()  # Normalize to probabilities
            idx = np.random.choice(len(flat_probs), p=flat_probs)
            y, x = np.unravel_index(idx, dt.shape)
            
            # Add random click (1 for positive since sampling from gt_mask)
            click_list.append((y, x, 1))
            not_clicked_map[y, x] = False
    except Exception as e:
        # print traceback
        traceback.print_exc()
        print(f"Error in init_clicks: {e}")
    finally:
        return click_list, eroded_mask, dt

def predict_next_click(gt_mask, pred_mask, click_list=[], not_clicked_map=None):
    """
    predict next click and update click list

    pred_mask: (H, W)
    
    Returns:
        Tuple[int, int, int]: (y, x, is_positive) coordinates of next click
    """
    assert gt_mask is not None, "Ground truth mask not given."
    
    if not_clicked_map is None:
        not_clicked_map = np.ones_like(gt_mask, dtype=bool)
    
    assert pred_mask.ndim == 2
    
    # Calculate false negative mask (ground truth is 1 but prediction is 0)
    fn_mask = np.logical_and(np.logical_and(gt_mask, np.logical_not(pred_mask)), not_clicked_map)
    # Calculate false positive mask (ground truth is 0 but prediction is 1)
    fp_mask = np.logical_and(np.logical_and(np.logical_not(gt_mask), pred_mask), not_clicked_map)

    # pad fn_mask and fp_mask with 1 pixel
    fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    # Compute distance transforms to find farthest points from boundaries
    fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
    fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

    # unpad fn_mask_dt and fp_mask_dt
    fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
    fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

    # Mask out already clicked points
    fn_mask_dt = fn_mask_dt * not_clicked_map
    fp_mask_dt = fp_mask_dt * not_clicked_map

    # Find maximum distances in each mask
    fn_max_dist = np.max(fn_mask_dt)
    fp_max_dist = np.max(fp_mask_dt)

    # Determine if next click should be positive (add) or negative (remove)
    is_positive = fn_max_dist > fp_max_dist
    
    # Get coordinates of point with maximum distance
    if is_positive:
        coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)
    else:
        coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)

    # Store click and update state
    click = (coords_y[0], coords_x[0], 1 if is_positive else 0)
    click_list.append(click)
    not_clicked_map[coords_y[0], coords_x[0]] = False
    
    return click, click_list, not_clicked_map

def to_sam_format(click_list, pad_size=0, device='cpu'):
    coords = torch.tensor([(click[1], click[0]) for click in click_list], device=device)
    # label: 1 for positive, 0 for negative, -1 for padding
    label = torch.tensor([click[2] for click in click_list], device=device)
    L_clicks = len(click_list)
    if pad_size > 0 and pad_size > L_clicks:
        coords = torch.cat([coords, torch.zeros(pad_size - L_clicks, 2, device=device)], dim=0).to(dtype=torch.float)
        label = torch.cat([label, torch.zeros(pad_size - L_clicks, dtype=torch.long, device=device) - 1], dim=0).to(dtype=torch.int)
    return coords, label