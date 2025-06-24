from typing import List, Tuple
import cv2
import numpy as np
import torch

class Clicker:

    def __init__(self, num_random_clicks: int = 2):
        self.click_list: List[Tuple[int, int, int]] = []
        # gt_mask: (H, W)
        self.gt_mask = None
        self.not_ignore_mask = None # ignore pixels that are -1
        self.not_clicked_map = None # mask out clicked pixels
        self.num_random_clicks = num_random_clicks

    def init_clicks(self) -> List[Tuple[int, int, int]]:
        """
        random sample some clickes predict initial clicks
        """
        try:
            for _ in range(self.num_random_clicks):
                # Erode the mask to get points away from edges
                # kernel = np.ones((3, 3), np.uint8)
                # eroded_mask = cv2.erode(self.gt_mask.astype(np.uint8), kernel, iterations=1)
                eroded_mask = self.gt_mask.astype(np.uint8)
                # pad eroded_mask with 1 pixel
                eroded_mask = np.pad(eroded_mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)

                # Compute distance transform - points closer to center have higher values
                dt = cv2.distanceTransform(eroded_mask, cv2.DIST_L2, 3)

                # unpad dt
                dt = dt[1:-1, 1:-1]

                # Sample a point based on the probability map
                flat_probs = ((dt*self.not_clicked_map)**2).flatten()
                flat_probs = flat_probs / flat_probs.sum()  # Normalize to probabilities
                idx = np.random.choice(len(flat_probs), p=flat_probs)
                y, x = np.unravel_index(idx, dt.shape)
                
                # Add random click (1 for positive since sampling from gt_mask)
                self.click_list.append((y, x, 1))
                self.not_clicked_map[y, x] = False
        except Exception as e:
            print(f"Error in init_clicks: {e}")
        finally:
            return self.click_list, eroded_mask, dt
    
    def set_gt_mask(self, gt_mask):
        """
        gt_mask: (H, W)
        """
        assert gt_mask.ndim == 2
        self.gt_mask = gt_mask == 1
        self.not_ignore_mask = gt_mask != -1
        self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)

    def predict_next_click(self, pred_mask) -> Tuple[int, int, int]:
        """
        predict next click and update click list

        pred_mask: (H, W)
        
        Returns:
            Tuple[int, int, int]: (y, x, is_positive) coordinates of next click
        """
        if self.gt_mask is None:
            raise ValueError("Ground truth mask not set. Call set_gt_mask first.")
        
        assert pred_mask.ndim == 2
        
        # Calculate false negative mask (ground truth is 1 but prediction is 0)
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        # Calculate false positive mask (ground truth is 0 but prediction is 1)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

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
        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

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
        self.click_list.append(click)
        self.not_clicked_map[coords_y[0], coords_x[0]] = False
        
        return click
    
    def to_sam_format(self, pad_size: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = torch.tensor([(click[1], click[0]) for click in self.click_list])
        # label: 1 for positive, 0 for negative, -1 for padding
        label = torch.tensor([click[2] for click in self.click_list])
        L_clicks = len(self.click_list)
        if pad_size > 0 and pad_size > L_clicks:
            coords = torch.cat([coords, torch.zeros(pad_size - L_clicks, 2)], dim=0)
            label = torch.cat([label, torch.zeros(pad_size - L_clicks, dtype=torch.long) - 1], dim=0)
        return coords, label


def init_clicks(gt_mask, num_random_clicks=2, click_list=[], not_clicked_map=None):
    click_list = []
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
    if gt_mask is None:
        raise ValueError("Ground truth mask not set. Call set_gt_mask first.")
    
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