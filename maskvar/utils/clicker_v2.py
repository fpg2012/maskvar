from typing import List, Tuple
import traceback

import cv2
import numpy as np

from .clicker import predict_next_click, to_sam_format


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), ((1, 1), (1, 1)), mode='constant', constant_values=0)
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 3)
    return dt[1:-1, 1:-1]


def _sample_from_component(component_mask: np.ndarray, not_clicked_map: np.ndarray, random_sample: bool):
    available = np.logical_and(component_mask, not_clicked_map)
    if not available.any():
        return None

    area = int(available.sum())
    if area <= 9:
        ys, xs = np.where(available)
        idx = np.random.choice(len(ys))
        return int(ys[idx]), int(xs[idx])

    dt = _distance_transform(component_mask)
    weights = (dt ** 2) * available
    weights_sum = float(weights.sum())

    if weights_sum <= 0:
        ys, xs = np.where(available)
        idx = np.random.choice(len(ys))
        return int(ys[idx]), int(xs[idx])

    if random_sample:
        probs = (weights / weights_sum).reshape(-1)
        idx = np.random.choice(len(probs), p=probs)
        y, x = np.unravel_index(idx, weights.shape)
        return int(y), int(x)

    idx = int(np.argmax(weights.reshape(-1)))
    y, x = np.unravel_index(idx, weights.shape)
    return int(y), int(x)


def init_clicks(gt_mask, num_random_clicks=1, not_clicked_map=None, random_sample=True):
    """
    Init positive clicks for interactive segmentation.

    Interface matches maskvar.utils.clicker.init_clicks:
        returns (click_list, eroded_mask, dt)

    The sampler prefers component interiors via distance-transform weights. When
    multiple components exist, the largest components are covered first.
    """
    assert random_sample or (not random_sample and num_random_clicks == 1), \
        f"num_random_clicks must be 1 if random_sample set to False, got {num_random_clicks}"

    gt_mask = np.asarray(gt_mask) > 0
    if gt_mask.sum() == 0:
        empty_mask = np.zeros_like(gt_mask, dtype=np.uint8)
        empty_dt = np.zeros_like(gt_mask, dtype=np.float32)
        return [], empty_mask, empty_dt

    if not_clicked_map is None:
        not_clicked_map = np.ones_like(gt_mask, dtype=bool)

    click_list: List[Tuple[int, int, int]] = []
    eroded_mask = gt_mask.astype(np.uint8)
    dt = _distance_transform(eroded_mask)

    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded_mask, connectivity=8)
        component_ids = list(range(1, num_labels))
        component_ids.sort(key=lambda cid: stats[cid, cv2.CC_STAT_AREA], reverse=True)

        for click_idx in range(num_random_clicks):
            if not np.logical_and(gt_mask, not_clicked_map).any():
                break

            if click_idx < len(component_ids):
                component_mask = labels == component_ids[click_idx]
            else:
                component_mask = gt_mask

            point = _sample_from_component(component_mask, not_clicked_map, random_sample)
            if point is None:
                point = _sample_from_component(gt_mask, not_clicked_map, random_sample)
            if point is None:
                break

            y, x = point
            click_list.append((y, x, 1))
            not_clicked_map[y, x] = False
    except Exception as e:
        traceback.print_exc()
        print(f"Error in clicker_v2.init_clicks: {e}")
    finally:
        return click_list, eroded_mask, dt
