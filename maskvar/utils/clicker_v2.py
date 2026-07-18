import os
import random
from typing import List, Tuple
import traceback

import cv2
import numpy as np
import torch

from maskvar.interaction import CLK_NEGATIVE, CLK_POSITIVE, ClickSampler
from maskvar.interaction.backend import cuda_sample_from_masks_batch

from .clicker import to_sam_format


_SAMPLERS = {}


def _select_backend(backend: str | None = None) -> str:
    if backend is not None:
        return backend
    env_backend = os.environ.get("MASKVAR_CLICK_BACKEND")
    if env_backend:
        return env_backend
    return None


def _get_sampler(backend: str | None = None) -> ClickSampler:
    backend = _select_backend(backend)
    sampler = _SAMPLERS.get(backend)
    if sampler is None:
        sampler = ClickSampler(seed=random.randrange(2**31), backend=backend)
        _SAMPLERS[backend] = sampler
    return sampler


def _old_clicks_to_interaction(click_list):
    result = []
    for y, x, label in click_list:
        mode = CLK_POSITIVE if int(label) == 1 else CLK_NEGATIVE
        result.append((int(y), int(x), mode))
    return result


def _interaction_click_to_old(click):
    y, x, mode = click
    if y is None or x is None or mode is None:
        return None
    label = 1 if mode == CLK_POSITIVE else 0
    return int(y), int(x), label


def sample_batched_click_conditions(
    masks: torch.Tensor,
    out_h: int,
    out_w: int,
    max_clicks: int,
    backend=None,
    random_click_counts: bool = True,
    num_clicks: int | None = None,
):
    """
    Sample positive click conditions for a batch of masks.

    masks: (B, 1, H, W) or (B, H, W), preferably already on CUDA.
    Returns:
        click_coords: (B, max_clicks, 2), row/col in output grid coordinates
        click_labels: (B, max_clicks), 1 for positive clicks and -1 for padding
    """
    if masks.ndim == 4:
        mask_2d = masks[:, 0]
    elif masks.ndim == 3:
        mask_2d = masks
    else:
        raise ValueError(f"Expected masks with shape BHW or B1HW, got {tuple(masks.shape)}")

    b, h, w = mask_2d.shape
    if max_clicks < 1:
        raise ValueError("max_clicks must be >= 1")

    selected_backend = _select_backend(backend)
    if selected_backend == "cuda" or (selected_backend is None and mask_2d.is_cuda and torch.cuda.is_available()):
        gt = mask_2d > 0
        pred = torch.zeros_like(gt, dtype=torch.bool)
        ignore = torch.zeros_like(gt, dtype=torch.bool)
        coords = torch.zeros((b, max_clicks, 2), device=mask_2d.device, dtype=torch.float32)
        labels = torch.full((b, max_clicks), -1, device=mask_2d.device, dtype=torch.long)

        if num_clicks is not None:
            click_counts = torch.full(
                (b,),
                min(max(int(num_clicks), 0), max_clicks),
                device=mask_2d.device,
                dtype=torch.long,
            )
        elif random_click_counts:
            click_counts = torch.randint(1, max_clicks + 1, (b,), device=mask_2d.device)
        else:
            click_counts = torch.full((b,), max_clicks, device=mask_2d.device, dtype=torch.long)
        active = torch.ones((b,), device=mask_2d.device, dtype=torch.bool)

        for click_idx in range(max_clicks):
            active = torch.logical_and(active, click_idx < click_counts)
            if not bool(active.any().item()):
                break

            random_values = torch.randint(0, 2**31 - 1, (b,), device=mask_2d.device, dtype=torch.long)
            result, meta = cuda_sample_from_masks_batch(
                pred_mask=pred,
                gt_mask=gt,
                ignore_mask=ignore,
                sfc_inner_k=1.7,
                random_values=random_values,
            )
            del meta
            valid = torch.logical_and(active, result[:, 2] == 1)
            if not bool(valid.any().item()):
                break

            ys = result[:, 0].to(torch.long).clamp(0, h - 1)
            xs = result[:, 1].to(torch.long).clamp(0, w - 1)
            batch_idx = torch.arange(b, device=mask_2d.device)
            coords[:, click_idx, 0] = ys.to(torch.float32) * (float(out_h) / float(h))
            coords[:, click_idx, 1] = xs.to(torch.float32) * (float(out_w) / float(w))
            labels[valid, click_idx] = 1
            ignore[batch_idx[valid], ys[valid], xs[valid]] = True

        coords[..., 0].clamp_(min=0, max=out_h - 1)
        coords[..., 1].clamp_(min=0, max=out_w - 1)
        return coords, labels

    point_coords = []
    point_labels = []
    for mask in masks:
        if num_clicks is not None:
            sampled_clicks = min(max(int(num_clicks), 0), max_clicks)
        elif random_click_counts:
            sampled_clicks = int(np.random.randint(1, max_clicks + 1))
        else:
            sampled_clicks = max_clicks
        mask_np = mask[0].detach().cpu().numpy() > 0 if mask.ndim == 3 else mask.detach().cpu().numpy() > 0
        click_list, _, _ = init_clicks(
            mask_np,
            num_random_clicks=sampled_clicks,
            random_sample=True,
            backend=backend,
        )
        coords_xy, labels = to_sam_format(click_list, pad_size=max_clicks, device=mask_2d.device)
        click_coords = torch.empty_like(coords_xy, dtype=torch.float32)
        click_coords[..., 0] = coords_xy[..., 1] * (float(out_h) / float(h))
        click_coords[..., 1] = coords_xy[..., 0] * (float(out_w) / float(w))
        click_coords = click_coords.clamp_min(0)
        click_coords[..., 0].clamp_(max=out_h - 1)
        click_coords[..., 1].clamp_(max=out_w - 1)
        point_coords.append(click_coords)
        point_labels.append(labels.long())

    return torch.stack(point_coords), torch.stack(point_labels)


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


def init_clicks(gt_mask, num_random_clicks=1, not_clicked_map=None, random_sample=True, backend=None):
    """
    Init positive clicks for interactive segmentation.

    Interface matches maskvar.utils.clicker.init_clicks:
        returns (click_list, eroded_mask, dt)

    Uses maskvar.interaction.ClickSampler while preserving the old clicker API.
    Existing callers still receive [(y, x, label)], eroded_mask, and dt.
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
    dt = _distance_transform(eroded_mask).astype(np.float32, copy=False)

    try:
        sampler = _get_sampler(backend)
        prev_output = np.zeros_like(gt_mask, dtype=np.uint8)
        ignore_mask = np.logical_not(not_clicked_map)
        maps = sampler.compute_maps(gt_mask.astype(np.uint8), prev_output, ignore_mask=ignore_mask)
        dt = maps.negative_distance.astype(np.float32, copy=False)

        for _ in range(num_random_clicks):
            if not np.logical_and(gt_mask, not_clicked_map).any():
                break

            if random_sample:
                click = sampler.sample_click(
                    gt=gt_mask.astype(np.uint8),
                    prev_output=prev_output,
                    clicks=_old_clicks_to_interaction(click_list),
                    ignore_mask=np.logical_not(not_clicked_map),
                    sfc_inner_k=1.7,
                )
            else:
                maps = sampler.compute_maps(
                    gt=gt_mask.astype(np.uint8),
                    prev_output=prev_output,
                    clicks=_old_clicks_to_interaction(click_list),
                    ignore_mask=np.logical_not(not_clicked_map),
                )
                flat_idx = int(np.argmax(maps.negative_distance.reshape(-1)))
                if float(maps.negative_distance.reshape(-1)[flat_idx]) <= 0:
                    click = (None, None, None)
                else:
                    y, x = np.unravel_index(flat_idx, maps.negative_distance.shape)
                    click = (int(y), int(x), CLK_POSITIVE)

            old_click = _interaction_click_to_old(click)
            if old_click is None:
                break

            y, x, _ = old_click
            click_list.append(old_click)
            not_clicked_map[y, x] = False
    except Exception as e:
        traceback.print_exc()
        print(f"Error in clicker_v2.init_clicks: {e}")
        if _select_backend(backend) == "cuda":
            raise
    return click_list, eroded_mask, dt


def predict_next_click(gt_mask, pred_mask, click_list=[], not_clicked_map=None, backend=None):
    """
    Predict the next interaction click and update the old-style click list.

    gt_mask: 0/1 mask of shape (H, W)
    pred_mask: 0/1 prediction of shape (H, W)
    click_list: updated in-place with (y, x, label), label 1 positive / 0 negative
    not_clicked_map: True means available for future clicks, updated in-place
    """
    assert gt_mask is not None, "Ground truth mask not given."
    assert pred_mask.ndim == 2

    if not_clicked_map is None:
        not_clicked_map = np.ones_like(gt_mask, dtype=bool)

    try:
        sampler = _get_sampler(backend)
        click = sampler.sample_click(
            gt=np.asarray(gt_mask).astype(np.uint8),
            prev_output=np.asarray(pred_mask).astype(np.uint8),
            clicks=_old_clicks_to_interaction(click_list),
            ignore_mask=np.logical_not(not_clicked_map),
            sfc_inner_k=1.7,
        )
        old_click = _interaction_click_to_old(click)
        if old_click is None:
            return (None, None, None), click_list, not_clicked_map
        y, x, _ = old_click
        click_list.append(old_click)
        not_clicked_map[y, x] = False
        return old_click, click_list, not_clicked_map
    except Exception as e:
        traceback.print_exc()
        print(f"Error in clicker_v2.predict_next_click: {e}")
        if _select_backend(backend) == "cuda":
            raise
        return (None, None, None), click_list, not_clicked_map
