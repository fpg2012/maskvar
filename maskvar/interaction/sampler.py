"""Unified interaction sampler."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .backend import (
    cpu_distance_pair,
    cuda_sample_from_masks,
)
from .types import CLK_NEGATIVE, CLK_POSITIVE, Click, ClickMaps


def _to_bool_numpy(mask: Any) -> np.ndarray:
    """Convert one mask-like input to one 2D bool NumPy array."""

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2D mask, but got shape {array.shape}.")
    return (array == 1) if array.dtype != np.bool_ else array


def _mask_shape(mask: Any) -> tuple[int, int]:
    """Return one 2D shape from one mask-like input."""

    if isinstance(mask, torch.Tensor):
        shape = tuple(int(v) for v in mask.shape)
    else:
        shape = tuple(int(v) for v in np.asarray(mask).shape)
    if len(shape) == 3 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) != 2:
        raise ValueError(f"Expected 2D mask, but got shape {shape}.")
    return shape


def _build_ignore_mask(
    shape: tuple[int, int],
    clicks: list[Click],
    ignore_mask: Any | None,
) -> np.ndarray:
    """Build one CPU ignore mask."""

    if ignore_mask is None:
        result = np.zeros(shape, dtype=np.bool_)
    else:
        result = _to_bool_numpy(ignore_mask).copy()
        if tuple(result.shape) != shape:
            raise ValueError("ignore_mask must match the mask shape.")
    for y, x, _ in clicks:
        if y is None or x is None:
            continue
        result[int(y), int(x)] = True
    return result


def _build_cuda_ignore_mask(
    shape: tuple[int, int],
    clicks: list[Click],
    ignore_mask: Any | None,
    device: torch.device,
) -> torch.Tensor:
    """Build one CUDA ignore mask."""

    if ignore_mask is None:
        result = torch.zeros(shape, device=device, dtype=torch.bool)
    elif isinstance(ignore_mask, torch.Tensor):
        result = ignore_mask.detach().to(device=device, dtype=torch.bool).contiguous().clone()
    else:
        result = torch.as_tensor(ignore_mask, device=device, dtype=torch.bool).contiguous().clone()
    if tuple(result.shape) != shape:
        raise ValueError("ignore_mask must match the mask shape.")
    if clicks:
        ys = [int(y) for y, x, _ in clicks if y is not None and x is not None]
        xs = [int(x) for y, x, _ in clicks if y is not None and x is not None]
        if ys:
            result[torch.tensor(ys, device=device), torch.tensor(xs, device=device)] = True
    return result


def _cpu_debug_from_pair(
    false_negative: np.ndarray,
    false_positive: np.ndarray,
    negative_dist: np.ndarray,
    positive_dist: np.ndarray,
) -> ClickMaps:
    """Build one CPU debug payload without candidate sampling."""

    return ClickMaps(
        false_negative=false_negative.astype(np.uint8),
        false_positive=false_positive.astype(np.uint8),
        negative_distance=negative_dist,
        positive_distance=positive_dist,
        candidate_mask=np.zeros_like(false_negative, dtype=np.uint8),
        negative_max=float(negative_dist.max()),
        positive_max=float(positive_dist.max()),
        threshold=0.0,
        mode=None,
    )


def _payload_to_click(payload: list[torch.Tensor]) -> Click:
    """Convert one CUDA payload to one click tuple."""

    result = payload[0]
    meta = payload[1]
    y = int(result[0].item())
    x = int(result[1].item())
    if y < 0 or x < 0 or (float(meta[0].item()) == 0.0 and float(meta[1].item()) == 0.0):
        return (None, None, None)
    return (y, x, CLK_POSITIVE if int(result[2].item()) == 1 else CLK_NEGATIVE)


def _payload_to_maps(payload: list[torch.Tensor], with_candidate: bool) -> ClickMaps:
    """Convert one CUDA payload to one click-map payload."""

    meta = payload[1]
    mode = None
    if with_candidate and (float(meta[0].item()) > 0.0 or float(meta[1].item()) > 0.0):
        mode = CLK_POSITIVE if int(meta[2].item()) == 1 else CLK_NEGATIVE
    return ClickMaps(
        false_negative=payload[5].cpu().numpy(),
        false_positive=payload[6].cpu().numpy(),
        negative_distance=payload[2].cpu().numpy(),
        positive_distance=payload[3].cpu().numpy(),
        candidate_mask=payload[4].cpu().numpy() if with_candidate else np.zeros_like(payload[5].cpu().numpy()),
        negative_max=float(meta[0].item()),
        positive_max=float(meta[1].item()),
        threshold=float(meta[3].item()) if with_candidate else 0.0,
        mode=mode,
    )


def _sample_cpu_click(
    negative_dist: np.ndarray,
    positive_dist: np.ndarray,
    rng: random.Random,
    sfc_inner_k: float,
) -> tuple[Click, np.ndarray, float, str | None]:
    """Sample one CPU click from distance maps."""

    negative_max = float(negative_dist.max())
    positive_max = float(positive_dist.max())
    if negative_max == 0.0 and positive_max == 0.0:
        return (None, None, None), np.zeros_like(negative_dist, dtype=np.uint8), 0.0, None

    if sfc_inner_k < 0.0:
        dist_scale = 0.0
    elif sfc_inner_k >= 1.0:
        dist_scale = 1.0 / (sfc_inner_k + np.finfo(np.float32).eps)
    else:
        raise ValueError("sfc_inner_k must be >= 1.0 or < 0.0.")

    if negative_max > positive_max:
        mode = CLK_POSITIVE
        threshold = float(dist_scale * negative_max)
        candidates = np.argwhere(negative_dist > threshold)
    else:
        mode = CLK_NEGATIVE
        threshold = float(dist_scale * positive_max)
        candidates = np.argwhere(positive_dist > threshold)

    candidate_mask = np.zeros_like(negative_dist, dtype=np.uint8)
    if len(candidates) == 0:
        return (None, None, None), candidate_mask, threshold, mode
    candidate_mask[candidates[:, 0], candidates[:, 1]] = 1
    index = rng.randrange(len(candidates))
    y, x = candidates[index].tolist()
    return (int(y), int(x), mode), candidate_mask, threshold, mode


@dataclass
class ClickSampler:
    """Unified click sampler with CPU and CUDA backends."""

    seed: int = 42
    backend: str | None = None

    def __post_init__(self) -> None:
        """Initialize RNG state."""

        self.reset(self.seed)

    def reset(self, seed: int | None = None) -> None:
        """Reset sampler RNG state."""

        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)
        self.cuda_rng = None
        if torch.cuda.is_available():
            self.cuda_rng = torch.Generator(device="cuda")
            self.cuda_rng.manual_seed(self.seed)

    def select_backend(self) -> str:
        """Return the active backend name."""

        if self.backend is not None:
            return self.backend
        return "cuda" if torch.cuda.is_available() else "cpu"

    def compute_maps(
        self,
        gt: Any,
        prev_output: Any,
        clicks: list[Click] | None = None,
        ignore_mask: Any | None = None,
    ) -> ClickMaps:
        """Compute one error-map payload without sampling."""

        click_list = [] if clicks is None else list(clicks)
        backend = self.select_backend()
        if backend == "cuda":
            pred_tensor = torch.as_tensor(prev_output, device="cuda")
            gt_tensor = torch.as_tensor(gt, device="cuda")
            if pred_tensor.ndim == 3 and pred_tensor.shape[0] == 1:
                pred_tensor = pred_tensor[0]
            if gt_tensor.ndim == 3 and gt_tensor.shape[0] == 1:
                gt_tensor = gt_tensor[0]
            ignore = _build_cuda_ignore_mask(
                shape=_mask_shape(pred_tensor),
                clicks=click_list,
                ignore_mask=ignore_mask,
                device=pred_tensor.device,
            )
            payload = cuda_sample_from_masks(
                pred_mask=(pred_tensor > 0),
                gt_mask=(gt_tensor == 1),
                ignore_mask=ignore,
                sfc_inner_k=-1.0,
                random_value=0,
                need_debug=True,
            )
            return _payload_to_maps(payload, with_candidate=False)

        pred_mask = _to_bool_numpy(np.asarray(prev_output) > 0)
        gt_mask = _to_bool_numpy(np.asarray(gt) == 1)
        ignore = _build_ignore_mask(pred_mask.shape, click_list, ignore_mask)
        false_negative = np.logical_and(np.logical_not(pred_mask), gt_mask)
        false_positive = np.logical_and(pred_mask, np.logical_not(gt_mask))
        false_negative = np.logical_and(false_negative, np.logical_not(ignore))
        false_positive = np.logical_and(false_positive, np.logical_not(ignore))
        negative_dist, positive_dist = cpu_distance_pair(false_negative, false_positive)
        return _cpu_debug_from_pair(
            false_negative,
            false_positive,
            negative_dist.numpy(),
            positive_dist.numpy(),
        )

    def sample_click(
        self,
        gt: Any,
        prev_output: Any,
        clicks: list[Click] | None = None,
        ignore_mask: Any | None = None,
        sfc_inner_k: float = 1.7,
        return_maps: bool = False,
    ) -> Click | tuple[Click, ClickMaps]:
        """Sample one click."""

        click_list = [] if clicks is None else list(clicks)
        backend = self.select_backend()
        if backend == "cuda":
            pred_tensor = torch.as_tensor(prev_output, device="cuda")
            gt_tensor = torch.as_tensor(gt, device="cuda")
            if pred_tensor.ndim == 3 and pred_tensor.shape[0] == 1:
                pred_tensor = pred_tensor[0]
            if gt_tensor.ndim == 3 and gt_tensor.shape[0] == 1:
                gt_tensor = gt_tensor[0]
            ignore = _build_cuda_ignore_mask(
                shape=_mask_shape(pred_tensor),
                clicks=click_list,
                ignore_mask=ignore_mask,
                device=pred_tensor.device,
            )
            payload = cuda_sample_from_masks(
                pred_mask=(pred_tensor > 0),
                gt_mask=(gt_tensor == 1),
                ignore_mask=ignore,
                sfc_inner_k=sfc_inner_k,
                random_value=int(torch.randint(0, 2**31 - 1, (1,), generator=self.cuda_rng, device="cuda").item()),
                need_debug=return_maps,
            )
            click = _payload_to_click(payload)
            if not return_maps:
                return click
            return click, _payload_to_maps(payload, with_candidate=True)

        maps = self.compute_maps(gt, prev_output, click_list, ignore_mask)
        click, candidate_mask, threshold, mode = _sample_cpu_click(
            maps.negative_distance,
            maps.positive_distance,
            self.rng,
            sfc_inner_k,
        )
        if not return_maps:
            return click
        maps.candidate_mask = candidate_mask
        maps.threshold = threshold
        maps.mode = mode
        return click, maps

    def sample_clicks(
        self,
        gt: Any,
        prev_output: Any,
        num_clicks: int,
        clicks: list[Click] | None = None,
        ignore_mask: Any | None = None,
        sfc_inner_k: float = 1.7,
    ) -> list[Click]:
        """Sample multiple clicks."""

        if num_clicks < 0:
            raise ValueError("num_clicks must be non-negative.")
        if num_clicks == 0:
            return []

        click_list = [] if clicks is None else list(clicks)
        backend = self.select_backend()
        if backend == "cuda":
            pred_tensor = torch.as_tensor(prev_output, device="cuda")
            gt_tensor = torch.as_tensor(gt, device="cuda")
            if pred_tensor.ndim == 3 and pred_tensor.shape[0] == 1:
                pred_tensor = pred_tensor[0]
            if gt_tensor.ndim == 3 and gt_tensor.shape[0] == 1:
                gt_tensor = gt_tensor[0]
            ignore = _build_cuda_ignore_mask(
                shape=_mask_shape(pred_tensor),
                clicks=click_list,
                ignore_mask=ignore_mask,
                device=pred_tensor.device,
            )
            payload = cuda_sample_from_masks(
                pred_mask=(pred_tensor > 0),
                gt_mask=(gt_tensor == 1),
                ignore_mask=ignore,
                sfc_inner_k=sfc_inner_k,
                random_value=int(torch.randint(0, 2**31 - 1, (1,), generator=self.cuda_rng, device="cuda").item()),
                need_debug=True,
            )
            maps = _payload_to_maps(payload, with_candidate=True)
            coords = np.argwhere(maps.candidate_mask > 0)
            mode = maps.mode
            if mode is None or len(coords) == 0:
                return [(None, None, None) for _ in range(num_clicks)]
            order = torch.randperm(len(coords), generator=self.cuda_rng).cpu().numpy()
            coords = coords[order[:min(num_clicks, len(coords))]]
            result = [(int(y), int(x), mode) for y, x in coords.tolist()]
            if len(result) < num_clicks:
                result.extend([(None, None, None) for _ in range(num_clicks - len(result))])
            return result

        maps = self.compute_maps(gt, prev_output, click_list, ignore_mask)
        _, candidate_mask, _, mode = _sample_cpu_click(
            maps.negative_distance,
            maps.positive_distance,
            self.rng,
            sfc_inner_k,
        )
        coords = np.argwhere(candidate_mask > 0)
        if mode is None or len(coords) == 0:
            return [(None, None, None) for _ in range(num_clicks)]
        coords_list = list(map(tuple, coords.tolist()))
        self.rng.shuffle(coords_list)
        result = [(int(y), int(x), mode) for y, x in coords_list[:num_clicks]]
        if len(result) < num_clicks:
            result.extend([(None, None, None) for _ in range(num_clicks - len(result))])
        return result
