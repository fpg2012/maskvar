"""Backend bindings for interaction sampling."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


_CPU_EXTENSION = None
_CUDA_EXTENSION = None


def _root() -> Path:
    """Return the interaction package root."""

    return Path(__file__).resolve().parent


def _as_cpu_mask_tensor(mask: Any) -> torch.Tensor:
    """Convert one mask-like input to one contiguous CPU uint8 mask tensor."""

    if isinstance(mask, torch.Tensor):
        return mask.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(mask, dtype=torch.uint8).contiguous()


def _as_cuda_bool(mask: Any) -> torch.Tensor:
    """Convert one mask-like input to one contiguous CUDA bool tensor."""

    if isinstance(mask, torch.Tensor):
        return mask.detach().to(device="cuda", dtype=torch.bool).contiguous()
    return torch.as_tensor(mask, device="cuda", dtype=torch.bool).contiguous()


def get_cpu_backend():
    """Load the CPU extension once."""

    global _CPU_EXTENSION
    if _CPU_EXTENSION is not None:
        return _CPU_EXTENSION

    try:
        _CPU_EXTENSION = importlib.import_module("maskvar.interaction._cpu_ext")
        return _CPU_EXTENSION
    except ImportError:
        pass

    source = _root() / "csrc" / "cpu_distance.cpp"
    _CPU_EXTENSION = load(
        name="ifv3_interaction_cpu_v4",
        sources=[str(source)],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _CPU_EXTENSION


def get_cuda_backend():
    """Load the CUDA extension once."""

    global _CUDA_EXTENSION
    if _CUDA_EXTENSION is not None:
        return _CUDA_EXTENSION

    try:
        ext = importlib.import_module("maskvar.interaction._cuda_ext")
        if hasattr(ext, "sample_from_masks_batch"):
            _CUDA_EXTENSION = ext
            return _CUDA_EXTENSION
    except ImportError:
        pass

    root = _root() / "csrc"
    _CUDA_EXTENSION = load(
        name="ifv3_interaction_cuda_v2",
        sources=[
            str(root / "cuda_distance.cpp"),
            str(root / "cuda_distance_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    return _CUDA_EXTENSION


def cpu_distance_pair(
    negative_mask: Any,
    positive_mask: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one CPU distance pair."""

    negative = _as_cpu_mask_tensor(negative_mask)
    positive = _as_cpu_mask_tensor(positive_mask)
    return get_cpu_backend().distance_pair(negative, positive, True)


def cuda_sample_from_masks(
    pred_mask: Any,
    gt_mask: Any,
    ignore_mask: Any,
    sfc_inner_k: float,
    random_value: int,
    need_debug: bool,
) -> list[torch.Tensor]:
    """Sample one click from CUDA masks."""

    pred = _as_cuda_bool(pred_mask)
    gt = _as_cuda_bool(gt_mask)
    ignore = _as_cuda_bool(ignore_mask)
    return get_cuda_backend().sample_from_masks(
        pred,
        gt,
        ignore,
        float(sfc_inner_k),
        int(random_value),
        bool(need_debug),
    )


def cuda_sample_from_masks_batch(
    pred_mask: Any,
    gt_mask: Any,
    ignore_mask: Any,
    sfc_inner_k: float,
    random_values: Any,
) -> list[torch.Tensor]:
    """Sample one click per mask from BHW CUDA masks."""

    pred = _as_cuda_bool(pred_mask)
    gt = _as_cuda_bool(gt_mask)
    ignore = _as_cuda_bool(ignore_mask)
    if isinstance(random_values, torch.Tensor):
        random_tensor = random_values.detach().to(dtype=torch.long).contiguous()
    else:
        random_tensor = torch.as_tensor(random_values, dtype=torch.long)
    return get_cuda_backend().sample_from_masks_batch(
        pred,
        gt,
        ignore,
        float(sfc_inner_k),
        random_tensor,
    )
