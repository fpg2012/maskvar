"""Smoke test for the maskvar.interaction click sampler integration.

Run from the repository root:

    conda run -n var_v2 python -m notebooks.test_interaction_clicker_v2 --backend cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/maskvar_torch_extensions")

import numpy as np
import torch

from maskvar.interaction import CLK_NEGATIVE, CLK_POSITIVE, ClickSampler
from maskvar.utils.clicker_v2 import init_clicks, predict_next_click, to_sam_format


def make_mask() -> np.ndarray:
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[8:28, 10:34] = 1
    mask[32:42, 45:58] = 1
    return mask


def assert_inside(mask: np.ndarray, click) -> None:
    y, x, label = click
    assert label == 1, f"expected positive init click, got {click}"
    assert mask[y, x] == 1, f"init click {click} is outside the GT mask"


def test_init_clicks(mask: np.ndarray, backend: str, num_clicks: int) -> None:
    not_clicked = np.ones_like(mask, dtype=bool)
    clicks, eroded_mask, dt = init_clicks(
        mask,
        num_random_clicks=num_clicks,
        not_clicked_map=not_clicked,
        random_sample=True,
        backend=backend,
    )
    assert len(clicks) == num_clicks, f"expected {num_clicks} init clicks, got {clicks}"
    assert eroded_mask.shape == mask.shape
    assert dt.shape == mask.shape
    assert len({(y, x) for y, x, _ in clicks}) == len(clicks), f"duplicate clicks: {clicks}"
    for click in clicks:
        assert_inside(mask, click)
        y, x, _ = click
        assert not not_clicked[y, x], "not_clicked_map was not updated"

    coords, labels = to_sam_format(clicks, pad_size=num_clicks + 2)
    assert tuple(coords.shape) == (num_clicks + 2, 2)
    assert tuple(labels.shape) == (num_clicks + 2,)
    assert labels[:num_clicks].tolist() == [1] * num_clicks
    assert labels[num_clicks:].tolist() == [-1, -1]


def test_predict_next_click(mask: np.ndarray, backend: str) -> None:
    not_clicked = np.ones_like(mask, dtype=bool)
    clicks = []

    pred_empty = np.zeros_like(mask, dtype=np.uint8)
    pos_click, clicks, not_clicked = predict_next_click(
        mask,
        pred_empty,
        click_list=clicks,
        not_clicked_map=not_clicked,
        backend=backend,
    )
    assert pos_click[2] == 1, f"empty prediction should produce positive click, got {pos_click}"
    assert mask[pos_click[0], pos_click[1]] == 1

    pred_fp = mask.copy()
    pred_fp[0:12, 50:63] = 1
    neg_click, clicks, not_clicked = predict_next_click(
        mask,
        pred_fp,
        click_list=clicks,
        not_clicked_map=not_clicked,
        backend=backend,
    )
    assert neg_click[2] == 0, f"false-positive region should produce negative click, got {neg_click}"
    assert mask[neg_click[0], neg_click[1]] == 0


def test_direct_sampler(mask: np.ndarray, backend: str, seed: int) -> None:
    sampler = ClickSampler(seed=seed, backend=backend)
    pred = np.zeros_like(mask, dtype=np.uint8)
    click, maps = sampler.sample_click(mask, pred, return_maps=True)
    y, x, mode = click
    assert mode == CLK_POSITIVE, f"expected positive direct sampler click, got {click}"
    assert mask[y, x] == 1
    assert maps.mode == CLK_POSITIVE
    assert maps.negative_max > 0

    pred_all = np.ones_like(mask, dtype=np.uint8)
    click = sampler.sample_click(mask, pred_all)
    y, x, mode = click
    assert mode == CLK_NEGATIVE, f"expected negative direct sampler click, got {click}"
    assert mask[y, x] == 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--num_clicks", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA backend requested, but torch.cuda.is_available() is False.")

    np.random.seed(args.seed)
    mask = make_mask()

    test_init_clicks(mask, args.backend, args.num_clicks)
    test_predict_next_click(mask, args.backend)
    test_direct_sampler(mask, args.backend, args.seed)

    print(
        f"interaction clicker_v2 smoke test passed "
        f"(backend={args.backend}, num_clicks={args.num_clicks}, cwd={Path.cwd()})"
    )


if __name__ == "__main__":
    main()
