"""Benchmark click sampling backends.

Run from the repository root on a GPU node:

    conda run -n var_v2 python -m notebooks.benchmark_interaction_clicker \
      --backend all --num_masks 512 --iters 4096 --warmup 128

The first run compiles torch extensions. Warmup is excluded from timing.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/tmp/maskvar_torch_extensions")

import numpy as np
import torch

from maskvar.interaction import ClickSampler
from maskvar.interaction.backend import cuda_sample_from_masks, cuda_sample_from_masks_batch
from maskvar.utils.clicker_v2 import sample_batched_click_conditions
from maskvar.utils import clicker as old_clicker


@dataclass
class Case:
    gt: np.ndarray
    pred: np.ndarray
    ignore: np.ndarray


def make_cases(num_masks: int, height: int, width: int, seed: int) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(num_masks):
        gt = np.zeros((height, width), dtype=np.uint8)
        pred = np.zeros((height, width), dtype=np.uint8)

        for _ in range(int(rng.integers(1, 4))):
            rh = int(rng.integers(max(4, height // 10), max(5, height // 3)))
            rw = int(rng.integers(max(4, width // 10), max(5, width // 3)))
            y0 = int(rng.integers(0, max(1, height - rh)))
            x0 = int(rng.integers(0, max(1, width - rw)))
            gt[y0 : y0 + rh, x0 : x0 + rw] = 1

        pred[:] = gt

        # Add a false-negative chunk.
        ys, xs = np.where(gt > 0)
        if len(ys) > 0:
            center = int(rng.integers(0, len(ys)))
            cy, cx = int(ys[center]), int(xs[center])
            r = int(rng.integers(3, max(4, min(height, width) // 10)))
            pred[max(0, cy - r) : min(height, cy + r), max(0, cx - r) : min(width, cx + r)] = 0

        # Add a false-positive chunk.
        rh = int(rng.integers(max(4, height // 12), max(5, height // 4)))
        rw = int(rng.integers(max(4, width // 12), max(5, width // 4)))
        y0 = int(rng.integers(0, max(1, height - rh)))
        x0 = int(rng.integers(0, max(1, width - rw)))
        pred[y0 : y0 + rh, x0 : x0 + rw] = 1

        ignore = np.zeros((height, width), dtype=bool)
        cases.append(Case(gt=gt, pred=pred, ignore=ignore))
    return cases


def bench(name: str, fn, iters: int, warmup: int, sync_cuda: bool = False):
    for i in range(warmup):
        fn(i)
    if sync_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for i in range(iters):
        fn(i)
    if sync_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    us_per_click = elapsed * 1e6 / iters
    clicks_per_sec = iters / elapsed
    return {
        "name": name,
        "elapsed": elapsed,
        "us_per_click": us_per_click,
        "clicks_per_sec": clicks_per_sec,
    }


def print_results(results):
    baseline = results[0]["us_per_click"] if results else None
    print(f"{'backend':32s} {'us/click':>12s} {'click/s':>12s} {'speedup':>10s}")
    print("-" * 72)
    for result in results:
        speedup = baseline / result["us_per_click"] if baseline else 1.0
        print(
            f"{result['name']:32s} "
            f"{result['us_per_click']:12.2f} "
            f"{result['clicks_per_sec']:12.1f} "
            f"{speedup:10.2f}x"
        )


def run_old_cv2(cases, iters, warmup):
    def one(i):
        case = cases[i % len(cases)]
        old_clicker.predict_next_click(
            gt_mask=case.gt > 0,
            pred_mask=case.pred > 0,
            click_list=[],
            not_clicked_map=np.ones_like(case.gt, dtype=bool),
        )

    return bench("old_cv2_predict_next_click", one, iters, warmup)


def run_interaction_sampler(cases, backend: str, iters: int, warmup: int):
    sampler = ClickSampler(seed=123, backend=backend)

    def one(i):
        case = cases[i % len(cases)]
        sampler.sample_click(
            gt=case.gt,
            prev_output=case.pred,
            clicks=[],
            ignore_mask=case.ignore,
        )

    return bench(
        f"interaction_{backend}_numpy_api",
        one,
        iters,
        warmup,
        sync_cuda=(backend == "cuda"),
    )


def run_cuda_tensor_api(cases, iters: int, warmup: int):
    sampler = ClickSampler(seed=123, backend="cuda")
    device_cases = [
        (
            torch.as_tensor(case.gt, device="cuda", dtype=torch.uint8),
            torch.as_tensor(case.pred, device="cuda", dtype=torch.uint8),
            torch.as_tensor(case.ignore, device="cuda", dtype=torch.bool),
        )
        for case in cases
    ]

    def one(i):
        gt, pred, ignore = device_cases[i % len(device_cases)]
        sampler.sample_click(gt=gt, prev_output=pred, clicks=[], ignore_mask=ignore)

    return bench("interaction_cuda_tensor_api", one, iters, warmup, sync_cuda=True)


def run_cuda_lowlevel(cases, iters: int, warmup: int):
    device_cases = [
        (
            torch.as_tensor(case.pred > 0, device="cuda", dtype=torch.bool),
            torch.as_tensor(case.gt == 1, device="cuda", dtype=torch.bool),
            torch.as_tensor(case.ignore, device="cuda", dtype=torch.bool),
        )
        for case in cases
    ]

    def one(i):
        pred, gt, ignore = device_cases[i % len(device_cases)]
        cuda_sample_from_masks(
            pred_mask=pred,
            gt_mask=gt,
            ignore_mask=ignore,
            sfc_inner_k=1.7,
            random_value=i,
            need_debug=False,
        )

    return bench("cuda_lowlevel_op_tensor", one, iters, warmup, sync_cuda=True)


def run_cuda_batch_lowlevel(cases, iters: int, warmup: int, batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    gt = torch.stack([
        torch.as_tensor(case.gt == 1, device="cuda", dtype=torch.bool)
        for case in cases
    ])
    pred = torch.stack([
        torch.as_tensor(case.pred > 0, device="cuda", dtype=torch.bool)
        for case in cases
    ])
    ignore = torch.stack([
        torch.as_tensor(case.ignore, device="cuda", dtype=torch.bool)
        for case in cases
    ])
    batch_size = min(batch_size, gt.shape[0])
    random_values = torch.arange(batch_size, dtype=torch.long)

    def one(i):
        start = (i * batch_size) % gt.shape[0]
        if start + batch_size <= gt.shape[0]:
            pred_b = pred[start : start + batch_size]
            gt_b = gt[start : start + batch_size]
            ignore_b = ignore[start : start + batch_size]
        else:
            indices = (torch.arange(batch_size, device=gt.device) + start) % gt.shape[0]
            pred_b = pred.index_select(0, indices)
            gt_b = gt.index_select(0, indices)
            ignore_b = ignore.index_select(0, indices)
        cuda_sample_from_masks_batch(
            pred_mask=pred_b,
            gt_mask=gt_b,
            ignore_mask=ignore_b,
            sfc_inner_k=1.7,
            random_values=random_values + i,
        )

    measured_batches = max(1, iters // batch_size)
    warmup_batches = max(1, warmup // batch_size)
    result = bench(
        f"cuda_batch_lowlevel_b{batch_size}",
        one,
        measured_batches,
        warmup_batches,
        sync_cuda=True,
    )
    result["us_per_click"] = result["elapsed"] * 1e6 / (measured_batches * batch_size)
    result["clicks_per_sec"] = (measured_batches * batch_size) / result["elapsed"]
    return result


def run_training_click_condition(
    cases,
    iters: int,
    warmup: int,
    batch_size: int,
    max_clicks: int,
    random_click_counts: bool,
):
    masks = torch.stack([
        torch.as_tensor(case.gt, device="cuda", dtype=torch.uint8).unsqueeze(0)
        for case in cases
    ])
    batch_size = min(batch_size, masks.shape[0])

    def one(i):
        start = (i * batch_size) % masks.shape[0]
        if start + batch_size <= masks.shape[0]:
            mask_b = masks[start : start + batch_size]
        else:
            indices = (torch.arange(batch_size, device=masks.device) + start) % masks.shape[0]
            mask_b = masks.index_select(0, indices)
        sample_batched_click_conditions(
            mask_b,
            out_h=64,
            out_w=64,
            max_clicks=max_clicks,
            backend="cuda",
            random_click_counts=random_click_counts,
        )

    measured_batches = max(1, iters // batch_size)
    warmup_batches = max(1, warmup // batch_size)
    result = bench(
        f"train_click_condition_b{batch_size}_m{max_clicks}",
        one,
        measured_batches,
        warmup_batches,
        sync_cuda=True,
    )
    sampled_clicks_per_sample = (max_clicks + 1) / 2.0 if random_click_counts else float(max_clicks)
    result["us_per_sample"] = result["elapsed"] * 1e6 / (measured_batches * batch_size)
    result["us_per_click"] = result["elapsed"] * 1e6 / (measured_batches * batch_size * sampled_clicks_per_sample)
    result["clicks_per_sec"] = (measured_batches * batch_size * sampled_clicks_per_sample) / result["elapsed"]
    return result


def burn_cuda_ms(target_ms: float, state: dict):
    if target_ms <= 0:
        return
    device = torch.device("cuda")
    if "mat" not in state:
        state["mat"] = torch.randn((512, 512), device=device)
    mat = state["mat"]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    elapsed = 0.0
    while elapsed < target_ms:
        start.record()
        for _ in range(8):
            mat = mat @ mat
        end.record()
        torch.cuda.synchronize()
        elapsed += start.elapsed_time(end)
    state["mat"] = mat


def measure_cuda_step(fn, iters: int, warmup: int):
    for i in range(warmup):
        fn(i)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for i in range(iters):
        fn(i)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / iters


def simulate_training_step(cases, batch_size: int, max_clicks: int, model_ms: float, iters: int, warmup: int):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA simulation requires torch.cuda.is_available().")

    masks_gpu = torch.stack([
        torch.as_tensor(case.gt, device="cuda", dtype=torch.uint8).unsqueeze(0)
        for case in cases
    ])
    batch_size = min(batch_size, masks_gpu.shape[0])
    cpu_cases = cases
    burn_state = {}

    def mask_batch(i):
        start = (i * batch_size) % masks_gpu.shape[0]
        if start + batch_size <= masks_gpu.shape[0]:
            return masks_gpu[start : start + batch_size]
        indices = (torch.arange(batch_size, device=masks_gpu.device) + start) % masks_gpu.shape[0]
        return masks_gpu.index_select(0, indices)

    def old_click_step(i):
        burn_cuda_ms(model_ms, burn_state)
        base = (i * batch_size) % len(cpu_cases)
        for j in range(batch_size):
            case = cpu_cases[(base + j) % len(cpu_cases)]
            for _ in range(max_clicks):
                old_clicker.predict_next_click(
                    gt_mask=case.gt > 0,
                    pred_mask=case.pred > 0,
                    click_list=[],
                    not_clicked_map=np.ones_like(case.gt, dtype=bool),
                )

    def new_click_step(i):
        burn_cuda_ms(model_ms, burn_state)
        sample_batched_click_conditions(
            mask_batch(i),
            out_h=64,
            out_w=64,
            max_clicks=max_clicks,
            backend="cuda",
            random_click_counts=False,
        )

    def model_only_step(_):
        burn_cuda_ms(model_ms, burn_state)

    model_only = measure_cuda_step(model_only_step, iters, warmup)
    old_total = measure_cuda_step(old_click_step, iters, warmup)
    new_total = measure_cuda_step(new_click_step, iters, warmup)
    return {
        "model_only_ms": model_only,
        "old_total_ms": old_total,
        "new_total_ms": new_total,
        "old_click_ms": max(0.0, old_total - model_only),
        "new_click_ms": max(0.0, new_total - model_only),
        "speedup": old_total / new_total if new_total > 0 else float("inf"),
    }


def print_simulation(result, batch_size: int, max_clicks: int, model_ms: float):
    old_click_share = result["old_click_ms"] / result["old_total_ms"] * 100.0 if result["old_total_ms"] else 0.0
    new_click_share = result["new_click_ms"] / result["new_total_ms"] * 100.0 if result["new_total_ms"] else 0.0
    print(f"simulated training step: batch={batch_size} max_clicks={max_clicks} target_model_ms={model_ms:.2f}")
    print(f"model_only_ms : {result['model_only_ms']:.2f}")
    print(f"old_total_ms  : {result['old_total_ms']:.2f}  click_ms={result['old_click_ms']:.2f}  share={old_click_share:.1f}%")
    print(f"new_total_ms  : {result['new_total_ms']:.2f}  click_ms={result['new_click_ms']:.2f}  share={new_click_share:.1f}%")
    print(f"end_to_end_speedup: {result['speedup']:.2f}x")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="all", choices=["all", "cpu", "cuda"])
    parser.add_argument("--num_masks", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--iters", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_clicks", type=int, default=2)
    parser.add_argument("--fixed_clicks", action="store_true")
    parser.add_argument("--simulate_step", action="store_true")
    parser.add_argument("--model_ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    cases = make_cases(args.num_masks, args.height, args.width, args.seed)
    if args.simulate_step:
        result = simulate_training_step(
            cases,
            batch_size=args.batch_size,
            max_clicks=args.max_clicks,
            model_ms=args.model_ms,
            iters=args.iters,
            warmup=args.warmup,
        )
        print_simulation(result, args.batch_size, args.max_clicks, args.model_ms)
        return

    results = [run_old_cv2(cases, args.iters, args.warmup)]

    if args.backend in {"all", "cpu"}:
        results.append(run_interaction_sampler(cases, "cpu", args.iters, args.warmup))

    if args.backend in {"all", "cuda"}:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA backend requested, but torch.cuda.is_available() is False.")
        results.append(run_interaction_sampler(cases, "cuda", args.iters, args.warmup))
        results.append(run_cuda_tensor_api(cases, args.iters, args.warmup))
        results.append(run_cuda_lowlevel(cases, args.iters, args.warmup))
        results.append(run_cuda_batch_lowlevel(cases, args.iters, args.warmup, args.batch_size))
        results.append(
            run_training_click_condition(
                cases,
                args.iters,
                args.warmup,
                args.batch_size,
                args.max_clicks,
                random_click_counts=not args.fixed_clicks,
            )
        )

    print(
        f"masks={args.num_masks} shape={args.height}x{args.width} "
        f"iters={args.iters} warmup={args.warmup}"
    )
    print_results(results)


if __name__ == "__main__":
    main()
