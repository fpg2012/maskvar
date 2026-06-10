"""
Analyze large interactive-segmentation eval JSON files without loading the
per-sample payload all at once.

Example:
    python misc_scripts/analyze_interactive_eval.py \
        out/interactive_eval_coconut_val/interactive_eval_coconut_hf_val_20260605_123718.json \
        --out out/interactive_eval_coconut_val/report.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


MODEL_RE = re.compile(r'^    "([^"]+)": \{$')


def parse_header(path: Path) -> dict:
    lines = []
    with path.open() as f:
        for line in f:
            if line.strip() == '"results": {':
                break
            lines.append(line)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].rstrip().endswith(","):
        lines[-1] = lines[-1].rstrip().rstrip(",") + "\n"
    return json.loads("".join(lines) + "}\n")


def iter_sample_records(path: Path):
    current_model = None
    in_samples = False
    acc = []
    brace_depth = 0

    with path.open() as f:
        for line in f:
            match = MODEL_RE.match(line.rstrip("\n"))
            if match and not in_samples:
                current_model = match.group(1)
                continue

            if current_model and line.strip() == '"samples": [' and not in_samples:
                in_samples = True
                continue

            if not in_samples:
                continue

            stripped = line.strip()
            if not acc and stripped == "]":
                in_samples = False
                current_model = None
                continue

            if not acc:
                if stripped.startswith("{"):
                    acc = [line]
                    brace_depth = line.count("{") - line.count("}")
                continue

            acc.append(line)
            brace_depth += line.count("{") - line.count("}")
            if brace_depth == 0:
                text = "".join(acc).strip()
                if text.endswith(","):
                    text = text[:-1]
                yield current_model, json.loads(text)
                acc = []


def empty_stats(max_clicks: int, thresholds: list[int]) -> dict:
    return {
        "n": 0,
        "sum_iou": [0.0] * max_clicks,
        "sum_iou_sq": [0.0] * max_clicks,
        "iou1": [],
        "final": [],
        "auc": [],
        "best": [],
        "noc_sum": {thr: 0.0 for thr in thresholds},
        "noc_hist": {thr: Counter() for thr in thresholds},
        "success_by_click": {thr: [0] * max_clicks for thr in thresholds},
    }


def first_hit(ious: list[float], threshold: int, miss_value: int) -> int:
    target = threshold / 100.0
    for idx, iou in enumerate(ious, start=1):
        if iou >= target:
            return idx
    return miss_value


def analyze(path: Path) -> tuple[dict, dict, dict]:
    header = parse_header(path)
    args = header["args"]
    max_clicks = int(args["max_clicks"])
    miss_value = max_clicks + 1
    thresholds = [int(x) for x in str(args["thresholds"]).split(",") if x]

    stats = defaultdict(lambda: empty_stats(max_clicks, thresholds))
    paired = defaultdict(dict)

    for model, obj in iter_sample_records(path):
        s = stats[model]
        ious = obj["ious"]
        sample_index = int(obj["sample_index"])
        s["n"] += 1
        for idx, iou in enumerate(ious):
            s["sum_iou"][idx] += iou
            s["sum_iou_sq"][idx] += iou * iou
        auc = sum(ious) / len(ious)
        s["iou1"].append(ious[0])
        s["final"].append(ious[-1])
        s["auc"].append(auc)
        s["best"].append(max(ious))

        for thr in thresholds:
            noc = int(obj["noc"].get(str(thr), first_hit(ious, thr, miss_value)))
            s["noc_sum"][thr] += noc
            s["noc_hist"][thr][noc] += 1
            for click_idx, iou in enumerate(ious):
                if iou >= thr / 100.0:
                    s["success_by_click"][thr][click_idx] += 1

        paired[sample_index][model] = {
            "iou1": ious[0],
            "final": ious[-1],
            "auc": auc,
            "best": max(ious),
            "noc": {thr: int(obj["noc"].get(str(thr), first_hit(ious, thr, miss_value))) for thr in thresholds},
        }

    return header, dict(stats), paired


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def quantiles(values: list[float], qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> list[float]:
    if not values:
        return [0.0 for _ in qs]
    vals = sorted(values)
    n = len(vals)
    return [vals[min(n - 1, int(q * (n - 1)))] for q in qs]


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def paired_rows(paired: dict, left: str, right: str, metrics: list[str]) -> list[list[str]]:
    pairs = [(idx, v[left], v[right]) for idx, v in paired.items() if left in v and right in v]
    rows = []
    for metric in metrics:
        diffs = [r[metric] - l[metric] for _, l, r in pairs]
        qs = quantiles(diffs)
        rows.append([
            metric,
            fmt(mean(diffs)),
            fmt(qs[2]),
            fmt(sum(d > 0 for d in diffs) / len(diffs)),
            fmt(qs[0]),
            fmt(qs[1]),
            fmt(qs[3]),
            fmt(qs[4]),
        ])
    return rows


def top_diffs(paired: dict, left: str, right: str, metric: str, count: int = 5):
    pairs = [(idx, v[left], v[right]) for idx, v in paired.items() if left in v and right in v]
    diffs = sorted(
        ((r[metric] - l[metric], idx, l[metric], r[metric]) for idx, l, r in pairs),
        key=lambda x: x[0],
    )
    return diffs[-count:][::-1], diffs[:count]


def render_report(path: Path, header: dict, stats: dict, paired: dict) -> str:
    args = header["args"]
    thresholds = [int(x) for x in str(args["thresholds"]).split(",") if x]
    max_clicks = int(args["max_clicks"])
    selected_thresholds = [thr for thr in (80, 85, 90, 95) if thr in thresholds]

    lines = [
        "# Interactive Segmentation Eval Report",
        "",
        f"- Source: `{path}`",
        f"- Dataset: `{args['dataset']}` split `{args['dataset_split']}`",
        f"- Samples: {next(iter(stats.values()))['n'] if stats else 0}",
        f"- Max clicks: {max_clicks}; miss value for NoC: {max_clicks + 1}",
        f"- Random seed: {args['seed']}; deterministic clicks: {args['deterministic_clicks']}",
        f"- SAM checkpoint: `{args.get('sam_checkpoint')}`",
        f"- RopeSAM checkpoint: `{args.get('rope_sam_checkpoint')}`",
        "",
    ]

    rows = []
    for model, s in stats.items():
        n = s["n"]
        row = [
            model,
            str(n),
            fmt(s["sum_iou"][0] / n),
            fmt(s["sum_iou"][4] / n) if max_clicks >= 5 else "n/a",
            fmt(s["sum_iou"][-1] / n),
            fmt(mean(s["auc"])),
            fmt(mean(s["best"])),
        ]
        for thr in selected_thresholds:
            row.append(fmt(s["noc_sum"][thr] / n))
            row.append(fmt(1.0 - s["noc_hist"][thr][max_clicks + 1] / n))
        rows.append(row)
    headers = ["model", "n", "IoU@1", "IoU@5", f"IoU@{max_clicks}", "AUC", "best"]
    for thr in selected_thresholds:
        headers.extend([f"NoC@{thr}", f"reach@{thr}"])
    lines.extend(["## Summary", "", table(headers, rows), ""])

    if "sam" in stats and "rope_sam" in stats:
        lines.extend([
            "## Paired RopeSAM Minus SAM",
            "",
            table(
                ["metric", "mean diff", "median", "win rate", "q10", "q25", "q75", "q90"],
                paired_rows(paired, "sam", "rope_sam", ["iou1", "final", "auc", "best"]),
            ),
            "",
        ])

        noc_rows = []
        pairs = [(v["sam"], v["rope_sam"]) for v in paired.values() if "sam" in v and "rope_sam" in v]
        for thr in thresholds:
            diffs = [l["noc"][thr] - r["noc"][thr] for l, r in pairs]
            qs = quantiles(diffs)
            noc_rows.append([
                str(thr),
                fmt(mean(diffs)),
                fmt(qs[2]),
                fmt(sum(d > 0 for d in diffs) / len(diffs)),
                fmt(qs[0]),
                fmt(qs[1]),
                fmt(qs[3]),
                fmt(qs[4]),
            ])
        lines.extend([
            "Positive NoC diff means RopeSAM needs fewer clicks.",
            "",
            table(["thr", "mean click saved", "median", "win rate", "q10", "q25", "q75", "q90"], noc_rows),
            "",
        ])

        gains, drops = top_diffs(paired, "sam", "rope_sam", "final")
        rows = [[str(idx), fmt(diff), fmt(sam), fmt(rope)] for diff, idx, sam, rope in gains]
        lines.extend(["## Largest Final-IoU Gains", "", table(["sample", "rope-sam", "sam final", "rope final"], rows), ""])
        rows = [[str(idx), fmt(diff), fmt(sam), fmt(rope)] for diff, idx, sam, rope in drops]
        lines.extend(["## Largest Final-IoU Drops", "", table(["sample", "rope-sam", "sam final", "rope final"], rows), ""])

    lines.extend(["## Success By Click", ""])
    for thr in selected_thresholds:
        rows = []
        for model, s in stats.items():
            n = s["n"]
            checkpoints = [1, 3, 5, max_clicks]
            rows.append([model] + [fmt(s["success_by_click"][thr][k - 1] / n) for k in checkpoints])
        lines.extend([f"Threshold {thr}", "", table(["model", "click1", "click3", "click5", f"click{max_clicks}"], rows), ""])

    lines.extend([
        "## Notes",
        "",
        "- The evaluator stores unreached thresholds as `max_clicks + 1`, so NoC values near 11 mean most samples failed within 10 clicks.",
        "- This report is generated by streaming the `samples` arrays and keeping only aggregates plus paired scalar metrics in memory.",
        "- SAM uses the normalized tensor emitted by `MaskLevelFlatDataset.preprocess_image`; this matches direct `Sam.image_encoder` use in this repo, but it is not the high-level `SamPredictor.set_image()` path.",
        "- The original run did not enable SAM multimask selection on the first click. Official SAM usage recommends multimask output for ambiguous single-click prompts, then choosing by predicted IoU. Re-run with `--sam_multimask_first_click` before treating the SAM baseline as final.",
        "- Masks are evaluated in the resized/padded mask frame. This keeps SAM and RopeSAM on the same target, but it is not the canonical SAM crop-back-to-original-image evaluation protocol.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze eval_interactive_seg.py JSON output.")
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    header, stats, paired = analyze(args.json_path)
    report = render_report(args.json_path, header, stats, paired)
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
