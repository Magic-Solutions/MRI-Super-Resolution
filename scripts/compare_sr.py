#!/usr/bin/env python3
"""Compare per-slice CSVs from scripts/evaluate_sr.py at the subject level.

Slices from one volume are correlated, so every statistic treats the volume
(one session per subject in the resubmission split) as the unit:

- per-volume means of each metric;
- mean +- SD over volumes and a bootstrap 95% CI over volumes;
- paired Wilcoxon signed-rank tests on the per-volume means, each method
  against --reference, with Holm correction across the comparisons.

The pooled mean over all slices (how the first submission's Table 1 was
computed) is reported for reference only.

Usage:
    python scripts/compare_sr.py results/*.csv --reference bicubic --out results/summary
writes <out>.md (tables) and <out>_per_volume.csv.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

METRICS = ["psnr", "psnr_fg", "ssim", "lpips"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", type=Path, nargs="+")
    p.add_argument("--reference", default=None, help="Method every other method is tested against")
    p.add_argument("--out", type=Path, required=True, help="Output prefix")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load(paths: list[Path]) -> tuple[dict, dict]:
    """Return per-volume means {method: {metric: {subject: mean}}} and pooled means {method: {metric: mean}}."""
    slices, per_volume, pooled = {}, {}, {}
    for path in paths:
        rows = list(csv.DictReader(path.open()))
        method = rows[0]["method"]
        assert method not in per_volume, f"method {method} appears in more than one CSV"
        slices[method] = {(r["subject"], int(r["slice"])) for r in rows}
        groups = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for m in METRICS:
                if r.get(m) not in (None, ""):
                    groups[m][r["subject"]].append(float(r[m]))
        per_volume[method] = {m: {s: float(np.mean(v)) for s, v in g.items()} for m, g in groups.items()}
        pooled[method] = {m: float(np.mean([x for v in g.values() for x in v])) for m, g in groups.items()}
    reference_set = next(iter(slices.values()))
    for method, s in slices.items():
        assert s == reference_set, f"{method} was evaluated on different slices than the other methods"
    return per_volume, pooled


def bootstrap_ci(values: np.ndarray, n: int, rng: np.random.Generator) -> tuple[float, float]:
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    per_volume, pooled = load(args.csvs)
    methods = list(per_volume)
    subjects = sorted(next(iter(per_volume.values()))["psnr"])
    metrics = [m for m in METRICS if all(m in per_volume[k] for k in methods)]
    rng = np.random.default_rng(args.seed)

    lines = [f"{len(subjects)} test volumes; statistics over volumes (bootstrap n={args.bootstrap}).", ""]
    for metric in metrics:
        lines += [f"### {metric}", "", "| Method | Mean ± SD | 95% CI | Pooled over slices |", "|---|---|---|---|"]
        for k in methods:
            v = np.array([per_volume[k][metric][s] for s in subjects])
            lo, hi = bootstrap_ci(v, args.bootstrap, rng)
            lines.append(f"| {k} | {v.mean():.4f} ± {v.std(ddof=1):.4f} | [{lo:.4f}, {hi:.4f}] | {pooled[k][metric]:.4f} |")
        lines.append("")

    if args.reference:
        assert args.reference in methods, f"--reference {args.reference} is not among {methods}"
        others = [k for k in methods if k != args.reference]
        for metric in metrics:
            ref = np.array([per_volume[args.reference][metric][s] for s in subjects])
            diffs, pvals = [], []
            for k in others:
                v = np.array([per_volume[k][metric][s] for s in subjects])
                diffs.append(float(np.mean(v - ref)))
                pvals.append(float(wilcoxon(v, ref).pvalue) if np.any(v != ref) else 1.0)
            adjusted = holm(pvals)
            lines += [f"### {metric}: paired Wilcoxon vs {args.reference} (Holm-corrected)", "",
                      "| Method | Mean difference | p | p (Holm) |", "|---|---|---|---|"]
            for k, d, p, pa in zip(others, diffs, pvals, adjusted):
                lines.append(f"| {k} | {d:+.4f} | {p:.4g} | {pa:.4g} |")
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text("\n".join(lines))
    with Path(f"{args.out}_per_volume.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "subject"] + metrics)
        for k in methods:
            for s in subjects:
                writer.writerow([k, s] + [per_volume[k][m][s] for m in metrics])
    print("\n".join(lines))


if __name__ == "__main__":
    main()
