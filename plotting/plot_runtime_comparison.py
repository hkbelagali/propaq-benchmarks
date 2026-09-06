#!/usr/bin/env python3
"""Reads every main-problem experiment folder and produces, per problem family:
  - a grouped-bar chart of propagation wall time (log scale) across backends, one panel per
    problem size (small multiples instead of a second axis).
  - a matching peak-RSS memory chart.
Plus a dedicated line chart for the random_near_clifford T-density sweep, and a scatter of
final term count vs wall time across everything, colored by backend.

A qubit-side (Jordan-Wigner) circuit and its native-fermionic companion (hubbard_trotter,
random_fermionic_circuit) share the same "label" string for the same nominal size, since
their generate_circuits.py scripts save both sides under matching filenames, so grouping by
"label" already puts them in the same panel without any special-casing here.

Usage: python3 plotting/plot_runtime_comparison.py [--outdir results/plots]
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_DIR))
from common.io_utils import load_experiment_results  # noqa: E402
from plotting.style import style_of, setup_axes, series_key, SERIES  # noqa: E402

import scienceplots

plt.style.use(["science", "grid"])

PROBLEMS = [
    "random_circuit", "random_near_clifford", "ising_trotter", "heisenberg_chain_trotter",
    "qaoa_maxcut", "ucj_h2", "hubbard_trotter", "random_fermionic_circuit",
]


def _ok(records):
    return [r for r in records if r.get("ok")]


def plot_family_bars(records, problem: str, outdir: Path) -> None:
    recs = [r for r in records if r["problem"] == problem]
    if not recs:
        return
    sizes = sorted({r["label"] for r in recs}, key=lambda s: (len(s), s))
    backends = sorted({series_key(r["backend"], r["basis"]) for r in recs},
                       key=lambda k: list(SERIES.keys()).index(k) if k in SERIES else 99)

    for metric, ylabel, fname_suffix in [
        ("wall_time_s", "Propagation wall time (s, log scale)", "runtime"),
        ("peak_rss_mb", "Peak RSS (MB, log scale)", "memory"),
    ]:
        fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 4.5), dpi=150,
                                  squeeze=False, sharey=True)
        axes = axes[0]
        for ax, size in zip(axes, sizes):
            setup_axes(ax)
            size_recs = [r for r in recs if r["label"] == size]
            by_key = {series_key(r["backend"], r["basis"]): r for r in size_recs}
            xs, heights, colors, labels = [], [], [], []
            for i, key in enumerate(backends):
                r = by_key.get(key)
                if r is None or r.get(metric) is None or not r.get("ok", True):
                    continue
                color, marker, label = style_of(*key.split(":"))
                xs.append(len(xs))
                heights.append(max(r[metric], 1e-6))
                colors.append(color)
                labels.append(label)
            ax.bar(xs, heights, color=colors, width=0.65, zorder=3)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_yscale("log")
            ax.set_title(size, fontsize=10)
        axes[0].set_ylabel(ylabel, fontsize=10)
        fig.suptitle(f"{problem}: {ylabel.split(' (')[0].lower()} by backend", fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(outdir / f"{problem}__{fname_suffix}.pgf", bbox_inches="tight")
        plt.close(fig)


def plot_near_clifford_sweep(records, outdir: Path) -> None:
    recs = [r for r in records if r["problem"] == "random_near_clifford"]
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    setup_axes(ax)
    by_key = collections.defaultdict(list)
    for r in recs:
        key = series_key(r["backend"], r["basis"])
        td = r["params"].get("t_density")
        if td is not None and r.get("wall_time_s") is not None:
            by_key[key].append((td, r["wall_time_s"]))
    for key, pts in by_key.items():
        pts.sort()
        color, marker, label = style_of(*key.split(":"))
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, color=color, label=label, linewidth=2, markersize=7)
    ax.set_xlabel("non-Clifford (RZ) gate density")
    ax.set_ylabel("Propagation wall time (s, log scale)")
    ax.set_yscale("log")
    ax.set_title("Near-Clifford circuit: runtime vs. non-Cliffordness")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "random_near_clifford__sweep.pgf", bbox_inches="tight")
    plt.close(fig)


def plot_terms_vs_time(records, outdir: Path) -> None:
    recs = [r for r in records if r.get("n_terms_final") is not None and r.get("wall_time_s") is not None]
    if not recs:
        return
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    setup_axes(ax)
    seen = set()
    for r in recs:
        key = series_key(r["backend"], r["basis"])
        color, marker, label = style_of(*key.split(":"))
        ax.scatter(r["n_terms_final"], r["wall_time_s"], color=color, marker=marker,
                   s=40, alpha=0.75, zorder=3, label=label if key not in seen else None)
        seen.add(key)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Final Pauli/Majorana term count (log scale)")
    ax.set_ylabel("Propagation wall time (s, log scale)")
    ax.set_title("Runtime vs. final term count, all problems")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "terms_vs_time.pgf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", default=str(BENCH_DIR / "experiments"))
    ap.add_argument("--outdir", default=str(BENCH_DIR / "results" / "plots"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    experiments_dir = Path(args.experiments_dir)

    records = []
    for problem in PROBLEMS:
        records.extend(load_experiment_results(experiments_dir / problem))
    records = _ok(records)

    for problem in sorted(set(PROBLEMS) - {"random_near_clifford"}):
        plot_family_bars(records, problem, outdir)
    plot_near_clifford_sweep(records, outdir)
    plot_terms_vs_time(records, outdir)
    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
