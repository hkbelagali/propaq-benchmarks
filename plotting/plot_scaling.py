#!/usr/bin/env python3
"""Reads experiments/thread_scaling/ and produces, per problem instance:
  - wall time vs. thread count (log-log), one line per backend, pauli-prop shown as a flat
    dashed single-threaded reference.
  - speedup vs. thread count (relative to each backend's own n_threads=1 run), with an ideal
    (y=x) linear-scaling reference line.

Grouping by "problem" rather than "label" is what puts the qubit-side (Jordan-Wigner)
hubbard_qubit run and the native-fermionic hubbard_native run (both tagged
problem="hubbard_trotter") on the same figure, since they describe the same nominal
instance measured two different ways.

Usage: python3 plotting/plot_scaling.py [--outdir results/plots]
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
from plotting.style import style_of, setup_axes, series_key  # noqa: E402

import scienceplots

plt.style.use(["science", "grid"])


def plot_instance(records, problem: str, outdir: Path) -> None:
    recs = [r for r in records if r["problem"] == problem and r.get("wall_time_s") is not None]
    if not recs:
        return
    by_backend = collections.defaultdict(list)
    for r in recs:
        key = series_key(r["backend"], r.get("basis", "pauli"))
        by_backend[key].append((r["n_threads"], r["wall_time_s"]))
    for pts in by_backend.values():
        pts.sort()

    ref = by_backend.pop("pauli_prop:pauli", None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    for ax in (ax1, ax2):
        setup_axes(ax)

    for key, pts in by_backend.items():
        color, marker, label = style_of(*key.split(":"))
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, marker=marker, color=color, label=label, linewidth=2, markersize=7)
        y0 = ys[0]
        speedup = [y0 / y for y in ys]
        ax2.plot(xs, speedup, marker=marker, color=color, label=label, linewidth=2, markersize=7)

    if ref:
        t0, y0 = ref[0]
        ax1.axhline(y0, color="#888888", linestyle="--", linewidth=1.5, label="pauli-prop (single-threaded)")

    all_threads = sorted({t for pts in by_backend.values() for t, _ in pts})
    if all_threads:
        ax2.plot(all_threads, all_threads, color="#888888", linestyle=":", linewidth=1.5, label="ideal linear")

    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Threads")
    ax1.set_yscale("log")
    ax1.set_ylabel("Propagation wall time (s, log scale)")
    ax1.set_title("Wall time vs. thread count")
    ax2.set_ylabel("Speedup relative to that backend's own 1-thread run")
    ax2.set_title("Speedup vs. thread count")
    ax1.legend(frameon=False, fontsize=8)
    ax2.legend(frameon=False, fontsize=8)
    fig.suptitle(problem, fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / f"scaling__{problem}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=str(BENCH_DIR / "experiments" / "thread_scaling"))
    ap.add_argument("--outdir", default=str(BENCH_DIR / "results" / "plots"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = [r for r in load_experiment_results(args.experiment) if r.get("ok")]
    problems = sorted({r["problem"] for r in records})
    for problem in problems:
        plot_instance(records, problem, outdir)
    print(f"Wrote {len(problems)} scaling plots to {outdir}")


if __name__ == "__main__":
    main()
