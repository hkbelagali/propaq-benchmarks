#!/usr/bin/env python3
"""Reads results/scaling_suite.npz and produces, per scaling problem instance:
  - wall time vs. thread count (log-log), one line per backend, pauli-prop shown as a flat
    dashed single-threaded reference.
  - speedup vs. thread count (relative to each backend's own n_threads=1 run), with an ideal
    (y=x) linear-scaling reference line.

Usage: python3 plotting/plot_scaling.py [--results results/scaling_suite.npz] [--outdir results/plots]
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
from common.io_utils import load_records_npz  # noqa: E402
from plotting.style import style_of, setup_axes, series_key  # noqa: E402

import scienceplots 

plt.style.use(["science", "grid"])


def _base_uid(uid: str) -> str:
    """Folds a "..._native(...)" uid (MajoranaPropagation.jl's native-fermionic scaling
    instance) back onto its matching qubit/JW-side uid, so both land in one plot instead of
    MajoranaPropagation.jl getting a figure of its own.
    """
    return uid.replace("_native(", "(")


def plot_instance(records, uid: str, outdir: Path) -> None:
    recs = [r for r in records if _base_uid(r["problem_uid"]) == uid and r.get("wall_time_s") is not None]
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
    fig.suptitle(uid, fontsize=12)
    fig.tight_layout()
    safe_uid = uid.replace("/", "-").replace(" ", "")
    fig.savefig(outdir / f"scaling__{safe_uid}.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(BENCH_DIR / "results" / "scaling_suite.npz"))
    ap.add_argument("--outdir", default=str(BENCH_DIR / "results" / "plots"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = [r for r in load_records_npz(args.results) if r.get("ok")]
    uids = sorted({_base_uid(r["problem_uid"]) for r in records})
    for uid in uids:
        plot_instance(records, uid, outdir)
    print(f"Wrote {len(uids)} scaling plots to {outdir}")


if __name__ == "__main__":
    main()
