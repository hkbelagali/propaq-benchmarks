#!/usr/bin/env python3
"""Plot propaq's thread-scaling behavior: propagation runtime vs Trotter step count,
one line per thread count, with each thread count's ideal (linear, T(1)/N) scaling
shown as a same-colored dashed reference (one combined "Ideal" legend entry).

Colors are the data-viz skill's validated 8-hue categorical palette (first 7 slots),
one fixed hue per thread count so every actual/ideal pair is unambiguous at a glance.
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
import scienceplots  # noqa: F401

plt.style.use(["science", "grid"])

BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_DIR))
from common.io_utils import load_records_npz  # noqa: E402

BENCHMARK_ID = "thread_scaling_ising_trotter_v1"

# data-viz skill's documented categorical palette, slots 1-7 (blue, orange, aqua,
# yellow, magenta, green, violet), one fixed hue per thread count.
THREAD_COLORS = {
    1:  "#2a78d6",
    2:  "#eb6834",
    4:  "#1baf7a",
    8:  "#eda100",
    16: "#e87ba4",
    32: "#008300",
    64: "#4a3aa7",
}
THREAD_MARKERS = {1: "o", 2: "s", 4: "^", 8: "D", 16: "v", 32: "P", 64: "X"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=BENCH_DIR / "results" / "thread_scaling.npz")
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "results" / "plots" / "thread_scaling__runtime_vs_steps")
    args = parser.parse_args()

    records = load_records_npz(str(args.results))
    records = [r for r in records if r.get("benchmark") == BENCHMARK_ID]
    if not records:
        raise ValueError(f"no {BENCHMARK_ID} records in {args.results}")
    latest = records[-1]
    config = (latest["nx"], latest["ny"], latest["J"], latest["h"], latest["dt"], latest["repeats"], latest["coeff_cutoff"])
    records = [
        r for r in records
        if (r["nx"], r["ny"], r["J"], r["h"], r["dt"], r["repeats"], r["coeff_cutoff"]) == config
    ]

    by_threads: dict[int, dict[int, float]] = collections.defaultdict(dict)
    for r in records:
        by_threads[r["n_threads"]][r["steps"]] = r["wall_time_s"]

    thread_counts = sorted(by_threads)
    if 1 not in by_threads:
        raise ValueError("no n_threads=1 baseline in results, the ideal-scaling reference needs it")
    baseline = by_threads[1]

    textwidth = 3.31314
    aspect_ratio = 6 / 8
    scale = 1.5
    width = textwidth * scale
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height))

    for n_threads in thread_counts:
        series = by_threads[n_threads]
        steps = sorted(series)
        wall = [series[s] for s in steps]
        color = THREAD_COLORS.get(n_threads, "#888888")
        marker = THREAD_MARKERS.get(n_threads, "o")
        ax.plot(steps, wall, marker=marker, color=color, linewidth=2, markersize=6,
                 label=f"{n_threads} thread" + ("s" if n_threads > 1 else ""), zorder=3)

        # Ideal (linear) scaling from the 1-thread baseline at the same step counts, same
        # color as its actual line, but unlabeled.
        # One combined legend entry for all of them is added below instead of one per
        # thread count.
        ideal_steps = [s for s in steps if s in baseline]
        if not ideal_steps:
            continue
        ideal = [baseline[s] / n_threads for s in ideal_steps]
        ax.plot(ideal_steps, ideal, linestyle="--", color=color, linewidth=1.5, alpha=0.8, zorder=2)
        ax.set_xlim(8, 16)
        ax.set_ylim(0.01, 1500)

    # One combined reference entry for what every dashed line means.
    ax.plot([], [], linestyle="--", color="#555555", linewidth=1.5, label="Ideal")

    ax.set_yscale("log")
    ax.set_xlabel("Trotter steps", fontsize=13)
    ax.set_ylabel("Propagation Wall Time (s)", fontsize=13)
    ax.legend(fancybox=False, edgecolor="black", loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=800)
    fig.savefig(args.out.with_suffix(".pgf"), bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
