#!/usr/bin/env python3
"""Plot direct and compiled-surrogate optimization timings.

Usage:
    python bench/plotting/plot_surrogate_optimization.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401  # Registers the SciencePlots styles.
except ModuleNotFoundError:
    # Keep this benchmark runnable in minimal environments.  Production plots
    # use the same SciencePlots stack as the other benchmark plotters above.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.grid": False,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "axes.spines.right": False,
        }
    )
else:
    plt.style.use(["science", "grid", "no-latex"])

BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_DIR))
from common.io_utils import load_records_npz  # noqa: E402

BENCHMARK_ID = "qaoa_maxcut_surrogate_optimization_v1"


def latest_record(path: Path) -> dict:
    """Load the most recently checkpointed optimization result."""
    records = load_records_npz(str(path))
    records = [record for record in records if record.get("benchmark") == BENCHMARK_ID]
    if not records:
        raise ValueError(f"no {BENCHMARK_ID} records in {path}")
    return records[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=BENCH_DIR / "experiments" / "surrogate_optimization" / "results_propaq.npz"
    )
    parser.add_argument(
        "--out", type=Path,
        default=BENCH_DIR / "results" / "plots" / "surrogate_optimization_runtime",
    )
    args = parser.parse_args()

    record = latest_record(args.results)
    labels = ["Numerical", "Evaluation", "Symbolic"]
    runtimes = [
        record["numerical_optimization_s"],
        record["compiled_optimization_s"],
        record["compiled_total_s"],
    ]


    textwidth = 3.31314
    aspect_ratio = 6/8
    scale = 1.3
    width = textwidth * scale
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height))

    bars = ax.bar(labels, runtimes, width=0.65, color=["C0", "C1", "C2"])
    ax.set_yscale("log")
    ax.grid(False)
    ax.set_ylim(top=max(runtimes) * 2.0)
    ax.set_ylabel("Optimization Wall Time (s)", fontsize=13)
    ax.tick_params(axis="x", labelsize=13)
    ax.bar_label(bars, labels=[f"{runtime:.3g} s" for runtime in runtimes], padding=3, fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=800)
    try:
        fig.savefig(args.out.with_suffix(".pgf"), bbox_inches="tight")
    except Exception as error:  # Local TeX is optional for the raster benchmark figure.
        print(f"Skipping PGF export: {error}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
