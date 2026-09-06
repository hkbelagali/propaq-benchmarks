#!/usr/bin/env python3
"""Plot discarded L1 weight and accuracy for the hybrid MPS-Heisenberg benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots  # noqa: F401
except ModuleNotFoundError:
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "serif",
        }
    )
else:
    plt.style.use(["science", "grid", "no-latex"])

BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_DIR))
from common.io_utils import load_records_npz  # noqa: E402

BENCHMARK_ID = "hybrid_mps_heisenberg_tfim_v3"


def load_latest_sweep(path: Path) -> list[dict]:
    """Load the latest coherent four-cutoff sweep from the npz checkpoint."""
    records = load_records_npz(str(path))
    records = [record for record in records if record.get("benchmark") == BENCHMARK_ID]
    if not records:
        raise ValueError(f"no {BENCHMARK_ID} records in {path}")
    latest = records[-1]
    config = (latest["n_qubits"], latest["mps_layers"], latest["heisenberg_layers"])
    records = [
        record for record in records
        if (record["n_qubits"], record["mps_layers"], record["heisenberg_layers"]) == config
    ]
    by_cutoff = {record["weight_cutoff"]: record for record in records}
    return [by_cutoff[cutoff] for cutoff in sorted(by_cutoff)]


def grouped_bars(ax, cutoffs, pure, hybrid, ylabel: str, title: str, mps=None) -> None:
    """Draw a log-scale paired bar chart for one hybrid diagnostic."""
    positions = np.arange(len(cutoffs))
    width = 0.25 if mps is not None else 0.36
    offset = width if mps is not None else width / 2
    ax.bar(positions - offset, pure, width, label="Heisenberg only", color="C0")
    ax.bar(positions, hybrid, width, label="Hybrid MPS-Heisenberg", color="C1")
    if mps is not None:
        ax.bar(positions + offset, mps, width, label=r"MPS only ($\chi \leq 4$)", color="C2")
    ax.set_yscale("log")
    ax.set_xticks(positions, [str(cutoff) for cutoff in cutoffs], fontsize=12)
    ax.set_xlabel("Pauli-weight cutoff", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=BENCH_DIR / "results" / "hybrid_mps_heisenberg.npz"
    )
    parser.add_argument(
        "--out", type=Path,
        default=BENCH_DIR / "results" / "plots" / "hybrid_mps_heisenberg_tradeoff.png",
    )
    args = parser.parse_args()

    records = load_latest_sweep(args.results)
    cutoffs = [record["weight_cutoff"] for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), dpi=180, sharex=True)
    grouped_bars(
        axes[0],
        cutoffs,
        [record["pure_discarded_l1"] for record in records],
        [record["hybrid_discarded_l1"] for record in records],
        r"Discarded coefficient $\ell_1$ norm",
        "Propagation loss",
    )
    grouped_bars(
        axes[1],
        cutoffs,
        [record["pure_absolute_error"] for record in records],
        [record["hybrid_absolute_error"] for record in records],
        "Absolute error to true value",
        "Accuracy",
        [record["mps_absolute_error"] for record in records],
    )
    axes[1].legend(fancybox=False, edgecolor="black", fontsize=9, loc="best")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
