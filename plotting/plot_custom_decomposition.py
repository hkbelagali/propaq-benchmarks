#!/usr/bin/env python3
"""Plot custom-versus-generic exchange decomposition propagation runtime."""

from __future__ import annotations

import argparse
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

BENCHMARK_ID = "exchange_custom_decomposition_v2_10q"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=BENCH_DIR / "experiments" / "custom_decomposition" / "results_propaq.npz")
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "results" / "plots" / "custom_decomposition__runtime_vs_layers")
    args = parser.parse_args()
    records = load_records_npz(str(args.results))
    records = [record for record in records if record.get("benchmark") == BENCHMARK_ID]
    if not records:
        raise ValueError(f"no {BENCHMARK_ID} records in {args.results}")
    latest = records[-1]
    records = [record for record in records if record.get("runs") == latest["runs"]]
    by_depth = {record["layers"]: record for record in records}
    records = [by_depth[depth] for depth in sorted(by_depth)]

    layers = [record["layers"] for record in records]
    fallback = [record["fallback_mean_s"] for record in records]
    custom = [record["custom_mean_s"] for record in records]
    textwidth = 3.31314
    aspect_ratio = 6/8
    scale = 1.3
    width = textwidth * scale
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(layers, fallback, marker="o", linewidth=2, label="Native Decomposition")
    ax.plot(layers, custom, marker="s", linewidth=2, label="Registered Decomposition")
    ax.set_yscale("log")
    ax.set_xticks(layers, labels=[str(depth) for depth in layers], fontsize=13)
    ax.set_xlabel("No. of Layers", fontsize=13)
    ax.set_ylabel("Propagation Wall Time (s)", fontsize=13)
    # ax.set_title("Custom non-Clifford gate decomposition accelerates Pauli propagation")
    ax.legend(fancybox=False, edgecolor='black', loc="best", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=800)
    fig.savefig(args.out.with_suffix(".pgf"), bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
