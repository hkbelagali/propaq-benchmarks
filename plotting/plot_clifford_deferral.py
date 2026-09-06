#!/usr/bin/env python3
"""Plot propaq's Clifford-deferral ablation: propagation runtime vs T-gate density,
with and without deferral, on identical circuits."""

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

BENCHMARK_ID = "clifford_deferral_ablation_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=BENCH_DIR / "experiments" / "clifford_deferral" / "results_propaq.npz")
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "results" / "plots" / "clifford_deferral__runtime_vs_t_density")
    args = parser.parse_args()

    records = load_records_npz(str(args.results))
    records = [record for record in records if record.get("benchmark") == BENCHMARK_ID]
    if not records:
        raise ValueError(f"no {BENCHMARK_ID} records in {args.results}")
    latest = records[-1]
    config = (latest["n_qubits"], latest["total_layers"], latest["n_threads"], latest["repeats"])
    records = [
        record for record in records
        if (record["n_qubits"], record["total_layers"], record["n_threads"], record["repeats"]) == config
    ]
    by_p = {record["p"]: record for record in records}
    records = [by_p[p] for p in sorted(by_p)]

    p = [record["p"] for record in records]
    t_on = [record["t_on_ms"] / 1e3 for record in records]
    t_off = [record["t_off_ms"] / 1e3 for record in records]

    textwidth = 3.31314
    aspect_ratio = 6 / 8
    scale = 1.3
    width = textwidth * scale
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(p, t_on, marker="o", linewidth=2, label="Clifford deferral on")
    ax.plot(p, t_off, marker="s", linewidth=2, label="Clifford deferral off")
    ax.set_yscale("log")
    ax.set_xlabel("T-gate insertion probability $p$", fontsize=13)
    ax.set_ylabel("Propagation Wall Time (s)", fontsize=13)
    ax.legend(fancybox=False, edgecolor="black", loc="best", fontsize=9)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix(".png"), bbox_inches="tight", dpi=800)
    fig.savefig(args.out.with_suffix(".pgf"), bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
