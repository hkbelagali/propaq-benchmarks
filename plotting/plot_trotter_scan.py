#!/usr/bin/env python3
"""Plots runtime-vs-Trotter-step curves for the fine step sweep in experiments/ising_trotter/
and experiments/hubbard_trotter/, one figure per (problem, lattice size), x-axis = Trotter
step, y-axis = propagation wall time (log scale), one line per backend, with an inset
showing final term count (log scale) against the same step axis.

Usage: python3 plotting/plot_trotter_scan.py [--outdir results/plots]
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
import matplotlib.ticker as mticker
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_DIR))
from common.io_utils import load_experiment_results  # noqa: E402

import scienceplots

plt.style.use(["science", "grid"])

PROBLEMS = ["ising_trotter", "hubbard_trotter"]


def _ok(records):
    return [r for r in records if r.get("ok")]


def _series_key(backend: str, basis: str) -> str:
    return f"{backend}:{basis}"


# Fixed backend order (also used for color-cycle indexing, below) and display labels, with no
# dependence on plotting/style.py, by request. Colors come straight from whatever "science" and
# "grid" (scienceplots) sets as axes.prop_cycle, not a hand-picked palette.
BACKEND_ORDER = [
    "pauli_prop:pauli",
    "pyrauli:pauli",
    "pauli_propagation_jl:pauli",
    "monoprop:pauli",
    "monoprop_native:majorana",
    "propaq:pauli",
    "propaq:majorana",
    "propaq_native:majorana",
    "majorana_propagation_jl:majorana",
]
BACKEND_LABELS = {
    "pauli_prop:pauli": "pauli-prop",
    "pyrauli:pauli": "pyrauli",
    "pauli_propagation_jl:pauli": "PauliPropagation.jl",
    "monoprop:pauli": "MonoProp",
    "monoprop_native:majorana": "MonoProp (native Majorana)",
    "propaq:pauli": "propaq (Pauli)",
    "propaq:majorana": "propaq (Majorana, JW)",
    "propaq_native:majorana": "propaq (native Majorana)",
    "majorana_propagation_jl:majorana": "MajoranaPropagation.jl",
}

# Cross-family markers only (stroke-only, no filled/solid shapes), one distinct type per
# backend series. '1'-'4' are matplotlib's tri_down/up/left/right, also stroke-only.
CROSS_MARKERS = {
    "pauli_prop:pauli": "+",
    "pyrauli:pauli": "x",
    "pauli_propagation_jl:pauli": "1",
    "monoprop:pauli": "P",
    "monoprop_native:majorana": "X",
    "propaq:pauli": "2",
    "propaq:majorana": "3",
    "propaq_native:majorana": "4",
    "majorana_propagation_jl:majorana": "d",
}


def _group_by_backend_and_step(records, size_label: str):
    """backend series-key -> {step -> record}, one record per (backend, step), no averaging
    since this experiment runs one trial per step count."""
    recs = [r for r in records if f"{int(r['params']['nx'])}x{int(r['params']['ny'])}" == size_label]
    by_backend_step: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for r in recs:
        steps = r.get("params", {}).get("steps")
        if steps is None or r.get("wall_time_s") is None:
            continue
        key = _series_key(r["backend"], r["basis"])
        by_backend_step[key][int(steps)] = r
    return by_backend_step


def plot_problem_size(records, problem: str, size_label: str, outdir: Path) -> None:
    by_backend_step = _group_by_backend_and_step(records, size_label)
    if not by_backend_step:
        return

    backends = sorted(by_backend_step, key=lambda k: BACKEND_ORDER.index(k) if k in BACKEND_ORDER else 99)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    textwidth = 3.31314
    aspect_ratio = 6 / 8
    scale = 1.3
    width = textwidth * scale
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height))
    # Full box border, both-axis dashed grid, and boxed legend all come straight from the
    # "science" and "grid" (scienceplots) style's own defaults, nothing extra applied here.
    ax_inset = ax.inset_axes([0.68, 0.04, 0.30, 0.30])
    ax_inset.patch.set_alpha(0.95)

    any_terms = False
    for i, key in enumerate(backends):
        step_map = by_backend_step[key]
        color = color_cycle[i % len(color_cycle)]
        label = BACKEND_LABELS.get(key, key)

        xs = sorted(step_map)
        ys = [step_map[s]["wall_time_s"] for s in xs]
        term_xs = [s for s in xs if step_map[s].get("n_terms_final") is not None]
        term_ys = [step_map[s]["n_terms_final"] for s in term_xs]

        marker = CROSS_MARKERS.get(key, "x")
        ax.plot(xs, ys, color=color, marker=marker, label=label,
                linewidth=1.6, markersize=8, markeredgewidth=1.5, zorder=3)

        if term_xs:
            any_terms = True
            ax_inset.plot(term_xs, term_ys, color=color, linewidth=1.3)

    ax.set_yscale("log")
    ax.set_xlabel("Trotter step", fontsize=13)
    ax.set_ylabel("Propagation Wall Time (s)", fontsize=13)
    ax.set_xlim(0, 26)
    ax.set_xticks([0, 5, 10, 15, 20, 25], labels=[0, 5, 10, 15, 20, 25], fontsize=13)

    legend = ax.legend(fancybox=False, edgecolor="black", loc="upper left", fontsize=9)
    legend.get_frame().set_linewidth(0.5)
    ax.grid(False)
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    if any_terms:
        ax_inset.set_yscale("log")
        ax_inset.set_title("Number of Terms", fontsize=9)
        ax_inset.grid(False)
        ax_inset.yaxis.set_minor_locator(mticker.NullLocator())
        ax_inset.set_xticks([])
    else:
        ax_inset.remove()

    fig.tight_layout()
    fname = outdir / f"{problem}__{size_label}__runtime_vs_step"
    fig.savefig(f"{fname}.pgf", bbox_inches="tight")
    fig.savefig(f"{fname}.png", bbox_inches="tight", dpi=800)
    plt.close(fig)
    print(f"  wrote {fname}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments-dir", default=str(BENCH_DIR / "experiments"))
    ap.add_argument("--outdir", default=str(BENCH_DIR / "results" / "plots"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    experiments_dir = Path(args.experiments_dir)

    for problem in PROBLEMS:
        records = _ok(load_experiment_results(experiments_dir / problem))
        sizes = sorted({f"{int(r['params']['nx'])}x{int(r['params']['ny'])}" for r in records
                        if "nx" in r.get("params", {}) and "ny" in r.get("params", {})})
        for size_label in sizes:
            plot_problem_size(records, problem, size_label, outdir)
    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
