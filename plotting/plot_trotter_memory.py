#!/usr/bin/env python3
"""Plots peak-RSS-vs-Trotter-step/layer curves for trotter/run_trotter_scan.py's output: one
figure per (system, lattice size), x-axis = Trotter step/layer, y-axis = peak resident set
size in MB (log scale, mean over --n-trials trials), one line per backend, with an inset
showing final term count (log scale) vs. the same step/layer axis.

Peak RSS comes from /usr/bin/time -v around every backend invocation (common/io_utils.py's
_RSS_RE), so it is one process's high-water mark for that single (step, backend) run, not a
memory-vs-time trace within the run.

Same layout as plot_trotter_scan.py (backend order, colors, markers, inset placement, PGF +
PNG output) with wall time swapped for peak RSS, so the two figures read as a pair.

Usage: python3 plotting/plot_trotter_memory.py [--results results/trotter_scan.npz] [--outdir results/plots]
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
from common.io_utils import load_records_npz  # noqa: E402

import scienceplots

plt.style.use(["science", "grid"])


def _ok(records):
    return [r for r in records if r.get("ok")]


def _series_key(backend: str, basis: str) -> str:
    return f"{backend}:{basis}"


# Fixed backend order (also used for color-cycle indexing, below) and display labels, with no
# dependence on plotting/style.py, by request. Colors come straight from whatever "science" +
# "grid" (scienceplots) sets as axes.prop_cycle, not a hand-picked palette.
BACKEND_ORDER = [
    "pauli_prop:pauli",
    "pyrauli:pauli",
    "pauli_propagation_jl:pauli",
    "monoprop:pauli",
    "monoprop:majorana",
    "propaq:pauli",
    "propaq:majorana",
    "majorana_propagation_jl:majorana",
]
BACKEND_LABELS = {
    "pauli_prop:pauli": "pauli-prop",
    "pyrauli:pauli": "pyrauli",
    "pauli_propagation_jl:pauli": "PauliPropagation.jl",
    "monoprop:pauli": "MonoProp",
    "monoprop:majorana": "MonoProp (Majorana)",
    "propaq:pauli": "propaq (Pauli)",
    "propaq:majorana": "propaq (Majorana)",
    "majorana_propagation_jl:majorana": "MajoranaPropagation.jl",
}

# Cross-family markers only (stroke-only, no filled/solid shapes), one distinct type per
# backend series. '1'-'4' are matplotlib's tri_down/up/left/right, also stroke-only.
CROSS_MARKERS = {
    "pauli_prop:pauli": "+",
    "pyrauli:pauli": "x",
    "pauli_propagation_jl:pauli": "1",
    "monoprop:pauli": "P",
    "monoprop:majorana": "X",
    "propaq:pauli": "2",
    "propaq:majorana": "3",
    "majorana_propagation_jl:majorana": "4",
}


def _group_by_backend_and_step(records, system: str, size_label: str):
    """backend series-key -> {n_step -> list of records (one per trial)}.

    Keeps only odd n_step (1, 3, 5, ...), every 2nd step, matching
    trotter/run_trotter_scan.py's --step-stride 2 convention, and consistent across every
    backend even where the underlying data has denser (stride-1) coverage for some of them.
    """
    recs = [r for r in records if r["problem"] == system and r["size_label"] == size_label]
    by_backend_step: dict[str, dict[int, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        n_step = r.get("n_step")
        if n_step is None or r.get("peak_rss_mb") is None:
            continue
        n_step = int(n_step)
        if n_step % 2 == 0:
            continue
        key = _series_key(r["backend"], r["basis"])
        by_backend_step[key][n_step].append(r)
    return by_backend_step


def plot_system_size(records, system: str, size_label: str, outdir: Path) -> None:
    by_backend_step = _group_by_backend_and_step(records, system, size_label)
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
    # "science"+"grid" (scienceplots) style's own defaults, nothing extra applied here.

    any_terms = False
    for i, key in enumerate(backends):
        step_map = by_backend_step[key]
        color = color_cycle[i % len(color_cycle)]
        label = BACKEND_LABELS.get(key, key)

        xs, means = [], []
        term_xs, term_ys = [], []
        for n_step in sorted(step_map):
            rss = np.array([r["peak_rss_mb"] for r in step_map[n_step] if r.get("peak_rss_mb") is not None])
            if len(rss) == 0:
                continue
            xs.append(n_step)
            means.append(float(rss.mean()))

            terms = np.array([r["n_terms_final"] for r in step_map[n_step] if r.get("n_terms_final") is not None])
            if len(terms) > 0:
                term_xs.append(n_step)
                term_ys.append(float(terms.mean()))

        if not xs:
            continue
        marker = CROSS_MARKERS.get(key, "x")
        ax.plot(xs, means, color=color, marker=marker, label=label,
                linewidth=1.6, markersize=8, markeredgewidth=1.5, zorder=3)
        
    ax.set_yscale("log")
    ax.set_xlabel("Trotter step", fontsize=13)
    ax.set_ylabel("Peak RSS (MB)", fontsize=13)
    ax.set_xlim(0, 26)
    ax.set_ylim(bottom=1e1, top=1e6)
    ax.set_xticks([0, 5, 10, 15, 20, 25], labels=[0, 5, 10, 15, 20, 25], fontsize=13)
    ax.set_yticks([1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
                  labels=[r"$10^{1}$", r"$10^{2}$", r"$10^{3}$", r"$10^{4}$", r"$10^{5}$", r"$10^{6}$"], fontsize=13)

    legend = ax.legend(fancybox=False, edgecolor='black', loc="upper left", fontsize=9)
    legend.get_frame().set_linewidth(0.5)
    ax.grid(False)
    ax.yaxis.set_minor_locator(mticker.NullLocator())

    fig.tight_layout()
    fname = outdir / f"{system}__{size_label}__peak_rss_vs_step"
    fig.savefig(f"{fname}.pgf", bbox_inches="tight")
    fig.savefig(f"{fname}.png", bbox_inches="tight", dpi=800)
    plt.close(fig)
    print(f"  wrote {fname}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(BENCH_DIR / "results" / "trotter_scan.npz"))
    ap.add_argument("--outdir", default=str(BENCH_DIR / "results" / "plots"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = _ok(load_records_npz(args.results))
    systems = sorted({r["problem"] for r in records})
    for system in systems:
        sizes = sorted({r["size_label"] for r in records if r["problem"] == system})
        for size_label in sizes:
            plot_system_size(records, system, size_label, outdir)
    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    main()
