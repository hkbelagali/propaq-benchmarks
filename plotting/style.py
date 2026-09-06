"""Shared plot styling: one fixed color per backend/basis combination (Okabe-Ito colorblind-
safe qualitative palette), assigned by identity and never re-cycled by rank, used consistently
across every figure this suite produces.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# backend or "backend:basis" -> (color, marker, display label)
SERIES = {
    "pauli_prop:pauli":            ("#E69F00", "o", "pauli-prop"),
    "pyrauli:pauli":                ("#56B4E9", "s", "pyrauli"),
    "pauli_propagation_jl:pauli":   ("#009E73", "^", "PauliPropagation.jl"),
    "monoprop:pauli":               ("#CC79A7", "P", "MonoProp"),
    "monoprop:majorana":            ("#882255", "X", "MonoProp (Majorana)"),
    "propaq:pauli":                 ("#0072B2", "D", "propaq (Pauli)"),
    "propaq:majorana":              ("#D55E00", "v", "propaq (Majorana)"),
    "majorana_propagation_jl:majorana": ("#CC79A7", "P", "MajoranaPropagation.jl"),
}


def series_key(backend: str, basis: str) -> str:
    return f"{backend}:{basis}"


def style_of(backend: str, basis: str):
    key = series_key(backend, basis)
    if key not in SERIES:
        return ("#888888", "x", f"{backend} ({basis})")
    return SERIES[key]


def setup_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#333333")


def new_figure(figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    setup_axes(ax)
    return fig, ax


PLOT_DEFAULTS = dict(
    facecolor="white",
)
