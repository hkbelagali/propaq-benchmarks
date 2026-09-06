#!/usr/bin/env python3
"""Benchmark coefficient zero-cutoff extrapolation on a spin-chain circuit.

This is the coefficient-cutoff analogue of ``zne.py``.  It uses a five-qubit
periodic transverse-field Ising Trotter circuit rather than the LUCJ molecular
example used for zero-noise extrapolation.  The untruncated propagation is the
reference and is deliberately kept separate from the cutoff sweep.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

N_THREADS = 64
os.environ["RAYON_NUM_THREADS"] = str(N_THREADS)

# Every figure in the suite lands in one directory; only the .npz stays next to this script.
PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"

from propaq.circuits import PauliCircuit
from propaq.datatypes import PauliTermSum
from propaq.extrapolators import CoefficientCutoffExtrapolator
from propaq.noise import TruncationPolicy
from propaq.propagators import PauliPropagator


# These are small enough to expose the cutoff trend without making any point
# effectively untruncated.  The resulting linear intercept is closer to the
# exact value than every individual truncated evaluation.
CUTOFF_VALUES = np.array(
    [0.0003, 0.0005, 0.0007, 0.0009, 0.0011, 0.0013, 0.0015, 0.0017]
)


def linear_fit(cutoff: float, intercept: float, slope: float) -> float:
    """First-order model for the small-cutoff bias."""
    return intercept + slope * cutoff


def build_circuit() -> QuantumCircuit:
    """Return four Trotter steps of a periodic transverse-field Ising chain."""
    n_qubits = 5
    circuit = QuantumCircuit(n_qubits)
    for step in range(4):
        # Inhomogeneous transverse fields avoid symmetries that would hide the
        # effect of coefficient truncation.
        for qubit in range(n_qubits):
            circuit.rx(0.11 + 0.04 * qubit + 0.01 * step, qubit)
        # Periodic ZZ couplings complete the Ising ring.
        for qubit in range(n_qubits - 1):
            circuit.rzz(0.27 + 0.03 * qubit, qubit, qubit + 1)
        circuit.rzz(0.31, n_qubits - 1, 0)
    return circuit


def run_benchmark() -> dict[str, np.ndarray | float]:
    """Evaluate the cutoff sweep, extrapolate it, and validate its accuracy."""
    circuit = PauliCircuit.from_qiskit(build_circuit())
    observable = PauliTermSum.from_sparse_pauli_op(
        SparsePauliOp.from_list([("ZZZZZ", 1.0)])
    )

    true_value = PauliPropagator(n_threads=N_THREADS).expectation_value(
        observable, circuit, initial_state=0
    ).expectation_value
    cutoff_propagator = PauliPropagator(
        n_threads=N_THREADS,
        truncation=TruncationPolicy(coeff_cutoff=float(CUTOFF_VALUES[0]))
    )
    result = CoefficientCutoffExtrapolator(
        fitting_fn=linear_fit, cutoff_values=CUTOFF_VALUES.tolist()
    ).run(cutoff_propagator, observable, circuit, initial_state=0, p0=[0.0, 0.0])

    values = np.asarray(result.expectation_values, dtype=float)
    cutoff_errors = np.abs(values - true_value)
    extrapolation_error = abs(result.zero_cutoff_value - true_value)
    best_cutoff_error = float(cutoff_errors.min())
    if not extrapolation_error < best_cutoff_error:
        raise RuntimeError(
            "zero-cutoff extrapolation must improve on every cutoff point: "
            f"extrapolation error={extrapolation_error:.3e}, "
            f"best cutoff error={best_cutoff_error:.3e}"
        )

    return {
        "cutoff_values": CUTOFF_VALUES,
        "expectation_values": values,
        "fit_params": result.fit_params,
        "zero_cutoff_value": result.zero_cutoff_value,
        "true_value": true_value,
        "cutoff_errors": cutoff_errors,
        "extrapolation_error": extrapolation_error,
    }


def plot(data: dict[str, np.ndarray | float], output: Path) -> None:
    """Render a SciencePlots-style zero-cutoff-extrapolation figure."""
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
    cutoffs = np.asarray(data["cutoff_values"])
    values = np.asarray(data["expectation_values"])
    fit_params = np.asarray(data["fit_params"])
    true_value = float(data["true_value"])
    zero_cutoff_value = float(data["zero_cutoff_value"])

    textwidth = 3.31314
    width = textwidth * 1.3
    fig, ax = plt.subplots(figsize=(width, width * 6 / 8))
    ax.plot(cutoffs * 1e3, values, "x", color="blue", label="Cutoff evaluations")
    fit_cutoffs = np.linspace(0.0, cutoffs.max(), 200)
    ax.plot(
        fit_cutoffs * 1e3,
        linear_fit(fit_cutoffs, *fit_params),
        ":",
        color="blue",
        label="Linear fit",
    )
    ax.plot(
        0,
        zero_cutoff_value,
        "+",
        markersize=10,
        color="red",
        label=f"ZCE value ({zero_cutoff_value:.4f})",
    )
    ax.axhline(
        true_value,
        color="orange",
        linewidth=1.5,
        label=f"Untruncated value ({true_value:.4f})",
    )
    ax.set_xlabel(r"Coefficient cutoff ($10^{-3}$)", fontsize=13)
    ax.set_ylabel(r"$\langle Z^{\otimes 5}\rangle$", fontsize=13)
    ax.legend(fancybox=False, edgecolor="black", loc="best", fontsize=9)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=800)
    fig.savefig(output.with_suffix(".pgf"), bbox_inches="tight")
    print(f"Wrote {output.with_suffix('.png')} and {output.with_suffix('.pgf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "zce_results.npz",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PLOTS_DIR / "zce_plot",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    data = run_benchmark()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.results, **data)
    print(
        "ZCE validation: "
        f"extrapolation error={float(data['extrapolation_error']):.3e}; "
        f"best cutoff error={float(np.min(data['cutoff_errors'])):.3e}"
    )
    print(f"Wrote {args.results}")
    if not args.no_plot:
        plot(data, args.out)


if __name__ == "__main__":
    main()
