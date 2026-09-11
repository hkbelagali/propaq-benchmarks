#!/usr/bin/env python3
"""
Run monoprop's Majorana propagation on the Hubbard Trotter circuits
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks.experiment_runner import run_on_saved_fermionic_circuits  # noqa: E402

MIN_ABS_COEFF = 1e-8
N_THREADS = 64


def ffsim_to_monoprop_fermi(operator: Any, n_orbitals: int) -> Any:
    """Convert ffsim's spin/orbital ladder representation to MonoProp's mode strings."""
    from monoprop import FermiOperator

    terms = []
    coeffs = []
    for term, coeff in operator.items():
        terms.append(tuple(
            (spin * n_orbitals + orbital, "+" if action else "-")
            for action, spin, orbital in term
        ))
        coeffs.append(coeff)
    return FermiOperator(terms, coeffs, num_modes=2 * n_orbitals)


def build_hubbard_native(params: dict[str, Any]):
    import ffsim
    from monoprop import Circuit, ExpGate, FermiOperator, MajoranaOperator

    nx, ny = params["nx"], params["ny"]
    n_sites = nx * ny
    dt, steps = params["dt"], params["steps"]
    hamiltonian = ffsim.fermi_hubbard_2d(
        nx, ny, tunneling=params["t"], interaction=params["U"],
    )
    h_fermi = ffsim_to_monoprop_fermi(hamiltonian, n_sites)
    h_majorana = h_fermi.get_majorana_operator()

    def order(item: tuple[tuple[int, ...], complex]) -> tuple[int, int]:
        mono, _ = item
        return len(mono), sum(1 << i for i in mono)

    generators = sorted(h_majorana.terms.items(), key=order)
    step_gates = [
        ExpGate(MajoranaOperator({mono: coeff}, num_modes=2 * n_sites))
        for mono, coeff in generators
    ]
    gates = tuple(step_gates * steps)
    circuit = Circuit(gates=gates, parameters=tuple([-dt] * len(gates)))

    target = n_sites - 1
    observable = FermiOperator(
        [(), ((target, "+"), (target, "-"))], [1.0, -2.0], num_modes=2 * n_sites,
    )
    initial_state = list(range(0, n_sites, 2))
    return circuit, observable, initial_state, n_sites


def propagate(source: dict[str, Any]) -> dict:
    from monoprop import MajoranaPropagator

    if source["problem"] != "hubbard_trotter":
        raise ValueError(f"unknown native MonoProp problem: {source['problem']!r}")
    os.environ["monoprop_NUM_THREADS"] = str(N_THREADS)

    t0 = time.perf_counter()
    circuit, observable, initial_state, n_sites = build_hubbard_native(source["params"])
    build_time_s = time.perf_counter() - t0

    prop = MajoranaPropagator(
        observable, initial_state, cutoff=4 * n_sites, lower_atol=MIN_ABS_COEFF,
    )
    t1 = time.perf_counter()
    prop.propagate(circuit)
    wall_time_s = time.perf_counter() - t1

    return {
        "n_qubits": n_sites,
        "gate_count": len(circuit.gates),
        "n_threads": N_THREADS,
        "wall_time_s": wall_time_s,
        "circuit_build_time_s": build_time_s,
        "expectation_value": float(prop.expectation_value()),
        "n_terms_final": int(prop.size()),
        "truncation_onenorm": None,
        "max_terms": None,
        "max_weight": None,
        "min_abs_coeff": MIN_ABS_COEFF,
    }


if __name__ == "__main__":
    run_on_saved_fermionic_circuits(HERE / "circuits_native", HERE, "monoprop_native", propagate)
