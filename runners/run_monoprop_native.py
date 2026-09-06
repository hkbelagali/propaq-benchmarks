#!/usr/bin/env python3
"""Runner for MonoProp's native Majorana propagation of the Hubbard Trotter model.

The circuit is constructed from the same ffsim Fermi-Hubbard Hamiltonian as
``run_propaq_native.py``. It is expanded into sorted Majorana monomial rotations rather
than Jordan-Wigner qubit gates, so it belongs in the native-fermionic comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import ffsim

try:
    from monoprop import Circuit, ExpGate, FermiOperator, MajoranaOperator, MajoranaPropagator
except ImportError as exc:  # pragma: no cover - optional local dependency
    raise SystemExit(
        "MonoProp is not importable in this Python environment. Install the sibling checkout "
        "with `python -m pip install -e ../../monoprop` before running this backend."
    ) from exc


def ffsim_to_monoprop_fermi(operator: "ffsim.FermionOperator", n_orbitals: int) -> FermiOperator:
    """Convert ffsim's spin/orbital ladder representation to MonoProp's mode strings."""
    terms = []
    coeffs = []
    for term, coeff in operator.items():
        terms.append(tuple(
            (spin * n_orbitals + orbital, "+" if action else "-")
            for action, spin, orbital in term
        ))
        coeffs.append(coeff)
    return FermiOperator(terms, coeffs, num_modes=2 * n_orbitals)


def build_hubbard_native(params: dict) -> tuple[Circuit, FermiOperator, list[int], int]:
    nx, ny = params["nx"], params["ny"]
    n_sites = nx * ny
    dt, steps = params["dt"], params["steps"]
    hamiltonian = ffsim.fermi_hubbard_2d(
        nx, ny, tunneling=params["t"], interaction=params["U"],
    )
    h_fermi = ffsim_to_monoprop_fermi(hamiltonian, n_sites)
    h_majorana = h_fermi.get_majorana_operator()

    # Same deterministic first-order splitting as the native ProPaQ runner: individual
    # Majorana monomials ordered by weight then bitmask. ExpGate evolves exp(+i theta H),
    # so theta=-dt gives the physical exp(-i H dt) Trotter factor.
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

    # Z on the final spin-up site: Z = I - 2 n. The checkerboard occupation agrees with
    # problems_qubit.hubbard_trotter_problem and the other native runners.
    target = n_sites - 1
    observable = FermiOperator(
        [(), ((target, "+"), (target, "-"))], [1.0, -2.0], num_modes=2 * n_sites,
    )
    initial_state = list(range(0, n_sites, 2))
    return circuit, observable, initial_state, n_sites


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--coeff-cutoff", type=float, default=1e-8)
    ap.add_argument("--n-threads", type=int, default=64)
    args = ap.parse_args()
    os.environ["monoprop_NUM_THREADS"] = str(args.n_threads)

    with open(args.problem) as f:
        source = json.load(f)
    if source["problem"] != "hubbard_trotter":
        raise ValueError(f"unknown native MonoProp problem: {source['problem']!r}")

    t0 = time.perf_counter()
    circuit, observable, initial_state, n_sites = build_hubbard_native(source["params"])
    build_time_s = time.perf_counter() - t0
    prop = MajoranaPropagator(
        observable,
        initial_state,
        cutoff=4 * n_sites,
        lower_atol=args.coeff_cutoff,
    )
    t1 = time.perf_counter()
    prop.propagate(circuit)
    wall_time_s = time.perf_counter() - t1

    print(json.dumps({
        "backend": "monoprop",
        "basis": "majorana",
        "problem": source["problem"],
        "n_qubits": n_sites,
        "gate_count": len(circuit.gates),
        "params": source["params"],
        "n_threads": args.n_threads,
        "wall_time_s": wall_time_s,
        "circuit_build_time_s": build_time_s,
        "expectation_value": float(prop.expectation_value()),
        "n_terms_final": int(prop.size()),
        "truncation_onenorm": None,
        "max_terms": None,
        "max_weight": None,
        "min_abs_coeff": args.coeff_cutoff,
    }))


if __name__ == "__main__":
    main()
