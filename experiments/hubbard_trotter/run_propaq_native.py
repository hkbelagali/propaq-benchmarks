#!/usr/bin/env python3
"""
Run propaq's Majorana propagator on the Hubbard Trotter circuits
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks.experiment_runner import run_on_saved_fermionic_circuits  # noqa: E402

warnings.simplefilter("ignore")

MIN_ABS_COEFF = 1e-8
N_THREADS = 64


class _FermionOpWrapper:
    def __init__(self, op: Any, norb: int):
        self._op = op
        self.norb = norb

    def _fermion_operator_(self) -> Any:
        return self._op


def build_hubbard_native(params: dict[str, Any]):
    """Return (full Trotter circuit, observable term sum, n_modes, n_sites)."""
    import ffsim
    from propaq.circuits.majorana.circuit import MajoranaCircuit
    from propaq.circuits.majorana.rotation import MajoranaRotation
    from propaq.datatypes import MajoranaTermSum
    from qiskit.quantum_info import SparsePauliOp

    nx, ny = params["nx"], params["ny"]
    t, U, dt, steps = params["t"], params["U"], params["dt"], params["steps"]
    n_sites = nx * ny
    n_qubits = 2 * n_sites
    n_modes = 4 * n_sites

    H = ffsim.fermi_hubbard_2d(nx, ny, tunneling=t, interaction=U)
    term_sum = MajoranaTermSum.from_ffsim(_FermionOpWrapper(H, norb=n_sites))

    items = sorted(term_sum.items(), key=lambda gc: (bin(gc[0].modes).count("1"), gc[0].modes))
    step_rotations = [MajoranaRotation(gen, 2.0 * coeff * dt) for gen, coeff in items]

    prep_rotations = []
    for site in range(0, n_sites, 2):
        for gen, angle in MajoranaTermSum.from_x(None, [site], n_modes).items():
            prep_rotations.append(MajoranaRotation(gen, angle))

    all_rotations = prep_rotations + step_rotations * steps
    circuit = MajoranaCircuit(all_rotations, n_modes)

    target_qubit = n_sites - 1
    label = ["I"] * n_qubits
    label[n_qubits - 1 - target_qubit] = "Z"
    obs_ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("".join(label)))

    return circuit, obs_ts, n_modes, n_sites


def propagate(source: dict[str, Any]) -> dict:
    from propaq import CoefficientTruncator, WeightTruncator
    from propaq.propagators import MajoranaPropagator
    import time

    if source["problem"] != "hubbard_trotter":
        raise ValueError(f"unknown fermionic problem for propaq_native: {source['problem']!r}")
    params = source["params"]

    t0 = time.perf_counter()
    circuit, obs_ts, n_modes, n_sites = build_hubbard_native(params)
    build_time_s = time.perf_counter() - t0

    truncators = [WeightTruncator(None), CoefficientTruncator(MIN_ABS_COEFF)]
    prop = MajoranaPropagator(truncation=truncators, n_threads=N_THREADS, progress_bar=False)
    t1 = time.perf_counter()
    res = prop.expectation_value(obs_ts, circuit, initial_state=0)
    wall_time_s = time.perf_counter() - t1
    n_terms_final = res.n_terms[-1] if res.n_terms else None

    return {
        "n_qubits": n_sites,
        "gate_count": len(circuit.rotations),
        "n_threads": N_THREADS,
        "wall_time_s": wall_time_s,
        "circuit_build_time_s": build_time_s,
        "expectation_value": float(res.expectation_value),
        "n_terms_final": int(n_terms_final) if n_terms_final is not None else None,
        "truncation_onenorm": None,
        "max_terms": None,
        "max_weight": None,
        "min_abs_coeff": MIN_ABS_COEFF,
    }


if __name__ == "__main__":
    run_on_saved_fermionic_circuits(HERE / "circuits_native", HERE, "propaq_native", propagate)
