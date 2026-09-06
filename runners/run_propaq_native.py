#!/usr/bin/env python3
"""Runner: propaq Majorana backend, fed a *native* fermionic circuit built directly from an
ffsim FermionOperator (via propaq's MajoranaTermSum.from_ffsim) instead of propaq's usual
MajoranaCircuit.from_qiskit path.

Why this exists: from_qiskit reinterprets an already-JW-mapped qubit gate sequence in Majorana
operators, which faithfully preserves the Jordan-Wigner string that gate sequence requires,
so it never actually gets the term-count locality advantage native Majorana propagation is
supposed to have (verified: propaq_pauli and propaq_majorana get bit-identical term counts on
the same qubit-suite circuit). Going through ffsim's FermionOperator instead skips the qubit/JW
embedding entirely. Majorana operators for different fermionic modes anticommute directly, so
hopping terms stay weight-2 and interaction terms weight-4 regardless of site distance, matching
MajoranaPropagation.jl's native construction (see runners/run_majorana_propagation.jl) far more
closely than the qubit-gate path did (verified: term-count gap shrank from ~7-9x to ~1.3-2x at
3x3 Hubbard).

Consumes the same fermionic-params JSON as run_majorana_propagation.jl (see
common/problems_fermionic.py), not the qubit-suite ProblemIR, so this is NOT interchangeable
with run_propaq.py --basis majorana on the same --problem path. It needs the *_native.json
sibling file instead. propaq_pauli has no equivalent native path (ffsim's FermionOperator only
converts to Majorana form here, deliberately not unified with propaq_pauli's qubit-gate
encoding, see the trotter/run_trotter_scan.py routing).

Only hubbard_trotter is implemented (mirrors run_majorana_propagation.jl's problem dispatch).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import ffsim
from qiskit.quantum_info import SparsePauliOp

# Benchmarks run the installed propaq (0.1.3 from PyPI), not the sibling source checkout,
# same treatment as runners/run_propaq.py.
from propaq import CoefficientTruncator, WeightTruncator
from propaq.circuits.majorana.circuit import MajoranaCircuit
from propaq.circuits.majorana.rotation import MajoranaRotation
from propaq.datatypes import MajoranaTermSum
from propaq.propagators import MajoranaPropagator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

warnings.simplefilter("ignore")


class _FermionOpWrapper:
    """MajoranaTermSum.from_ffsim expects an object satisfying ffsim's SupportsFermionOperator
    protocol (a `_fermion_operator_()` method). ffsim.fermi_hubbard_2d returns a bare
    FermionOperator, which doesn't implement that protocol on itself. This shim bridges the two
    so from_ffsim's default `n_modes = 4 * norb` still works."""

    def __init__(self, op: "ffsim.FermionOperator", norb: int):
        self._op = op
        self.norb = norb

    def _fermion_operator_(self) -> "ffsim.FermionOperator":
        return self._op


def build_hubbard_native(params: dict[str, Any]) -> tuple[MajoranaCircuit, MajoranaTermSum, int, int]:
    """Returns (full Trotter circuit, observable term sum, n_modes, n_sites)."""
    nx, ny = params["nx"], params["ny"]
    t, U, dt, steps = params["t"], params["U"], params["dt"], params["steps"]
    n_sites = nx * ny
    n_qubits = 2 * n_sites
    n_modes = 4 * n_sites

    H = ffsim.fermi_hubbard_2d(nx, ny, tunneling=t, interaction=U)
    term_sum = MajoranaTermSum.from_ffsim(_FermionOpWrapper(H, norb=n_sites))

    # G = exp(-i*theta*M/2) (MajoranaRotation's convention) for a Trotter step of
    # exp(-i * coeff * M * dt) -> theta = 2*coeff*dt.
    #
    # term_sum.items() has no defined order, confirmed empirically to vary run-to-run (a
    # fresh Rust-side HashMap seed per process). Since the individual Hamiltonian terms
    # (hopping vs. on-site interaction) generally don't commute, applying them as first-order
    # Trotter rotations in a different order each run is a truly different circuit, not
    # floating-point noise. It was giving a different, non-reproducible Trotterization on
    # every single invocation (term counts varying up to 15x, expectation value drifting by
    # up to ~1e-3). Sorting into a fixed order makes this deterministic and reproducible.
    #
    # The specific order also matters for how many terms survive propagation, not just for
    # reproducibility. Sorting by ascending Majorana weight (2-operator hopping terms applied
    # before 4-operator on-site-interaction terms) measured 65 final terms at 3x3/step=1 vs.
    # 130 for an arbitrary bitmask sort and 187 for descending weight, letting the cheaper,
    # lower-weight generators merge/cancel before the circuit gets more entangled beats
    # MajoranaPropagation.jl's own native construction (76 terms) at this size.
    items = sorted(term_sum.items(), key=lambda gc: (bin(gc[0].modes).count("1"), gc[0].modes))
    step_rotations = [MajoranaRotation(gen, 2.0 * coeff * dt) for gen, coeff in items]

    # Checkerboard initial occupation on spin-up sites only, matching problems_qubit.py's
    # `for site in range(0, n_sites, 2): qc.x(up(site))`. This reuses propaq's own from_x
    # conversion (JW-string form for a single-qubit X) rather than hand-deriving it, and
    # ffsim's mode order here (spin*norb+orb, spin=0=up) makes up(site) == site directly.
    prep_rotations = []
    for site in range(0, n_sites, 2):
        for gen, angle in MajoranaTermSum.from_x(None, [site], n_modes).items():
            prep_rotations.append(MajoranaRotation(gen, angle))

    all_rotations = prep_rotations + step_rotations * steps
    circuit = MajoranaCircuit(all_rotations, n_modes)

    # Observable: Z on the same physical qubit as the qubit-suite convention (spin-up, last
    # site), built via a SparsePauliOp + the already-validated from_sparse_pauli_op path
    # rather than hand-rolling the Majorana form.
    target_qubit = n_sites - 1
    label = ["I"] * n_qubits
    label[n_qubits - 1 - target_qubit] = "Z"
    obs_ts = MajoranaTermSum.from_sparse_pauli_op(SparsePauliOp("".join(label)))

    return circuit, obs_ts, n_modes, n_sites


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--weight-cutoff", type=int, default=None)
    ap.add_argument("--coeff-cutoff", type=float, default=1e-8)
    ap.add_argument("--n-threads", type=int, default=64)
    args = ap.parse_args()

    with open(args.problem) as f:
        d = json.load(f)
    problem_name = d["problem"]
    params = d["params"]
    if problem_name != "hubbard_trotter":
        raise ValueError(f"unknown fermionic problem for propaq-native: {problem_name!r}")

    t0 = time.perf_counter()
    circuit, obs_ts, n_modes, n_sites = build_hubbard_native(params)
    build_time_s = time.perf_counter() - t0

    truncators = [WeightTruncator(args.weight_cutoff), CoefficientTruncator(args.coeff_cutoff)]
    # FlushSchedule is gone in propaq 0.1.3, see runners/run_propaq.py.
    prop = MajoranaPropagator(truncation=truncators,
                              n_threads=args.n_threads, progress_bar=False)
    t1 = time.perf_counter()
    res = prop.expectation_value(obs_ts, circuit, initial_state=0)
    wall_time_s = time.perf_counter() - t1
    n_terms_final = res.n_terms[-1] if res.n_terms else None

    result = {
        "backend": "propaq",
        "basis": "majorana",
        "problem": problem_name,
        "n_qubits": n_sites,  # matches run_majorana_propagation.jl's convention (n_sites, not n_modes)
        "gate_count": len(circuit.rotations),
        "params": params,
        "n_threads": args.n_threads,
        "wall_time_s": wall_time_s,
        "circuit_build_time_s": build_time_s,
        "expectation_value": float(res.expectation_value),
        "n_terms_final": int(n_terms_final) if n_terms_final is not None else None,
        "truncation_onenorm": None,
        "max_terms": None,
        "max_weight": args.weight_cutoff,
        "min_abs_coeff": args.coeff_cutoff,
        # Sparse-backend storage accounting: resident key bytes at the end of the
        # run, and the peak temporary dense workspace held live during it. The
        # resident metric deliberately excludes the workspace.
        # What was asked for, and what actually ran. They differ when the
        # requested engine declines (unsupported width, f32 storage) and falls
        # back, which would otherwise be invisible in the results.
        "engine_requested": os.environ.get("PROPAQ_ENGINE", "soa"),
        "engine": getattr(res, "engine", "soa"),
        "sparse_key_bytes": int(getattr(res, "sparse_key_bytes", 0)),
        "workspace_peak_bytes": int(getattr(res, "workspace_peak_bytes", 0)),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
