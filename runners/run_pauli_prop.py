#!/usr/bin/env python3
"""Runner: pauli-prop backend (Pauli basis only, confirmed single-threaded, no thread control).

Prints one JSON line to stdout: the standard result record consumed by orchestrate.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.circuit_ir import ProblemIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--max-terms", type=int, default=200_000)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--n-threads", type=int, default=1)  # accepted for CLI uniformity, unused (single-threaded)
    args = ap.parse_args()

    import pauli_prop

    ir = ProblemIR.load(args.problem)
    qc = ir.to_qiskit()
    obs = ir.observable.to_sparse_pauli_op()

    t0 = time.perf_counter()
    cliff, residual = pauli_prop.evolve_through_cliffords(qc)
    evolved, one_norm = pauli_prop.propagate_through_circuit(
        obs, residual, max_terms=args.max_terms, atol=args.atol, frame="h"
    )
    evolved.paulis = evolved.paulis.evolve(cliff, frame="h")
    expval = float(evolved.coeffs[~evolved.paulis.x.any(axis=1)].sum())
    wall_time_s = time.perf_counter() - t0

    result = {
        "backend": "pauli_prop",
        "basis": "pauli",
        "problem": ir.problem,
        "n_qubits": ir.n_qubits,
        "gate_count": ir.gate_count(),
        "params": ir.params,
        "n_threads": 1,
        "wall_time_s": wall_time_s,
        "expectation_value": expval,
        "n_terms_final": int(len(evolved)),
        "truncation_onenorm": float(one_norm),
        "max_terms": args.max_terms,
        "max_weight": None,
        "min_abs_coeff": args.atol,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
