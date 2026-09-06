#!/usr/bin/env python3
"""Runner: pyrauli backend (Pauli basis only).

Thread control is process-wide via OMP_NUM_THREADS (must be set in the environment before
the process starts, before pyrauli's first parallel region runs, see scaling/run_scaling.py),
plus a per-call `runtime=` choice (pyrauli.seq vs pyrauli.par) selected by --n-threads: 1 maps
to sequential, >1 maps to the OpenMP-parallel runtime.

IMPORTANT gotcha (found during IR validation): `pyrauli.from_qiskit` on a SparsePauliOp needs
`reverse=True` to match Qiskit's little-endian Pauli-label convention. pyrauli's own native
Observable string convention is big-endian (leftmost char = qubit 0). Without it, expectation
values come out silently wrong (verified: exactly 0 for entangled circuits, sign-flipped for
simple ones). The circuit converter (`from_qiskit` on a QuantumCircuit) needs no such flag.
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
    ap.add_argument("--weight-cutoff", type=int, default=None)
    ap.add_argument("--coeff-cutoff", type=float, default=1e-8)
    ap.add_argument("--keep-n", type=int, default=None)
    ap.add_argument("--n-threads", type=int, default=1)
    args = ap.parse_args()

    import pyrauli

    ir = ProblemIR.load(args.problem)
    qc = ir.to_qiskit()
    obs = ir.observable.to_sparse_pauli_op()

    truncators = []
    if args.coeff_cutoff is not None:
        truncators.append(pyrauli.CoefficientTruncator(args.coeff_cutoff))
    if args.weight_cutoff is not None:
        truncators.append(pyrauli.WeightTruncator(args.weight_cutoff))
    if args.keep_n is not None:
        truncators.append(pyrauli.KeepNTruncator(args.keep_n))
    truncator = pyrauli.MultiTruncator(truncators) if truncators else pyrauli.NeverTruncator()

    runtime = pyrauli.seq if args.n_threads <= 1 else pyrauli.par

    t0 = time.perf_counter()
    pcirc = pyrauli.from_qiskit(qc)
    pcirc.set_truncator(truncator)
    pcirc.set_truncate_policy(pyrauli.AlwaysAfterSplittingPolicy())
    build_time_s = time.perf_counter() - t0
    pobs = pyrauli.from_qiskit(obs, reverse=True)

    t1 = time.perf_counter()
    ev, err = pcirc.expectation_value(pobs, runtime=runtime)
    wall_time_s = time.perf_counter() - t1

    result = {
        "backend": "pyrauli",
        "basis": "pauli",
        "problem": ir.problem,
        "n_qubits": ir.n_qubits,
        "gate_count": ir.gate_count(),
        "params": ir.params,
        "n_threads": args.n_threads,
        "wall_time_s": wall_time_s,
        "circuit_build_time_s": build_time_s,
        "expectation_value": float(ev),
        "n_terms_final": None,
        "truncation_onenorm": float(err),
        "max_terms": args.keep_n,
        "max_weight": args.weight_cutoff,
        "min_abs_coeff": args.coeff_cutoff,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
