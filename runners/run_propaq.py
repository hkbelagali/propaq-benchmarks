#!/usr/bin/env python3
"""Runner: propaq backend, --basis pauli|majorana.

Pauli basis consumes the qubit-suite IR via PauliCircuit.from_qiskit (native rz/rx/ry, the
rest auto-transpiled into propaq's own basis). Majorana basis consumes the *same* qubit
circuit via MajoranaCircuit.from_qiskit's Jordan-Wigner mapping, giving a same-package,
same-input Pauli-vs-Majorana comparison point in addition to the cross-package one.

n_threads is a real constructor kwarg here (a dedicated per-instance Rayon thread pool),
so this runner is also used directly by the parallelism-scaling sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

# Before numpy loads, since OpenBLAS reads these at import and never re-reads them.
# qiskit pulls in two OpenBLAS copies, each of which starts one spinning worker per
# core (192 apiece on this host) and keeps them runnable long after the last BLAS
# call. propaq's engine pins one worker per partition to a core, so those spinners
# stop being harmless idle-core noise and start time-slicing against pinned workers.
# This measured 449ms against 184ms at 64 threads on 6x6 Ising-Trotter step 13. No backend
# here wants threaded BLAS, and every one of them declares its own thread count.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Benchmarks run whichever propaq is installed in the launching interpreter (propaq 0.1.3 from
# PyPI), not the sibling source checkout. Inserting that checkout on sys.path would shadow the
# installed wheel with an unbuilt source tree.
from propaq import CoefficientTruncator, WeightTruncator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.circuit_ir import ProblemIR  # noqa: E402

warnings.simplefilter("ignore")  # propaq's transpile-fallback UserWarnings go to stderr, keep logs clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--basis", choices=["pauli", "majorana"], default="pauli")
    ap.add_argument("--weight-cutoff", type=int, default=None)
    ap.add_argument("--coeff-cutoff", type=float, default=1e-8)
    ap.add_argument("--n-threads", type=int, default=64)  # None = system core count
    args = ap.parse_args()

    ir = ProblemIR.load(args.problem)
    qc = ir.to_qiskit()
    obs = ir.observable.to_sparse_pauli_op()
    truncators = [
        WeightTruncator(args.weight_cutoff),
        CoefficientTruncator(args.coeff_cutoff),
    ]

    if args.basis == "pauli":
        from propaq.circuits import PauliCircuit
        from propaq.datatypes import PauliTermSum
        from propaq.propagators import PauliPropagator

        t0 = time.perf_counter()
        circuit = PauliCircuit.from_qiskit(qc)
        build_time_s = time.perf_counter() - t0
        obs_ts = PauliTermSum.from_sparse_pauli_op(obs)
        # propaq 0.1.3 removed FlushSchedule. The engine folds duplicates on insert, so
        # there is no outbox to flush and the old merge_max_terms=1 setting has no analogue.
        prop = PauliPropagator(
            truncation=truncators,
            n_threads=args.n_threads, progress_bar=True)
        t1 = time.perf_counter()
        res = prop.expectation_value(obs_ts, circuit, initial_state=0)
        wall_time_s = time.perf_counter() - t1
        n_terms_final = res.n_terms[-1] if res.n_terms else None
    else:
        from propaq.circuits import MajoranaCircuit
        from propaq.datatypes import MajoranaTermSum
        from propaq.propagators import MajoranaPropagator

        t0 = time.perf_counter()
        circuit = MajoranaCircuit.from_qiskit(qc, n_modes=2 * ir.n_qubits)
        build_time_s = time.perf_counter() - t0
        obs_ts = MajoranaTermSum.from_sparse_pauli_op(obs)
        prop = MajoranaPropagator(truncation=truncators,
                                  n_threads=args.n_threads, progress_bar=False)
        t1 = time.perf_counter()
        res = prop.expectation_value(obs_ts, circuit, initial_state=0)
        wall_time_s = time.perf_counter() - t1
        n_terms_final = res.n_terms[-1] if res.n_terms else None

    result = {
        "backend": "propaq",
        "basis": args.basis,
        "problem": ir.problem,
        "n_qubits": ir.n_qubits,
        "gate_count": ir.gate_count(),
        "params": ir.params,
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
