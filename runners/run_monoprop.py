#!/usr/bin/env python3
"""Runner for MonoProp's Pauli propagator.

MonoProp's Qiskit adapter accepts PauliEvolution gates and Pauli rotations, but not the
benchmark IR's Clifford ``h``/``cx`` instructions. The runner therefore decomposes the
reconstructed Qiskit circuit exactly into {rx, ry, rz, rxx}, all accepted by MonoProp.

Install MonoProp in the benchmark Python environment first, for example:
  python -m pip install -e ../../monoprop
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from qiskit import transpile

try:
    from monoprop import PauliPropagator
    from monoprop.qiskit_conversion import from_qiskit_circuit, from_qiskit_operator
except ImportError as exc:  # pragma: no cover - optional local dependency
    raise SystemExit(
        "MonoProp is not importable in this Python environment. Install the sibling checkout "
        "with `python -m pip install -e ../../monoprop` before running this backend."
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.circuit_ir import ProblemIR  # noqa: E402


# Fallback for logical circuits containing instructions MonoProp's Qiskit adapter does not
# accept directly (such as h/cx). Do not apply it to a circuit that is already supported:
# coefficient truncation happens after every rotation, so an otherwise exact decomposition
# changes the approximation trajectory and must not be used for comparable Ising results.
MONOPROP_BASIS = ["rx", "ry", "rz", "rxx"]


def occupied_qubits(bitstring: int, n_qubits: int) -> list[int]:
    """Decode ProblemIR's little-endian computational-basis integer."""
    return [q for q in range(n_qubits) if bitstring & (1 << q)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True)
    ap.add_argument("--coeff-cutoff", type=float, default=1e-8)
    # Kept for the common runner interface. MonoProp's serial Python frontend has no matching
    # per-instance thread-pool option; MPI/partition configuration belongs to MonoProp itself.
    ap.add_argument("--n-threads", type=int, default=1)
    args = ap.parse_args()
    # MonoProp reads this when it constructs the C++ propagator. ``auto`` partitioning then
    # uses one partition per physical core, capped here to the suite-wide thread request.
    os.environ["monoprop_NUM_THREADS"] = str(args.n_threads)

    ir = ProblemIR.load(args.problem)
    initial_state = occupied_qubits(ir.initial_state, ir.n_qubits)

    t0 = time.perf_counter()
    qc = ir.to_qiskit()
    try:
        circuit = from_qiskit_circuit(qc, initial_state)
        conversion_mode = "direct"
    except ValueError as direct_error:
        # The fallback is an exact unitary decomposition, but the result is only comparable
        # under a nonzero coefficient cutoff to other backends using the same decomposition.
        # Record that choice so it cannot be mistaken for direct shared-IR propagation.
        try:
            qc = transpile(qc, basis_gates=MONOPROP_BASIS, optimization_level=0)
            circuit = from_qiskit_circuit(qc, initial_state)
        except ValueError:
            raise direct_error
        conversion_mode = "decomposed"
    observable = from_qiskit_operator(ir.observable.to_sparse_pauli_op())
    build_time_s = time.perf_counter() - t0

    # A cutoff equal to register width is structurally exact. Only the common coefficient
    # threshold truncates, matching the other Trotter runners' no-weight-cutoff setup.
    prop = PauliPropagator(
        observable, initial_state, cutoff=ir.n_qubits, lower_atol=args.coeff_cutoff,
    )
    t1 = time.perf_counter()
    prop.propagate(circuit)
    wall_time_s = time.perf_counter() - t1

    print(json.dumps({
        "backend": "monoprop",
        "basis": "pauli",
        "problem": ir.problem,
        "n_qubits": ir.n_qubits,
        "gate_count": ir.gate_count(),
        "converted_gate_count": len(qc.data),
        "conversion_mode": conversion_mode,
        "params": ir.params,
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
