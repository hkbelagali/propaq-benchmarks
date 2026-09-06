"""Compare truncated Heisenberg propagation with propaq's hybrid MPS method.

The circuit is bipartite: a six-layer nearest-neighbour TFIM preparation is
well suited to an MPS Schrödinger calculation, while a three-layer local TFIM
readout block is propagated backwards in the Heisenberg picture.  At each
Pauli-weight cutoff this benchmark records the coefficient L1 norm discarded
by propagation and the error relative to an exact state-vector reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_DIR))

from common import io_utils  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "propaq"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

N_THREADS = 64
os.environ["RAYON_NUM_THREADS"] = str(N_THREADS)
# quimb uses Numba's on-disk cache; the shared home cache is not writable on
# every benchmark host, so use a safe local cache unless the caller chose one.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "propaq-quimb-numba-cache"))

import numpy as np
from qiskit import QuantumCircuit
import qiskit.qasm2 as qasm2
from qiskit.quantum_info import SparsePauliOp, Statevector
import quimb.tensor as qtn

from propaq import Logger, PauliPropagator, PauliTermSum
from propaq.circuits import PauliCircuit
from propaq.hybrid import hybrid_expectation_value
from propaq.noise import TruncationPolicy


N_QUBITS = 12
MPS_LAYERS = 10
HEISENBERG_LAYERS = 4
WEIGHT_CUTOFFS = (3, 4, 5, 6)
MPS_MAX_BOND = 4
BENCHMARK_ID = "hybrid_mps_heisenberg_tfim_v3"


def tfim_block(n_layers: int, coupling_base: float) -> QuantumCircuit:
    """Build a shallow 1D TFIM block with nearest-neighbour entanglement only."""
    circuit = QuantumCircuit(N_QUBITS)
    for layer in range(n_layers):
        for qubit in range(N_QUBITS - 1):
            circuit.rzz(coupling_base + 0.03 * ((qubit + layer) % 7), qubit, qubit + 1)
        for qubit in range(N_QUBITS):
            circuit.rx(0.22 + 0.02 * ((qubit + layer) % 5), qubit)
    return circuit


def build_problem() -> tuple[QuantumCircuit, QuantumCircuit, PauliTermSum]:
    """Return (MPS preparation, Heisenberg readout, local observable)."""
    mps_preparation = tfim_block(MPS_LAYERS, 0.41)
    heisenberg_readout = tfim_block(HEISENBERG_LAYERS, 0.55)
    observable = PauliTermSum.from_sparse_pauli_op(
        SparsePauliOp("I" * (N_QUBITS // 2) + "Z" + "I" * (N_QUBITS // 2 - 1))
    )
    return mps_preparation, heisenberg_readout, observable


def wrap_term_sum(raw) -> PauliTermSum:
    """Restore the Python wrapper after Rust propagation returns its base type."""
    wrapped = PauliTermSum(dtype=raw.dtype)
    wrapped.merge(raw)
    return wrapped


def build_mps(circuit: QuantumCircuit, max_bond: int | None = None):
    """Simulate a 1D circuit with quimb and optionally truncate its MPS bond."""
    mps = qtn.CircuitMPS.from_openqasm2_str(qasm2.dumps(circuit)).psi
    if max_bond is not None:
        mps.compress(max_bond=max_bond)
    return mps


def hybrid_value(theta: PauliTermSum, mps) -> float:
    """Contract a propagated observable against quimb's MPS via propaq."""
    return hybrid_expectation_value(theta, mps, initial_state=0)


def discarded_l1(log_path: Path) -> float:
    """Sum propaq's per-truncation discarded coefficient L1 diagnostics."""
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    return float(
        sum(event.get("discarded_coeff_l1", 0.0) for event in events if event.get("event") == "truncation")
    )


def propagate_with_diagnostics(
    observable: PauliTermSum, circuit: QuantumCircuit, weight_cutoff: int
) -> tuple[PauliTermSum, float]:
    """Propagate while collecting the L1 norm of terms discarded by truncation."""
    with tempfile.TemporaryDirectory(prefix="propaq-hybrid-") as directory:
        log_path = Path(directory) / "propagation.jsonl"
        propagator = PauliPropagator(
            n_threads=N_THREADS,
            truncation=TruncationPolicy(weight_cutoff=weight_cutoff, coeff_cutoff=0.0),
            logger=Logger(str(log_path), log_every=1),
        )
        theta = propagator.propagate(observable, PauliCircuit.from_qiskit(circuit))
        return wrap_term_sum(theta), discarded_l1(log_path)


def run_benchmark() -> list[dict]:
    """Run both paths and verify that hybrid loses less L1 weight and is closer."""
    mps_preparation, heisenberg_readout, observable = build_problem()
    full_circuit = mps_preparation.compose(heisenberg_readout)
    true_value = float(Statevector(full_circuit).expectation_value(observable.to_sparse_pauli_op()).real)
    exact_preparation_mps = build_mps(mps_preparation)
    truncated_full_mps = build_mps(full_circuit, max_bond=MPS_MAX_BOND)
    mps_value = hybrid_expectation_value(observable, truncated_full_mps, initial_state=0)
    mps_error = abs(mps_value - true_value)

    # Verify the split itself before assessing truncated results.
    exact_theta = wrap_term_sum(
        PauliPropagator(n_threads=N_THREADS).propagate(
            observable, PauliCircuit.from_qiskit(heisenberg_readout)
        )
    )
    split_discrepancy = abs(hybrid_value(exact_theta, exact_preparation_mps) - true_value)
    # QASM conversion plus MPS contraction introduces a small floating-point
    # discrepancy for the deeper 12-qubit preparation, still far below every
    # truncation error reported by this benchmark.
    if split_discrepancy > 1e-5:
        raise RuntimeError(f"hybrid split disagrees with the exact reference by {split_discrepancy:.3e}")

    records = []
    for cutoff in WEIGHT_CUTOFFS:
        pure_theta, pure_discarded_l1 = propagate_with_diagnostics(observable, full_circuit, cutoff)
        hybrid_theta, hybrid_discarded_l1 = propagate_with_diagnostics(
            observable, heisenberg_readout, cutoff
        )
        pure_value = PauliPropagator(n_threads=N_THREADS).expectation_value(
            pure_theta, PauliCircuit([]), initial_state=0
        ).expectation_value
        hybrid_estimate = hybrid_value(hybrid_theta, exact_preparation_mps)
        pure_error = abs(pure_value - true_value)
        hybrid_error = abs(hybrid_estimate - true_value)
        if not hybrid_discarded_l1 < pure_discarded_l1:
            raise RuntimeError(f"cutoff {cutoff}: hybrid discarded L1 is not lower")
        if not hybrid_error < pure_error:
            raise RuntimeError(f"cutoff {cutoff}: hybrid error is not lower")
        if not hybrid_error < mps_error:
            raise RuntimeError(f"cutoff {cutoff}: hybrid error is not lower than truncated MPS")
        records.append(
            {
                "benchmark": BENCHMARK_ID,
                "n_qubits": N_QUBITS,
                "mps_layers": MPS_LAYERS,
                "heisenberg_layers": HEISENBERG_LAYERS,
                "weight_cutoff": cutoff,
                "n_threads": N_THREADS,
                "true_value": true_value,
                "pure_value": pure_value,
                "hybrid_value": hybrid_estimate,
                "pure_discarded_l1": pure_discarded_l1,
                "hybrid_discarded_l1": hybrid_discarded_l1,
                "pure_absolute_error": pure_error,
                "hybrid_absolute_error": hybrid_error,
                "mps_max_bond": MPS_MAX_BOND,
                "mps_value": mps_value,
                "mps_absolute_error": mps_error,
                "split_discrepancy": split_discrepancy,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=BENCH_DIR / "results" / "hybrid_mps_heisenberg.jsonl",
    )
    parser.add_argument(
        "--out", type=Path,
        default=BENCH_DIR / "results" / "hybrid_mps_heisenberg.npz",
    )
    args = parser.parse_args()
    records = run_benchmark()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for record in records:
        io_utils.append_jsonl(str(args.checkpoint), record)
        io_utils.save_records_npz(io_utils.load_jsonl(str(args.checkpoint)), str(args.out))
    for record in records:
        print(
            f"weight={record['weight_cutoff']}: discarded L1 "
            f"{record['pure_discarded_l1']:.3e} -> {record['hybrid_discarded_l1']:.3e}; "
            f"absolute error {record['pure_absolute_error']:.3e} -> {record['hybrid_absolute_error']:.3e}"
            f" -> MPS(chi<={MPS_MAX_BOND}) {record['mps_absolute_error']:.3e}"
        )
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
