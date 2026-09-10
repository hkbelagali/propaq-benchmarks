#!/usr/bin/env python3
"""Compare truncated Heisenberg propagation with propaq's hybrid MPS method.

At each Pauli-weight cutoff this records the coefficient L1 norm discarded by propagation
and the error relative to an exact state-vector reference, for a pure full-depth Heisenberg
propagation, propaq's hybrid MPS-Heisenberg split, and a truncated-bond-dimension MPS-only
baseline.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks import io_utils  # noqa: E402
from propaq_benchmarks.circuit_ir import ProblemIR  # noqa: E402

PACKAGE_ROOT = HERE.parents[2] / "propaq"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

N_THREADS = 64
os.environ["RAYON_NUM_THREADS"] = str(N_THREADS)
# quimb uses Numba's on-disk cache; the shared home cache is not writable on
# every benchmark host, so use a safe local cache unless the caller chose one.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "propaq-quimb-numba-cache"))

warnings.simplefilter("ignore")

import qiskit.qasm2 as qasm2  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
import quimb.tensor as qtn  # noqa: E402

from propaq import Logger, PauliPropagator, PauliTermSum  # noqa: E402
from propaq.circuits import PauliCircuit  # noqa: E402
from propaq.hybrid import hybrid_expectation_value  # noqa: E402
from propaq.noise import TruncationPolicy  # noqa: E402

WEIGHT_CUTOFFS = (3, 4, 5, 6)
MPS_MAX_BOND = 4
BENCHMARK_ID = "hybrid_mps_heisenberg_tfim_v3"


def wrap_term_sum(raw) -> PauliTermSum:
    """Restore the Python wrapper after Rust propagation returns its base type."""
    wrapped = PauliTermSum(dtype=raw.dtype)
    wrapped.merge(raw)
    return wrapped


def build_mps(circuit, max_bond: int | None = None):
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


def propagate_with_diagnostics(observable: PauliTermSum, circuit, weight_cutoff: int) -> tuple[PauliTermSum, float]:
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


def run_benchmark(mps_preparation, heisenberg_readout, observable: PauliTermSum, n_qubits: int, mps_layers: int, heisenberg_layers: int, cutoffs: tuple[int, ...]) -> list[dict]:
    """Run every path and verify that hybrid loses less L1 weight and is closer."""
    full_circuit = mps_preparation.compose(heisenberg_readout)
    true_value = float(Statevector(full_circuit).expectation_value(observable.to_sparse_pauli_op()).real)
    exact_preparation_mps = build_mps(mps_preparation)
    truncated_full_mps = build_mps(full_circuit, max_bond=MPS_MAX_BOND)
    mps_value = hybrid_expectation_value(observable, truncated_full_mps, initial_state=0)
    mps_error = abs(mps_value - true_value)

    # Verify the split itself before assessing truncated results.
    exact_theta = wrap_term_sum(
        PauliPropagator(n_threads=N_THREADS).propagate(observable, PauliCircuit.from_qiskit(heisenberg_readout))
    )
    split_discrepancy = abs(hybrid_value(exact_theta, exact_preparation_mps) - true_value)
    # QASM conversion plus MPS contraction introduces a small floating-point
    # discrepancy for the deeper 12-qubit preparation, still far below every
    # truncation error reported by this benchmark.
    if split_discrepancy > 1e-5:
        raise RuntimeError(f"hybrid split disagrees with the exact reference by {split_discrepancy:.3e}")

    records = []
    for cutoff in cutoffs:
        pure_theta, pure_discarded_l1 = propagate_with_diagnostics(observable, full_circuit, cutoff)
        hybrid_theta, hybrid_discarded_l1 = propagate_with_diagnostics(observable, heisenberg_readout, cutoff)
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
                "n_qubits": n_qubits,
                "mps_layers": mps_layers,
                "heisenberg_layers": heisenberg_layers,
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
    circuits_dir = HERE / "circuits"
    checkpoint = HERE / "results_propaq.jsonl"
    npz_path = HERE / "results_propaq.npz"

    mps_ir = ProblemIR.load(str(circuits_dir / "mps_preparation.json"))
    heisenberg_ir = ProblemIR.load(str(circuits_dir / "heisenberg_readout.json"))
    mps_preparation = mps_ir.to_qiskit()
    heisenberg_readout = heisenberg_ir.to_qiskit()
    observable = PauliTermSum.from_sparse_pauli_op(mps_ir.observable.to_sparse_pauli_op())

    done = {
        r["weight_cutoff"] for r in io_utils.load_jsonl(str(checkpoint))
        if r.get("ok") and r.get("benchmark") == BENCHMARK_ID
    }
    pending = [c for c in WEIGHT_CUTOFFS if c not in done]
    if not pending:
        print("all weight cutoffs already checkpointed")
        return

    try:
        records = run_benchmark(
            mps_preparation, heisenberg_readout, observable,
            mps_ir.n_qubits, mps_ir.params["mps_layers"], mps_ir.params["heisenberg_layers"],
            tuple(pending),
        )
    except Exception as exc:  # noqa: BLE001 - a failed sweep should still leave a record behind
        record = {"ok": False, "error": str(exc), "benchmark": BENCHMARK_ID}
        io_utils.append_jsonl(str(checkpoint), record)
        io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))
        raise

    for record in records:
        record["ok"] = True
        record["label"] = f"weight{record['weight_cutoff']}"
        record["backend"] = "propaq"
        io_utils.append_jsonl(str(checkpoint), record)
        io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))
    for record in records:
        print(
            f"weight={record['weight_cutoff']}: discarded L1 "
            f"{record['pure_discarded_l1']:.3e} -> {record['hybrid_discarded_l1']:.3e}; "
            f"absolute error {record['pure_absolute_error']:.3e} -> {record['hybrid_absolute_error']:.3e}"
            f" -> MPS(chi<={MPS_MAX_BOND}) {record['mps_absolute_error']:.3e}"
        )
    print(f"Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
