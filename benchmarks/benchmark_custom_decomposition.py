"""Compare generic and custom non-Clifford exchange-gate decompositions versus depth.

The workload is a 10-qubit brickwork non-Clifford exchange ansatz interleaved
with RX rotations. Qiskit's generic unitary fallback expands every exchange
gate into many native rotations; the registered XXPlusYY decomposition avoids
that expansion while preserving the exact unitary.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_DIR))

from common import io_utils  # noqa: E402

N_THREADS = 64
os.environ["RAYON_NUM_THREADS"] = str(N_THREADS)

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, XXPlusYYGate
from qiskit.quantum_info import SparsePauliOp

from propaq import PauliPropagator, PauliTermSum
from propaq.circuits import PauliCircuit, _registry, register_qiskit_gate

N_QUBITS = 10
DEFAULT_LAYERS = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
DEFAULT_RUNS = 1
EXCHANGE_ANGLE = 0.73
EXCHANGE_GATE = UnitaryGate(XXPlusYYGate(EXCHANGE_ANGLE, 0.0).to_matrix(), label="exchange")
BENCHMARK_ID = "exchange_custom_decomposition_v2_10q"


def exchange_terms(instr, q_indices, width, rep):
    """Direct native representation of the fixed non-Clifford exchange unitary."""
    del instr
    return [rep.xx_plus_yy_terms(EXCHANGE_ANGLE, q_indices[0], q_indices[1], width)]


def build_circuit(n_layers: int) -> QuantumCircuit:
    """Build a 10-qubit connected brickwork ansatz with non-Clifford exchange gates."""
    circuit = QuantumCircuit(N_QUBITS)
    for layer in range(n_layers):
        # Alternate two periodic matchings; each qubit interacts once per layer
        # and with both neighbours over successive layers.
        start = layer % 2
        for qubit in range(start, N_QUBITS + start, 2):
            circuit.append(EXCHANGE_GATE, [qubit % N_QUBITS, (qubit + 1) % N_QUBITS])
        for qubit in range(N_QUBITS):
            circuit.rx(0.37 + 0.03 * ((layer + qubit) % 5), qubit)
    return circuit


def observable() -> PauliTermSum:
    """A noncommuting eight-term observable that sustains a substantial workload."""
    labels = (
        "XZXZXZXZXZ", "ZXZXZXZXZX", "XYZZYXXZYZ", "YXXZYZZXYX",
        "ZYYXXZXYZY", "XYYZZXXYZZ", "YYXZZYXXZY", "ZXXYYZZXYX",
    )
    return PauliTermSum.from_sparse_pauli_op(
        SparsePauliOp.from_list([(label, 1.0 / (index + 1)) for index, label in enumerate(labels)])
    )


def benchmark_depth(n_layers: int, runs: int) -> dict:
    circuit = build_circuit(n_layers)
    obs = observable()

    # Generic Qiskit-transpilation fallback is measured before registry installation.
    _registry._QISKIT_REGISTRY.pop("unitary", None)
    _registry._VALIDATED = {key for key in _registry._VALIDATED if key[0] != "unitary"}
    fallback_circuit = PauliCircuit.from_qiskit(circuit)
    register_qiskit_gate("unitary", exchange_terms, validate=True)
    custom_circuit = PauliCircuit.from_qiskit(circuit)

    # Registration validates its first dispatch against propaq's fallback, outside timing.
    if len(custom_circuit.rotations) >= len(fallback_circuit.rotations):
        raise RuntimeError("custom exchange decomposition did not reduce rotation count")

    PauliPropagator(n_threads=N_THREADS).expectation_value(obs, fallback_circuit, initial_state=0)
    PauliPropagator(n_threads=N_THREADS).expectation_value(obs, custom_circuit, initial_state=0)
    fallback_prop = PauliPropagator(n_threads=N_THREADS, progress_bar=True)
    custom_prop = PauliPropagator(n_threads=N_THREADS, progress_bar=True)

    fallback_times, custom_times = [], []
    # Internal propaq bars expose live term growth during every timed propagation.
    for _ in range(runs):
        start = time.perf_counter()
        fallback_prop.expectation_value(obs, fallback_circuit, initial_state=0)
        fallback_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        custom_prop.expectation_value(obs, custom_circuit, initial_state=0)
        custom_times.append(time.perf_counter() - start)

    fallback_energy = fallback_prop.expectation_value(obs, fallback_circuit, initial_state=0).expectation_value
    custom_energy = custom_prop.expectation_value(obs, custom_circuit, initial_state=0).expectation_value
    agreement = abs(fallback_energy - custom_energy)
    if agreement > 1e-9:
        raise RuntimeError(f"decompositions disagree by {agreement:.3e}")
    return {
        "benchmark": BENCHMARK_ID,
        "layers": n_layers,
        "n_qubits": N_QUBITS,
        "n_threads": N_THREADS,
        "runs": runs,
        "fallback_rotations": len(fallback_circuit.rotations),
        "custom_rotations": len(custom_circuit.rotations),
        "fallback_mean_s": sum(fallback_times) / runs,
        "custom_mean_s": sum(custom_times) / runs,
        "energy_discrepancy": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=BENCH_DIR / "results" / "custom_decomposition.jsonl",
    )
    parser.add_argument(
        "--out", type=Path,
        default=BENCH_DIR / "results" / "custom_decomposition.npz",
    )
    args = parser.parse_args()
    if any(depth < 1 for depth in args.layers) or args.runs < 1:
        parser.error("--layers and --runs must be positive")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    records = io_utils.load_jsonl(str(args.checkpoint))
    completed = {
        record["layers"] for record in records
        if record.get("benchmark") == BENCHMARK_ID and record.get("runs") == args.runs
    }
    print(f"Non-Clifford exchange custom-decomposition sweep: {N_QUBITS} qubits, {N_THREADS} threads")
    for depth in args.layers:
        if depth in completed:
            print(f"layers={depth}: skip (already checkpointed)")
            continue
        record = benchmark_depth(depth, args.runs)
        io_utils.append_jsonl(str(args.checkpoint), record)
        io_utils.save_records_npz(io_utils.load_jsonl(str(args.checkpoint)), str(args.out))
        speedup = record["fallback_mean_s"] / record["custom_mean_s"]
        print(f"layers={depth}: fallback={record['fallback_mean_s']:.3f}s, custom={record['custom_mean_s']:.3f}s, {speedup:.1f}x")


if __name__ == "__main__":
    main()
