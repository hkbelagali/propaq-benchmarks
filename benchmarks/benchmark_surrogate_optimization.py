"""Compare direct and compiled-surrogate QAOA optimization.

The workload is depth-two QAOA for MaxCut on a 12-vertex 3-regular graph.  It
is deliberately large enough that direct numerical propagation is substantial,
while the symbolic model remains practical to compile.  Both paths use the
same deterministic COBYLA optimization budget and starting point.

Run from the repository root:

    python bench/benchmark_surrogate_optimization.py
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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from propaq import (
    PauliPropagator,
    PauliSurrogatePropagator,
    PauliTermSum,
    SurrogatePauliCircuit,
    VariationalSurrogateModel,
)
from propaq.circuits import PauliCircuit


N_QUBITS = 12
N_LAYERS = 2
MAXITER = 1000
MINIMUM_SPEEDUP = 3.0
BENCHMARK_ID = "qaoa_maxcut_surrogate_optimization_v1"
# C_12 plus opposite vertices, giving every vertex degree three.
MAXCUT_EDGES = tuple((qubit, (qubit + 1) % N_QUBITS) for qubit in range(N_QUBITS)) + tuple(
    (qubit, qubit + N_QUBITS // 2) for qubit in range(N_QUBITS // 2)
)


def build_problem() -> tuple[QuantumCircuit, ParameterVector, PauliTermSum]:
    """Build depth-two QAOA and the MaxCut cost Hamiltonian."""
    parameters = ParameterVector("theta", 2 * N_LAYERS)
    circuit = QuantumCircuit(N_QUBITS)
    circuit.h(range(N_QUBITS))
    for layer in range(N_LAYERS):
        gamma, beta = parameters[2 * layer : 2 * layer + 2]
        for qubit_a, qubit_b in MAXCUT_EDGES:
            circuit.rzz(-gamma, qubit_a, qubit_b)
        for qubit in range(N_QUBITS):
            circuit.rx(2 * beta, qubit)

    terms = [("I" * N_QUBITS, len(MAXCUT_EDGES) / 2)]
    for qubit_a, qubit_b in MAXCUT_EDGES:
        label = ["I"] * N_QUBITS
        label[qubit_a] = label[qubit_b] = "Z"
        terms.append(("".join(label), -0.5))
    return circuit, parameters, PauliTermSum.from_sparse_pauli_op(SparsePauliOp.from_list(terms))


class _BudgetExhausted(Exception):
    """Raised inside the objective once the evaluation budget is spent."""


class _FixedBudgetResult:
    """The subset of an OptimizeResult this benchmark reads, for a budget-capped run."""

    def __init__(self, x: np.ndarray, fun: float, nfev: int) -> None:
        self.x = x
        self.fun = fun
        self.nfev = nfev


def optimize(objective, initial_point: np.ndarray, maxiter: int, tol: float):
    """Run a derivative-free optimization for exactly ``maxiter`` objective evaluations.

    COBYLA's own stopping rules cannot be relied on to spend the budget: at ``tol=1e-10``
    it stopped after 166 evaluations and at ``tol=0`` after 117, and in both cases after a
    *different* count for each of the two objectives, so the two timings covered different
    amounts of work and their ratio was not a like-for-like speedup. Rather than tune a
    tolerance until the counts happen to agree, the budget is enforced here: the objective
    is wrapped in a counter that aborts the solve the moment the quota is met. Each path
    therefore follows its own genuine COBYLA trajectory for an identical number of
    evaluations, which is what the timing comparison needs.
    """
    calls = 0
    best_x, best_fun = np.asarray(initial_point, dtype=float), np.inf

    def counted(values: np.ndarray) -> float:
        nonlocal calls, best_x, best_fun
        if calls >= maxiter:
            raise _BudgetExhausted
        calls += 1
        value = objective(values)
        if value < best_fun:
            best_fun, best_x = value, np.array(values, dtype=float)
        return value

    # COBYLA's trust region collapses after ~117 evaluations on this problem whatever the
    # tolerance, so spending the budget takes restarts, each resuming from the best point
    # found so far. The loop exits early only if a restart consumes no evaluations at all,
    # which would otherwise spin forever.
    while calls < maxiter:
        before = calls
        try:
            minimize(
                counted,
                best_x if np.isfinite(best_fun) else initial_point,
                method="COBYLA",
                options={"maxiter": maxiter - calls, "rhobeg": 0.4, "tol": tol},
            )
        except _BudgetExhausted:
            break
        if calls == before:
            break
    return _FixedBudgetResult(best_x, best_fun, calls)


def benchmark(maxiter: int, minimum_speedup: float, tol: float) -> dict:
    """Compile once, then compare complete numerical and surrogate optimizations."""
    circuit, parameters, observable = build_problem()
    initial_point = np.array([0.31, -0.22, 0.17, 0.28])
    parameter_list = list(parameters)

    # Build and validate the compiled objective before timing either optimizer.
    surrogate_circuit = SurrogatePauliCircuit.from_qiskit(circuit)
    start = time.perf_counter()
    raw_model = PauliSurrogatePropagator(n_threads=N_THREADS, progress_bar=True).build(
        observable, surrogate_circuit, initial_state=0
    )
    compilation_seconds = time.perf_counter() - start
    compiled_objective = VariationalSurrogateModel(
        raw_model, surrogate_circuit.parameter_sources, surrogate_circuit.qiskit_parameters
    )

    def numerical_objective(values: np.ndarray) -> float:
        bound = circuit.assign_parameters(dict(zip(parameter_list, values)))
        numeric_circuit = PauliCircuit.from_qiskit(bound)
        return PauliPropagator(n_threads=N_THREADS, progress_bar=False).expectation_value(
            observable, numeric_circuit, initial_state=0
        ).expectation_value

    # The two objectives must agree at the common starting point before running
    # an optimizer; this catches parameter-ordering or conversion errors.
    initial_discrepancy = abs(numerical_objective(initial_point) - compiled_objective.evaluate(initial_point))
    if initial_discrepancy > 1e-9:
        raise RuntimeError(f"compiled objective disagrees at the initial point by {initial_discrepancy:.3e}")

    start = time.perf_counter()
    numerical_result = optimize(numerical_objective, initial_point, maxiter, tol)
    numerical_seconds = time.perf_counter() - start

    start = time.perf_counter()
    compiled_result = optimize(compiled_objective.evaluate, initial_point, maxiter, tol)
    compiled_optimization_seconds = time.perf_counter() - start
    speedup = numerical_seconds / compiled_optimization_seconds

    # A speedup is only a like-for-like ratio if both paths did the same amount of work.
    if int(numerical_result.nfev) != int(compiled_result.nfev):
        raise RuntimeError(
            f"evaluation counts differ (numerical={numerical_result.nfev}, "
            f"compiled={compiled_result.nfev}); the timings are not comparable"
        )
    if speedup < minimum_speedup:
        raise RuntimeError(
            f"compiled optimization speedup was only {speedup:.2f}x; expected at least "
            f"{minimum_speedup:.2f}x"
        )

    final_discrepancy = abs(
        numerical_objective(compiled_result.x) - compiled_objective.evaluate(compiled_result.x)
    )
    if final_discrepancy > 1e-9:
        raise RuntimeError(f"compiled objective disagrees at the final point by {final_discrepancy:.3e}")

    return {
        "benchmark": BENCHMARK_ID,
        "ansatz": "depth-2 QAOA MaxCut on a 12-vertex 3-regular graph",
        "n_qubits": N_QUBITS,
        "layers": N_LAYERS,
        "n_parameters": len(parameters),
        "n_gates": len(circuit.data),
        "n_threads": N_THREADS,
        "optimizer": "COBYLA",
        "maxiter": maxiter,
        "tol": tol,
        "seconds_per_numerical_evaluation": numerical_seconds / int(numerical_result.nfev),
        "seconds_per_compiled_evaluation": compiled_optimization_seconds / int(compiled_result.nfev),
        "compilation_s": compilation_seconds,
        "numerical_optimization_s": numerical_seconds,
        "compiled_optimization_s": compiled_optimization_seconds,
        "compiled_total_s": compilation_seconds + compiled_optimization_seconds,
        "optimization_speedup": speedup,
        "numerical_nfev": int(numerical_result.nfev),
        "compiled_nfev": int(compiled_result.nfev),
        "numerical_final_value": float(numerical_result.fun),
        "compiled_final_value": float(compiled_result.fun),
        "initial_objective_discrepancy": initial_discrepancy,
        "final_objective_discrepancy": final_discrepancy,
        "compiled_terms": raw_model.n_terms,
        "compiled_monomials": raw_model.n_monomials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--minimum-speedup", type=float, default=MINIMUM_SPEEDUP)
    parser.add_argument("--tol", type=float, default=0.0,
                        help="COBYLA trust-region stopping radius. 0 (the default) disables "
                             "early convergence so both objectives run the full --maxiter "
                             "budget and their timings cover identical work")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BENCH_DIR / "results" / "surrogate_optimization.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BENCH_DIR / "results" / "surrogate_optimization.npz",
    )
    args = parser.parse_args()
    if args.maxiter < 2 or args.minimum_speedup <= 0 or args.tol < 0:
        parser.error("--maxiter must be at least 2, --minimum-speedup must be positive, "
                     "and --tol must be non-negative")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    record = benchmark(args.maxiter, args.minimum_speedup, args.tol)
    io_utils.append_jsonl(str(args.checkpoint), record)
    io_utils.save_records_npz(io_utils.load_jsonl(str(args.checkpoint)), str(args.out))
    print(
        f"numerical optimization={record['numerical_optimization_s']:.2f}s; "
        f"compiled optimization={record['compiled_optimization_s']:.2f}s; "
        f"speedup={record['optimization_speedup']:.1f}x"
    )
    print(
        f"compiled-object build={record['compilation_s']:.2f}s; "
        f"total compiled workflow={record['compiled_total_s']:.2f}s"
    )
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
