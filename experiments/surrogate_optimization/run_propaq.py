#!/usr/bin/env python3
"""Compare direct and compiled-surrogate QAOA optimization.

The workload is depth-two QAOA for MaxCut on a 12-vertex 3-regular graph. It is
deliberately large enough that direct numerical propagation is substantial, while the
symbolic model remains practical to compile. Both paths use the same deterministic COBYLA
optimization budget and starting point.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import io_utils  # noqa: E402

N_THREADS = 64
os.environ["RAYON_NUM_THREADS"] = str(N_THREADS)

import numpy as np  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import ParameterVector  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from propaq import (  # noqa: E402
    PauliPropagator,
    PauliSurrogatePropagator,
    PauliTermSum,
    SurrogatePauliCircuit,
    VariationalSurrogateModel,
)
from propaq.circuits import PauliCircuit  # noqa: E402

MAXITER = 1000
MINIMUM_SPEEDUP = 3.0
TOL = 0.0
BENCHMARK_ID = "qaoa_maxcut_surrogate_optimization_v1"


def build_problem(n_qubits: int, n_layers: int, maxcut_edges: list, initial_point: list):
    """Build depth-two QAOA and the MaxCut cost Hamiltonian."""
    parameters = ParameterVector("theta", 2 * n_layers)
    circuit = QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits))
    for layer in range(n_layers):
        gamma, beta = parameters[2 * layer : 2 * layer + 2]
        for qubit_a, qubit_b in maxcut_edges:
            circuit.rzz(-gamma, qubit_a, qubit_b)
        for qubit in range(n_qubits):
            circuit.rx(2 * beta, qubit)

    terms = [("I" * n_qubits, len(maxcut_edges) / 2)]
    for qubit_a, qubit_b in maxcut_edges:
        label = ["I"] * n_qubits
        label[qubit_a] = label[qubit_b] = "Z"
        terms.append(("".join(label), -0.5))
    observable = PauliTermSum.from_sparse_pauli_op(SparsePauliOp.from_list(terms))
    return circuit, parameters, observable


class _BudgetExhausted(Exception):
    """Raised inside the objective once the evaluation budget is spent."""


class _FixedBudgetResult:
    """The subset of an OptimizeResult this benchmark reads, for a budget-capped run."""

    def __init__(self, x: np.ndarray, fun: float, nfev: int) -> None:
        self.x = x
        self.fun = fun
        self.nfev = nfev


def optimize(objective, initial_point: np.ndarray, maxiter: int, tol: float):
    """Run a derivative-free optimization for exactly maxiter objective evaluations.

    COBYLA's own stopping rules cannot be relied on to spend the budget, so the budget is
    enforced here instead: the objective is wrapped in a counter that aborts the solve the
    moment the quota is met, so each path follows its own genuine COBYLA trajectory for an
    identical number of evaluations.
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

    # COBYLA's trust region collapses after roughly 117 evaluations on this problem
    # whatever the tolerance, so spending the budget takes restarts, each resuming from
    # the best point found so far.
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


def benchmark(spec: dict, maxiter: int, minimum_speedup: float, tol: float) -> dict:
    """Compile once, then compare complete numerical and surrogate optimizations."""
    circuit, parameters, observable = build_problem(
        spec["n_qubits"], spec["n_layers"], [tuple(e) for e in spec["maxcut_edges"]], spec["initial_point"],
    )
    initial_point = np.array(spec["initial_point"])
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
        "n_qubits": spec["n_qubits"],
        "layers": spec["n_layers"],
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
    circuits_dir = HERE / "circuits"
    checkpoint = HERE / "results_propaq.jsonl"
    npz_path = HERE / "results_propaq.npz"

    spec_path = circuits_dir / "qaoa_maxcut.json"
    if not spec_path.exists():
        raise SystemExit(f"no saved circuit spec at {spec_path}, run generate_circuits.py first")
    with open(spec_path) as f:
        spec = json.load(f)

    done = {
        r.get("maxiter") for r in io_utils.load_jsonl(str(checkpoint))
        if r.get("ok") and r.get("benchmark") == BENCHMARK_ID
    }
    if MAXITER in done:
        print(f"maxiter={MAXITER}: skip (already checkpointed)")
        return

    record = benchmark(spec, MAXITER, MINIMUM_SPEEDUP, TOL)
    record["ok"] = True
    record["label"] = "qaoa_maxcut"
    record["backend"] = "propaq"
    io_utils.append_jsonl(str(checkpoint), record)
    io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))
    print(
        f"numerical optimization={record['numerical_optimization_s']:.2f}s; "
        f"compiled optimization={record['compiled_optimization_s']:.2f}s; "
        f"speedup={record['optimization_speedup']:.1f}x"
    )
    print(
        f"compiled-object build={record['compilation_s']:.2f}s; "
        f"total compiled workflow={record['compiled_total_s']:.2f}s"
    )
    print(f"Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
