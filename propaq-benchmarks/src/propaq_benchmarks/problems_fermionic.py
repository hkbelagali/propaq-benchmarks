"""Native-fermionic problem *parameters* for MajoranaPropagation.jl.

MajoranaPropagation.jl has no Qiskit bridge, so it cannot consume the qubit-suite JSON IR
(circuit_ir.py). It builds its own circuits directly from physical parameters using its
native fermionic gate builders (see experiments/hubbard_trotter/run_majorana_propagation_jl.jl and experiments/random_fermionic_circuit/run_majorana_propagation_jl.jl). These two functions
just record the *same* physical parameters used by the matching qubit-suite problems
(problems_qubit.hubbard_trotter_problem / random_fermionic_circuit_problem) into a small JSON
file, so the two construction paths ("qiskit gates + JW, for propaq-Majorana" vs "native
fermionic gates, for MajoranaPropagation.jl") describe the same model/instance even though the
resulting term-by-term trajectories differ.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class FermionicProblem:
    problem: str
    params: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"problem": self.problem, "params": self.params}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f)


def hubbard_trotter_fermionic(
    nx: int = 3, ny: int = 3, t: float = 1.0, U: float = 2.0, dt: float = 0.1, steps: int = 2,
) -> FermionicProblem:
    return FermionicProblem(
        "hubbard_trotter",
        {"nx": nx, "ny": ny, "t": t, "U": U, "dt": dt, "steps": steps, "n_sites": nx * ny},
    )


def random_fermionic_circuit_fermionic(
    n_modes: int = 12, n_gates: int = 40, seed: int = 0, max_weight_per_gate: int = 4,
) -> FermionicProblem:
    return FermionicProblem(
        "random_fermionic_circuit",
        {"n_modes": n_modes, "n_gates": n_gates, "seed": seed, "max_weight_per_gate": max_weight_per_gate},
    )
