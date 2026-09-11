"""
Fermionic problem parameters for MajoranaPropagation.jl
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
