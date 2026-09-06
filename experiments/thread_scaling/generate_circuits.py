#!/usr/bin/env python3
"""Build and save the fixed circuit instances the thread-count sweep runs at every thread count.

Run this once before any run_<backend> file in this folder.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import problems_fermionic, problems_qubit  # noqa: E402


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    native_dir = HERE / "circuits_native"
    native_dir.mkdir(exist_ok=True)

    ir = problems_qubit.random_circuit_problem(n_qubits=10, depth=60, seed=0)
    ir.save(str(circuits_dir / "random_circuit.json"))
    print(f"saved circuits/random_circuit.json ({ir.n_qubits} qubits, {ir.gate_count()} gates)")

    hub_ir = problems_qubit.hubbard_trotter_problem(nx=5, ny=5, steps=4)
    hub_ir.save(str(circuits_dir / "hubbard_qubit.json"))
    print(f"saved circuits/hubbard_qubit.json ({hub_ir.n_qubits} qubits, {hub_ir.gate_count()} gates)")

    hub_native = problems_fermionic.hubbard_trotter_fermionic(nx=5, ny=5, steps=4)
    hub_native.save(str(native_dir / "hubbard_native.json"))
    print("saved circuits_native/hubbard_native.json")


if __name__ == "__main__":
    main()
