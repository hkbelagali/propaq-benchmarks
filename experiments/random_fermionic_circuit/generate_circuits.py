#!/usr/bin/env python3
"""Build and save every random_fermionic_circuit instance this experiment compares backends on.

Run this once (or again after changing SIZES below) before any run_<backend> file in this
folder. Every run_<backend> file then loads whatever is saved in circuits/ (or
circuits_native/ for the native Majorana Julia backend) and does not build circuits itself.

Each size builds both the qubit-suite circuit (Jordan-Wigner mapped, saved to circuits/) and
the native-fermionic circuit (saved to circuits_native/) under the same label, so the two
trees describe the same nominal problem size even though the two saved circuits are not
bit-identical (different RNG, native fermionic gates vs JW-mapped qubit gates, by design, see
common/problems_fermionic.py's module docstring).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import problems_fermionic, problems_qubit  # noqa: E402

# (n_modes, n_gates) pairs, shared by both the qubit-suite and native-fermionic builders.
SIZES = [(8, 20), (14, 40), (18, 60), (20, 80)]


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_native_dir = HERE / "circuits_native"
    circuits_dir.mkdir(exist_ok=True)
    circuits_native_dir.mkdir(exist_ok=True)
    for n_modes, n_gates in SIZES:
        label = f"n_modes{n_modes}_n_gates{n_gates}"

        ir = problems_qubit.random_fermionic_circuit_problem(n_modes, n_gates, seed=0)
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")

        fp = problems_fermionic.random_fermionic_circuit_fermionic(n_modes, n_gates, seed=0)
        native_path = circuits_native_dir / f"{label}.json"
        fp.save(str(native_path))
        print(f"saved {native_path} ({n_modes} modes, {n_gates} gates)")


if __name__ == "__main__":
    main()
