#!/usr/bin/env python3
"""Build and save every hubbard_trotter circuit instance this experiment compares backends on.

Run this once (or again after changing SIZES below) before any run_<backend> file in this
folder. Every run_<backend> file then loads whatever is saved in circuits/ or circuits_native/
and does not build circuits itself.

This experiment has two independently-constructed sides that describe the same nominal
(nx, ny, steps) instance without being bit-identical circuits. circuits/ holds the qubit-suite
Jordan-Wigner-mapped ProblemIR (common.problems_qubit.hubbard_trotter_problem), consumed by the
5 qubit-side run_<backend> files. circuits_native/ holds the much simpler native-fermionic
FermionicProblem (common.problems_fermionic.hubbard_trotter_fermionic, just a params dict with
no gate list or observable), consumed by the 3 native-side run_<backend> files that build their
own circuit directly from physical parameters.

circuits_native/ additionally gets a fine Trotter-step curve, one saved circuit per step count
from 1 to 25 at a fixed 3x3 lattice, native-only. Only the 3 native-fermionic backends
(majorana_propagation_jl, propaq_native, monoprop_native) can reach these deeper steps, since
hubbard_trotter's term count blows up several steps sooner through the qubit/Jordan-Wigner path
than through the native fermionic one, so the qubit-side backends stay on the small SIZES list
above instead. plotting/plot_trotter_scan.py and plot_trotter_memory.py plot runtime and peak
RSS against this fine curve.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import problems_fermionic, problems_qubit  # noqa: E402

# (nx, ny, steps): shared across both sides so each pair names the same nominal instance
# even though the two circuits are built along entirely different paths.
SIZES = [(2, 2, 1), (3, 3, 2), (4, 4, 3), (5, 5, 4)]

FINE_CURVE_NX, FINE_CURVE_NY = 3, 3
FINE_CURVE_STEPS = range(1, 26)


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    circuits_native_dir = HERE / "circuits_native"
    circuits_native_dir.mkdir(exist_ok=True)

    for nx, ny, steps in SIZES:
        label = f"{nx}x{ny}_steps{steps}"

        ir = problems_qubit.hubbard_trotter_problem(nx=nx, ny=ny, steps=steps)
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")

        fp = problems_fermionic.hubbard_trotter_fermionic(nx=nx, ny=ny, steps=steps)
        native_path = circuits_native_dir / f"{label}.json"
        fp.save(str(native_path))
        print(f"saved {native_path} (nx={nx}, ny={ny}, steps={steps})")

    for steps in FINE_CURVE_STEPS:
        label = f"{FINE_CURVE_NX}x{FINE_CURVE_NY}_steps{steps}"
        fp = problems_fermionic.hubbard_trotter_fermionic(nx=FINE_CURVE_NX, ny=FINE_CURVE_NY, steps=steps)
        native_path = circuits_native_dir / f"{label}.json"
        fp.save(str(native_path))
        print(f"saved {native_path} (nx={FINE_CURVE_NX}, ny={FINE_CURVE_NY}, steps={steps})")


if __name__ == "__main__":
    main()
