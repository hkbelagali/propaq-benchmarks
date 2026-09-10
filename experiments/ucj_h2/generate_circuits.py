#!/usr/bin/env python3
"""Build and save the ucj_h2 circuit instance this experiment compares backends on.

Run this once before any run_<backend> file in this folder. Every run_<backend> file then
loads whatever is saved in circuits/ and does not build circuits itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from propaq_benchmarks import problems_qubit  # noqa: E402

# A single instance at the defaults (bond_length=0.74, n_reps=1, seed=0), the H2/STO-6G
# UCJ ansatz used by this repo's existing propaq/benchmarks/bench_ucj.py.
LABEL = "h2_sto6g_nreps1"


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    ir = problems_qubit.ucj_h2_problem(bond_length=0.74, n_reps=1, seed=0)
    path = circuits_dir / f"{LABEL}.json"
    ir.save(str(path))
    print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
