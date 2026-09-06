#!/usr/bin/env python3
"""Build and save every ising_trotter circuit instance this experiment compares backends on.

Run this once (or again after changing SIZES below) before any run_<backend> file in this
folder. Every run_<backend> file then loads whatever is saved in circuits/ and does not
build circuits itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import problems_qubit  # noqa: E402

# (nx, ny, steps): two small warm-up sizes, plus a fine Trotter-step curve at the 6x6=36
# qubit lattice from the original request, one saved circuit per step count from 1 to 25.
# The fine curve is what plotting/plot_trotter_scan.py and plot_trotter_memory.py plot
# runtime and peak RSS against, one point per step count instead of a handful of discrete
# sizes. Running the slowest backend across all 25 steps takes a while, each step count is
# an independent from-scratch circuit build and propagation run, not a single run
# instrumented with mid-circuit checkpoints, so cost grows like steps squared overall, not
# steps. That is fine at this lattice size, revisit with real mid-circuit checkpointing in
# a runner if a much larger lattice ever makes that the bottleneck.
WARMUP_SIZES = [(3, 3, 10), (4, 4, 12)]
FINE_CURVE_NX, FINE_CURVE_NY = 6, 6
FINE_CURVE_STEPS = range(1, 26)


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    sizes = [(nx, ny, steps) for nx, ny, steps in WARMUP_SIZES]
    sizes += [(FINE_CURVE_NX, FINE_CURVE_NY, steps) for steps in FINE_CURVE_STEPS]
    for nx, ny, steps in sizes:
        ir = problems_qubit.ising_trotter_problem(nx=nx, ny=ny, steps=steps)
        label = f"{nx}x{ny}_steps{steps}"
        path = circuits_dir / f"{label}.json"
        ir.save(str(path))
        print(f"saved {path} ({ir.n_qubits} qubits, {ir.gate_count()} gates)")


if __name__ == "__main__":
    main()
