#!/usr/bin/env python3
"""Save the natoms sweep parameters for the hybrid_ucj_heisenberg experiment.

The UCJ operator here is fit fresh from a CCSD amplitude calculation (pyscf and ffsim)
inside run_propaq.py for each natoms value, exactly as the original benchmark script did
on every invocation. ffsim's UCJOpSpinBalanced does not fit ProblemIR's qubit-gate-list
shape, and there is no way to serialize its fitted numeric arrays and verify the round
trip in this sandbox, since ffsim is not installed here. Each saved file therefore holds
only the natoms value the run script needs to rebuild the operator, not a full ProblemIR.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

NATOMS_SWEEP = (4, 6)


def main() -> None:
    circuits_dir = HERE / "circuits"
    circuits_dir.mkdir(exist_ok=True)
    for natoms in NATOMS_SWEEP:
        path = circuits_dir / f"natoms{natoms}.json"
        with open(path, "w") as f:
            json.dump({"problem": "hybrid_ucj_heisenberg", "natoms": natoms}, f)
        print(f"saved {path} (natoms={natoms})")


if __name__ == "__main__":
    main()
