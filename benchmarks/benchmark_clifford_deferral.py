"""Ablation comparing propaq's Pauli propagation with and without Clifford deferral.

The same circuits are run both ways while sweeping the proportion of layers that insert
a T gate.

Deferral is toggled via the PROPAQ_DISABLE_CLIFFORD_DEFERRAL env var, read at the start
of every propagation call in crates/pauli/src/engine.rs and crates/majorana/src/engine.rs.
It forces defer_cliffords = false, so every Clifford rotation takes the generic branching
path instead of being absorbed into the frame tableau in O(1).
propaq must be rebuilt (slurm/rebuild_native.sh) after any change to those files for this
script to see it.

Each circuit layer applies a randomized Clifford round (random single-qubit Clifford on
every qubit plus a random perfect-matching CX layer) to scramble the observable's Pauli
support.
A deterministic brickwork pattern was tried first and largely failed to produce
anticommuting overlap with the T gates, so term counts stayed roughly constant regardless
of T count.
With probability p per layer, a T gate on a random qubit is also inserted.

Term count is governed by the expected T count (TOTAL_LAYERS times p), not by N_QUBITS.
Each T gate can roughly double the live term count in the worst case, so pushing
TOTAL_LAYERS and PROPORTIONS up together runs into exponential blowup fast (confirmed
experimentally, 80 layers at p up to 0.30 tops out around 2 to 3 million terms in a
couple seconds even at N_QUBITS=64 or 128, doubling both layers and the max proportion at
once killed the process on an out of memory error).
N_QUBITS is the safe axis to scale for more per-thread work at 64 threads, since it only
raises the ceiling rather than the branching factor.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Set before numpy/qiskit load, since OpenBLAS reads these at import and never re-reads
# them.
# qiskit pulls in two OpenBLAS copies, each of which starts one spinning worker per core
# and keeps them runnable long after the last BLAS call.
# propaq's engine pins one worker per partition to a core, so those spinners stop being
# harmless idle-core noise and start time-slicing against pinned workers.
# See the PauliPropagator docstring in crates/pauli/src/propagator.rs, measured 449ms
# against 184ms at 64 threads on an unrelated circuit purely from this.
# No backend here wants threaded BLAS.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import random
import sys
import time
import warnings
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

BENCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_DIR))

from common import io_utils  # noqa: E402

warnings.simplefilter("ignore")  # propaq's transpile-fallback UserWarnings go to stderr

N_QUBITS = 64
TOTAL_LAYERS = 80
PROPORTIONS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25]
SEED = 123
REPEATS = 5
COEFF_CUTOFF = 1e-9
N_THREADS = 64
BENCHMARK_ID = "clifford_deferral_ablation_v1"


def build_circuit(n_qubits: int, total_layers: int, p: float, seed: int) -> QuantumCircuit:
    rng = random.Random(seed)
    qc = QuantumCircuit(n_qubits)
    for _ in range(total_layers):
        for q in range(n_qubits):
            getattr(qc, rng.choice(["h", "s", "x", "sdg"]))(q)
        qubits = list(range(n_qubits))
        rng.shuffle(qubits)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(qubits[i], qubits[i + 1])
        if rng.random() < p:
            qc.t(rng.randrange(n_qubits))
    return qc


def time_run(circuit, obs, trunc, n_threads, repeats):
    from propaq.propagators import PauliPropagator

    prop = PauliPropagator(None, trunc, n_threads=n_threads)
    prop.expectation_value(obs, circuit, initial_state=0)  # warmup
    times = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = prop.expectation_value(obs, circuit, initial_state=0)
        times.append(time.perf_counter() - t0)
    return min(times), result.expectation_value, result.n_terms[-1]


def run_proportion(p: float, args: argparse.Namespace) -> dict:
    from propaq.circuits import PauliCircuit
    from propaq.datatypes import PauliTermSum
    from propaq.noise import TruncationPolicy

    qc = build_circuit(args.n_qubits, args.total_layers, p, args.seed)
    circuit = PauliCircuit.from_qiskit(qc)
    obs = PauliTermSum.from_sparse_pauli_op(SparsePauliOp("Z" + "I" * (args.n_qubits - 1)))
    trunc = TruncationPolicy(weight_cutoff=1_000_000, coeff_cutoff=args.coeff_cutoff)

    os.environ.pop("PROPAQ_DISABLE_CLIFFORD_DEFERRAL", None)
    t_on, v_on, n_on = time_run(circuit, obs, trunc, args.n_threads, args.repeats)

    os.environ["PROPAQ_DISABLE_CLIFFORD_DEFERRAL"] = "1"
    t_off, v_off, n_off = time_run(circuit, obs, trunc, args.n_threads, args.repeats)
    os.environ.pop("PROPAQ_DISABLE_CLIFFORD_DEFERRAL", None)

    # Deferral is exact. It changes only how many Clifford steps are folded into the frame
    # tableau versus branched into the term store, never the physics.
    # n_terms legitimately differs between the two (the branching path leaves a
    # negligible but nonzero cosine-branch residue in an append-only store that never
    # reclaims it, see the set_defer_cliffords doc in crates/core/src/partitioned.rs), so
    # only the expectation value is asserted, not the term count.
    agreement = abs(v_on - v_off)
    if agreement > 1e-6:
        raise RuntimeError(f"p={p}: deferral changed the expectation value by {agreement:.3e}")

    return {
        "benchmark": BENCHMARK_ID,
        "n_qubits": args.n_qubits,
        "total_layers": args.total_layers,
        "n_threads": args.n_threads,
        "repeats": args.repeats,
        "p": p,
        "t_on_ms": t_on * 1e3,
        "t_off_ms": t_off * 1e3,
        "n_on": n_on,
        "n_off": n_off,
        "speedup": t_off / t_on,
        "v_on": v_on,
        "v_off": v_off,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-qubits", type=int, default=N_QUBITS)
    parser.add_argument("--total-layers", type=int, default=TOTAL_LAYERS)
    parser.add_argument("--proportions", type=float, nargs="+", default=PROPORTIONS)
    parser.add_argument("--n-threads", type=int, default=N_THREADS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--coeff-cutoff", type=float, default=COEFF_CUTOFF)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=BENCH_DIR / "results" / "clifford_deferral_ablation.jsonl",
    )
    parser.add_argument(
        "--out", type=Path,
        default=BENCH_DIR / "results" / "clifford_deferral_ablation.npz",
    )
    args = parser.parse_args()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    records = io_utils.load_jsonl(str(args.checkpoint))
    completed = {
        record["p"] for record in records
        if record.get("benchmark") == BENCHMARK_ID
        and record.get("n_qubits") == args.n_qubits
        and record.get("total_layers") == args.total_layers
        and record.get("n_threads") == args.n_threads
        and record.get("repeats") == args.repeats
    }

    print(
        f"Clifford-deferral ablation: {args.n_qubits} qubits, {args.total_layers} layers, "
        f"{args.n_threads} threads, {args.repeats} repeats"
    )
    header = f"{'p':>6} {'defer_on(ms)':>13} {'N_on':>9} {'defer_off(ms)':>14} {'N_off':>9} {'speedup':>9} {'<Z>_on':>10} {'<Z>_off':>10}"
    print(header)
    for p in args.proportions:
        if p in completed:
            print(f"{p:6.3f}  skip (already checkpointed)")
            continue
        record = run_proportion(p, args)
        io_utils.append_jsonl(str(args.checkpoint), record)
        io_utils.save_records_npz(io_utils.load_jsonl(str(args.checkpoint)), str(args.out))
        print(
            f"{p:6.3f} {record['t_on_ms']:13.3f} {record['n_on']:9d} {record['t_off_ms']:14.3f} "
            f"{record['n_off']:9d} {record['speedup']:8.2f}x {record['v_on']:10.5f} {record['v_off']:10.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
