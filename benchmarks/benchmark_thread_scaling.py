"""Thread-scaling benchmark: propaq's Pauli-propagation runtime vs number of Trotter
steps, swept across thread counts, on the same 2D transverse-field Ising Trotter
circuit propaq's own CHANGELOG benchmarks against monoprop (see
common/problems_qubit.py:ising_trotter_problem). Runtime grows with the step count
(more RZZ/RX layers to propagate through), which is the axis this script holds fixed
per point and sweeps to get a range of problem sizes at each thread count.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import sys
import time
import warnings
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_DIR))

from common import io_utils, problems_qubit  # noqa: E402

warnings.simplefilter("ignore")  # propaq's transpile-fallback UserWarnings go to stderr

NX, NY = 3, 3
STEPS_SWEEP = [1, 2, 3, 4, 5, 6]
THREAD_SWEEP = [1, 2, 4, 8, 16, 32, 64]
REPEATS = 5
COEFF_CUTOFF = 1e-8
BENCHMARK_ID = "thread_scaling_ising_trotter_v1"


def time_run(circuit, obs, trunc, n_threads: int, repeats: int):
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


def run_point(steps: int, n_threads: int, args: argparse.Namespace) -> dict:
    from propaq.circuits import PauliCircuit
    from propaq.datatypes import PauliTermSum
    from propaq.noise import TruncationPolicy

    ir = problems_qubit.ising_trotter_problem(
        nx=args.nx, ny=args.ny, J=args.J, h=args.h, dt=args.dt, steps=steps,
    )
    circuit = PauliCircuit.from_qiskit(ir.to_qiskit())
    obs = PauliTermSum.from_sparse_pauli_op(ir.observable.to_sparse_pauli_op())
    trunc = TruncationPolicy(weight_cutoff=1_000_000, coeff_cutoff=args.coeff_cutoff)

    t, v, n = time_run(circuit, obs, trunc, n_threads, args.repeats)
    return {
        "benchmark": BENCHMARK_ID,
        "nx": args.nx, "ny": args.ny, "J": args.J, "h": args.h, "dt": args.dt,
        "steps": steps, "n_threads": n_threads, "repeats": args.repeats,
        "coeff_cutoff": args.coeff_cutoff,
        "wall_time_s": t, "expectation_value": v, "n_terms": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nx", type=int, default=NX)
    parser.add_argument("--ny", type=int, default=NY)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.5)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--steps", type=int, nargs="+", default=STEPS_SWEEP)
    parser.add_argument("--threads", type=int, nargs="+", default=THREAD_SWEEP)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--coeff-cutoff", type=float, default=COEFF_CUTOFF)
    parser.add_argument(
        "--checkpoint", type=Path, default=BENCH_DIR / "results" / "thread_scaling.jsonl",
    )
    parser.add_argument(
        "--out", type=Path, default=BENCH_DIR / "results" / "thread_scaling.npz",
    )
    args = parser.parse_args()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    records = io_utils.load_jsonl(str(args.checkpoint))
    completed = {
        (r["steps"], r["n_threads"]) for r in records
        if r.get("benchmark") == BENCHMARK_ID
        and r.get("nx") == args.nx and r.get("ny") == args.ny
        and r.get("J") == args.J and r.get("h") == args.h and r.get("dt") == args.dt
        and r.get("repeats") == args.repeats and r.get("coeff_cutoff") == args.coeff_cutoff
    }

    print(
        f"Thread scaling: {args.nx}x{args.ny} Ising Trotter, steps={args.steps}, "
        f"threads={args.threads}, {args.repeats} repeats"
    )
    print(f"{'steps':>6} {'threads':>8} {'wall(s)':>10} {'n_terms':>10}")
    for steps in args.steps:
        for n_threads in args.threads:
            if (steps, n_threads) in completed:
                print(f"{steps:6d} {n_threads:8d}  skip (already checkpointed)")
                continue
            rec = run_point(steps, n_threads, args)
            io_utils.append_jsonl(str(args.checkpoint), rec)
            io_utils.save_records_npz(io_utils.load_jsonl(str(args.checkpoint)), str(args.out))
            print(f"{steps:6d} {n_threads:8d} {rec['wall_time_s']:10.4f} {rec['n_terms']:10d}", flush=True)


if __name__ == "__main__":
    main()
