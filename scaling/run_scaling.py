#!/usr/bin/env python3
"""Parallelism-scaling sweep: fixed problem instances, varying thread count, per backend.

CHECKPOINTED / RESUMABLE, see orchestrate.py's docstring for the mechanism, identical here.
Re-running the same command after a kill resumes from the next incomplete (problem, backend,
n_threads) task.

Backends with real thread control:
  - propaq (pauli & majorana): --n-threads N -> a dedicated per-instance Rayon thread pool.
  - pyrauli: process-wide via OMP_NUM_THREADS env var (set here per subprocess call) plus
    runtime=seq/par chosen by the runner from --n-threads.
  - PauliPropagation.jl / MajoranaPropagation.jl: `julia -t N` (VectorPauliSum/VectorMajoranaSum
    backend, AcceleratedKernels-parallel).
pauli-prop is confirmed single-threaded and is run once as a flat reference.

Usage: python3 scaling/run_scaling.py [--quick] [--checkpoint results/scaling_suite.jsonl]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_DIR))

from common import io_utils, problems_qubit, problems_fermionic  # noqa: E402
import orchestrate as orch  # noqa: E402

RESULTS_RAW = BENCH_DIR / "results" / "raw"
RESULTS_RAW.mkdir(parents=True, exist_ok=True)

THREAD_SWEEP = [1, 2, 4, 8, 16, 32]


def run_one(uid: str, backend_key: str, n_threads: int, cmd: list[str], env=None) -> dict:
    print(f"  [{uid}] {backend_key} n_threads={n_threads} ...", end=" ", flush=True)
    t0 = time.time()
    rec = io_utils.run_backend(cmd, env=env)
    rec["size_label"] = uid
    rec["problem_uid"] = uid
    rec["backend_key"] = f"{backend_key}@t{n_threads}"
    rec.setdefault("backend", backend_key.split("_")[0] if "propaq" in backend_key else backend_key)
    rec["n_threads"] = n_threads  # authoritative requested value
    print(f"{'OK' if rec.get('ok') else 'FAIL'} wall_time_s={rec.get('wall_time_s')} ({time.time()-t0:.1f}s)")
    if not rec.get("ok"):
        print("    ", rec.get("error"), (rec.get("stderr_tail") or "")[-500:])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(BENCH_DIR / "results" / "scaling_suite.npz"))
    ap.add_argument("--checkpoint", default=str(BENCH_DIR / "results" / "scaling_suite.jsonl"))
    args = ap.parse_args()

    done = io_utils.load_jsonl(args.checkpoint)
    done_keys = {(r["problem_uid"], r["backend_key"]) for r in done if r.get("ok")}
    print(f"Resuming: {len(done_keys)} tasks already completed in {args.checkpoint}")

    def maybe_run(uid, backend_key, n_threads, cmd, env=None):
        key = f"{backend_key}@t{n_threads}"
        if (uid, key) in done_keys:
            print(f"  [{uid}] {key} ... skip (already done)")
            return
        rec = run_one(uid, backend_key, n_threads, cmd, env=env)
        io_utils.append_jsonl(args.checkpoint, rec)
        io_utils.save_records_npz(io_utils.load_jsonl(args.checkpoint), args.out)

    # --- Qubit-suite scaling instance: a large-ish random circuit ---
    n, depth = (14, 20) if args.quick else (10, 60)
    ir = problems_qubit.random_circuit_problem(n, depth, seed=0)
    path = str(RESULTS_RAW / "scaling_random_circuit.json")
    ir.save(path)
    uid = f"random_circuit_scaling(n={n},depth={depth})"
    print(f"=== Qubit scaling problem: {uid} ===")

    for t in THREAD_SWEEP:
        maybe_run(uid, "propaq_pauli", t, orch.cmd_propaq(path, "pauli", t))
    for t in THREAD_SWEEP:
        maybe_run(uid, "propaq_majorana", t, orch.cmd_propaq(path, "majorana", t))
    for t in THREAD_SWEEP:
        env = dict(os.environ, OMP_NUM_THREADS=str(t))
        maybe_run(uid, "pyrauli", t, orch.cmd_pyrauli(path, t), env=env)
    for t in THREAD_SWEEP:
        maybe_run(uid, "pauli_propagation_jl", t, orch.cmd_pauli_propagation_jl(path, t))
    maybe_run(uid, "pauli_prop", 1, orch.cmd_pauli_prop(path, 1))

    # --- Fermionic-native scaling instance: a larger Hubbard lattice ---
    nx, ny, steps = (3, 3, 2) if args.quick else (5, 5, 4)
    hub_ir = problems_qubit.hubbard_trotter_problem(nx, ny, steps=steps)
    hub_path = str(RESULTS_RAW / "scaling_hubbard_qubit.json")
    hub_ir.save(hub_path)
    hub_uid = f"hubbard_scaling({nx}x{ny},steps={steps})"
    print(f"=== Fermionic scaling problem (qubit/JW side): {hub_uid} ===")
    for t in THREAD_SWEEP:
        maybe_run(hub_uid, "propaq_majorana", t, orch.cmd_propaq(hub_path, "majorana", t))

    hub_fp = problems_fermionic.hubbard_trotter_fermionic(nx, ny, steps=steps)
    hub_fp_path = str(RESULTS_RAW / "scaling_hubbard_native.json")
    hub_fp.save(hub_fp_path)
    hub_native_uid = f"hubbard_scaling_native({nx}x{ny},steps={steps})"
    print(f"=== Fermionic scaling problem (native side): {hub_native_uid} ===")
    for t in THREAD_SWEEP:
        maybe_run(hub_native_uid, "majorana_propagation_jl", t, orch.cmd_majorana_propagation_jl(hub_fp_path, t))

    all_records = io_utils.load_jsonl(args.checkpoint)
    n_ok = sum(1 for r in all_records if r.get("ok"))
    print(f"\n{len(all_records)} total records ({n_ok} ok). Checkpoint: {args.checkpoint}  Snapshot: {args.out}")


if __name__ == "__main__":
    main()
