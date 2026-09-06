#!/usr/bin/env python3
"""Top-level driver for the main (non-scaling) cross-package benchmark suite.

CHECKPOINTED / RESUMABLE: every completed (problem, backend) run is appended immediately
(fsync'd) to a JSONL checkpoint file. On startup, the checkpoint is re-loaded and any task
whose (problem_uid, backend_key) pair already has a successful record is skipped. So if this
process is killed mid-run (e.g. a SLURM walltime SIGKILL), simply re-running the exact same
command resumes from the next incomplete task instead of starting over. The .npz snapshot is
regenerated from the checkpoint after every single task, so results/main_suite.npz is always
a valid, up-to-date (if partial) dataset even if the run never reaches the end.

Usage:
  python3 orchestrate.py [--quick] [--checkpoint results/main_suite.jsonl] [--out results/main_suite.npz]
  (--quick shrinks every problem for a fast smoke test)

To force-redo a task, delete its line(s) from the checkpoint JSONL (or delete the whole file
to redo everything).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR))

from common import io_utils, problems_qubit, problems_fermionic  # noqa: E402

RESULTS_RAW = BENCH_DIR / "results" / "raw"
RESULTS_RAW.mkdir(parents=True, exist_ok=True)
# Resolve via PATH (i.e. whatever `module load Julia/...` set up on *this* node). This
# cluster has architecture-specific module trees (x86_64/generic vs x86_64/amd/zen4), so a
# path hardcoded on one node type can be wrong on another. No silent fallback to a hardcoded
# path here anymore: that previously turned a missing `module load` into an opaque
# "no_json_output"/exit-127 failure on *every* Julia task instead of one clear error up front.
JULIA_BIN = shutil.which("julia")
if JULIA_BIN is None:
    sys.exit(
        "ERROR: `julia` not found on PATH. Run `module load Julia/1.11.3-linux-x86_64` "
        "(matching bench/julia_env's instantiated version) before running this script."
    )
JULIA_PROJECT = str(BENCH_DIR / "julia_env")
PY = sys.executable

# pauli-prop's max_terms=None is documented as "no cap" but actually crashes (propagation.py's
# `if max_terms < 1` doesn't guard None), so "no cap" is a large int here instead. propaq,
# pyrauli, and both Julia backends have no term-count cap at all, only the coeff cutoff below,
# so this needs to be set high enough to never actually bind, matching the other four backends'
# truncation behavior exactly. Confirmed empirically that a larger value costs nothing on small
# circuits (no upfront buffer scales with it, a tiny circuit's peak RSS was identical, ~87MB,
# at max_terms=50M vs 2B), so there's no reason to keep it tight.
PAULI_PROP_MAX_TERMS = 2_000_000_000
MIN_ABS_COEFF = 1e-6


def cmd_pauli_prop(problem_path: str, n_threads: int) -> list[str]:
    return [PY, str(BENCH_DIR / "runners" / "run_pauli_prop.py"),
            "--problem", problem_path, "--max-terms", str(PAULI_PROP_MAX_TERMS), "--atol", str(MIN_ABS_COEFF)]


def cmd_propaq(problem_path: str, basis: str, n_threads: int) -> list[str]:
    return [PY, str(BENCH_DIR / "runners" / "run_propaq.py"),
            "--problem", problem_path, "--basis", basis,
            "--coeff-cutoff", str(MIN_ABS_COEFF),
            "--n-threads", str(n_threads)]


def cmd_pyrauli(problem_path: str, n_threads: int) -> list[str]:
    return [PY, str(BENCH_DIR / "runners" / "run_pyrauli.py"),
            "--problem", problem_path, "--coeff-cutoff", str(MIN_ABS_COEFF),
            "--n-threads", str(n_threads)]


def cmd_pauli_propagation_jl(problem_path: str, n_threads: int) -> list[str]:
    return [JULIA_BIN, f"--project={JULIA_PROJECT}", "-t", str(n_threads),
            str(BENCH_DIR / "runners" / "run_pauli_propagation.jl"),
            "--problem", problem_path, "--min-abs-coeff", str(MIN_ABS_COEFF)]


def cmd_majorana_propagation_jl(problem_path: str, n_threads: int) -> list[str]:
    return [JULIA_BIN, f"--project={JULIA_PROJECT}", "-t", str(n_threads),
            str(BENCH_DIR / "runners" / "run_majorana_propagation.jl"),
            "--problem", problem_path, "--min-abs-coeff", str(MIN_ABS_COEFF)]


# Backends that consume the qubit-suite (COMMON_BASIS) IR.
QUBIT_BACKENDS = {
    "pauli_prop": lambda p, t: cmd_pauli_prop(p, t),
    "pauli_propagation_jl": lambda p, t: cmd_pauli_propagation_jl(p, t),
    "pyrauli": lambda p, t: cmd_pyrauli(p, t),
    "propaq_pauli": lambda p, t: cmd_propaq(p, "pauli", t),
    "propaq_majorana": lambda p, t: cmd_propaq(p, "majorana", t),
}


def _add_qiskit(plan: list, name: str, size_label: str, build_fn, *args, **kwargs) -> None:
    """Build one qubit-suite problem (Qiskit circuit construction + canonicalize/transpile
    happens inside build_fn) and print a message the moment it's ready, before moving on."""
    t0 = time.time()
    ir = build_fn(*args, **kwargs)
    print(f"  prepared qiskit circuit for [{name}: {size_label}] "
          f"({ir.n_qubits} qubits, {ir.gate_count()} gates, {time.time() - t0:.2f}s to build)")
    plan.append((name, size_label, ir))


def build_plan(quick: bool) -> tuple[list[tuple[str, str, object]], list[tuple[str, str, object]]]:
    q = quick
    plan = []
    print("Preparing qiskit circuits for the qubit-suite problems...")
    # 1. random_circuit. n is chaotic past 12 for propaq/pyrauli's uncapped term count (n=14
    # ran faster than n=13, n=16 timed out at >300s), so depth at fixed n=12 scales smoothly instead.
    for n, depth in ([(6, 6)] if q else [(10, 20), (12, 20), (12, 28), (12, 36)]):
        _add_qiskit(plan, "random_circuit", f"n={n},depth={depth}",
                    problems_qubit.random_circuit_problem, n, depth, seed=0)
    # 2. random_near_clifford: T-density sweep, base size bumped from n=12,depth=16 (<0.1s
    # everywhere) to n=13,depth=18 (32s at the heaviest density, t_density=0.6).
    for td in ([0.3] if q else [0.1, 0.3, 0.6, 1.0]):
        _add_qiskit(plan, "random_near_clifford", f"n=13,depth=18,t_density={td}",
                    problems_qubit.random_near_clifford_problem, 13, 18, td, seed=0)
    # 3. ising_trotter (6x6 per the user's request, plus a smaller warm-up size). Grid capped at
    # 6x6: 5x5 and 7x7 both exceeded 200s, so the largest point grows via Trotter step count instead.
    for nx, ny, steps in ([(3, 3, 10)] if q else [(3, 3, 10), (4, 4, 12), (6, 6, 15), (6, 6, 20)]):
        _add_qiskit(plan, "ising_trotter", f"{nx}x{ny},steps={steps}",
                    problems_qubit.ising_trotter_problem, nx, ny, steps=steps)
    # 4. heisenberg_chain_trotter
    for n, steps in ([(10, 2)] if q else [(12, 2), (20, 4), (26, 6), (30, 8)]):
        _add_qiskit(plan, "heisenberg_chain_trotter", f"n={n},steps={steps}",
                    problems_qubit.heisenberg_chain_trotter_problem, n, steps=steps)
    # 5. qaoa_maxcut
    for n, p in ([(8, 2)] if q else [(10, 2), (16, 3), (24, 4), (36, 5)]):
        _add_qiskit(plan, "qaoa_maxcut", f"n={n},p={p}",
                    problems_qubit.qaoa_maxcut_problem, n, p, seed=0)
    # 6. ucj_h2, fixed at H2/STO-6G (4 qubits). n_reps barely moves wall time (0.06s even at
    # n_reps=3), so it's left as a single instance. A bigger molecule/basis is a separate decision.
    _add_qiskit(plan, "ucj_h2", "H2-STO-6G,n_reps=1", problems_qubit.ucj_h2_problem)
    # 7. hubbard_trotter (qubit/JW version)
    for nx, ny, steps in ([(2, 2, 1)] if q else [(2, 2, 1), (3, 3, 2), (4, 4, 3), (5, 5, 4)]):
        _add_qiskit(plan, "hubbard_trotter", f"{nx}x{ny},steps={steps}",
                    problems_qubit.hubbard_trotter_problem, nx, ny, steps=steps)
    # 8. random_fermionic_circuit (qubit/JW version)
    for nm, ng in ([(6, 10)] if q else [(8, 20), (14, 40), (18, 60), (20, 80)]):
        _add_qiskit(plan, "random_fermionic_circuit", f"n_modes={nm},n_gates={ng}",
                    problems_qubit.random_fermionic_circuit_problem, nm, ng, seed=0)

    # Fermionic-native companion problems, MajoranaPropagation.jl only (no Qiskit circuit,
    # just physical parameters). Mirrors the qubit-suite size lists above so every size has a match.
    fplan = []
    for nx, ny, steps in ([(2, 2, 1)] if q else [(2, 2, 1), (3, 3, 2), (4, 4, 3), (5, 5, 4)]):
        fplan.append(("hubbard_trotter", f"{nx}x{ny},steps={steps}(native)",
                      problems_fermionic.hubbard_trotter_fermionic(nx, ny, steps=steps)))
    for nm, ng in ([(6, 10)] if q else [(8, 20), (14, 40), (18, 60), (20, 80)]):
        fplan.append(("random_fermionic_circuit", f"n_modes={nm},n_gates={ng}(native)",
                      problems_fermionic.random_fermionic_circuit_fermionic(nm, ng, seed=0)))
    return plan, fplan


def _uid(problem_name: str, size_label: str) -> str:
    return f"{problem_name}__{size_label}".replace(" ", "").replace("/", "-")


def run_one(uid: str, size_label: str, problem_name: str, backend_key: str, cmd: list[str],
            env: dict[str, str] | None = None) -> dict:
    print(f"  [{uid}] {backend_key} ...", end=" ", flush=True)
    t0 = time.time()
    rec = io_utils.run_backend(cmd, env=env)
    rec["size_label"] = size_label
    rec["problem_uid"] = uid
    rec["backend_key"] = backend_key
    rec.setdefault("problem", problem_name)
    if not rec.get("ok") or "backend" not in rec:
        rec.setdefault("backend", backend_key.split("_")[0] if "propaq" in backend_key else backend_key)
        rec.setdefault("basis", "majorana" if "majorana" in backend_key else "pauli")
    print(f"{'OK' if rec.get('ok') else 'FAIL'} ({time.time() - t0:.1f}s wall)")
    if not rec.get("ok"):
        print("    ", rec.get("error"), (rec.get("stderr_tail") or "")[-500:])
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny sizes, for a fast smoke test")
    ap.add_argument("--out", default=str(BENCH_DIR / "results" / "main_suite.npz"))
    ap.add_argument("--checkpoint", default=str(BENCH_DIR / "results" / "main_suite.jsonl"))
    ap.add_argument("--n-threads", type=int, default=64,
                     help="threads for propaq/pyrauli/PauliPropagation.jl/MajoranaPropagation.jl "
                          "(pauli-prop is single-threaded regardless, per README)")
    args = ap.parse_args()

    done = io_utils.load_jsonl(args.checkpoint)
    done_keys = {(r["problem_uid"], r["backend_key"]) for r in done if r.get("ok")}
    print(f"Resuming: {len(done_keys)} tasks already completed in {args.checkpoint}")

    plan, fplan = build_plan(args.quick)

    n_total = sum(len(QUBIT_BACKENDS) for _ in plan) + len(fplan)
    n_run = 0

    for problem_name, size_label, ir in plan:
        uid = _uid(problem_name, size_label)
        path = str(RESULTS_RAW / f"{uid}.json")
        ir.save(path)
        for backend_key, cmdfn in QUBIT_BACKENDS.items():
            n_run += 1
            if (uid, backend_key) in done_keys:
                print(f"  [{uid}] {backend_key} ... skip (already done, {n_run}/{n_total})")
                continue
            # pyrauli's OpenMP thread count is process-wide via OMP_NUM_THREADS, set before the
            # process starts. --n-threads alone only picks the seq/par runtime (see run_pyrauli.py).
            env = dict(os.environ, OMP_NUM_THREADS=str(args.n_threads)) if backend_key == "pyrauli" else None
            rec = run_one(uid, size_label, problem_name, backend_key, cmdfn(path, args.n_threads), env=env)
            io_utils.append_jsonl(args.checkpoint, rec)
            io_utils.save_records_npz(io_utils.load_jsonl(args.checkpoint), args.out)

    for problem_name, size_label, fp in fplan:
        uid = _uid(problem_name, size_label)
        path = str(RESULTS_RAW / f"{uid}.json")
        fp.save(path)
        backend_key = "majorana_propagation_jl"
        n_run += 1
        if (uid, backend_key) in done_keys:
            print(f"  [{uid}] {backend_key} ... skip (already done, {n_run}/{n_total})")
            continue
        rec = run_one(uid, size_label, problem_name, backend_key,
                      cmd_majorana_propagation_jl(path, args.n_threads))
        io_utils.append_jsonl(args.checkpoint, rec)
        io_utils.save_records_npz(io_utils.load_jsonl(args.checkpoint), args.out)

    all_records = io_utils.load_jsonl(args.checkpoint)
    n_ok = sum(1 for r in all_records if r.get("ok"))
    print(f"\n{len(all_records)} total records ({n_ok} ok, {len(all_records) - n_ok} failed). "
          f"Checkpoint: {args.checkpoint}  Snapshot: {args.out}")


if __name__ == "__main__":
    main()
