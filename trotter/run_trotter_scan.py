#!/usr/bin/env python3
"""Trotter-step/layer runtime scan: for each Trotter-parameterized system (ising_trotter,
hubbard_trotter), builds circuits at increasing Trotter step/layer counts (1..max_steps) on a
fixed lattice size, runs every backend for --n-trials independent trials at each step count,
and checkpoints every run. This is the data behind plotting/plot_trotter_scan.py's cumulative-
runtime-vs-Trotter-step curves (log scale, inset term-count curves, trial error bars).

Each step count is an independent, from-scratch circuit build + propagation run (not a single
run instrumented with mid-circuit checkpoints). This is simpler and much lower-risk than adding
incremental-timing hooks to five heterogeneous backend runners (two Rust, one C++, two Julia),
at the cost of O(steps^2) total propagation work instead of O(steps). Fine at the demo size
(3x3 lattice, <=25 steps). Revisit with real mid-circuit checkpointing in each runner if this
becomes the bottleneck at larger lattices.

hubbard_trotter's term count blows up fast (measured ~525 -> ~2.48M terms from step 1 to step 6
at 3x3, all backends/bases, the on-site interaction gate is what does it), so the later steps
at max_steps=10 are expected to run long, and at larger lattices (6x6 Hubbard is 72 qubits)
longer still. Every run here passes timeout=None to io_utils.run_backend, meaning no cap at all, by
request, unlike orchestrate.py/scaling/run_scaling.py which keep that function's 1800s default.

CHECKPOINTED / RESUMABLE, identical mechanism to orchestrate.py: every completed
(system, lattice_size, n_step, backend, trial) run is fsync'd to a JSONL checkpoint immediately,
and re-running the same command skips everything already recorded. The .npz snapshot is
regenerated after every task.

Usage:
  module load Julia/1.11.3-linux-x86_64
  python3 trotter/run_trotter_scan.py --quick        # 1 step, 1 trial, 3x3 only, smoke test
  python3 trotter/run_trotter_scan.py                 # Hubbard only: 5x5, full step range, 1 trial
  python3 trotter/run_trotter_scan.py --systems ising_trotter --sizes 6x6  # explicit Ising rerun
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_DIR))

from common import io_utils, problems_qubit, problems_fermionic  # noqa: E402
import orchestrate as orch  # noqa: E402 (reuses cmd_* backend-command builders + the julia-on-PATH check)

RESULTS_RAW = BENCH_DIR / "results" / "raw" / "trotter"
RESULTS_RAW.mkdir(parents=True, exist_ok=True)

# One entry per Trotter-step-parameterized system this scan covers. Both builders share the
# (nx, ny, ..., steps=) signature (see common/problems_qubit.py), so a single driver loop
# handles both. `native_fermionic=True` additionally runs MajoranaPropagation.jl on
# problems_fermionic's independently-constructed fermionic version at matching (nx, ny, n_step),
# mirroring orchestrate.py's fplan pattern. ising_trotter has no native-fermionic analog.
SYSTEMS = {
    "ising_trotter": {
        "builder": problems_qubit.ising_trotter_problem,
        "max_steps": 25,
        "native_fermionic": False,
        "qubit_backends": True,
        # propaq's Majorana basis reaches a spin model only through Jordan-Wigner, whose
        # Z-strings make every term's weight run to the end of the register, the wrong
        # comparison for Ising, and unbounded in cost at 36 qubits. It stays on the two
        # fermionic systems, where it consumes a natively fermionic circuit instead.
        "qubit_majorana": False,
        "monoprop_qubit_backend": True,
    },
    "hubbard_trotter": {
        "builder": problems_qubit.hubbard_trotter_problem,
        "max_steps": 25,
        "native_fermionic": True,
        # Majorana engines only, by request: majorana_propagation_jl, propaq_majorana and
        # monoprop_majorana, all reached through the `native_fermionic` block below. The
        # Pauli backends (pauli-prop, pyrauli, PauliPropagation.jl, and the Pauli arms of
        # propaq and MonoProp) reach Hubbard only through a Jordan-Wigner mapped qubit
        # circuit, whose term count blows up several steps sooner than the fermionic one.
        # They could not pass step 7 at 6x6 and would consume the scan before the native
        # backends reached the depths this run is for.
        "qubit_backends": False,
        "monoprop_qubit_backend": False,
    },
}

QUBIT_BACKENDS = orch.QUBIT_BACKENDS  # pauli_prop, pauli_propagation_jl, pyrauli, propaq_pauli, propaq_majorana

# propaq 0.1.3 has a single engine serving both bases, so the reason this was set
# (a Pauli-only engine A/B that Majorana could not participate in) is gone.
SKIP_PROPAQ_MAJORANA = False


def cmd_monoprop(problem_path: str, n_threads: int) -> list[str]:
    return [orch.PY, str(BENCH_DIR / "runners" / "run_monoprop.py"),
            "--problem", problem_path, "--coeff-cutoff", str(orch.MIN_ABS_COEFF),
            "--n-threads", str(n_threads)]


def cmd_propaq_native(problem_path: str, n_threads: int) -> list[str]:
    return [orch.PY, str(BENCH_DIR / "runners" / "run_propaq_native.py"),
            "--problem", problem_path, "--coeff-cutoff", str(orch.MIN_ABS_COEFF),
            "--n-threads", str(n_threads)]


def cmd_monoprop_native(problem_path: str, n_threads: int) -> list[str]:
    return [orch.PY, str(BENCH_DIR / "runners" / "run_monoprop_native.py"),
            "--problem", problem_path, "--coeff-cutoff", str(orch.MIN_ABS_COEFF),
            "--n-threads", str(n_threads)]

def _uid(system: str, nx: int, ny: int, n_step: int) -> str:
    return f"{system}__{nx}x{ny}__step{n_step}"



def _parse_sizes(s: str) -> list[tuple[int, int]]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        nx_s, ny_s = tok.lower().split("x")
        out.append((int(nx_s), int(ny_s)))
    return out


def run_one(uid: str, size_label: str, system: str, n_step: int, trial: int,
            backend_key: str, cmd: list[str], env: dict[str, str] | None = None,
            timeout: float | None = None) -> dict:
    print(f"  [{uid}] {backend_key} trial={trial} ...", end=" ", flush=True)
    t0 = time.time()
    rec = io_utils.run_backend(cmd, env=env, timeout=timeout)
    rec["size_label"] = size_label
    rec["problem_uid"] = uid
    rec["backend_key"] = backend_key
    rec["n_step"] = n_step
    rec["trial"] = trial
    # Which machine produced this timing. Records had no provenance field, which made a
    # curve spanning amr-131, acm-060 and amr-001 indistinguishable from one machine's
    # numbers, and those nodes differ by 1.15-1.8x on identical work (different memory
    # configurations, plus co-tenant load on the shared ones). Cheap to record, and the
    # only way to tell afterwards whether a comparison is apples-to-apples.
    rec["host"] = socket.gethostname()
    rec.setdefault("problem", system)
    if not rec.get("ok") or "backend" not in rec:
        rec.setdefault("backend", backend_key.split("_")[0] if "propaq" in backend_key else backend_key)
        rec.setdefault("basis", "majorana" if "majorana" in backend_key else "pauli")
    print(f"{'OK' if rec.get('ok') else 'FAIL'} ({time.time() - t0:.1f}s wall)")
    if not rec.get("ok"):
        print("    ", rec.get("error"), (rec.get("stderr_tail") or "")[-500:])
    return rec


def main() -> None:
    # Reject option-prefix typos such as --size. Using it previously silently selected --sizes
    # while retaining the default system set, which can launch unwanted scans.
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--quick", action="store_true", help="1 step, 1 trial, 3x3 only, for a smoke test")
    ap.add_argument("--sizes", default="5x5",
                     help="comma-separated NxN lattice sizes to run, e.g. 3x3,4x4,5x5,6x6")
    ap.add_argument("--n-trials", type=int, default=1)
    ap.add_argument("--systems", default="hubbard_trotter",
                     help="comma-separated subset of: " + ",".join(SYSTEMS))
    ap.add_argument("--out", default=str(BENCH_DIR / "results" / "trotter_scan.npz"))
    ap.add_argument("--checkpoint", default=str(BENCH_DIR / "results" / "trotter_scan.jsonl"))
    ap.add_argument("--n-threads", type=int, default=64,
                     help="matches orchestrate.py's default, for consistency with the rest of "
                          "the suite's results")
    ap.add_argument("--backends", default=None,
                     help="comma-separated allowlist of backend keys to run this pass, e.g. "
                          "'propaq_majorana,monoprop_majorana'. Lets a cheap backend's full "
                          "step range be collected before an expensive one is given a long "
                          "timeout, instead of the scan spending the allocation on one "
                          "backend's timeouts at every deep step. Default: all backends")
    ap.add_argument("--timeout", type=float, default=None,
                     help="per-backend-run wall-time cap in seconds. A run that exceeds it is "
                          "killed (whole process group) and recorded as failed, and the scan "
                          "continues, which is what keeps one backend's term-count blow-up "
                          "from consuming an allocation the other backends still need. "
                          "Default: no cap")
    ap.add_argument("--max-steps", type=int, default=None,
                     help="cap the Trotter-step range below each system's own max_steps, to "
                          "scope a run to the wall time available. The scan is checkpointed, "
                          "so a later run with a higher cap extends the same curves rather "
                          "than redoing them")
    ap.add_argument("--step-stride", type=int, default=1,
                     help="only run every Nth Trotter step/layer (e.g. 2 -> 1,3,5,...) to cut "
                          "cost on a system with cheap-enough per-step blowup that most of the "
                          "cost is the number of distinct step counts run, not any one of them "
                          "(e.g. ising_trotter). Combine with --systems to scope it")
    args = ap.parse_args()

    sizes = [(3, 3)] if args.quick else _parse_sizes(args.sizes)
    n_trials = 1 if args.quick else args.n_trials
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    done = io_utils.load_jsonl(args.checkpoint)
    done_keys = {(r["problem_uid"], r["backend_key"], int(r.get("trial") or 0)) for r in done if r.get("ok")}
    print(f"Resuming: {len(done_keys)} tasks already completed in {args.checkpoint}")

    selected = ({b.strip() for b in args.backends.split(",") if b.strip()}
                if args.backends else None)
    if selected:
        print(f"Backend allowlist for this pass: {', '.join(sorted(selected))}")

    def maybe_run(uid, size_label, system, n_step, backend_key, cmd, trial, env=None):
        if selected is not None and backend_key not in selected:
            return
        if (uid, backend_key, trial) in done_keys:
            print(f"  [{uid}] {backend_key} trial={trial} ... skip (already done)")
            return
        rec = run_one(uid, size_label, system, n_step, trial, backend_key, cmd, env=env,
                      timeout=args.timeout)
        io_utils.append_jsonl(args.checkpoint, rec)
        io_utils.save_records_npz(io_utils.load_jsonl(args.checkpoint), args.out)

    for system in systems:
        cfg = SYSTEMS[system]
        max_steps = 1 if args.quick else cfg["max_steps"]
        if args.max_steps is not None:
            max_steps = min(max_steps, args.max_steps)
        for nx, ny in sizes:
            size_label = f"{nx}x{ny}"
            print(f"=== {system} {size_label} (steps 1..{max_steps} stride {args.step_stride}, "
                  f"{n_trials} trial(s)) ===")
            for n_step in range(1, max_steps + 1, args.step_stride):
                uid = _uid(system, nx, ny, n_step)

                if cfg["qubit_backends"] or cfg.get("monoprop_qubit_backend", False):
                    ir = cfg["builder"](nx, ny, steps=n_step)
                    path = str(RESULTS_RAW / f"{uid}.json")
                    ir.save(path)

                    # propaq's own from_qiskit declares a native rotation basis (xx_plus_yy, p,
                    # rz, cp, x, swap, rx, ry, rzz, rxx, ryy, rzx, see
                    # propaq/circuits/_gates.py's NATIVE_GATES) that both ising_trotter and
                    # hubbard_trotter are entirely built from as originally constructed.
                    # Canonicalizing onto COMMON_BASIS for the other three backends introduces
                    # gates outside that set (h, for one, 69 of them for a single
                    # 3x3/step=1 hubbard_trotter circuit), which propaq then has to
                    # re-decompose *again* internally, on top of the first transpile. Giving
                    # propaq the uncanonicalized circuit directly avoids that redundant
                    # double-decomposition.
                    ir_native = cfg["builder"](nx, ny, steps=n_step, canonicalize=False)
                    path_native = str(RESULTS_RAW / f"{uid}__propaq_native.json")
                    ir_native.save(path_native)

                    backends = dict(QUBIT_BACKENDS) if cfg["qubit_backends"] else {}
                    if cfg.get("monoprop_qubit_backend", False):
                        backends["monoprop_pauli"] = cmd_monoprop
                    for backend_key, cmdfn in backends.items():
                        # propaq_majorana gets a truly different encoding of the problem for
                        # native_fermionic systems (built straight from an ffsim FermionOperator
                        # via propaq's from_ffsim, no qubit gates or Jordan-Wigner string at all,
                        # see runners/run_propaq_native.py) instead of reinterpreting the
                        # qubit circuit in Majorana operators. propaq_pauli has no such native
                        # path and stays on the qubit-gate encoding regardless. These two now
                        # deliberately diverge.
                        if backend_key == "propaq_majorana" and (
                            cfg["native_fermionic"]
                            or SKIP_PROPAQ_MAJORANA
                            or not cfg.get("qubit_majorana", True)
                        ):
                            continue
                        backend_path = path_native if backend_key.startswith("propaq") else path
                        for trial in range(n_trials):
                            # pyrauli's OpenMP thread count is process-wide via OMP_NUM_THREADS,
                            # set before the process starts (matches orchestrate.py's handling).
                            env = (dict(os.environ, OMP_NUM_THREADS=str(args.n_threads))
                                   if backend_key == "pyrauli" else None)
                            maybe_run(uid, size_label, system, n_step, backend_key,
                                      cmdfn(backend_path, args.n_threads), trial, env=env)

                if cfg["native_fermionic"]:
                    fp = problems_fermionic.hubbard_trotter_fermionic(nx, ny, steps=n_step)
                    fpath = str(RESULTS_RAW / f"{uid}__native.json")
                    fp.save(fpath)
                    for trial in range(n_trials):
                        maybe_run(uid, size_label, system, n_step, "majorana_propagation_jl",
                                  orch.cmd_majorana_propagation_jl(fpath, args.n_threads), trial)
                    if not SKIP_PROPAQ_MAJORANA:
                        for trial in range(n_trials):
                            maybe_run(uid, size_label, system, n_step, "propaq_majorana",
                                      cmd_propaq_native(fpath, args.n_threads), trial)
                    for trial in range(n_trials):
                        maybe_run(uid, size_label, system, n_step, "monoprop_majorana",
                                  cmd_monoprop_native(fpath, args.n_threads), trial)

    all_records = io_utils.load_jsonl(args.checkpoint)
    n_ok = sum(1 for r in all_records if r.get("ok"))
    print(f"\n{len(all_records)} total records ({n_ok} ok, {len(all_records) - n_ok} failed). "
          f"Checkpoint: {args.checkpoint}  Snapshot: {args.out}")


if __name__ == "__main__":
    main()
