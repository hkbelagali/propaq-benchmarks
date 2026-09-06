#!/usr/bin/env python3
"""Fill in the propaq rows the trotter scan is missing, without re-running other backends.

`results/trotter_scan.jsonl` had its propaq records removed when the monoprop-shaped
engine landed, so the file carries every other backend but nothing to compare them
against. Re-running `run_trotter_scan.py` would rebuild the gaps in *every* backend's
coverage too, which is hours of work nobody asked for. This runs propaq alone, on
exactly the problems that already carry a row for `--match-backend`, so the scan gains
a like-for-like comparison point and nothing else moves.

Records are appended in `run_trotter_scan.run_one`'s format and are resumable on the
same (problem_uid, backend_key, trial) key, so re-running skips what is already there.

Usage:
  python3 trotter/fill_propaq_rows.py --engine monoprop
  python3 trotter/fill_propaq_rows.py --engine monoprop --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(BENCH_DIR / "trotter"))

from common import io_utils  # noqa: E402
import orchestrate as orch  # noqa: E402
import run_trotter_scan as scan  # noqa: E402

RESULTS_RAW = BENCH_DIR / "results" / "raw" / "trotter"


def main() -> None:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--engine", choices=("soa", "monoprop"), default="monoprop",
                    help="value for PROPAQ_ENGINE; records are tagged propaq_pauli_<engine>")
    ap.add_argument("--match-backend", default="monoprop_pauli",
                    help="only run problems that already carry a row for this backend, so every "
                         "new record has something to be compared against")
    ap.add_argument("--checkpoint", default=str(BENCH_DIR / "results" / "trotter_scan.jsonl"))
    ap.add_argument("--out", default=str(BENCH_DIR / "results" / "trotter_scan.npz"))
    ap.add_argument("--n-threads", type=int, default=64)
    ap.add_argument("--pair-rule", action="store_true",
                    help="run with PROPAQ_PAIR_RULE=1, which matches monoprop's truncation rule "
                         "and so its term count; tags records propaq_pauli_<engine>_pair")
    ap.add_argument("--rerun-match-backend", action="store_true",
                    help="also re-measure --match-backend on this node, tagged <backend>_rerun. "
                         "The rows already in the scan were taken in other sessions on other "
                         "nodes and are cold, so a ratio taken across them is not a measurement "
                         "of either engine; this is what makes the file internally comparable")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    backend_key = f"propaq_pauli_{args.engine}" + ("_pair" if args.pair_rule else "")

    done = io_utils.load_jsonl(args.checkpoint)
    done_keys = {(r["problem_uid"], r["backend_key"], int(r.get("trial") or 0))
                 for r in done if r.get("ok")}

    # The problems worth running, in the order the scan would have produced them.
    targets = []
    seen = set()
    for rec in done:
        if rec.get("backend_key") != args.match_backend or not rec.get("ok"):
            continue
        uid = rec["problem_uid"]
        if uid in seen:
            continue
        seen.add(uid)
        targets.append((uid, rec["size_label"], rec["problem"], int(rec["n_step"]),
                        int(rec.get("trial") or 0)))
    targets.sort(key=lambda t: (t[2], t[1], t[3]))

    print(f"{len(targets)} problems carry a {args.match_backend} row; "
          f"writing {backend_key} for the ones not already done")
    for uid, size_label, system, n_step, trial in targets:
        if args.rerun_match_backend:
            rerun_key = f"{args.match_backend}_rerun"
            if (uid, rerun_key, trial) not in done_keys:
                cmd = scan.cmd_monoprop(str(RESULTS_RAW / f"{uid}.json"), args.n_threads)
                if args.dry_run:
                    print(f"  [{uid}] would rerun {rerun_key}")
                else:
                    rec = scan.run_one(uid, size_label, system, n_step, trial, rerun_key, cmd)
                    io_utils.append_jsonl(args.checkpoint, rec)

        if (uid, backend_key, trial) in done_keys:
            print(f"  [{uid}] skip (already done)")
            continue
        # propaq reads the uncanonicalized circuit, as run_trotter_scan does. Without
        # that file there is nothing to run. This script deliberately does not build
        # problems, so that it cannot disturb the inputs other backends were measured on.
        problem_path = RESULTS_RAW / f"{uid}__propaq_native.json"
        if not problem_path.exists():
            print(f"  [{uid}] skip (no {problem_path.name})")
            continue
        cmd = orch.cmd_propaq(str(problem_path), "pauli", args.n_threads)
        # Always explicit. The engine defaults the pair rule on, so leaving it
        # unset would run it for records tagged as not using it.
        env = dict(os.environ, PROPAQ_ENGINE=args.engine,
                   PROPAQ_PAIR_RULE="1" if args.pair_rule else "0")
        if args.dry_run:
            print(f"  [{uid}] would run {' '.join(cmd)}")
            continue
        rec = scan.run_one(uid, size_label, system, n_step, trial, backend_key, cmd, env=env)
        rec["propaq_engine"] = args.engine
        rec["propaq_pair_rule"] = bool(args.pair_rule)
        io_utils.append_jsonl(args.checkpoint, rec)
        io_utils.save_records_npz(io_utils.load_jsonl(args.checkpoint), args.out)


if __name__ == "__main__":
    main()
