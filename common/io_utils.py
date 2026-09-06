"""Shared subprocess-timing and npz (de)serialization helpers used across the benchmark suite.

Every backend runner is invoked as an isolated subprocess (fresh memory, no state leakage
between runs, and one language-agnostic way to measure peak RSS via /usr/bin/time -v,
since Python's resource module, Julia's GC stats, and Rust's allocator all report memory
differently).
The runner's own internal wall_time_s (measured around just the propagation call,
excluding process, JIT, and import startup) is the authoritative timing number.
GNU time's wall clock is kept too as process_wall_time_s for context.

Every benchmark script in this repo appends its result records to a JSONL checkpoint file
with append_jsonl (so a killed run can resume by skipping already-completed records) and
snapshots the whole checkpoint to a single .npz file with save_records_npz after every
record.
The npz snapshot is what every plotting script reads.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
from typing import Any

import numpy as np

_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
_WALL_RE = re.compile(r"Elapsed \(wall clock\) time.*: (\d+):(\d+\.\d+|\d+)")


def run_backend(cmd: list[str], env: dict[str, str] | None = None, timeout: float | None = 1800) -> dict[str, Any]:
    """Run a runner script under /usr/bin/time -v and return a merged result record.

    The runner's JSON stdout line and GNU time's stderr are parsed together.
    On failure this returns a record with ok=False and the captured stderr tail instead of
    raising, so a single bad run does not abort the whole sweep.

    timeout=None waits indefinitely (subprocess.communicate's own accepted meaning), used
    by trotter/run_trotter_scan.py, which wants no cap even under Hubbard's term-count
    blowup.
    orchestrate.py and scaling/run_scaling.py do not override the 1800s default, since
    those run unattended under a SLURM wall-time budget and still want a single bad run to
    fail rather than block the whole allocation.

    The process is launched in its own process group (start_new_session=True) and killed
    via that group on timeout.
    subprocess.run(..., timeout=)'s default handling only signals the direct child
    (/usr/bin/time), which dies without a chance to also kill its own child (the actual
    runner), leaking an orphaned, still-resource-consuming process for every timeout.
    That matters here, a term-count blowup is exactly the case where a run times out and
    is memory hungry at the same time.
    """
    full_cmd = ["/usr/bin/time", "-v"] + cmd
    proc = subprocess.Popen(
        full_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()  # reap now-dead children, avoid a zombie
        return {"ok": False, "error": "timeout", "cmd": " ".join(cmd)}

    record: dict[str, Any] = {"ok": proc.returncode == 0, "cmd": " ".join(cmd)}

    m = _RSS_RE.search(stderr)
    record["peak_rss_mb"] = float(m.group(1)) / 1024.0 if m else None
    m = _WALL_RE.search(stderr)
    if m:
        record["process_wall_time_s"] = float(m.group(1)) * 60 + float(m.group(2))

    # The runner's JSON is the last non-empty line of stdout.
    json_line = None
    for line in reversed(stdout.strip().splitlines()):
        if line.strip().startswith("{"):
            json_line = line.strip()
            break
    if json_line is not None:
        try:
            record.update(json.loads(json_line))
        except json.JSONDecodeError:
            record["ok"] = False
            record["error"] = "bad_json"
    else:
        record["ok"] = False
        record["error"] = "no_json_output"

    if not record["ok"]:
        record["stderr_tail"] = "\n".join(stderr.strip().splitlines()[-40:])
        record["stdout_tail"] = "\n".join(stdout.strip().splitlines()[-20:])

    return record


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    """Append one record as a line to a checkpoint file, fsync'd so it survives a hard kill."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


_JSON_KEYS_COLUMN = "__json_keys__"


def save_records_npz(records: list[dict[str, Any]], path: str) -> None:
    """Flatten a list of (possibly heterogeneous) result-record dicts into one .npz file.

    Every key that appears in any record becomes one array, column-major, one entry per
    record.
    A key whose values are all booleans becomes a bool array.
    A key whose values are all numbers becomes a float64 array, with NaN for a record that
    is missing it.
    A key whose values are all strings becomes an object array of str, with an empty
    string for a record that is missing it.
    Anything else (a dict, a list, or a key with no non-missing value at all) is JSON
    encoded into a string column instead, so no information is dropped, and the set of
    JSON-encoded keys is stored alongside so load_records_npz can decode them back.
    """
    keys = sorted({k for r in records for k in r})
    arrays: dict[str, np.ndarray] = {}
    json_keys: list[str] = []
    for key in keys:
        values = [r.get(key) for r in records]
        present = [v for v in values if v is not None]
        if present and all(isinstance(v, bool) for v in present):
            arrays[key] = np.array([bool(v) for v in values], dtype=bool)
        elif present and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
            arrays[key] = np.array([float(v) if v is not None else np.nan for v in values], dtype=np.float64)
        elif present and all(isinstance(v, str) for v in present):
            arrays[key] = np.array([v if v is not None else "" for v in values], dtype=object)
        else:
            json_keys.append(key)
            arrays[key] = np.array([json.dumps(v) for v in values], dtype=object)
    arrays[_JSON_KEYS_COLUMN] = np.array(json_keys, dtype=object)
    np.savez_compressed(path, **arrays)


def load_records_npz(path: str) -> list[dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    json_keys = set(data[_JSON_KEYS_COLUMN].tolist()) if _JSON_KEYS_COLUMN in data.files else set()
    keys = [k for k in data.files if k != _JSON_KEYS_COLUMN]
    n = len(data[keys[0]]) if keys else 0
    records: list[dict[str, Any]] = []
    for i in range(n):
        rec: dict[str, Any] = {}
        for key in keys:
            arr = data[key]
            v = arr[i]
            if key in json_keys:
                rec[key] = json.loads(str(v))
            elif arr.dtype == bool:
                rec[key] = bool(v)
            elif np.issubdtype(arr.dtype, np.floating):
                rec[key] = None if np.isnan(v) else float(v)
            else:
                rec[key] = str(v)
        records.append(rec)
    return records
