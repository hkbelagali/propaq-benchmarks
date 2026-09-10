#!/usr/bin/env python3
"""Benchmark native UniformNoiseModel ABI plugins on a fixed QAOA MaxCut circuit."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BENCH_DIR = ROOT.parent
sys.path.insert(0, str(BENCH_DIR))

from propaq_benchmarks import io_utils  # noqa: E402

BUILD = ROOT / "build"
RESULTS = ROOT / "results"
# Data (checkpoint, CSV, npz) stays next to the experiment.
# The figure goes to the suite-wide plots directory, so every plot in the repo lives in
# one place.
PLOTS = BENCH_DIR / "results" / "plots"
DAMPING = 0.005
BENCHMARK_VERSION = 2  # v1 plugins accidentally hard-coded damping=0.001.


def shared_library(name: str) -> str:
    if sys.platform == "darwin":
        return f"lib{name}.dylib"
    if sys.platform == "win32":
        return f"{name}.dll"
    return f"lib{name}.so"


def command(args: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True, env=env)


def build_plugins(skip_build: bool, rebuild: bool) -> dict[str, Path]:
    c_lib = BUILD / shared_library("uniform_noise_c")
    rust_lib = ROOT / "rust" / "target" / "release" / shared_library("propaq_uniform_noise_benchmark_plugin")
    julia_raw_lib = BUILD / "julia" / "lib" / shared_library("uniform_noise_julia")
    julia_lib = BUILD / shared_library("uniform_noise_julia_plugin")
    libraries = {"C plugin": c_lib, "Rust plugin": rust_lib, "AOT Julia plugin": julia_lib}
    if skip_build:
        missing = [str(path) for path in libraries.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("--skip-build requested but libraries are missing: " + ", ".join(missing))
        return libraries

    BUILD.mkdir(exist_ok=True)
    compiler = os.environ.get("CC", "cc")
    if rebuild or not c_lib.exists():
        if sys.platform == "darwin":
            command([compiler, "-O3", "-dynamiclib", "c/uniform_noise.c", "-o", str(c_lib), "-lm"])
        elif sys.platform == "win32":
            raise RuntimeError("This experiment's build commands currently target Unix-like platforms.")
        else:
            command([compiler, "-O3", "-fPIC", "-shared", "c/uniform_noise.c", "-o", str(c_lib), "-lm"])
    if rebuild or not rust_lib.exists():
        command(["cargo", "build", "--release", "--manifest-path", "rust/Cargo.toml"])
    # Julia 1.11-era PackageCompiler can hang or crash while producing its
    # temporary sysimage with more than one Julia thread. The plugin itself is
    # thread-safe and propaq calls it concurrently; this only serializes AOT
    # compilation, which is outside the measured region.
    julia_env = os.environ.copy()
    julia_env["JULIA_NUM_THREADS"] = "1"
    if rebuild or not julia_raw_lib.exists():
        command(["julia", "build_julia.jl"], env=julia_env)
    if rebuild or not julia_lib.exists():
        rpath = "$ORIGIN/julia/lib"
        command([compiler, "-O3", "-fPIC", "-shared", "c/julia_noise_shim.c", "-o", str(julia_lib),
                 "-ldl", "-lpthread", f"-Wl,-rpath,{rpath}"])
    return libraries


def qaoa_problem(n_qubits: int, layers: int, seed: int):
    """Create a deterministic 3-regular MaxCut QAOA circuit and Z-mid observable."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    rng = np.random.default_rng(seed)
    gammas = rng.uniform(0.0, np.pi, layers)
    betas = rng.uniform(0.0, np.pi / 2.0, layers)
    circuit = QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits))
    # Cycle plus antipodal matching, a fixed simple 3-regular graph.
    edges = [(q, (q + 1) % n_qubits) for q in range(n_qubits)]
    edges += [(q, q + n_qubits // 2) for q in range(n_qubits // 2)]
    for gamma, beta in zip(gammas, betas):
        for left, right in edges:
            circuit.rzz(2.0 * gamma, left, right)
        for qubit in range(n_qubits):
            circuit.rx(2.0 * beta, qubit)
    label = ["I"] * n_qubits
    label[n_qubits - 1 - n_qubits // 2] = "Z"  # Qiskit labels are big-endian.
    return circuit, SparsePauliOp("".join(label))


def prepare_timing(noise, circuit, observable, threads: int):
    from propaq import CoefficientTruncator
    from propaq.circuits import PauliCircuit
    from propaq.datatypes import PauliTermSum
    from propaq.propagators import PauliPropagator

    native_circuit = PauliCircuit.from_qiskit(circuit)
    native_observable = PauliTermSum.from_sparse_pauli_op(observable)
    # propaq 0.1.3 removed FlushSchedule; the engine folds duplicates on insert.
    kwargs = dict(truncation=[CoefficientTruncator(1e-8)], n_threads=threads,
                  progress_bar=True)
    # Exclude Rayon pool creation and library page faults from measured propagation.
    PauliPropagator(noise=noise, **kwargs).expectation_value(native_observable, native_circuit, initial_state=0)
    return native_circuit, native_observable, kwargs


def time_once(native_circuit, native_observable, kwargs) -> float:
    from propaq.propagators import PauliPropagator

    propagator = PauliPropagator(**kwargs)
    start = time.perf_counter()
    propagator.expectation_value(native_observable, native_circuit, initial_state=0)
    return time.perf_counter() - start


def load_checkpoint(path: Path, config: dict[str, object]) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # An interrupted append can leave a partial final line.
                if line_number == sum(1 for _ in path.open()):
                    break
                raise
            if record.get("kind") == "metadata":
                if record.get("config") != config:
                    raise ValueError(f"checkpoint {path} belongs to a different benchmark configuration; choose --checkpoint")
            elif record.get("kind") == "sample":
                records.append(record["row"])
    return records


def append_checkpoint(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_outputs(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with (RESULTS / "runtime_vs_qaoa_layers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    io_utils.save_records_npz(rows, str(RESULTS / "runtime_vs_qaoa_layers.npz"))


def plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    })
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import scienceplots  # Registers the "science" and "grid" styles.

    plt.style.use(["science", "grid"])

    summary: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        summary.setdefault(str(row["model"]), {}).setdefault(int(row["layers"]), []).append(float(row["runtime_s"]))
    textwidth = 3.31314
    width = textwidth * 1.3
    fig, ax = plt.subplots(figsize=(width, width * 6 / 8))
    for model, per_layer in summary.items():
        layers = sorted(per_layer)
        ax.plot(layers, [np.median(per_layer[p]) for p in layers], marker="x",
                linewidth=1.6, markersize=8, markeredgewidth=1.5, label=model, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("QAOA layer", fontsize=13)
    ax.set_ylabel("Propagation Wall Time (s)", fontsize=13)
    all_layers = sorted({int(row["layers"]) for row in rows})
    ax.set_xticks(all_layers, labels=all_layers, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    legend = ax.legend(fancybox=False, edgecolor="black", loc="best", fontsize=9)
    legend.get_frame().set_linewidth(0.5)
    ax.grid(False)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pgf"), bbox_inches="tight")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--damping", type=float, default=DAMPING,
                        help=f"uniform per-weight damping rate (default: {DAMPING})")
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="rebuild all plugin libraries")
    parser.add_argument("--checkpoint", type=Path, default=RESULTS / "runtime_vs_qaoa_layers.jsonl",
                        help="durable JSONL checkpoint, reruns automatically resume it")
    args = parser.parse_args()
    if args.qubits < 4 or args.qubits % 2 or args.repeats < 1 or args.threads < 1 or args.damping < 0 or any(p < 1 for p in args.layers):
        parser.error("--qubits must be even and >=4; layers, repeats, threads, and damping must be non-negative")

    config = {"benchmark_version": BENCHMARK_VERSION, "layers": args.layers, "repeats": args.repeats, "qubits": args.qubits,
              "damping": args.damping, "threads": args.threads, "seed": args.seed}
    rows = load_checkpoint(args.checkpoint, config)
    if not args.checkpoint.exists():
        append_checkpoint(args.checkpoint, {"kind": "metadata", "config": config})
    from propaq.noise import NativeNoiseModel, UniformNoiseModel
    libraries = build_plugins(args.skip_build, args.rebuild)
    config_json = json.dumps({"damping": args.damping})
    models = {"native": UniformNoiseModel(args.damping)}
    models.update({name: NativeNoiseModel(str(path), config=config_json) for name, path in libraries.items()})
    RESULTS.mkdir(exist_ok=True)
    completed = {(str(row["model"]), int(row["layers"]), int(row["sample"])) for row in rows}
    for layers in args.layers:
        circuit, observable = qaoa_problem(args.qubits, layers, args.seed)
        for name, noise in models.items():
            pending = [sample for sample in range(args.repeats) if (name, layers, sample) not in completed]
            if pending:
                native_circuit, native_observable, kwargs = prepare_timing(noise, circuit, observable, args.threads)
                for sample in pending:
                    value = time_once(native_circuit, native_observable, dict(noise=noise, **kwargs))
                    row = {"model": name, "layers": layers, "sample": sample, "runtime_s": value,
                           "qubits": args.qubits, "damping": args.damping, "threads": args.threads,
                           "seed": args.seed}
                    append_checkpoint(args.checkpoint, {"kind": "sample", "row": row})
                    rows.append(row)
                    completed.add((name, layers, sample))
                    write_outputs(rows)
                    print(f"p={layers:2d} {name:16s} sample={sample} runtime={value:.6f}s", flush=True)
            samples = [float(row["runtime_s"]) for row in rows if row["model"] == name and row["layers"] == layers]
            print(f"p={layers:2d} {name:16s} median={np.median(samples):.6f}s ({len(samples)}/{args.repeats} samples)")
    write_outputs(rows)
    plot(rows, PLOTS / "runtime_vs_qaoa_layers.png")
    metadata = {"platform": platform.platform(), "python": sys.version, "layers": args.layers,
                "repeats": args.repeats, "qubits": args.qubits, "damping": args.damping,
                "threads": args.threads, "libraries": {key: str(value) for key, value in libraries.items()}}
    io_utils.save_records_npz([metadata], str(RESULTS / "metadata.npz"))


if __name__ == "__main__":
    main()
