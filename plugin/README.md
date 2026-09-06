# Uniform-noise plugin benchmark

This experiment compares propaq's built-in `UniformNoiseModel` with ABI plugins
implementing the identical `exp(-damping * term_weight)` model in C, AOT-compiled
Julia, and Rust. It measures only `expectation_value`: circuit construction,
plugin compilation, Julia AOT compilation, and plotting are outside the timer.

## Run

From this directory:

```bash
python3 run.py --layers 1 2 3 4 5 6 --repeats 7
```

This builds the C, Rust, and Julia AOT libraries. Results are written to
`results/runtime_vs_qaoa_layers.csv` and `results/runtime_vs_qaoa_layers.npz`.
The figure and its matching PGF go to the suite-wide
`../results/plots/runtime_vs_qaoa_layers.{png,pgf}`, where every plot in the repo lives.
Only this experiment's data stays here.
Plot styling matches the Trotter scan's SciencePlots and LaTeX configuration.

Progress is durable and resumable. Every timing sample is appended and fsynced to
`results/runtime_vs_qaoa_layers.jsonl`.
Rerun the exact command after an interruption and completed samples are skipped.
Use `--checkpoint PATH` to keep independent parameter sweeps separate.

To render a plot from a completed or partial run without running more measurements:

```bash
python3 plot.py --results results/runtime_vs_qaoa_layers.npz
```

The default workload is a fixed 16-qubit, 3-regular MaxCut QAOA instance (a cycle plus
antipodal matching).
Each plotted point is the median of the requested repetitions.
The CSV and npz files contain every sample.
All models use `damping=0.01`, one worker, and the same coefficient cutoff.
The damping value is passed to every ABI plugin through `propaq_noise_create`'s JSON
configuration, so it matches the built-in model.
Use `--damping` or `--threads` to change runtime benchmark parameters.

Prerequisites: `numpy`, `matplotlib`, `qiskit`, local `propaq`, a C compiler,
Cargo, and Julia with `PackageCompiler.jl`. `--skip-build` uses existing libs.
After a successful build, ordinary invocations reuse the compiled libraries.
Pass `--rebuild` after changing plugin source.

`run.py` builds the Julia library with `JULIA_NUM_THREADS=1`. This avoids a Julia
1.11/PackageCompiler temporary-sysimage failure seen with multithreaded compilation.
It does not affect benchmark propagation, which uses `--threads`.
If Julia still segfaults, update PackageCompiler to at least v2.1.22 and retry.

The PackageCompiler output is loaded through a small C shim. The shim runs the required
`init_julia` routine once before propaq invokes any Julia ABI symbol.
The raw PackageCompiler `.so` must not be passed to `NativeNoiseModel` directly.
It intentionally retains standard-library artifacts such as OpenBLAS, which are needed
while initializing Julia in Python.
The shim dispatches every Julia ABI call to one dedicated Julia-owned thread.
propaq may invoke plugins concurrently from Rayon workers, and the embedded Julia
runtime cannot be entered from those foreign threads.
