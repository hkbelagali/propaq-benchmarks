# propaq-benchmark-suite / bench

Cross-package benchmark suite comparing six Heisenberg-picture (Pauli/Majorana) operator
backpropagation implementations on the same problems, with the same resources.

| Package | Language | Basis | Thread control |
|---|---|---|---|
| [pauli-prop](../pauli-prop) | Rust+Python | Pauli | none (confirmed single-threaded) |
| [PauliPropagation.jl](../PauliPropagation.jl) | Julia | Pauli | `julia -t N` (`VectorPauliSum`) |
| [pyrauli](../pyrauli) | C++/Python | Pauli | `OMP_NUM_THREADS` env var + `runtime=par` |
| [propaq](../propaq) | Rust+Python | **Pauli and Majorana** | `n_threads=` kwarg (per-instance Rayon pool) |
| [MajoranaPropagation.jl](../MajoranaPropagation.jl) | Julia | Majorana | `julia -t N` (`VectorMajoranaSum`) |
| [monoprop](../monoprop) | C++/Python | **Pauli and Majorana** | `monoprop_NUM_THREADS` env var |

`propaq` and `monoprop` each supporting both bases on the same input circuit is what makes
a same-package, same-input Pauli-vs-Majorana comparison possible, in addition to the
cross-package ones.

## Directory layout

Every experiment lives in its own folder under `experiments/`. Each one saves its circuits
once, then has one `run_<backend>` file per package that loads those saved circuits and
reports results directly. There is no orchestrator process spawning these as subprocesses,
running `python3 run_propaq.py` (or `julia ... run_pauli_propagation_jl.jl`) is the whole
interface.

```
bench/
├── propaq-benchmarks/             shared circuit IR, problem builders, checkpoint/npz helpers, packaged
│   ├── pyproject.toml              pinned deps (numpy, qiskit), pip install -e'd for the Python side
│   ├── src/propaq_benchmarks/
│   │   ├── circuit_ir.py            ProblemIR: serializes a transpiled circuit + observable
│   │   ├── experiment_runner.py    loads saved circuits and checkpoints results, used by every run_<backend>.py
│   │   ├── io_utils.py             JSONL checkpoint + npz (de)serialization, and the plotting-side merge helper
│   │   ├── problems_qubit.py        the 8 qubit-suite problem builders
│   │   └── problems_fermionic.py    native-fermionic companion builders (hubbard_trotter, random_fermionic_circuit)
│   └── julia/BenchCommon/          Julia package, Pkg.develop-ed into julia_env alongside the two Julia backends
│       ├── Project.toml
│       └── src/
│           ├── BenchCommon.jl       top-level module, includes the two below as nested submodules
│           ├── circuit_ir.jl        Julia-side reader for the same IR (BenchCommon.CircuitIR)
│           └── experiment_runner.jl the same runner, for the two Julia-only backends (BenchCommon.ExperimentRunner)
├── experiments/
│   ├── ising_trotter/              one of the 8 main problems, see the full layout below
│   ├── random_circuit/             same file layout as ising_trotter
│   ├── random_near_clifford/       same file layout
│   ├── heisenberg_chain_trotter/   same file layout
│   ├── qaoa_maxcut/                same file layout
│   ├── ucj_h2/                     same file layout, one saved circuit instead of four
│   ├── hubbard_trotter/            qubit-side layout plus a native-fermionic side, see below
│   ├── random_fermionic_circuit/   qubit-side layout plus a native-fermionic side, see below
│   ├── thread_scaling/             cross-backend thread-count sweep on two fixed circuits
│   ├── clifford_deferral/          propaq only, Clifford deferral on vs off
│   ├── custom_decomposition/       propaq only
│   ├── hybrid_mps_heisenberg/      propaq only
│   ├── hybrid_ucj_heisenberg/      propaq only
│   ├── surrogate_optimization/     propaq only
│   └── propaq_thread_scaling/      propaq only, Trotter step count x thread count sweep
├── plotting/                      matplotlib scripts, one per experiment, reading its results_<backend>.npz files
├── extrapolators/                 zero-noise and zero-coefficient extrapolation studies
├── plugin/                        native ABI noise-plugin benchmark (C, Rust, AOT Julia)
├── slurm/                         rebuild_native.sh, for building native extensions on a compute node
├── julia_env/                     shared Julia Project.toml/Manifest.toml, all three Julia packages Pkg.develop-ed into it
└── results/
    └── plots/                     every figure in the suite, flat, both .png and .pgf
```

`experiments/ising_trotter/`'s full layout, which every one of the 8 main problems follows:

```
experiments/ising_trotter/
├── generate_circuits.py            builds and saves every circuit instance, run this once
├── circuits/                       the saved ProblemIR json files, one per size
│   ├── 3x3_steps10.json             a warm-up size
│   ├── 4x4_steps12.json             a second warm-up size
│   └── 6x6_steps1.json .. 6x6_steps25.json   a fine Trotter-step curve, one circuit per step
├── run_pauli_prop.py                python3 run_pauli_prop.py, no arguments needed
├── run_pyrauli.py
├── run_monoprop.py
├── run_propaq.py                    runs both the Pauli and Majorana basis
└── run_pauli_propagation_jl.jl      julia --project=../../julia_env -t 64 run_pauli_propagation_jl.jl
```

`ising_trotter` and `hubbard_trotter` each save a fine Trotter-step curve (one circuit per
step count from 1 to 25, at a fixed lattice size) alongside their small discrete sizes,
since `plotting/plot_trotter_scan.py` and `plotting/plot_trotter_memory.py` plot runtime and
peak RSS against Trotter step, not against problem size. `hubbard_trotter`'s fine curve is
native-fermionic only (`circuits_native/`, not `circuits/`), since its qubit/Jordan-Wigner
side blows up several steps sooner than the native side and cannot reach as deep. Running
the slowest backend across the full 25-step curve at 6x6 takes a while, each step count is
an independent from-scratch circuit build and propagation run, not a single run
instrumented with mid-circuit checkpoints, so cost grows like steps squared overall.

`hubbard_trotter` and `random_fermionic_circuit` additionally have a `circuits_native/`
folder (an independently built native-fermionic circuit, see "How a fair comparison is
constructed" below) and extra native-only backend files: both get
`run_majorana_propagation_jl.jl`, and `hubbard_trotter` additionally gets
`run_propaq_native.py` and `run_monoprop_native.py` (checkpointed to
`results_propaq_native.*`/`results_monoprop_native.*`, distinct from the qubit-side
`run_propaq.py`/`run_monoprop.py`'s own `results_propaq.*`/`results_monoprop.*` in the same
folder, since these are genuinely different measurements of the same nominal instance).

Running any `run_<backend>` file writes `results_<backend>.jsonl` (an appendable
checkpoint, fsynced after every circuit) and `results_<backend>.npz` (a snapshot of the
same records, rebuilt after every circuit, which is what every plotting script reads)
directly into that experiment's folder. Killing a run and re-running the same command
resumes from the next circuit not already recorded with `ok=true`.

## How a fair comparison is constructed

**Qubit/Pauli-basis problems** are built once in Qiskit, then transpiled (`transpile(...,
basis_gates=["rz","rx","ry","rzz","cx","h"])`, once, in `propaq_benchmarks/circuit_ir.py`) onto a
common gate basis every backend supports natively or via an exact (non-approximating)
decomposition. The resulting gate list and observable are serialized (`propaq_benchmarks/circuit_ir.py`'s
`ProblemIR`) by each experiment's `generate_circuits.py`, then replayed identically by
`pauli-prop` and `PauliPropagation.jl`, or handed to `propaq`/`pyrauli`/`monoprop`'s own
`from_qiskit`-style importer (which may further transpile into that package's native basis
internally, which is expected and how a real user of that library would use it). Every
qubit-suite circuit was validated end to end. Exact `Statevector` expectation value versus
every Pauli backend agreed to about 1e-6 or better (see git history for the validation
runs).

`propaq`'s and `monoprop`'s Majorana modes also consume these same qubit circuits via
Jordan-Wigner mapping, giving a same-input Pauli-vs-Majorana comparison for every
qubit-suite problem.

**Fermionic-native problems** (`hubbard_trotter`, `random_fermionic_circuit`) additionally
get a second, independently constructed instance built directly from
`MajoranaPropagation.jl`'s own fermionic gate builders (`hubbard_circ_fermionic_sites`,
random `MajoranaRotation`s), with matching physical parameters (t, U, dt, steps / n_modes,
n_gates) to the qubit-suite/JW version. This is not a bit-identical circuit (different RNG,
native fermionic gates versus JW-mapped qubit gates), but it is the same physical model at
the same size. `MajoranaPropagation.jl` has no Qiskit bridge, so an exact shared IR is not
possible for it, this is the closest fair comparison available. `hubbard_trotter` goes
further and also feeds this native instance directly to propaq (via ffsim's
`FermionOperator`, skipping the Jordan-Wigner embedding entirely) and to monoprop, since
going through Jordan-Wigner never gives Majorana propagation its expected term-count
locality advantage. Both of those are checkpointed under distinct backend names
(`propaq_native`, `monoprop_native`) so they are never confused with the same package's
ordinary Jordan-Wigner measurement in the same folder.

**Observable convention**: a single local `Z` (or `ZZ` for UCJ-H2, matching this repo's
pre-existing `propaq/benchmarks/bench_ucj.py` convention) on or near the middle qubit for
every problem. This keeps the initial term count at 1 everywhere so runtime differences
reflect propagation cost, not observable-parsing overhead.

## The 8 problems

1. **random_circuit**, a brickwork random circuit, {rz,rx,ry,h} + {cx,rzz}.
2. **random_near_clifford**, a Clifford skeleton (H/CX) with a swept density of
   non-Clifford RZ ("magic") gates. Probes how each backend's term count scales with
   non-Cliffordness, since Clifford gates never branch a Pauli/Majorana sum.
3. **ising_trotter**, a 2D transverse-field Ising model Trotter circuit, default
   6x6=36 qubits per the requested size, bricklayer ZZ+X layers.
4. **heisenberg_chain_trotter**, a 1D Heisenberg XXZ chain Trotter circuit
   (non-integrable regime, complementing Ising's free-fermion-dual integrable case).
5. **qaoa_maxcut**, a QAOA circuit for MaxCut on a random 3-regular graph (a canonical
   NISQ/variational workload).
6. **ucj_h2**, a unitary cluster Jastrow ansatz for H2/STO-6G, built with `ffsim`+`pyscf`.
7. **hubbard_trotter**, a 2D Fermi-Hubbard model Trotter circuit (native to both Majorana
   packages, and also runs on the Pauli backends via JW).
8. **random_fermionic_circuit**, random hopping/pairing/on-site rotations, the fermionic
   analog of problem 1.

Problems 3 and 6 through 8 directly answer the three examples in the original request
(Ising Trotter, UCJ-H2, and Hubbard/random-fermionic as the "anything else"). Problems 2,
4, and 5 round out the suite with a truncation-scaling probe, a non-integrable spin model,
and a canonical NISQ ansatz.

## Notes on propaq 0.1.3 (PyPI)

The suite runs the published `propaq==0.1.3` wheel from PyPI, not a sibling source
checkout. Install with `pip install --user propaq==0.1.3`. It is a manylinux wheel, so
unlike the other native backends it needs no per-node rebuild. `FlushSchedule` and the
`PROPAQ_ENGINE` env var from earlier propaq versions are gone in 0.1.3, one engine now
serves both bases with no outbox to flush, so neither appears anywhere in this suite.

The native noise plugin ABI `propaq` exposes (see `plugin/`) was renamed and widened in
0.1.3. `propaq_noise_damping_factor`/`_batch` became `propaq_noise_factor`/`_batch`, which
now also carry the basis kind, the term's raw key words, and the layer position, and a new
optional `propaq_noise_depends` declares which of those a plugin actually reads (0 =
weight alone, bit 0 = key, bit 1 = layer). All three `plugin/` implementations (C, Rust,
AOT Julia) and the C shim that fronts the Julia library declare `depends = 0`, so propaq
collapses each to one weight-indexed table, verified bit-identical against the built-in
`UniformNoiseModel` at damping 0, 0.005, and 0.05.

`monoprop` and `pyrauli` both compile with `-march=native` and must be built on the node
that runs them (`monoprop`'s flag is `monoprop_ENABLE_ARCH_FLAGS`, on by default). A build
from a dev node with different CPU features than the compute node dies with `SIGILL`
there. `monoprop` also needs the `Boost/1.88.0-GCC-14.3.0` module to configure.

## Known upstream issues found and worked around

- **pyrauli**: `pyrauli.from_qiskit(sparse_pauli_op)` needs `reverse=True`. pyrauli's own
  `Observable` string convention is big-endian (leftmost char = qubit 0), the opposite of
  Qiskit's little-endian convention, and `from_qiskit` does not correct for this by
  default. Confirmed empirically. Without it, expectation values are silently wrong
  (exactly 0 for entangled circuits, sign-flipped for product states). Handled in every
  experiment's `run_pyrauli.py`.
- **MajoranaPropagation.jl v0.3.0**: `propagate()` on any `AbstractMajoranaSum` (both
  `MajoranaSum` and `VectorMajoranaSum`) crashes with `UndefVarError`. It calls three
  `PauliPropagation.PathProperties` internal helpers
  (`_check_wrapping_into_paulifreqtracker`, `_checkfreqandsinfields`,
  `_check_unwrap_from_paulifreqtracker`) as bare names, but only imports
  PauliPropagation's exported names. `VectorMajoranaSum` is also missing a `coefftype`
  method entirely. Both patched at the top of every `run_majorana_propagation_jl.jl`
  (documented inline there) rather than upstream.
- **pauli-prop**: `propagate_through_circuit`'s docstring says `max_terms=None` disables
  the term-count cap, but the implementation (`propagation.py`'s `if max_terms < 1:` check)
  crashes with `TypeError` on `None` instead. Worked around in every `run_pauli_prop.py` by
  passing a large finite cap (`MAX_TERMS = 2_000_000_000`) instead of `None`, chosen high
  enough to never actually bind (confirmed a larger cap costs nothing on small circuits, no
  buffer pre-allocates to its size). This matches pauli-prop's truncation to every other
  backend's (coefficient-cutoff only).

## Running one experiment

Every experiment folder is self-contained. From the repo root:

```bash
cd experiments/ising_trotter
python3 generate_circuits.py       # builds and saves every circuit instance, run once
python3 run_pauli_prop.py          # or run_pyrauli.py, run_monoprop.py, run_propaq.py
module load Julia/1.11.3-linux-x86_64
julia --project=../../julia_env -t 64 run_pauli_propagation_jl.jl
```

No CLI flags for the common case, no orchestrator, no subprocess dispatch across backends.
Each `run_<backend>` file loads every saved circuit in `circuits/` (and `circuits_native/`
where present), runs that one package, and writes `results_<backend>.jsonl` and
`results_<backend>.npz` right there in the folder, resumable as described above.

pyrauli needs `OMP_NUM_THREADS` set before its process starts, since OpenMP reads it once
at the first parallel region. `experiments/thread_scaling/run_pyrauli.py`, the one
experiment that sweeps thread count, re-executes itself once per thread count with that
env var set for the child process. This is the only subprocess use left anywhere in this
suite, everything else runs the whole comparison in one process.

The two Julia-only backends (PauliPropagation.jl on the qubit side, MajoranaPropagation.jl
on the native-fermionic side) fix their thread count at process launch (`julia -t N`), so
in `experiments/thread_scaling/` they record one row per invocation instead of sweeping
in-process. Run `run_pauli_propagation_jl.jl` (or `run_majorana_propagation_jl.jl`) once
per thread count you want a point for, each invocation appends its own row.

## Plotting

```bash
python3 plotting/plot_runtime_comparison.py
python3 plotting/plot_scaling.py
python3 plotting/plot_trotter_scan.py
python3 plotting/plot_trotter_memory.py
```

Each plotting script reads the relevant experiment folders directly
(`propaq_benchmarks/io_utils.py`'s `load_experiment_results` merges every `results_<backend>.*` file
in a folder into one list) and produces, per problem family, a grouped-bar chart of
propagation wall time and peak RSS (log scale, small multiples across problem sizes), plus
a dedicated line chart for the `random_near_clifford` T-density sweep and a terms-vs-time
scatter across every run. `plotting/plot_scaling.py` reads `experiments/thread_scaling/`
for the cross-backend wall-time-and-speedup-vs-thread-count charts. `plot_trotter_scan.py`
and `plot_trotter_memory.py` read `experiments/ising_trotter/` and
`experiments/hubbard_trotter/`'s fine step curves for runtime-vs-step and peak-RSS-vs-step
line charts, one figure per lattice size, with a term-count inset on the runtime figure.
Every other experiment folder has its own matching `plotting/plot_<name>.py`.

PNGs and PGFs land in `results/plots/`, the single output directory for every figure in the
suite, flat, with no per-experiment subdirectories. Result data
(`results_<backend>.{jsonl,npz}`) stays in the experiment folder that produced it.

## Resource budget

Default truncation is identical across every backend that has a coefficient-cutoff knob,
`min_abs_coeff=1e-6` is the only real cutoff (terms below this magnitude are dropped), and
there is no weight cutoff. Term count is otherwise unbounded everywhere. pauli-prop's
`max_terms` is set to 2,000,000,000 purely as a required non-`None` placeholder (see
"Known upstream issues" above), high enough to never actually bind. Problem sizes were
chosen so the heaviest single run lands in the tens-of-seconds-to-minutes range on a
dedicated machine, not a shared login node, since per-run wall time is sensitive to
contention from other users' processes. `propaq`'s and `monoprop`'s runners default to
`n_threads=64` (or the equivalent env var), matching the core count this suite was
developed on, edit that constant at the top of a `run_<backend>.py` file to match a
different machine.
