# propaq-benchmarks

This repo contains code to reproduce benchmarks and experiments from propaq's [paper](arxiv.org/abs/2609.07730). It's organized
into a `propaq-benchmarks` module, which contains code to build and parse circuits across backends, and a set of experiments under `experiments/`.
Its requirements also pin the versions of the various backends used in the paper.

## Build

```bash
pip install propaq-benchmarks/
julia --project=julia_env -e 'using Pkg; Pkg.instantiate()'
```

## Run an experiment

```bash
cd experiments/ising_trotter        # or any other folder under experiments/
python3 generate_circuits.py        # builds and saves circuits, run once
python3 run_pauli_prop.py           # or run_pyrauli.py, run_monoprop.py, run_propaq.py
julia --project=../../julia_env -t 64 run_pauli_propagation_jl.jl
```