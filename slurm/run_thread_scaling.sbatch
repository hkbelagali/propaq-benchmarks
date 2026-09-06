#!/bin/bash
# IMPORTANT: run slurm/rebuild_native.sh on this compute node (once, before first use, and
# again any time you switch to a different node type). propaq/pauli-prop/pyrauli were
# originally built on a dev node with different CPU features and will SIGILL here otherwise.
#SBATCH --job-name=propaq-thread-scaling
#SBATCH --partition=general-short         # any node here, not pinned to a specific one
#SBATCH --account=general
#SBATCH --time=03:45:00                   # extrapolated from the ~2.86x/step growth measured
                                           #   at steps 8-12: step 14 ~9.8min, step 15 ~28min
                                           #   at 1 thread alone (x2 for warmup+repeat), and the
                                           #   n_threads=1 column alone sums to ~1.5hr, so this
                                           #   stays close to general-short's 4hr cap
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=logs/thread_scaling_%j.out
#SBATCH --requeue

set -euo pipefail
# Absolute, not $(dirname "$0")/.. , because under SLURM's forced `bash --login` wrapper a login
# shell can change PWD before this script body runs and $0 keeps whatever relative path
# was passed to `sbatch`, so a dirname-based cd can land somewhere other than bench/ and
# `mkdir -p logs` fails there with permission denied (hit this exactly on the previous job).
cd /mnt/ffs24/home/belagal1/propaq-benchmark-suite/bench
mkdir -p logs

# SLURM runs this script under `bash --login`, which re-sources this node's login profile
# and can put a *different* python3 ahead of the one propaq is actually installed into (a
# plain login-shell SSH into amr-242 resolved python3 to an unrelated system Python 3.11
# with no propaq at all, not the Miniforge 3.12 the editable install targets). Pinning the
# interpreter by absolute path sidesteps that instead of trusting login-shell PATH.
PY=/opt/software-current/2023.06/x86_64/generic/software/Miniforge3/25.11.0-1/bin/python3

"$PY" - <<'EOF'
import propaq
from propaq._rust_core import rust_available
assert rust_available()
print("propaq OK (rust_available=True)")
EOF

# Checkpointed: safe to resubmit. Each (steps, n_threads) pair is one task, and progress lives
# in results/thread_scaling.jsonl. Same 6x6 Ising-Trotter problem as the previous run, but
# windowed to steps 8-15 instead of 2-12. For this local-Hamiltonian circuit, propagated
# support only grows within a light cone bounded by step count, not qubit count, so a wider
# grid does NOT make small step counts more expensive (confirmed empirically: step=2 costs
# ~0.001s on 6x6, 8x8, 10x10, and 12x12 alike). Shifting the window is what actually avoids
# the overhead-dominated regime. Measured steps 8-12 at 1.11s/2.95s/10.2s/25.0s/71.7s (1
# thread), ~2.86x per step. Steps 13-15 are extrapolated, not measured, at ~3.4min/9.8min/
# ~28min respectively for that same n_threads=1 column.
"$PY" benchmarks/benchmark_thread_scaling.py \
    --nx 6 --ny 6 \
    --steps 8 9 10 11 12 13 14 15 \
    --threads 1 2 4 8 16 32 64 \
    --repeats 1
