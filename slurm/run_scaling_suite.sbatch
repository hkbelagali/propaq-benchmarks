#!/bin/bash
# IMPORTANT: run slurm/rebuild_native.sh on this compute node (once, before first use, and
# again any time you switch to a different node type). propaq/pauli-prop/pyrauli were
# originally built on a dev node with different CPU features and will SIGILL here otherwise.
#SBATCH --job-name=propaq-bench-scaling
#SBATCH --partition=general-long          # EDIT: pick a partition available to your account
#SBATCH --account=general                 # EDIT: your SLURM account
#SBATCH --time=04:00:00                   # EDIT: wall-clock budget, job checkpoints, so a short
                                           #        limit just means more (cheap) resubmissions
#SBATCH --cpus-per-task=32                # must be >= max(THREAD_SWEEP) in scaling/run_scaling.py
#SBATCH --mem=64G
#SBATCH --output=logs/scaling_suite_%j.out
#SBATCH --requeue

set -euo pipefail
cd "$(dirname "$0")/.."   # bench/
mkdir -p logs

module load Julia/1.11.3-linux-x86_64

python3 - <<'EOF'
import pauli_prop, propaq, pyrauli
from propaq._rust_core import rust_available
assert rust_available()
print("Python backends OK: pauli_prop, propaq (rust_available=True), pyrauli")
EOF

# Checkpointed the same way as run_main_suite.sbatch, safe to resubmit. Each (problem, backend,
# n_threads) triple is one task, and results/scaling_suite.jsonl / .npz track progress.
python3 scaling/run_scaling.py --out results/scaling_suite.npz --checkpoint results/scaling_suite.jsonl
