#!/bin/bash
# IMPORTANT: run slurm/rebuild_native.sh on this compute node (once, before first use, and
# again any time you switch to a different node type). propaq/pauli-prop/pyrauli were
# originally built on a dev node with different CPU features and will SIGILL here otherwise.
#SBATCH --job-name=propaq-bench-main
#SBATCH --partition=general-long          # EDIT: pick a partition available to your account
#SBATCH --account=general                 # EDIT: your SLURM account (see `sacctmgr -n show associations user=$USER`)
#SBATCH --time=04:00:00                   # EDIT: wall-clock budget, job checkpoints, so a short
                                           #        limit just means more (cheap) resubmissions
#SBATCH --cpus-per-task=64                # matches orchestrate.py's --n-threads default (64);
                                           #   EDIT both together if you change one
#SBATCH --mem=64G
#SBATCH --output=logs/main_suite_%j.out
#SBATCH --requeue                         # let SLURM auto-resubmit on preemption (scavenger etc.)

set -euo pipefail
cd "$(dirname "$0")/.."   # bench/
mkdir -p logs

module load Julia/1.11.3-linux-x86_64

# Sanity check before spending walltime: make sure every backend still imports on this node.
python3 - <<'EOF'
import pauli_prop, propaq, pyrauli
from propaq._rust_core import rust_available
assert rust_available()
print("Python backends OK: pauli_prop, propaq (rust_available=True), pyrauli")
EOF

# Checkpointed: safe to resubmit this exact script as many times as needed. Progress lives in
# results/main_suite.jsonl (one line per completed (problem, backend) task); results/main_suite.npz
# is refreshed after every task, so it's always a valid (possibly partial) snapshot.
python3 orchestrate.py --out results/main_suite.npz --checkpoint results/main_suite.jsonl
