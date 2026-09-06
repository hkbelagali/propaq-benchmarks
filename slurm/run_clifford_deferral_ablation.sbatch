#!/bin/bash
# IMPORTANT: run slurm/rebuild_native.sh on this compute node (once, before first use, and
# again any time you switch to a different node type). propaq/pauli-prop/pyrauli were
# originally built on a dev node with different CPU features and will SIGILL here otherwise.
#SBATCH --job-name=propaq-clifford-deferral
#SBATCH --partition=general-short         # amr-242 is in scavenger and general-short
#SBATCH --nodelist=amr-242                # pin to the node the run was sized for
#SBATCH --account=general
#SBATCH --time=04:00:00                   # general-short's limit, job checkpoints per T-density
                                           #   point, so a short limit just means resubmitting
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=logs/clifford_deferral_%j.out
#SBATCH --requeue

set -euo pipefail
# Absolute, not $(dirname "$0")/.. , because under SLURM's forced `bash --login` wrapper (see the
# sbatch banner on submit) a login shell can change PWD before this script body runs, and
# $0 keeps whatever relative path was passed to `sbatch`, so the old dirname-based cd landed
# somewhere other than bench/ and `mkdir -p logs` failed there with permission denied.
cd /mnt/ffs24/home/belagal1/propaq-benchmark-suite/bench
mkdir -p logs

# SLURM runs this script under `bash --login`, which re-sources this node's login profile
# and can put a *different* python3 ahead of the one propaq is actually installed into (bit
# us directly: a plain login-shell SSH into amr-242 resolved python3 to an unrelated system
# Python 3.11 with no propaq at all, not the Miniforge 3.12 the editable install targets).
# Pinning the interpreter by absolute path sidesteps that instead of trusting login-shell PATH.
PY=/opt/software-current/2023.06/x86_64/generic/software/Miniforge3/25.11.0-1/bin/python3

"$PY" - <<'EOF'
import propaq
from propaq._rust_core import rust_available
assert rust_available()
print("propaq OK (rust_available=True)")
EOF

# Checkpointed: safe to resubmit. Each T-gate-density point is one task, and progress lives in
# results/clifford_deferral_ablation.jsonl. p=0.45 alone costs ~3-20 min per propagation
# (deferral off, ~116M terms), sized from interactive probing on this node up through
# p=0.45. p=0.50 was not pursued further (~10x growth per 0.05 step observed, so it would
# likely mean 20-30+ min single runs and was cancelled after 4 min unfinished).
"$PY" benchmarks/benchmark_clifford_deferral.py \
    --proportions 0.0 0.025 0.05 0.075 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45
