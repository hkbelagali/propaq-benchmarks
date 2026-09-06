#!/bin/bash
# Run this ON THE COMPUTE NODE (interactive salloc session, or as a tiny sbatch job) before
# submitting run_main_suite.sbatch / run_scaling_suite.sbatch.
#
# Why: propaq was originally built on the dev node with Rust's `-C target-cpu=native`, which
# crashed at *runtime* with SIGILL on a compute node with fewer CPU features. Worse, `native`
# turned out unsafe even to build with on some nodes here: rustc's own cpuid-based detection
# crashed rustc itself with SIGILL mid-compile (an LLVM/cpuid quirk, seen on at least one node
# in this cluster, plausibly cpuid reporting features a hypervisor/cgroup masks off). So
# propaq/.cargo/config.toml now pins a fixed, portable target (x86-64-v3) instead of `native`.
# This is a real fix, not a per-node workaround, so propaq no longer needs rebuilding every
# time you switch node type. pyrauli's prebuilt PyPI wheel had a similar runtime SIGILL and has
# no such config to fix, so it's still rebuilt from source here. Re-run this script (idempotent)
# after a fresh checkout or if pyrauli ever needs picking up a different node's build again.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root (propaq-benchmark-suite/)

module load Rust/1.88.0-GCCcore-14.3.0 maturin/1.9.1-GCCcore-14.3.0 CMake/4.0.3-GCCcore-14.3.0
MFPY=/opt/software-current/2023.06/x86_64/generic/software/Miniforge3/25.11.0-1/bin

echo "=== nproc / CPU model on this node ==="
nproc
lscpu | grep "Model name"

echo "=== rebuilding pauli-prop (portable already, but harmless/fast to redo) ==="
(cd pauli-prop && "$MFPY/pip3" install --user -e '.[test]')

echo "=== rebuilding propaq (portable x86-64-v3; only needed after a fresh checkout) ==="
(cd propaq && "$MFPY/pip3" install --user -e '.[dev]')

echo "=== rebuilding pyrauli from source (was a prebuilt PyPI wheel; build here instead) ==="
(cd pyrauli && "$MFPY/pip3" install --user -e '.[qiskit,test]')

echo "=== sanity check ==="
"$MFPY/python3" - <<'EOF'
import pauli_prop, propaq, pyrauli
from propaq._rust_core import rust_available
assert rust_available()
print("OK: pauli_prop, propaq (rust_available=True), pyrauli all import on this node")
EOF
