#!/usr/bin/env python3
"""Hybrid Schrodinger-Heisenberg simulation of a UCJ ansatz on linear hydrogen chains.

A unitary cluster Jastrow (UCJ) operator is a fixed sequence of n_reps reps (an orbital
rotation followed by a diagonal Coulomb evolution), plus an optional final orbital
rotation. Slicing the op at any rep boundary yields two exact UCJ operators whose
composition reproduces the original op exactly, with no approximation. This puts the early
reps on propaq's Schrodinger/MPS side (quimb, bond dimension left uncapped) and the late
reps plus the final rotation on the Heisenberg/Majorana side (propaq, no weight cutoff),
and checks the hybrid contraction against an exact dense-statevector reference at every cut
point, including the two pure endpoints, with no truncation active anywhere. A separate
pure full-depth Heisenberg baseline (a weight-cutoff sweep over the whole ansatz) shows the
accuracy and term-count cost pure Heisenberg propagation pays for the same problem, for
contrast.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common import io_utils  # noqa: E402

PACKAGE_ROOT = HERE.parents[2] / "propaq"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

N_THREADS = 64
os.environ["RAYON_NUM_THREADS"] = str(N_THREADS)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# quimb uses Numba's on-disk cache; the shared home cache is not writable on
# every benchmark host, so use a safe local cache unless the caller chose one.
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "propaq-quimb-numba-cache"))

import qiskit  # noqa: E402
import qiskit.qasm2 as qasm2  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402
import quimb.tensor as qtn  # noqa: E402

from propaq import Logger  # noqa: E402
from propaq.circuits import MajoranaCircuit  # noqa: E402
from propaq.datatypes import MajoranaTermSum  # noqa: E402
from propaq.hybrid import hybrid_expectation_value  # noqa: E402
from propaq.noise import TruncationPolicy  # noqa: E402
from propaq.propagators import MajoranaPropagator  # noqa: E402

ATOM = "H"
ATOMIC_DISTANCE = 1.0
BASIS = "sto-6g"
N_REPS = 4
HEISENBERG_WEIGHT_CUTOFFS = (2, 3, 4, 6)
BENCHMARK_ID = "hybrid_ucj_heisenberg_v1"
# Standard universal basis (not bench_ucj.py's hardware-native cp/xx_plus_yy basis) so
# qasm2.dumps + quimb's OpenQASM2 importer can consume the circuit directly.
MPS_BASIS_GATES = ["rz", "sx", "x", "cx"]


def _linear_geometry(natoms: int) -> str:
    return "; ".join(f"{ATOM} 0 0 {i * ATOMIC_DISTANCE}" for i in range(natoms))


def build_ucj_op(natoms: int):
    """Fit a UCJ operator to CCSD amplitudes for a linear H-chain.

    This ansatz's orbital-rotation layers are fully dense, so its entanglement growth
    isn't bounded by chain length in either picture.
    """
    import ffsim
    import pyscf
    import pyscf.cc

    mol = pyscf.gto.Mole()
    mol.verbose = 0
    mol.build(atom=_linear_geometry(natoms), basis=BASIS)

    scf = pyscf.scf.RHF(mol)
    scf.verbose = 0
    scf.run()

    norb = mol.nao_nr()
    n_electrons = int(sum(scf.mo_occ))
    nelec = (n_electrons // 2, n_electrons // 2)

    ccsd = pyscf.cc.CCSD(scf)
    ccsd.verbose = 0
    ccsd.run()

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=ccsd.t2, t1=ccsd.t1, n_reps=N_REPS, optimize=True, options=dict(maxiter=1000),
    )
    return ucj_op, norb, nelec


def hf_int(norb: int, nelec: tuple[int, int]) -> int:
    """Hartree-Fock reference state as a JW bitstring integer (alpha block, then beta block)."""
    n_alpha, n_beta = nelec
    return sum(1 << k for k in range(n_alpha)) | sum(1 << (norb + k) for k in range(n_beta))


def split_ucj_op(op, cut_rep: int):
    """Slice op at rep boundary cut_rep, so op equals heisenberg_op(schrodinger_op(state)).

    Reps [0, cut_rep) go to the Schrodinger half (state-prep side, growing out from HF).
    Reps [cut_rep, n_reps) plus any final rotation go to the Heisenberg half (closest to
    the observable). Both halves are exact UCJ operators, with no approximation in the
    split itself.
    """
    import ffsim

    schrodinger = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=op.diag_coulomb_mats[:cut_rep],
        orbital_rotations=op.orbital_rotations[:cut_rep],
    )
    heisenberg = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=op.diag_coulomb_mats[cut_rep:],
        orbital_rotations=op.orbital_rotations[cut_rep:],
        final_orbital_rotation=op.final_orbital_rotation,
    )
    return schrodinger, heisenberg


def build_observable(n_qubits: int) -> SparsePauliOp:
    """A single ZZ near the middle of the chain, matching bench_ucj.py's convention."""
    return SparsePauliOp("ZZ" + "I" * (n_qubits - 2))


def reference_value(op, norb: int, nelec: tuple[int, int], observable: SparsePauliOp) -> float:
    """Exact expectation value via ffsim's CI-vector simulator (no dense operator matrix built)."""
    import ffsim

    hf_vec = ffsim.hartree_fock_state(norb, nelec)
    vec_out = ffsim.apply_unitary(hf_vec, op, norb=norb, nelec=nelec)
    qvec = ffsim.qiskit.ffsim_vec_to_qiskit_vec(vec_out, norb, nelec)
    return float(Statevector(qvec).expectation_value(observable).real)


def build_schrodinger_mps(op, norb: int, nelec: tuple[int, int]):
    """Exact quimb MPS for |Psi> = (schrodinger UCJ reps) |HF>; bond dimension never capped."""
    import ffsim

    n_qubits = 2 * norb
    qubits = qiskit.QuantumRegister(n_qubits, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(op), qubits)
    compiled = qiskit.transpile(circuit, basis_gates=MPS_BASIS_GATES, optimization_level=3)
    return qtn.CircuitMPS.from_openqasm2_str(qasm2.dumps(compiled)).psi


def max_bond(mps) -> int:
    return max((mps.bond_size(i, i + 1) for i in range(mps.L - 1)), default=1)


def discarded_l1(log_path: Path) -> float:
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    return float(
        sum(event.get("discarded_coeff_l1", 0.0) for event in events if event.get("event") == "truncation")
    )


def run_hybrid_split(op, norb, nelec, observable_majorana, cut_rep: int, true_value: float) -> dict:
    """One (op, cut_rep) hybrid point: exact on both sides, no truncation anywhere."""
    schrodinger_op, heisenberg_op = split_ucj_op(op, cut_rep)
    n_modes = 4 * norb

    t0 = time.perf_counter()
    mps = build_schrodinger_mps(schrodinger_op, norb, nelec)
    t1 = time.perf_counter()
    heisenberg_circuit = MajoranaCircuit.from_ffsim_ucj(heisenberg_op, n_modes)
    theta = MajoranaPropagator(n_threads=N_THREADS).propagate(observable_majorana.copy(), heisenberg_circuit)
    t2 = time.perf_counter()
    value = hybrid_expectation_value(theta, mps, initial_state=0)
    t3 = time.perf_counter()

    return {
        "cut_rep": cut_rep,
        "theta_terms": len(theta),
        "mps_max_bond": max_bond(mps),
        "schrodinger_time_s": t1 - t0,
        "heisenberg_time_s": t2 - t1,
        "contraction_time_s": t3 - t2,
        "hybrid_value": value,
        "absolute_error": abs(value - true_value),
    }


def run_pure_heisenberg_sweep(op, norb, nelec, observable_majorana, true_value: float, cutoffs: tuple[int, ...]) -> list[dict]:
    """Pure full-depth Heisenberg propagation of the entire ansatz, at several weight cutoffs."""
    n_modes = 4 * norb
    circuit = MajoranaCircuit.from_ffsim_ucj(op, n_modes)
    empty_circuit = MajoranaCircuit([], n_modes)
    initial_state = hf_int(norb, nelec)

    records = []
    for cutoff in cutoffs:
        with tempfile.TemporaryDirectory(prefix="propaq-ucj-hybrid-") as directory:
            log_path = Path(directory) / "propagation.jsonl"
            propagator = MajoranaPropagator(
                n_threads=N_THREADS,
                truncation=TruncationPolicy(weight_cutoff=cutoff, coeff_cutoff=0.0),
                logger=Logger(str(log_path), log_every=1),
            )
            t0 = time.perf_counter()
            theta = propagator.propagate(observable_majorana.copy(), circuit)
            t1 = time.perf_counter()
            value = MajoranaPropagator(n_threads=N_THREADS).expectation_value(
                theta, empty_circuit, initial_state=initial_state
            ).expectation_value
            discarded = discarded_l1(log_path)
        record = {
            "weight_cutoff": cutoff,
            "theta_terms": len(theta),
            "discarded_l1": discarded,
            "time_s": t1 - t0,
            "pure_value": value,
            "absolute_error": abs(value - true_value),
        }
        print(
            f"  pure heisenberg cutoff={cutoff}: terms={record['theta_terms']} "
            f"error={record['absolute_error']:.3e} ({record['time_s']:.2f}s)",
            flush=True,
        )
        records.append(record)
    return records


# This ansatz's dense orbital-rotation layers make its entanglement growth intrinsic to the
# operator, not something a smarter cut point dodges.
# At natoms=6 (n_modes=24), every cut point already back-propagates the observable to a
# near-saturated approximately 4.2 million exact terms, costing about 100s per point in the
# MPS contraction alone, not in a way any cut choice avoids.
# So natoms=6 only samples the two pure endpoints plus the balanced middle instead of the
# full range, while natoms=4 stays fast and gets the full sweep.
def cut_reps_for(natoms: int) -> tuple[int, ...]:
    if natoms <= 4:
        return tuple(range(N_REPS + 1))
    return (0, N_REPS // 2, N_REPS)


def run_one_natoms(natoms: int) -> list[dict]:
    print(f"=== natoms={natoms} ===", flush=True)
    op, norb, nelec = build_ucj_op(natoms)
    n_qubits = 2 * norb
    print(f"  built UCJ op: norb={norb} nelec={nelec} n_qubits={n_qubits}", flush=True)
    observable_sparse = build_observable(n_qubits)
    observable_majorana = MajoranaTermSum.from_sparse_pauli_op(observable_sparse)
    true_value = reference_value(op, norb, nelec, observable_sparse)
    print(f"  exact reference value: {true_value:.6f}", flush=True)

    pure_records = run_pure_heisenberg_sweep(
        op, norb, nelec, observable_majorana, true_value, HEISENBERG_WEIGHT_CUTOFFS
    )

    records = []
    for cut_rep in cut_reps_for(natoms):
        hybrid = run_hybrid_split(op, norb, nelec, observable_majorana, cut_rep, true_value)
        # Threshold is floating-point-noise scale, not truncation-scale: summing millions of
        # terms in the MPS contraction accumulates rounding error up to about 1e-6; a real
        # correctness bug would show as an O(0.01-1) error, like the pure Heisenberg
        # baseline above.
        if hybrid["absolute_error"] > 1e-4:
            raise RuntimeError(
                f"natoms={natoms} cut_rep={cut_rep}: hybrid split disagrees with the exact "
                f"reference by {hybrid['absolute_error']:.3e}, with no truncation active"
            )
        record = {
            "benchmark": BENCHMARK_ID,
            "natoms": natoms,
            "norb": norb,
            "n_qubits": n_qubits,
            "n_reps": N_REPS,
            "n_threads": N_THREADS,
            "true_value": true_value,
            "pure_heisenberg_sweep": pure_records,
            **hybrid,
        }
        records.append(record)
        print(
            f"  cut_rep={cut_rep}: theta_terms={hybrid['theta_terms']:7d} "
            f"mps_max_bond={hybrid['mps_max_bond']:5d} "
            f"error={hybrid['absolute_error']:.3e} "
            f"(schrodinger {hybrid['schrodinger_time_s']:.2f}s, "
            f"heisenberg {hybrid['heisenberg_time_s']:.2f}s, "
            f"contraction {hybrid['contraction_time_s']:.2f}s)",
            flush=True,
        )
    print(
        "  pure full-depth Heisenberg: "
        + ", ".join(
            f"cutoff={r['weight_cutoff']} terms={r['theta_terms']} error={r['absolute_error']:.3e}"
            for r in pure_records
        )
    )
    return records


def main() -> None:
    circuits_dir = HERE / "circuits"
    checkpoint = HERE / "results_propaq.jsonl"
    npz_path = HERE / "results_propaq.npz"

    circuit_files = sorted(circuits_dir.glob("*.json"))
    if not circuit_files:
        raise SystemExit(f"no saved circuits in {circuits_dir}, run generate_circuits.py first")

    done = {r["natoms"] for r in io_utils.load_jsonl(str(checkpoint)) if r.get("ok") and r.get("benchmark") == BENCHMARK_ID}

    for path in circuit_files:
        with open(path) as f:
            spec = json.load(f)
        natoms = spec["natoms"]
        if natoms in done:
            print(f"natoms={natoms}: skip (already checkpointed)")
            continue
        try:
            records = run_one_natoms(natoms)
        except Exception as exc:  # noqa: BLE001 - a bad natoms point should not abort the sweep
            record = {"ok": False, "error": str(exc), "natoms": natoms, "benchmark": BENCHMARK_ID}
            io_utils.append_jsonl(str(checkpoint), record)
            io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))
            print(f"natoms={natoms}: FAIL ({exc})")
            continue
        for record in records:
            record["ok"] = True
            record["label"] = f"natoms{natoms}_cut{record['cut_rep']}"
            record["backend"] = "propaq"
            io_utils.append_jsonl(str(checkpoint), record)
            io_utils.save_records_npz(io_utils.load_jsonl(str(checkpoint)), str(npz_path))
    print(f"Checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
