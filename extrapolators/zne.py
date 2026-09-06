#!/usr/bin/env python3
"""Zero-noise extrapolation for a two-layer LUCJ ansatz on a linear H6 chain.

Builds the UCJ ansatz from CCSD t1/t2 amplitudes, propagates the Jordan-Wigner mapped
electronic Hamiltonian back through it in the Majorana basis under a uniform damping
channel, and extrapolates the resulting energies to zero damping.

The computation was previously commented out and the script only replotted a cached
``zne_results.npz``.  It is restored here so ``python3 extrapolators/zne.py`` recomputes
end to end; pass ``--replot`` to redraw the figure from an existing results file.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

N_THREADS = 64

# propaq pins its rayon workers, so a threaded BLAS spinner on the same core is a large loss.
# Must be set before numpy is imported by anything below.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("pgf")
matplotlib.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "text.usetex": True,
        "pgf.rcfonts": False,
    }
)

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  # Registers the SciencePlots styles.

plt.style.use(["science", "grid"])

import ffsim
import numpy as np
import pyscf
import pyscf.cc
import pyscf.mcscf
import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import ElectronicIntegrals

from propaq.circuits import MajoranaCircuit
from propaq.datatypes import MajoranaTermSum
from propaq.extrapolators import ZeroNoiseExtrapolator
from propaq.noise import TruncationPolicy, UniformNoiseModel
from propaq.propagators import MajoranaPropagator

ATOM = "H"
NATOMS = 6
NLAYERS = 2
ATOMIC_DISTANCE = 1.0
BASIS = "sto-6g"

NOISE_VALUES = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004]

RESULTS_PATH = Path(__file__).resolve().parent / "zne_results.npz"
# Every figure in the suite lands in one directory; only the .npz stays next to this script.
PLOT_PATH = Path(__file__).resolve().parent.parent / "results" / "plots" / "zne_plot"

zne = ZeroNoiseExtrapolator(
    fitting_fn=lambda x, a, b: a + b * x,
    noise_values=NOISE_VALUES,
)


def generate_linear_geometry(atom: str, natoms: int, atomic_distance: float = 1.0) -> str:
    """Return a linear chain geometry for use in PySCF molecule construction."""
    return "; ".join([f"{atom} 0 0 {i * atomic_distance}" for i in range(natoms)])


def build_problem():
    """Return the transpiled LUCJ circuit, its Hamiltonian, and the reference energies."""
    warnings.formatwarning = lambda msg, *args, **kwargs: f"Warning: {msg}\n"

    mol = pyscf.gto.Mole()
    mol.build(
        atom=generate_linear_geometry(ATOM, NATOMS, ATOMIC_DISTANCE),
        basis=BASIS,
    )

    # No frozen orbitals: the whole space is active at this size.
    active_space = range(0, mol.nao_nr())

    scf = pyscf.scf.RHF(mol).run()

    norb = len(active_space)
    n_electrons = int(sum(scf.mo_occ[active_space]))
    n_alpha = (n_electrons + mol.spin) // 2
    n_beta = (n_electrons - mol.spin) // 2
    nelec = (n_alpha, n_beta)
    cas = pyscf.mcscf.CASCI(scf, norb, nelec)
    mo = cas.sort_mo(active_space, base=0)
    hcore, nuclear_repulsion_energy = cas.get_h1cas(mo)
    eri = pyscf.ao2mo.restore(1, cas.get_h2cas(mo), norb)

    print(f"norb = {norb}")
    print(f"nelec = {nelec}")

    ccsd = pyscf.cc.CCSD(
        scf, frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
    ).run()
    t1 = ccsd.t1
    t2 = ccsd.t2

    # Use exactly 2*norb qubits so transpilation adds no ancillas and physical qubit
    # indices match the molecular spin-orbital ordering.
    coupling_map = CouplingMap.from_full(2 * norb, bidirectional=True)
    backend = GenericBackendV2(
        2 * norb,
        coupling_map=coupling_map,
        basis_gates=["cp", "xx_plus_yy", "p", "x", "swap"],
    )

    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2=t2,
        t1=t1,
        n_reps=NLAYERS,
        # optimize=True enables the "compressed" factorization.
        optimize=True,
        options=dict(maxiter=10_000),
    )

    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    compiled = qiskit.transpile(
        circuit,
        backend=backend,
        optimization_level=3,
        initial_layout=list(range(2 * norb)),
    )

    print(f"Number of qubits: {compiled.num_qubits}")
    print(f"Gate counts: {compiled.count_ops()}")

    h2e_phys = np.einsum("prqs->pqrs", eri)  # chemist -> physicist notation
    elec_ints = ElectronicIntegrals.from_raw_integrals(hcore, h2e_phys)
    elec_hamiltonian = ElectronicEnergy(elec_ints)
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(elec_hamiltonian.second_q_op())
    hamiltonian = (
        hamiltonian
        + SparsePauliOp("I" * (2 * norb), coeffs=[nuclear_repulsion_energy])
    ).simplify()
    # Descending coefficient magnitude, so the largest contributions enter first.
    sorted_indices = np.argsort(-np.abs(hamiltonian.coeffs))
    hamiltonian = hamiltonian[sorted_indices]
    print(f"Hamiltonian has {len(hamiltonian)} Pauli terms.")

    obs_ham = MajoranaTermSum.from_sparse_pauli_op(hamiltonian)
    mc = MajoranaCircuit.from_qiskit(compiled.copy(), 4 * norb)

    return obs_ham, mc, scf.e_tot, ccsd.e_tot


def make_propagator(damping: float) -> MajoranaPropagator:
    """Return a propagator at one damping strength, with the shared truncation policy."""
    return MajoranaPropagator(
        noise=UniformNoiseModel(damping=damping),
        truncation=TruncationPolicy(
            weight_cutoff=None,
            coeff_cutoff=1e-10,
            truncation_range=(None, 10_000_000),
        ),
        n_threads=N_THREADS,
        progress_bar=True,
    )


def run_benchmark() -> None:
    """Compute the noiseless reference and the damping sweep, then save both."""
    obs_ham, mc, hf_energy, ccsd_energy = build_problem()

    # Noiseless LUCJ value for reference.
    result_ham = make_propagator(0.0).expectation_value(obs_ham, mc, initial_state=0)
    print(f"Noiseless LUCJ energy = {result_ham.expectation_value:.6f} Ha")

    result_zne = zne.run(make_propagator(NOISE_VALUES[0]), obs_ham, mc)
    print(f"ZNE zero-noise energy = {result_zne.zero_noise_value:.6f} Ha")

    np.savez(
        RESULTS_PATH,
        noise_values=result_zne.noise_values,
        expectation_values=result_zne.expectation_values,
        fit_params=result_zne.fit_params,
        zero_noise_value=result_zne.zero_noise_value,
        noiseless_value=result_ham.expectation_value,
        hf_energy=hf_energy,
        ccsd_energy=ccsd_energy,
    )
    print(f"Wrote {RESULTS_PATH}")


def plot() -> None:
    """Render the zero-noise-extrapolation figure from the saved results."""
    data = np.load(RESULTS_PATH, allow_pickle=True)

    textwidth = 3.31314
    aspect_ratio = 6 / 8
    scale = 1.3
    width = textwidth * scale
    height = width * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height))

    ax.plot(data["noise_values"], data["expectation_values"], "x", label="Noisy data", color="blue")

    # Fit line from zero to max noise, extrapolated back to x=0.
    noise_range = np.linspace(0, max(data["noise_values"]), 200)
    fit_curve = [zne.fitting_fn(x, *data["fit_params"]) for x in noise_range]
    ax.plot(noise_range, fit_curve, ":", color="blue")

    ax.plot(0, data["zero_noise_value"], "+", markersize=10,
            label=f'ZNE value ({data["zero_noise_value"]:.4f} Ha)', color="red")

    ax.axhline(data["noiseless_value"], color="orange", linestyle="-",
               label=f'Noiseless ({data["noiseless_value"]:.4f} Ha)', linewidth=1.5)
    ax.axhline(data["hf_energy"], color="purple", linestyle=":", markersize=10,
               label=f'HF energy ({data["hf_energy"]:.4f} Ha)', linewidth=1.5)
    ax.axhline(data["ccsd_energy"], color="green", linestyle=":", markersize=10,
               label=f'CCSD energy ({data["ccsd_energy"]:.4f} Ha)', linewidth=1.5)

    ax.set_xticks([0.001, 0.002, 0.003, 0.004], labels=[1, 2, 3, 4], fontsize=12)
    yticks = [-3.6, -3.4, -3.2, -3.0, -2.8, -2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4]
    ax.set_yticks(yticks, labels=yticks, fontsize=12)
    ax.set_xlabel("Noise Value ($10^{-3}$)", fontsize=13)
    ax.set_ylabel("Energy (Ha)", fontsize=13)
    ax.set_ylim(-3.4, -1.3)
    legend = ax.legend(fancybox=False, edgecolor="black", loc="upper left", fontsize=9)
    legend.get_frame().set_linewidth(0.5)

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH.with_suffix(".pgf"), bbox_inches="tight")
    fig.savefig(PLOT_PATH.with_suffix(".png"), bbox_inches="tight", dpi=800)
    print(f"Wrote {PLOT_PATH.with_suffix('.png')} and {PLOT_PATH.with_suffix('.pgf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replot", action="store_true",
                        help="skip the propagation and redraw from the existing results file")
    args = parser.parse_args()

    if not args.replot:
        run_benchmark()
    plot()


if __name__ == "__main__":
    main()
