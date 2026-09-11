"""
Qubit problems for benchmarks
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from .circuit_ir import ProblemIR, qiskit_to_ir


def _z_mid_observable(n_qubits: int) -> SparsePauliOp:
    label = ["I"] * n_qubits
    label[n_qubits // 2] = "Z"
    return SparsePauliOp("".join(label))


def _zz_observable(n_qubits: int, i: int = 0, j: int = 1) -> SparsePauliOp:
    label = ["I"] * n_qubits
    label[i] = "Z"
    label[j] = "Z"
    return SparsePauliOp("".join(label))


def random_circuit_problem(n_qubits: int, depth: int, seed: int = 0, two_q_prob: float = 0.5) -> ProblemIR:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    single_q_gates = ["rz", "rx", "ry", "h"]
    for layer in range(depth):
        offset = layer % 2
        # single-qubit layer
        for q in range(n_qubits):
            gate = single_q_gates[rng.integers(len(single_q_gates))]
            if gate == "h":
                qc.h(q)
            else:
                angle = float(rng.uniform(0, 2 * math.pi))
                getattr(qc, gate)(angle, q)
        # brickwork two-qubit layer
        for q in range(offset, n_qubits - 1, 2):
            if rng.random() < two_q_prob:
                if rng.random() < 0.5:
                    qc.cx(q, q + 1)
                else:
                    qc.rzz(float(rng.uniform(0, 2 * math.pi)), q, q + 1)
    return qiskit_to_ir(
        qc,
        _z_mid_observable(n_qubits),
        "random_circuit",
        {"n_qubits": n_qubits, "depth": depth, "seed": seed, "two_q_prob": two_q_prob},
    )


def random_near_clifford_problem(
    n_qubits: int, depth: int, t_density: float, seed: int = 0
) -> ProblemIR:
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)
    for layer in range(depth):
        offset = layer % 2
        for q in range(n_qubits):
            if rng.random() < t_density:
                qc.rz(float(rng.uniform(0, 2 * math.pi)), q)
            else:
                qc.h(q)
        for q in range(offset, n_qubits - 1, 2):
            qc.cx(q, q + 1)
    return qiskit_to_ir(
        qc,
        _z_mid_observable(n_qubits),
        "random_near_clifford",
        {"n_qubits": n_qubits, "depth": depth, "t_density": t_density, "seed": seed},
    )


def _grid_edges(nx: int, ny: int, periodic: bool) -> list[tuple[int, int]]:
    def idx(x: int, y: int) -> int:
        return y * nx + x

    edges = []
    for y in range(ny):
        for x in range(nx):
            if x + 1 < nx:
                edges.append((idx(x, y), idx(x + 1, y)))
            elif periodic and nx > 2:
                edges.append((idx(x, y), idx(0, y)))
            if y + 1 < ny:
                edges.append((idx(x, y), idx(x, y + 1)))
            elif periodic and ny > 2:
                edges.append((idx(x, y), idx(x, 0)))
    return edges


def ising_trotter_problem(
    nx: int = 6, ny: int = 6, J: float = 1.0, h: float = 0.5, dt: float = 0.1,
    steps: int = 4, periodic: bool = False, canonicalize: bool = True,
) -> ProblemIR:
    n_qubits = nx * ny
    edges = _grid_edges(nx, ny, periodic)
    qc = QuantumCircuit(n_qubits)
    for _ in range(steps):
        for (i, j) in edges:
            qc.rzz(-2.0 * J * dt, i, j)
        for q in range(n_qubits):
            qc.rx(-2.0 * h * dt, q)
    return qiskit_to_ir(
        qc,
        _z_mid_observable(n_qubits),
        "ising_trotter",
        {"nx": nx, "ny": ny, "J": J, "h": h, "dt": dt, "steps": steps,
         "periodic": periodic, "n_edges": len(edges)},
        canonicalize_circuit=canonicalize,
    )


def heisenberg_chain_trotter_problem(
    n_qubits: int = 20, Jxy: float = 1.0, Jz: float = 1.0, dt: float = 0.1,
    steps: int = 4, periodic: bool = False,
) -> ProblemIR:
    qc = QuantumCircuit(n_qubits)
    bonds = [(i, i + 1) for i in range(n_qubits - 1)]
    if periodic and n_qubits > 2:
        bonds.append((n_qubits - 1, 0))
    for _ in range(steps):
        for (i, j) in bonds:

            qc.h(i); qc.h(j)
            qc.rzz(-2.0 * Jxy * dt, i, j)
            qc.h(i); qc.h(j)
            qc.rx(math.pi / 2, i); qc.rx(math.pi / 2, j)
            qc.rzz(-2.0 * Jxy * dt, i, j)
            qc.rx(-math.pi / 2, i); qc.rx(-math.pi / 2, j)
            qc.rzz(-2.0 * Jz * dt, i, j)
    return qiskit_to_ir(
        qc,
        _z_mid_observable(n_qubits),
        "heisenberg_chain_trotter",
        {"n_qubits": n_qubits, "Jxy": Jxy, "Jz": Jz, "dt": dt, "steps": steps, "periodic": periodic},
    )


def qaoa_maxcut_problem(n_qubits: int = 16, p: int = 3, seed: int = 0, degree: int = 3) -> ProblemIR:
    rng = np.random.default_rng(seed)
    import networkx 
    graph = networkx.random_regular_graph(degree, n_qubits, seed=seed)
    edges = list(graph.edges())

    gammas = rng.uniform(0, math.pi, size=p)
    betas = rng.uniform(0, math.pi / 2, size=p)

    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    for layer in range(p):
        for (i, j) in edges:
            qc.rzz(2.0 * gammas[layer], i, j)
        for q in range(n_qubits):
            qc.rx(2.0 * betas[layer], q)
    return qiskit_to_ir(
        qc,
        _z_mid_observable(n_qubits),
        "qaoa_maxcut",
        {"n_qubits": n_qubits, "p": p, "seed": seed, "degree": degree, "n_edges": len(edges)},
    )


def ucj_h2_problem(bond_length: float = 0.74, n_reps: int = 1, seed: int = 0) -> ProblemIR:
    import pyscf
    import ffsim

    mol = pyscf.gto.Mole()
    mol.build(atom=f"H 0 0 0; H 0 0 {bond_length}", basis="sto-6g")
    scf = pyscf.scf.RHF(mol).run(verbose=0)
    n_orbitals = scf.mo_coeff.shape[1]
    n_electrons = mol.nelectron
    norb = n_orbitals
    nelec = (n_electrons // 2, n_electrons // 2)

    mol_data = ffsim.MolecularData.from_scf(scf)
    ccsd = pyscf.cc.CCSD(scf).run(verbose=0)
    ucj_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(ccsd.t2, t1=ccsd.t1, n_reps=n_reps)

    from qiskit.circuit import QuantumRegister
    qubits = QuantumRegister(2 * norb)
    qc = QuantumCircuit(qubits)
    qc.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    qc.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)

    n_qubits = qc.num_qubits
    return qiskit_to_ir(
        qc,
        _zz_observable(n_qubits, 0, 1),
        "ucj_h2",
        {"bond_length": bond_length, "n_reps": n_reps, "norb": norb, "nelec": list(nelec),
         "n_qubits": n_qubits},
    )

def hubbard_trotter_problem(
    nx: int = 3, ny: int = 3, t: float = 1.0, U: float = 2.0, dt: float = 0.1,
    steps: int = 2, periodic: bool = False, canonicalize: bool = True,
) -> ProblemIR:
    from qiskit.circuit.library import XXPlusYYGate

    n_sites = nx * ny
    n_qubits = 2 * n_sites
    edges = _grid_edges(nx, ny, periodic)

    def up(site: int) -> int:
        return site

    def dn(site: int) -> int:
        return n_sites + site

    qc = QuantumCircuit(n_qubits)
    for site in range(0, n_sites, 2):
        qc.x(up(site))
    for _ in range(steps):
        for spin_map in (up, dn):
            for (i, j) in edges:
                qc.append(XXPlusYYGate(-2.0 * t * dt), [spin_map(i), spin_map(j)])
        for site in range(n_sites):
            qc.cp(-U * dt, up(site), dn(site))
    return qiskit_to_ir(
        qc,
        _z_mid_observable(n_qubits),
        "hubbard_trotter",
        {"nx": nx, "ny": ny, "t": t, "U": U, "dt": dt, "steps": steps,
         "periodic": periodic, "n_sites": n_sites, "n_qubits": n_qubits},
        canonicalize_circuit=canonicalize,
    )

def random_fermionic_circuit_problem(n_modes: int = 12, n_gates: int = 40, seed: int = 0) -> ProblemIR:
    from qiskit.circuit.library import XXPlusYYGate

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_modes)
    for site in range(0, n_modes, 2):
        qc.x(site)
    for _ in range(n_gates):
        i, j = rng.choice(n_modes, size=2, replace=False)
        i, j = int(i), int(j)
        angle = float(rng.uniform(0, 2 * math.pi))
        if rng.random() < 0.5:
            qc.append(XXPlusYYGate(angle), [i, j])
        else:
            qc.cp(angle, i, j)
    return qiskit_to_ir(
        qc,
        _z_mid_observable(n_modes),
        "random_fermionic_circuit",
        {"n_modes": n_modes, "n_gates": n_gates, "seed": seed},
    )


ALL_QUBIT_PROBLEM_BUILDERS = {
    "random_circuit": random_circuit_problem,
    "random_near_clifford": random_near_clifford_problem,
    "ising_trotter": ising_trotter_problem,
    "heisenberg_chain_trotter": heisenberg_chain_trotter_problem,
    "qaoa_maxcut": qaoa_maxcut_problem,
    "ucj_h2": ucj_h2_problem,
    "hubbard_trotter": hubbard_trotter_problem,
    "random_fermionic_circuit": random_fermionic_circuit_problem,
}
