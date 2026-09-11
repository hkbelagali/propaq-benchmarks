"""
Circuit representations for the benchmarks. We choose a basis 
that is native to the packages being benchmarked on the Python side.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

COMMON_BASIS = ["rz", "rx", "ry", "rzz", "cx", "h"]


@dataclass
class GateOp:
    name: str
    qubits: list[int]
    angle: float | None = None

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "qubits": self.qubits}
        if self.angle is not None:
            d["angle"] = self.angle
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "GateOp":
        return GateOp(name=d["name"], qubits=list(d["qubits"]), angle=d.get("angle"))


@dataclass
class ObservableIR:
    paulis: list[str]
    coeffs: list[float]

    def to_json(self) -> dict[str, Any]:
        return {"paulis": self.paulis, "coeffs": self.coeffs}

    @staticmethod
    def from_json(d: dict[str, Any]) -> "ObservableIR":
        return ObservableIR(paulis=list(d["paulis"]), coeffs=[float(c) for c in d["coeffs"]])

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        return SparsePauliOp(self.paulis, coeffs=self.coeffs)

    @staticmethod
    def from_sparse_pauli_op(op: SparsePauliOp) -> "ObservableIR":
        return ObservableIR(
            paulis=[p.to_label() for p in op.paulis],
            coeffs=[complex(c).real for c in op.coeffs],
        )


@dataclass
class ProblemIR:
    problem: str
    n_qubits: int
    params: dict[str, Any]
    gates: list[GateOp]
    observable: ObservableIR
    initial_state: int = 0  # computational-basis bitstring index, |0...0> by default

    def to_json(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "n_qubits": self.n_qubits,
            "params": self.params,
            "initial_state": self.initial_state,
            "gates": [g.to_json() for g in self.gates],
            "observable": self.observable.to_json(),
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "ProblemIR":
        return ProblemIR(
            problem=d["problem"],
            n_qubits=d["n_qubits"],
            params=d["params"],
            gates=[GateOp.from_json(g) for g in d["gates"]],
            observable=ObservableIR.from_json(d["observable"]),
            initial_state=d.get("initial_state", 0),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @staticmethod
    def load(path: str) -> "ProblemIR":
        with open(path) as f:
            return ProblemIR.from_json(json.load(f))

    def to_qiskit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for g in self.gates:
            if g.name == "rz":
                qc.rz(g.angle, g.qubits[0])
            elif g.name == "rx":
                qc.rx(g.angle, g.qubits[0])
            elif g.name == "ry":
                qc.ry(g.angle, g.qubits[0])
            elif g.name == "rzz":
                qc.rzz(g.angle, g.qubits[0], g.qubits[1])
            elif g.name == "cx":
                qc.cx(g.qubits[0], g.qubits[1])
            elif g.name == "h":
                qc.h(g.qubits[0])
            elif g.name == "x":
                qc.x(g.qubits[0])
            elif g.name == "cp":
                qc.cp(g.angle, g.qubits[0], g.qubits[1])
            elif g.name == "xx_plus_yy":
                from qiskit.circuit.library import XXPlusYYGate
                qc.append(XXPlusYYGate(g.angle), [g.qubits[0], g.qubits[1]])
            else:
                raise ValueError(f"gate {g.name} not supported by ProblemIR.to_qiskit()")
        return qc

    def gate_count(self) -> int:
        return len(self.gates)


def canonicalize(qc: QuantumCircuit) -> QuantumCircuit:
    """Transpile qiskit circuits into the shared basis""" 
    return transpile(qc, basis_gates=COMMON_BASIS, optimization_level=1, seed_transpiler=0)


def qiskit_to_ir(
    qc: QuantumCircuit,
    observable: SparsePauliOp,
    problem_name: str,
    params: dict[str, Any],
    canonicalize_circuit: bool = True,
) -> ProblemIR:
    if canonicalize_circuit:
        qc = canonicalize(qc)
    gates: list[GateOp] = []
    for instr in qc.data:
        name = instr.operation.name
        qubits = [qc.find_bit(q).index for q in instr.qubits]
        angle = float(instr.operation.params[0]) if instr.operation.params else None
        gates.append(GateOp(name=name, qubits=qubits, angle=angle))
    return ProblemIR(
        problem=problem_name,
        n_qubits=qc.num_qubits,
        params=params,
        gates=gates,
        observable=ObservableIR.from_sparse_pauli_op(observable),
    )
