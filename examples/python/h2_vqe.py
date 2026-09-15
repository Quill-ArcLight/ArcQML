"""使用 ArcQML Python API 求解 H2/STO-3G 基态能量。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import arcqml


NUM_QUBITS = 4
ACTIVE_ELECTRONS = 2
LAYERS = 6
STEPS = 100
EXACT_GROUND_ENERGY = -1.1361894542078266
HARTREE_FOCK_ENERGY = -1.1173490350562805
CHEMICAL_ACCURACY = 1.6e-3
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "h2_hamiltonian.json"


def load_hamiltonian(path: Path) -> arcqml.PauliSum:
    """从 examples/data 的公共 JSON 定义构造 PauliSum。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    hamiltonian = arcqml.PauliSum(num_qubits=payload["num_qubits"])
    for term in payload["terms"]:
        coefficient = term["coefficient"]
        operations = term["paulis"]
        if not operations:
            hamiltonian.add_identity(coefficient=coefficient)
            continue
        hamiltonian.add_term(
            paulis="".join(operation["pauli"] for operation in operations),
            qubits=[operation["qubit"] for operation in operations],
            coefficient=coefficient,
        )
    return hamiltonian


def build_ansatz() -> arcqml.Circuit:
    """构造 Hartree-Fock 初态和 6 层硬件高效 ansatz。"""
    circuit = arcqml.Circuit(num_qubits=NUM_QUBITS)
    for qubit in range(ACTIVE_ELECTRONS):
        circuit.x(qubit=qubit)

    parameter_index = 0
    for _layer in range(LAYERS):
        for qubit in range(NUM_QUBITS):
            ry = 0.04 * math.sin(0.37 * (parameter_index + 1))
            parameter_index += 1
            rz = 0.04 * math.sin(0.37 * (parameter_index + 1))
            parameter_index += 1
            circuit.ry(angle=ry, qubit=qubit)
            circuit.rz(angle=rz, qubit=qubit)
        for control in range(NUM_QUBITS):
            circuit.cnot(control=control, target=(control + 1) % NUM_QUBITS)
    return circuit


def evaluate(
    simulator: arcqml.StateVectorSimulator,
    circuit: arcqml.Circuit,
    hamiltonian: arcqml.PauliSum,
) -> float:
    """无梯度计算当前能量。"""
    with arcqml.no_grad():
        return float(simulator.run(circuit=circuit, observable=hamiltonian).item())


def main() -> None:
    hamiltonian = load_hamiltonian(path=DATA_PATH)
    circuit = build_ansatz()
    simulator = arcqml.StateVectorSimulator(num_qubits=NUM_QUBITS)
    optimizer = arcqml.Adam(learning_rate=0.05)

    print(f"H2/STO-3G, Jordan-Wigner, {NUM_QUBITS} qubits")
    print(f"Hamiltonian terms : {hamiltonian.num_terms}")
    print(f"Trainable params  : {circuit.num_parameters}")
    print(f"Hartree-Fock      : {HARTREE_FOCK_ENERGY:.12f} Ha")
    print(f"Exact ground      : {EXACT_GROUND_ENERGY:.12f} Ha")

    for step in range(1, STEPS + 1):
        energy = simulator.run(circuit=circuit, observable=hamiltonian)
        energy.backward()
        optimizer.step(circuit=circuit)
        optimizer.zero_grad(circuit=circuit)

        if step == 1 or step % 10 == 0:
            current = evaluate(
                simulator=simulator,
                circuit=circuit,
                hamiltonian=hamiltonian,
            )
            error = abs(current - EXACT_GROUND_ENERGY)
            print(f"step {step:>3}: energy={current:.12f} Ha, |error|={error:.3e} Ha")

    final_energy = evaluate(
        simulator=simulator,
        circuit=circuit,
        hamiltonian=hamiltonian,
    )
    error = abs(final_energy - EXACT_GROUND_ENERGY)
    reached = "reached" if error < CHEMICAL_ACCURACY else "not reached"
    print(f"\nVQE energy        : {final_energy:.12f} Ha")
    print(f"Absolute error    : {error:.6e} Ha")
    print(f"Chemical accuracy: {reached} (threshold {CHEMICAL_ACCURACY:.1e} Ha)")


if __name__ == "__main__":
    main()
