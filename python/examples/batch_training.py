"""使用通用 Tensor API 训练行主序批量量子线路。"""

import numpy as np
import arcqml

EPOCHS = 5


def build_circuit():
    """构造用于 batch 训练的参数化量子线路。"""
    circuit = arcqml.Circuit(2)
    circuit.ry(0.65, 0)
    circuit.rz(-0.35, 0)
    circuit.ry(0.15, 1)
    circuit.cnot(0, 1)
    circuit.ry(-0.45, 1)
    circuit.rz(0.20, 1)
    return circuit


def build_observable():
    """构造示例 Hamiltonian。"""
    observable = arcqml.PauliSum.z(2, 0, coefficient=0.60)
    observable.add_x(1, coefficient=0.25)
    observable.add_y(0, coefficient=-0.20)
    return observable


def training_states():
    """返回三个计算基初态组成的 C 连续行主序 batch。"""
    return np.ascontiguousarray(
        [
            [1.0 + 0.0j, 0.0j, 0.0j, 0.0j],
            [0.0j, 1.0 + 0.0j, 0.0j, 0.0j],
            [0.0j, 0.0j, 1.0 + 0.0j, 0.0j],
        ],
        dtype=np.complex128,
    )


def train():
    """执行统一的行主序 batch 训练循环。"""
    circuit = build_circuit()
    observable = build_observable()
    simulator = arcqml.BatchStateVectorSimulator.from_amplitudes(2, training_states())
    optimizer = arcqml.Adam(learning_rate=0.08)
    targets = arcqml.tensor(np.array([0.20, -0.15, 0.10], dtype=np.float64))

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(circuit)
        predictions = simulator.run(circuit, observable)
        loss = arcqml.mse_loss(predictions, targets)
        loss.backward()
        gradients = circuit.gradients()
        optimizer.step(circuit)

        print(
            f"epoch {epoch}: loss={loss.item():.10f}, "
            f"predictions={predictions.numpy()}, gradients={gradients}"
        )


if __name__ == "__main__":
    train()