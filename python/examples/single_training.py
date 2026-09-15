"""使用通用 Tensor API 训练单态量子线路。"""

import arcqml

EPOCHS = 5


def build_circuit():
    """构造用于单态训练的参数化量子线路。"""
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


def train():
    """执行统一的单态训练循环。"""
    circuit = build_circuit()
    observable = build_observable()
    simulator = arcqml.StateVectorSimulator(2)
    optimizer = arcqml.Adam(learning_rate=0.08)
    target = arcqml.tensor(0.20)

    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad(circuit)
        prediction = simulator.run(circuit, observable)
        loss = arcqml.mse_loss(prediction, target)
        loss.backward()
        gradients = circuit.gradients()
        optimizer.step(circuit)

        print(
            f"epoch {epoch}: loss={loss.item():.10f}, "
            f"prediction={prediction.item():+.8f}, gradients={gradients}"
        )


if __name__ == "__main__":
    train()