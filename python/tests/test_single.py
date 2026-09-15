"""验证单态模拟器的通用训练接口。"""

import math
import unittest

import arcqml


class SingleSimulatorTests(unittest.TestCase):
    """覆盖单态 run、loss、backward 和状态 API。"""

    def test_generic_training_step(self):
        """验证单态训练遵循统一的五步流程。"""
        circuit = arcqml.Circuit(1)
        circuit.ry(math.pi / 3.0, 0)
        observable = arcqml.PauliSum.z(1, 0)
        simulator = arcqml.StateVectorSimulator(1)
        optimizer = arcqml.Adam(learning_rate=0.1)

        optimizer.zero_grad(circuit)
        prediction = simulator.run(circuit, observable)
        loss = arcqml.mse_loss(prediction, arcqml.tensor(0.0))
        loss.backward()
        gradients = circuit.gradients()
        optimizer.step(circuit)

        self.assertAlmostEqual(prediction.item(), 0.5, places=12)
        self.assertAlmostEqual(loss.item(), 0.125, places=12)
        self.assertAlmostEqual(
            gradients["ry_q0_theta_0"], -math.sqrt(3.0) / 4.0, places=12
        )

    def test_no_grad_run(self):
        """验证 no_grad 中的 run 不会连接自动微分图。"""
        circuit = arcqml.Circuit(1)
        circuit.ry(0.2, 0)
        observable = arcqml.PauliSum.z(1, 0)
        simulator = arcqml.StateVectorSimulator(1)

        with arcqml.no_grad():
            output = simulator.run(circuit, observable)

        self.assertFalse(output.requires_grad)