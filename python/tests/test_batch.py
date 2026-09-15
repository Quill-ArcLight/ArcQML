"""验证行主序批量模拟器的通用训练接口。"""

import math
import unittest

import numpy as np
import arcqml


class BatchSimulatorTests(unittest.TestCase):
    """覆盖行主序 batch run、普通 loss 与 no_grad。"""

    def test_generic_batch_training_step(self):
        """验证 batch 训练使用标准 Tensor 反向传播流程。"""
        circuit = arcqml.Circuit(1)
        circuit.ry(math.pi / 3.0, 0)
        observable = arcqml.PauliSum.z(1, 0)
        states = np.ascontiguousarray(
            [[1.0 + 0.0j, 0.0j], [0.0j, 1.0 + 0.0j]],
            dtype=np.complex128,
        )
        simulator = arcqml.BatchStateVectorSimulator.from_amplitudes(1, states)
        optimizer = arcqml.Adam(learning_rate=0.1)

        optimizer.zero_grad(circuit)
        logits = simulator.run(circuit, observable)
        loss = arcqml.binary_cross_entropy_with_logits(
            logits, arcqml.tensor(np.array([1.0, 0.0], dtype=np.float64))
        )
        loss.backward()
        gradients = circuit.gradients()
        optimizer.step(circuit)

        self.assertAlmostEqual(
            loss.item(), math.log1p(math.exp(-0.5)), places=12
        )
        self.assertTrue(math.isfinite(gradients["ry_q0_theta_0"]))

    def test_no_grad_batch_run(self):
        """验证 batch 评估不建立自动微分图。"""
        circuit = arcqml.Circuit(1)
        circuit.ry(0.2, 0)
        observable = arcqml.PauliSum.z(1, 0)
        states = np.ascontiguousarray([[1.0 + 0.0j, 0.0j]], dtype=np.complex128)
        simulator = arcqml.BatchStateVectorSimulator.from_amplitudes(1, states)

        with arcqml.no_grad():
            output = simulator.run(circuit, observable)

        self.assertFalse(output.requires_grad)