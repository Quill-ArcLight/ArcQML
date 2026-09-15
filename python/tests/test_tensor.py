"""验证 Python Tensor 通用接口。"""

import unittest

import arcqml


class TensorTests(unittest.TestCase):
    """覆盖 Tensor 创建和图控制语义。"""

    def test_tensor_metadata(self):
        """验证数值列表创建的 Tensor 保留形状和数据类型。"""
        value = arcqml.tensor([1.0, 2.0, 3.0])

        self.assertEqual(value.shape, [3])
        self.assertEqual(value.dtype, "float64")
        self.assertFalse(value.requires_grad)

    def test_detach(self):
        """验证 detach 会断开自动微分图。"""
        source = arcqml.tensor(0.5, requires_grad=True)
        detached = source.detach()

        self.assertTrue(source.requires_grad)
        self.assertFalse(detached.requires_grad)
        self.assertAlmostEqual(detached.item(), 0.5, places=12)