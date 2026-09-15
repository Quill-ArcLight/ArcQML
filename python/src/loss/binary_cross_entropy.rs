use crate::error::runtime_error;
use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// 计算二元交叉熵 logits 损失，并返回保留自动微分图的标量 Tensor。
#[pyfunction]
pub(crate) fn binary_cross_entropy_with_logits(
    logits: PyRef<'_, PyTensor>,
    targets: PyRef<'_, PyTensor>,
) -> PyResult<PyTensor> {
    arcqml_loss::binary_cross_entropy_with_logits_loss(&logits.inner, &targets.inner)
        .map(PyTensor::from_inner)
        .map_err(runtime_error)
}
