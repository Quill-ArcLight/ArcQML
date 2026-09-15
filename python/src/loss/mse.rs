use crate::error::runtime_error;
use crate::tensor::PyTensor;
use pyo3::prelude::*;

/// 计算两个 Tensor 的半均方误差，并返回保留自动微分图的标量 Tensor。
#[pyfunction]
pub(crate) fn mse_loss(
    prediction: PyRef<'_, PyTensor>,
    target: PyRef<'_, PyTensor>,
) -> PyResult<PyTensor> {
    arcqml_loss::mse_loss(&prediction.inner, &target.inner)
        .map(PyTensor::from_inner)
        .map_err(runtime_error)
}
