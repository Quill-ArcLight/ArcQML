use crate::error::runtime_error;
use arcqml_core::{Tensor, TensorData};
use num_complex::Complex64;
use numpy::{PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// 将 Python 数值、列表或 `numpy.ndarray` 转换为新的 Rust `Tensor`。
pub(crate) fn tensor_from_python(value: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, f64>>() {
        return tensor_from_f64_array(&array);
    }
    if let Ok(array) = value.extract::<PyReadonlyArrayDyn<'_, Complex64>>() {
        return tensor_from_c64_array(&array);
    }
    if let Ok(scalar) = value.extract::<f64>() {
        return Tensor::new(scalar).map_err(runtime_error);
    }
    if let Ok(scalar) = value.extract::<Complex64>() {
        return Tensor::new(scalar).map_err(runtime_error);
    }
    if let Ok(values) = value.extract::<Vec<f64>>() {
        return Tensor::new(values).map_err(runtime_error);
    }
    if let Ok(values) = value.extract::<Vec<Vec<f64>>>() {
        return Tensor::new(values).map_err(runtime_error);
    }
    if let Ok(values) = value.extract::<Vec<Complex64>>() {
        return Tensor::new(values).map_err(runtime_error);
    }
    if let Ok(values) = value.extract::<Vec<Vec<Complex64>>>() {
        return Tensor::new(values).map_err(runtime_error);
    }
    Err(PyTypeError::new_err(
        "`Tensor` 仅接受数值、数值列表或 C 连续的 NumPy `float64`/`complex128` 数组",
    ))
}

/// 从 C 连续的 NumPy `float64` 数组创建并保持原 `shape` 的 `Tensor`。
fn tensor_from_f64_array(values: &PyReadonlyArrayDyn<'_, f64>) -> PyResult<Tensor> {
    let data = values
        .as_slice()
        .map_err(|_| PyTypeError::new_err("NumPy float64 数组必须采用 C 连续存储"))?;
    Tensor::new(TensorData::FlatF64 {
        data: data.to_vec(),
        shape: values.shape().to_vec(),
    })
    .map_err(runtime_error)
}

/// 从 C 连续的 NumPy `complex128` 数组创建并保持原 `shape` 的 `Tensor`。
fn tensor_from_c64_array(values: &PyReadonlyArrayDyn<'_, Complex64>) -> PyResult<Tensor> {
    let data = values
        .as_slice()
        .map_err(|_| PyTypeError::new_err("NumPy complex128 数组必须采用 C 连续存储"))?;
    Tensor::new(TensorData::FlatC64 {
        data: data.to_vec(),
        shape: values.shape().to_vec(),
    })
    .map_err(runtime_error)
}
