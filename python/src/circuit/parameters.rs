use crate::error::{runtime_error, value_error};
use arcqml_circuit::{Circuit, ParameterId};
use arcqml_core::Tensor;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use std::collections::BTreeMap;

/// 将按参数名排序的 Rust 映射转换为 Python `dict`。
pub(crate) fn dictionary_from(
    values: BTreeMap<String, f64>,
    py: Python<'_>,
) -> PyResult<Py<PyDict>> {
    let dictionary = PyDict::new(py);
    for (name, value) in values {
        dictionary.set_item(name, value)?;
    }
    Ok(dictionary.unbind())
}

/// 读取线路全部参数的当前梯度，并以参数名到 `f64` 的映射返回。
pub(crate) fn gradients_for(circuit: &Circuit) -> PyResult<BTreeMap<String, f64>> {
    circuit
        .parameters()
        .iter()
        .enumerate()
        .map(|(index, parameter)| {
            let name = circuit
                .parameter_name(ParameterId::new(index))
                .map_err(value_error)?;
            let gradient = parameter
                .grad()
                .ok_or_else(|| PyRuntimeError::new_err(format!("参数 {name} 尚未产生梯度")))?;
            Ok((name, scalar_to_f64(&gradient)?))
        })
        .collect()
}

/// 读取线路全部参数的当前数值，并以参数名到 `f64` 的映射返回。
pub(crate) fn parameter_values_for(circuit: &Circuit) -> PyResult<BTreeMap<String, f64>> {
    circuit
        .parameters()
        .iter()
        .enumerate()
        .map(|(index, parameter)| {
            let name = circuit
                .parameter_name(ParameterId::new(index))
                .map_err(value_error)?;
            Ok((name, scalar_to_f64(&parameter.tensor())?))
        })
        .collect()
}

/// 将单元素 `F32`/`F64` `Tensor` 读取为 Python 统一使用的 `f64` 标量。
fn scalar_to_f64(tensor: &Tensor) -> PyResult<f64> {
    tensor.value().map_err(runtime_error)
}

/// 清空线路中所有可训练参数已累计的梯度。
pub(crate) fn zero_gradients(circuit: &Circuit) {
    for parameter in circuit.parameters() {
        parameter.zero_grad();
    }
}
