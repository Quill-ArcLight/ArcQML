use crate::error::runtime_error;
use crate::simulator::PyStateVectorSimulator;
use crate::tensor::PyTensor;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

/// 计算计算基概率，并返回保留自动微分关系的 `Tensor`。
#[pyfunction]
fn probabilities(simulator: PyRef<'_, PyStateVectorSimulator>) -> PyResult<PyTensor> {
    arcqml_analysis::probabilities(&simulator.inner)
        .map(PyTensor::from_inner)
        .map_err(runtime_error)
}

/// 计算指定量子比特的边缘概率，并返回保留自动微分关系的 `Tensor`。
#[pyfunction]
fn marginal_probabilities(
    simulator: PyRef<'_, PyStateVectorSimulator>,
    qubits: Vec<usize>,
) -> PyResult<PyTensor> {
    arcqml_analysis::marginal_probabilities(&simulator.inner, &qubits)
        .map(PyTensor::from_inner)
        .map_err(runtime_error)
}

/// 计算两个单态模拟器当前状态的保真度，并返回可微标量 `Tensor`。
#[pyfunction]
fn fidelity(
    lhs: PyRef<'_, PyStateVectorSimulator>,
    rhs: PyRef<'_, PyStateVectorSimulator>,
) -> PyResult<PyTensor> {
    arcqml_analysis::fidelity(&lhs.inner, &rhs.inner)
        .map(PyTensor::from_inner)
        .map_err(runtime_error)
}

/// 计算指定量子比特的 Bloch 向量 `(x, y, z)`。
#[pyfunction]
fn bloch_vector(
    simulator: PyRef<'_, PyStateVectorSimulator>,
    qubit: usize,
) -> PyResult<(f64, f64, f64)> {
    let vector = arcqml_analysis::bloch_vector(&simulator.inner, qubit).map_err(runtime_error)?;
    Ok((vector.x, vector.y, vector.z))
}

/// 将分析函数注册到 `analysis` 子模块。
pub(crate) fn add_to_module(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let analysis = PyModule::new(module.py(), "analysis")?;
    analysis.add_function(wrap_pyfunction!(probabilities, &analysis)?)?;
    analysis.add_function(wrap_pyfunction!(marginal_probabilities, &analysis)?)?;
    analysis.add_function(wrap_pyfunction!(fidelity, &analysis)?)?;
    analysis.add_function(wrap_pyfunction!(bloch_vector, &analysis)?)?;
    module.add_submodule(&analysis)
}
