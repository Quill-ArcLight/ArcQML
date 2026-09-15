mod parameters;

use crate::circuit::parameters::{
    dictionary_from, gradients_for, parameter_values_for, zero_gradients,
};
use crate::error::value_error;
use arcqml_circuit::Circuit;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Python `Circuit`；底层委托给 `arcqml_circuit::Circuit`。
#[pyclass(name = "Circuit")]
pub struct PyCircuit {
    pub(crate) inner: Circuit,
}

#[pymethods]
impl PyCircuit {
    /// 创建一条包含 `num_qubits` 个量子比特的线路。
    #[new]
    pub(crate) fn new(num_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Circuit::new(num_qubits).map_err(value_error)?,
        })
    }

    #[getter]
    /// 返回线路中的量子比特数量 `num_qubits`。
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    /// 返回线路当前登记的可训练参数总数 `num_parameters`。
    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    /// 在指定量子比特上添加 `H` 门。
    fn h(&mut self, qubit: usize) -> PyResult<()> {
        self.inner.h(qubit).map_err(value_error)?;
        Ok(())
    }

    /// 在指定量子比特上添加 `X` 门。
    fn x(&mut self, qubit: usize) -> PyResult<()> {
        self.inner.x(qubit).map_err(value_error)?;
        Ok(())
    }

    /// 在指定量子比特上添加 `Y` 门。
    fn y(&mut self, qubit: usize) -> PyResult<()> {
        self.inner.y(qubit).map_err(value_error)?;
        Ok(())
    }

    /// 在指定量子比特上添加 `Z` 门。
    fn z(&mut self, qubit: usize) -> PyResult<()> {
        self.inner.z(qubit).map_err(value_error)?;
        Ok(())
    }

    /// 添加可训练的 `RX(angle)` 旋转门。
    fn rx(&mut self, angle: f64, qubit: usize) -> PyResult<()> {
        self.inner.rx(angle, qubit).map_err(value_error)?;
        Ok(())
    }

    /// 添加可训练的 `RY(angle)` 旋转门。
    fn ry(&mut self, angle: f64, qubit: usize) -> PyResult<()> {
        self.inner.ry(angle, qubit).map_err(value_error)?;
        Ok(())
    }

    /// 添加可训练的 `RZ(angle)` 旋转门。
    fn rz(&mut self, angle: f64, qubit: usize) -> PyResult<()> {
        self.inner.rz(angle, qubit).map_err(value_error)?;
        Ok(())
    }

    /// 添加可训练的 `Phase(angle)` 旋转门。
    fn phase(&mut self, angle: f64, qubit: usize) -> PyResult<()> {
        self.inner.phase(angle, qubit).map_err(value_error)?;
        Ok(())
    }

    /// 添加可训练的 `U3(theta, phi, lambda)` 门。
    fn u3(&mut self, theta: f64, phi: f64, lambda: f64, qubit: usize) -> PyResult<()> {
        self.inner
            .u3(theta, phi, lambda, qubit)
            .map_err(value_error)?;
        Ok(())
    }

    /// 添加一个 `CNOT(control, target)` 门。
    fn cnot(&mut self, control: usize, target: usize) -> PyResult<()> {
        self.inner.cnot(control, target).map_err(value_error)?;
        Ok(())
    }

    /// 清空线路所有参数当前累计的梯度；通常在优化器 `step()` 后调用。
    fn zero_grad(&self) {
        zero_gradients(&self.inner);
    }

    /// 返回参数名到当前参数值的 `dict` 映射。
    fn parameter_values(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        dictionary_from(parameter_values_for(&self.inner)?, py)
    }

    /// 返回参数名到当前累计梯度的 `dict` 映射；无梯度时返回错误。
    fn gradients(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        dictionary_from(gradients_for(&self.inner)?, py)
    }
}
