use crate::circuit::PyCircuit;
use crate::error::runtime_error;
use crate::observable::PyPauliSum;
use crate::tensor::PyTensor;
use arcqml_core::{Tensor, TensorData};
use arcqml_sim::StateVectorSimulator;
use num_complex::Complex64;
use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods, PyType};

/// Python 单态 `StateVectorSimulator`。
#[pyclass(name = "StateVectorSimulator")]
pub struct PyStateVectorSimulator {
    pub(crate) inner: StateVectorSimulator,
}

#[pymethods]
impl PyStateVectorSimulator {
    /// 创建初态为 `|0…0⟩` 的单态模拟器。
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            inner: StateVectorSimulator::new(num_qubits).map_err(runtime_error)?,
        })
    }

    /// 从 C 连续 NumPy `complex128` 初态创建单态模拟器。
    #[classmethod]
    fn from_amplitudes(
        _type: &Bound<'_, PyType>,
        num_qubits: usize,
        amplitudes: PyReadonlyArray1<'_, Complex64>,
    ) -> PyResult<Self> {
        let state = state_tensor_from_numpy(num_qubits, &amplitudes)?;
        Ok(Self {
            inner: StateVectorSimulator::from_state_tensor(num_qubits, state)
                .map_err(runtime_error)?,
        })
    }

    #[getter]
    /// 返回模拟器的量子比特数 `num_qubits`。
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// 将模拟器重置为 `|0…0⟩` 初态。
    fn reset(&mut self) -> PyResult<()> {
        self.inner.reset().map_err(runtime_error)?;
        Ok(())
    }

    /// 通过 `apply_circuit(circuit)` 在当前量子态逐门执行电路，并永久更新模拟器状态。
    fn apply_circuit(&mut self, circuit: PyRef<'_, PyCircuit>) -> PyResult<()> {
        self.inner
            .apply_circuit(&circuit.inner)
            .map_err(runtime_error)?;
        Ok(())
    }

    /// 通过 `run(circuit, observable)` 创建一个量子伴随自动微分节点，并返回可微期望值 `Tensor`；不修改当前状态。
    fn run(
        &self,
        circuit: PyRef<'_, PyCircuit>,
        observable: PyRef<'_, PyPauliSum>,
    ) -> PyResult<PyTensor> {
        self.inner
            .run(&circuit.inner, &observable.inner)
            .map(PyTensor::from_inner)
            .map_err(runtime_error)
    }

    /// 返回当前量子态的可微复数振幅 `Tensor`。
    fn amplitudes(&self) -> PyResult<PyTensor> {
        self.inner
            .amplitudes()
            .map(PyTensor::from_inner)
            .map_err(runtime_error)
    }

    /// 对当前量子态进行末端全量子比特 `sample_counts()`，并返回各 bitstring 的出现次数。
    #[pyo3(signature = (shots, *, seed=None))]
    fn sample_counts(
        &self,
        py: Python<'_>,
        shots: usize,
        seed: Option<u64>,
    ) -> PyResult<Py<PyDict>> {
        let counts = self
            .inner
            .sample_counts(shots, seed)
            .map_err(runtime_error)?;
        let result = PyDict::new(py);
        for (bitstring, count) in counts {
            result.set_item(bitstring, count)?;
        }
        Ok(result.unbind())
    }
}

/// 从 C 连续 NumPy `complex128` 振幅构造单态 `Tensor`。
fn state_tensor_from_numpy(
    num_qubits: usize,
    amplitudes: &PyReadonlyArray1<'_, Complex64>,
) -> PyResult<Tensor> {
    let expected_dimension = 1usize
        .checked_shl(num_qubits as u32)
        .ok_or_else(|| PyValueError::new_err("量子比特数过大，无法计算状态向量维度"))?;
    if amplitudes.len() != expected_dimension {
        return Err(PyValueError::new_err(format!(
            "初态长度必须为 {expected_dimension}，实际为 {}",
            amplitudes.len()
        )));
    }
    let values = amplitudes
        .as_slice()
        .map_err(|_| PyValueError::new_err("初态必须是 C 连续存储的 NumPy complex128 一维数组"))?;
    Tensor::new(TensorData::FlatC64 {
        data: values.to_vec(),
        shape: vec![expected_dimension],
    })
    .map_err(runtime_error)
}
