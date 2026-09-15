use crate::circuit::PyCircuit;
use crate::error::runtime_error;
use crate::observable::PyPauliSum;
use crate::tensor::PyTensor;
use arcqml_core::{Tensor, TensorData};
use arcqml_sim::BatchStateVectorSimulator;
use num_complex::Complex64;
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

/// Python 行主序 `BatchStateVectorSimulator`。
#[pyclass(name = "BatchStateVectorSimulator")]
pub struct PyBatchStateVectorSimulator {
    inner: BatchStateVectorSimulator,
}

#[pymethods]
impl PyBatchStateVectorSimulator {
    /// 创建 `batch_size` 个 `|0…0⟩` 初态。
    #[new]
    fn new(num_qubits: usize, batch_size: usize) -> PyResult<Self> {
        Ok(Self {
            inner: BatchStateVectorSimulator::new(num_qubits, batch_size).map_err(runtime_error)?,
        })
    }

    /// 从 C 连续 NumPy `complex128` 行主序初态创建批量模拟器。
    #[classmethod]
    fn from_amplitudes(
        _type: &Bound<'_, PyType>,
        num_qubits: usize,
        amplitudes: PyReadonlyArray2<'_, Complex64>,
    ) -> PyResult<Self> {
        let state = state_tensor_from_numpy(num_qubits, &amplitudes)?;
        Ok(Self {
            inner: BatchStateVectorSimulator::from_state_tensor(num_qubits, state)
                .map_err(runtime_error)?,
        })
    }

    #[getter]
    /// 返回每个 batch 样本的量子比特数 `num_qubits`。
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    /// 返回当前 batch 大小 `batch_size`。
    fn batch_size(&self) -> usize {
        self.inner.batch_size()
    }

    /// 将所有 batch 样本重置为 `|0…0⟩`。
    fn reset(&mut self) -> PyResult<()> {
        self.inner.reset().map_err(runtime_error)?;
        Ok(())
    }

    /// 通过 `apply_circuit(circuit)` 在当前 batch 状态逐门执行电路，并永久更新模拟器状态。
    fn apply_circuit(&mut self, circuit: PyRef<'_, PyCircuit>) -> PyResult<()> {
        self.inner
            .apply_circuit(&circuit.inner)
            .map_err(runtime_error)?;
        Ok(())
    }

    /// 通过唯一的行主序 batch `run(circuit, observable)` 路径返回逐样本可微期望值 `Tensor`；只在 batch 顶层并行。
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

    /// 返回当前 batch 量子态的可微复数振幅 `Tensor`。
    fn amplitudes(&self) -> PyResult<PyTensor> {
        self.inner
            .amplitudes()
            .map(PyTensor::from_inner)
            .map_err(runtime_error)
    }
}

/// 从 C 连续 NumPy `complex128` 行主序初态构造 batch `Tensor`。
fn state_tensor_from_numpy(
    num_qubits: usize,
    amplitudes: &PyReadonlyArray2<'_, Complex64>,
) -> PyResult<Tensor> {
    let shape = amplitudes.shape();
    let batch_size = shape[0];
    let dimension = shape[1];
    if batch_size == 0 {
        return Err(PyValueError::new_err("初态 batch 不能为空"));
    }
    let expected_dimension = 1usize
        .checked_shl(num_qubits as u32)
        .ok_or_else(|| PyValueError::new_err("量子比特数过大，无法计算状态向量维度"))?;
    if dimension != expected_dimension {
        return Err(PyValueError::new_err(format!(
            "初态列数必须为 {expected_dimension}，实际为 {dimension}"
        )));
    }
    let values = amplitudes
        .as_slice()
        .map_err(|_| PyValueError::new_err("初态必须是 C 连续存储的 NumPy complex128 二维数组"))?;
    Tensor::new(TensorData::FlatC64 {
        data: values.to_vec(),
        shape: vec![batch_size, dimension],
    })
    .map_err(runtime_error)
}
