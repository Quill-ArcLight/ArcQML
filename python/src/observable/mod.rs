use crate::error::value_error;
use arcqml_observable::{Pauli, PauliOp, PauliString, SparsePauliOp};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// `PauliSum`：用于计算期望值的 Pauli Hamiltonian。
#[pyclass(name = "PauliSum")]
pub struct PyPauliSum {
    pub(crate) inner: SparsePauliOp,
}

#[pymethods]
impl PyPauliSum {
    /// 创建作用于 `num_qubits` 个量子比特的空 `PauliSum`。
    #[new]
    fn new(num_qubits: usize) -> PyResult<Self> {
        Ok(Self {
            inner: SparsePauliOp::zero(num_qubits).map_err(value_error)?,
        })
    }

    #[getter]
    /// 返回该可观测量作用的量子比特数量 `num_qubits`。
    fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    #[getter]
    /// 返回 `PauliSum` 当前包含的项数 `num_terms`。
    fn num_terms(&self) -> usize {
        self.inner.len()
    }

    /// 构造只含一个 `Pauli-X` 项的 `PauliSum`。
    #[staticmethod]
    #[pyo3(signature = (num_qubits, qubit, coefficient=1.0))]
    fn x(num_qubits: usize, qubit: usize, coefficient: f64) -> PyResult<Self> {
        Self::single(num_qubits, qubit, Pauli::X, coefficient)
    }

    /// 构造只含一个 `Pauli-Y` 项的 `PauliSum`。
    #[staticmethod]
    #[pyo3(signature = (num_qubits, qubit, coefficient=1.0))]
    fn y(num_qubits: usize, qubit: usize, coefficient: f64) -> PyResult<Self> {
        Self::single(num_qubits, qubit, Pauli::Y, coefficient)
    }

    /// 构造只含一个 `Pauli-Z` 项的 `PauliSum`。
    #[staticmethod]
    #[pyo3(signature = (num_qubits, qubit, coefficient=1.0))]
    pub(crate) fn z(num_qubits: usize, qubit: usize, coefficient: f64) -> PyResult<Self> {
        Self::single(num_qubits, qubit, Pauli::Z, coefficient)
    }

    /// 向当前 `PauliSum` 追加一个 `Pauli-X` 项。
    #[pyo3(signature = (qubit, coefficient=1.0))]
    fn add_x(&mut self, qubit: usize, coefficient: f64) -> PyResult<()> {
        self.add_single(qubit, Pauli::X, coefficient)
    }

    /// 向当前 `PauliSum` 追加一个 `Pauli-Y` 项。
    #[pyo3(signature = (qubit, coefficient=1.0))]
    fn add_y(&mut self, qubit: usize, coefficient: f64) -> PyResult<()> {
        self.add_single(qubit, Pauli::Y, coefficient)
    }

    /// 向当前 `PauliSum` 追加一个 `Pauli-Z` 项。
    #[pyo3(signature = (qubit, coefficient=1.0))]
    fn add_z(&mut self, qubit: usize, coefficient: f64) -> PyResult<()> {
        self.add_single(qubit, Pauli::Z, coefficient)
    }

    /// 向当前 `PauliSum` 追加一个多量子比特 `PauliString` 项。
    #[pyo3(signature = (paulis, qubits, coefficient=1.0))]
    fn add_term(&mut self, paulis: &str, qubits: Vec<usize>, coefficient: f64) -> PyResult<()> {
        let symbols = paulis.chars().collect::<Vec<_>>();
        if symbols.len() != qubits.len() {
            return Err(PyValueError::new_err(format!(
                "`paulis` 的长度为 {}，但 `qubits` 的长度为 {}",
                symbols.len(),
                qubits.len()
            )));
        }

        let mut ops = Vec::with_capacity(symbols.len());
        for (symbol, qubit) in symbols.into_iter().zip(qubits) {
            let pauli = match symbol.to_ascii_uppercase() {
                'I' => Pauli::I,
                'X' => Pauli::X,
                'Y' => Pauli::Y,
                'Z' => Pauli::Z,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "不支持的 Pauli 符号 {symbol:?}；只允许 `I`、`X`、`Y` 或 `Z`"
                    )));
                }
            };
            ops.push(PauliOp::new(qubit, pauli));
        }

        let string = PauliString::new(self.inner.num_qubits(), ops).map_err(value_error)?;
        self.inner
            .add_pauli_string(coefficient, string)
            .map_err(value_error)?;
        Ok(())
    }

    /// 向当前 `PauliSum` 追加一个单位算符项。
    #[pyo3(signature = (coefficient=1.0))]
    fn add_identity(&mut self, coefficient: f64) -> PyResult<()> {
        let string = PauliString::identity(self.inner.num_qubits()).map_err(value_error)?;
        self.inner
            .add_pauli_string(coefficient, string)
            .map_err(value_error)?;
        Ok(())
    }
}

impl PyPauliSum {
    /// 用给定系数和单比特 Pauli 类型创建底层 `SparsePauliOp`。
    fn single(num_qubits: usize, qubit: usize, pauli: Pauli, coefficient: f64) -> PyResult<Self> {
        Ok(Self {
            inner: SparsePauliOp::single(num_qubits, qubit, pauli, coefficient)
                .map_err(value_error)?,
        })
    }

    /// 将一个单比特 Pauli 项添加到已有的底层 `SparsePauliOp` 中。
    fn add_single(&mut self, qubit: usize, pauli: Pauli, coefficient: f64) -> PyResult<()> {
        let pauli_string =
            PauliString::single(self.inner.num_qubits(), qubit, pauli).map_err(value_error)?;
        self.inner
            .add_pauli_string(coefficient, pauli_string)
            .map_err(value_error)?;
        Ok(())
    }
}
