use crate::{
    ObservableError::*, ObservableResult, Pauli, PauliOp, PauliString,
    compiled_hamiltonian::CompiledHamiltonian,
};
use arcqml_circuit::Qubit;
use num_complex::Complex64;
use serde_json;
use std::{collections::BTreeMap, sync::OnceLock};

/// 稀疏 Pauli 算符中的一项 `coefficient × pauli_string`。
#[derive(Debug, Clone, PartialEq)]
pub struct PauliTerm {
    coefficient: f64,
    pauli_string: PauliString,
}

impl PauliTerm {
    /// 创建具有实数系数的 Pauli 项。
    ///
    /// # Errors
    ///
    /// 当 `coefficient` 不是有限数时返回错误。
    pub fn new(coefficient: f64, pauli_string: PauliString) -> ObservableResult<Self> {
        if !coefficient.is_finite() {
            return Err(NonFiniteCoefficientError {
                value: coefficient.to_string(),
            });
        }

        Ok(Self {
            coefficient,
            pauli_string,
        })
    }

    /// 返回系数。
    pub fn coefficient(&self) -> f64 {
        self.coefficient
    }

    /// 返回该项的 Pauli 字符串。
    pub fn pauli_string(&self) -> &PauliString {
        &self.pauli_string
    }

    /// 返回该项的 Pauli 字符串；这是 [`PauliTerm::pauli_string`] 的简写。
    pub fn string(&self) -> &PauliString {
        self.pauli_string()
    }
}

/// 由实系数 Pauli 字符串之和表示的稀疏 Hermitian 算符。
///
/// 数学定义为 `H = Σ_j c_j P_j`。不同项可以包含相同的 [`PauliString`]；调用
/// [`SparsePauliOp::simplify`] 可以合并它们。内部编译缓存不属于数学定义，克隆时会被丢弃。
#[derive(Debug)]
pub struct SparsePauliOp {
    num_qubits: usize,
    terms: Vec<PauliTerm>,
    compiled_hamiltonian: OnceLock<CompiledHamiltonian>,
}

impl Clone for SparsePauliOp {
    /// 克隆可观测量定义，但不复制可重新生成的内部执行缓存。
    fn clone(&self) -> Self {
        Self {
            num_qubits: self.num_qubits,
            terms: self.terms.clone(),
            compiled_hamiltonian: OnceLock::new(),
        }
    }
}

impl PartialEq for SparsePauliOp {
    /// 比较可观测量的公开数学定义，忽略内部执行缓存。
    fn eq(&self, other: &Self) -> bool {
        self.num_qubits == other.num_qubits && self.terms == other.terms
    }
}

/// 通用 Pauli observable。
pub type PauliObservable = SparsePauliOp;

/// 多个 Pauli 项的和。
pub type PauliSum = SparsePauliOp;

/// 物理语境中的 Hamiltonian。
pub type Hamiltonian = SparsePauliOp;

impl SparsePauliOp {
    /// 从一组 Pauli 项创建稀疏算符。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零，或任一项的量子比特数与 `num_qubits` 不一致时返回错误。
    pub fn new(num_qubits: usize, terms: Vec<PauliTerm>) -> ObservableResult<Self> {
        if num_qubits == 0 {
            return Err(EmptyQubitError);
        }

        for term in terms.iter() {
            if term.pauli_string().num_qubits() != num_qubits {
                return Err(QubitCountMismatchError {
                    expected: num_qubits,
                    actual: term.pauli_string().num_qubits(),
                });
            }
        }

        Ok(Self {
            num_qubits,
            terms,
            compiled_hamiltonian: OnceLock::new(),
        })
    }

    /// 创建不含任何项的零算符。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零时返回错误。
    pub fn zero(num_qubits: usize) -> ObservableResult<Self> {
        Self::new(num_qubits, Vec::new())
    }

    /// 从单个 Pauli 字符串及其实系数创建稀疏算符。
    ///
    /// # Errors
    ///
    /// 当 `coefficient` 不是有限数，或 `pauli_string` 的量子比特数为零时返回错误。
    pub fn from_pauli_string(
        coefficient: f64,
        pauli_string: PauliString,
    ) -> ObservableResult<Self> {
        let num_qubits = pauli_string.num_qubits();
        let term = PauliTerm::new(coefficient, pauli_string)?;

        Self::new(num_qubits, vec![term])
    }

    /// 创建仅在指定量子比特上非平凡的 Pauli 可观测量。
    ///
    /// # Errors
    ///
    /// 当量子比特数为零、`qubit` 越界，或 `coefficient` 不是有限数时返回错误。
    pub fn single(
        num_qubits: usize,
        qubit: impl Into<Qubit>,
        pauli: Pauli,
        coefficient: f64,
    ) -> ObservableResult<Self> {
        let string = PauliString::single(num_qubits, qubit, pauli)?;

        Self::from_pauli_string(coefficient, string)
    }

    /// 创建 `coefficient × I` 恒等可观测量。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零，或 `coefficient` 不是有限数时返回错误。
    pub fn identity(num_qubits: usize, coefficient: f64) -> ObservableResult<Self> {
        Self::constant(num_qubits, coefficient)
    }

    /// 创建值为 `value` 的常数算符 `value × I`。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零，或 `value` 不是有限数时返回错误。
    pub fn constant(num_qubits: usize, value: f64) -> ObservableResult<Self> {
        let string = PauliString::identity(num_qubits)?;
        let term = PauliTerm::new(value, string)?;

        Self::new(num_qubits, vec![term])
    }

    /// 从 ArcQML 稀疏 Pauli JSON 格式解析算符。
    ///
    /// 根对象必须包含非零 `num_qubits` 和 `terms` 数组；每一项包含有限实数
    /// `coefficient` 以及由 `{ "qubit", "pauli" }` 对象组成的 `paulis` 数组。
    ///
    /// # Errors
    ///
    /// 当 JSON 语法或字段类型无效、Pauli 名称不是 `X`/`Y`/`Z`、量子比特
    /// 越界或重复、项的量子比特数不一致，或系数不是有限数时返回错误。
    pub fn from_json(input: &str) -> ObservableResult<Self> {
        let root = serde_json::from_str::<serde_json::Value>(input)
            .map_err(|error| json_error(error.to_string()))?;
        let root = root
            .as_object()
            .ok_or_else(|| json_error("根节点必须是 JSON 对象"))?;
        let num_qubits = json_usize(root.get("num_qubits"), "num_qubits")?;
        let terms_json = json_array(root.get("terms"), "terms")?;
        let mut terms = Vec::with_capacity(terms_json.len());

        for (term_index, term_json) in terms_json.iter().enumerate() {
            let term = term_json
                .as_object()
                .ok_or_else(|| json_error(format!("terms[{term_index}] 必须是 JSON 对象")))?;
            let coefficient = json_f64(
                term.get("coefficient"),
                &format!("terms[{term_index}].coefficient"),
            )?;
            let paulis_json =
                json_array(term.get("paulis"), &format!("terms[{term_index}].paulis"))?;
            let mut ops = Vec::with_capacity(paulis_json.len());

            for (pauli_index, pauli_json) in paulis_json.iter().enumerate() {
                let pauli_object = pauli_json.as_object().ok_or_else(|| {
                    json_error(format!(
                        "terms[{term_index}].paulis[{pauli_index}] 必须是 JSON 对象"
                    ))
                })?;
                let qubit = json_usize(
                    pauli_object.get("qubit"),
                    &format!("terms[{term_index}].paulis[{pauli_index}].qubit"),
                )?;
                let pauli = match json_string(
                    pauli_object.get("pauli"),
                    &format!("terms[{term_index}].paulis[{pauli_index}].pauli"),
                )? {
                    "X" => Pauli::X,
                    "Y" => Pauli::Y,
                    "Z" => Pauli::Z,
                    value => {
                        return Err(json_error(format!(
                            "terms[{term_index}].paulis[{pauli_index}].pauli 必须是 X、Y 或 Z，得到 {value}"
                        )));
                    }
                };
                ops.push(PauliOp::new(qubit, pauli));
            }

            let string = PauliString::new(num_qubits, ops)?;
            terms.push(PauliTerm::new(coefficient, string)?);
        }

        Self::new(num_qubits, terms)
    }

    /// 返回算符作用的量子比特数量。
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// 返回当前保存的 Pauli 项数；尚未隐式合并重复项。
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// 判断算符是否不含任何 Pauli 项，即是否为零算符的空表示。
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// 按插入顺序返回全部 Pauli 项。
    pub fn terms(&self) -> &[PauliTerm] {
        &self.terms
    }

    #[doc(hidden)]
    /// 为状态矢量模拟器计算可观测量期望值，并自动复用内部编译缓存。
    pub fn statevector_expectation(&self, amplitudes: &[Complex64]) -> ObservableResult<f64> {
        self.compiled_hamiltonian()?.expectation(amplitudes)
    }

    #[doc(hidden)]
    /// 为状态矢量自动微分一次计算期望值和 H|ψ⟩。
    pub fn statevector_expectation_and_apply(
        &self,
        amplitudes: &[Complex64],
    ) -> ObservableResult<(f64, Vec<Complex64>)> {
        self.compiled_hamiltonian()?
            .expectation_and_apply(amplitudes)
    }
    #[doc(hidden)]
    /// 为已在外层按 batch 并行的状态矢量路径串行计算期望值和 H|ψ⟩。
    pub fn statevector_expectation_and_apply_serial(
        &self,
        amplitudes: &[Complex64],
    ) -> ObservableResult<(f64, Vec<Complex64>)> {
        self.compiled_hamiltonian()?
            .expectation_and_apply_serial(amplitudes)
    }

    /// 惰性构建并返回与当前公开定义一致的内部执行计划。
    fn compiled_hamiltonian(&self) -> ObservableResult<&CompiledHamiltonian> {
        if let Some(compiled) = self.compiled_hamiltonian.get() {
            return Ok(compiled);
        }

        let compiled = CompiledHamiltonian::compile(self)?;
        let _ = self.compiled_hamiltonian.set(compiled);
        Ok(self
            .compiled_hamiltonian
            .get()
            .expect("内部 Hamiltonian 缓存在写入后必须存在"))
    }

    /// 在末尾添加一项，并使内部执行缓存失效。
    ///
    /// # Errors
    ///
    /// 当 `term` 的量子比特数与当前算符不一致时返回错误。
    pub fn add_term(&mut self, term: PauliTerm) -> ObservableResult<&mut Self> {
        if term.pauli_string().num_qubits() != self.num_qubits {
            return Err(QubitCountMismatchError {
                expected: self.num_qubits,
                actual: term.pauli_string().num_qubits(),
            });
        }

        let _ = self.compiled_hamiltonian.take();
        self.terms.push(term);

        Ok(self)
    }

    /// 使用给定系数在末尾添加一个 Pauli 字符串项。
    ///
    /// # Errors
    ///
    /// 当 `coefficient` 不是有限数，或 `pauli_string` 的量子比特数与当前算符
    /// 不一致时返回错误。
    pub fn add_pauli_string(
        &mut self,
        coefficient: f64,
        pauli_string: PauliString,
    ) -> ObservableResult<&mut Self> {
        self.add_term(PauliTerm::new(coefficient, pauli_string)?)
    }

    /// 返回所有项系数乘以 `factor` 后的新算符。
    ///
    /// # Errors
    ///
    /// 当 `factor` 不是有限数，或缩放后的任一系数溢出为非有限数时返回错误。
    pub fn scale(&self, factor: f64) -> ObservableResult<Self> {
        if !factor.is_finite() {
            return Err(NonFiniteCoefficientError {
                value: factor.to_string(),
            });
        }

        let mut terms = Vec::with_capacity(self.terms.len());

        for term in self.terms.iter() {
            terms.push(PauliTerm::new(
                term.coefficient() * factor,
                term.pauli_string().clone(),
            )?);
        }

        Self::new(self.num_qubits, terms)
    }

    /// 合并相同的 Pauli 字符串，并删除绝对值不大于 `tolerance` 的系数。
    ///
    /// # Errors
    ///
    /// 当 `tolerance` 不是有限非负数，或合并后的系数成为非有限数时返回错误。
    pub fn simplify(&self, tolerance: f64) -> ObservableResult<Self> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(InvalidToleranceError {
                value: tolerance.to_string(),
            });
        }

        let mut map = BTreeMap::<PauliString, f64>::new();

        for term in self.terms.iter() {
            let entry = map.entry(term.pauli_string().clone()).or_insert(0.0);
            *entry += term.coefficient();
        }

        let mut terms = Vec::new();

        for (pauli_string, coefficient) in map.into_iter() {
            if coefficient.abs() > tolerance {
                terms.push(PauliTerm::new(coefficient, pauli_string)?);
            }
        }

        Self::new(self.num_qubits, terms)
    }
}

/// 将 JSON 解析失败转换为统一的可观测量错误。
fn json_error(message: impl Into<String>) -> crate::ObservableError {
    JsonDeserializationError {
        message: message.into(),
    }
}

/// 从 JSON 字段读取非负整数索引或数量。
fn json_usize(value: Option<&serde_json::Value>, field: &str) -> ObservableResult<usize> {
    let value = value
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| json_error(format!("{field} 必须是非负整数")))?;
    usize::try_from(value).map_err(|_| json_error(format!("{field} 超出 usize 范围")))
}

/// 从 JSON 字段读取实数系数。
fn json_f64(value: Option<&serde_json::Value>, field: &str) -> ObservableResult<f64> {
    value
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| json_error(format!("{field} 必须是数值")))
}

/// 从 JSON 字段读取数组。
fn json_array<'a>(
    value: Option<&'a serde_json::Value>,
    field: &str,
) -> ObservableResult<&'a Vec<serde_json::Value>> {
    value
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| json_error(format!("{field} 必须是数组")))
}

/// 从 JSON 字段读取字符串。
fn json_string<'a>(value: Option<&'a serde_json::Value>, field: &str) -> ObservableResult<&'a str> {
    value
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| json_error(format!("{field} 必须是字符串")))
}
