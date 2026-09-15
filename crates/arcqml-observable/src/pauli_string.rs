use crate::{ObservableError::*, ObservableResult, Pauli};
use arcqml_circuit::Qubit;
use num_complex::Complex64;
use std::collections::BTreeMap;

/// 指定量子比特上的一个 Pauli 算符。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PauliOp {
    qubit: Qubit,
    pauli: Pauli,
}

impl PauliOp {
    /// 创建指定量子比特上的 Pauli 操作；量子比特范围由所属 [`PauliString`] 校验。
    pub fn new(qubit: impl Into<Qubit>, pauli: Pauli) -> Self {
        Self {
            qubit: qubit.into(),
            pauli,
        }
    }

    /// 返回目标量子比特。
    pub fn qubit(&self) -> Qubit {
        self.qubit
    }

    /// 返回该位置的 Pauli 算符。
    pub fn pauli(&self) -> Pauli {
        self.pauli
    }
}

/// 固定量子比特数上的 Pauli 张量积。
///
/// 内部只保存非恒等项，并按量子比特下标升序排列；未保存的位置隐含为 [`Pauli::I`]。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PauliString {
    num_qubits: usize,
    ops: Vec<PauliOp>,
}

impl PauliString {
    /// 创建并规范化 Pauli 字符串。
    ///
    /// 输入中的恒等项会被删除，其余项按量子比特下标排序。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零、非恒等项的量子比特越界，或同一量子比特出现多个
    /// 非恒等项时返回错误。
    pub fn new(num_qubits: usize, ops: Vec<PauliOp>) -> ObservableResult<Self> {
        if num_qubits == 0 {
            return Err(EmptyQubitError);
        }

        let mut ops = ops
            .into_iter()
            .filter(|op| !op.pauli().is_identity())
            .collect::<Vec<_>>(); // 过滤所有 I 项

        ops.sort_by_key(|op| op.qubit().index()); // 按 qubit 从小到大排列

        let mut last_index = None;

        for op in ops.iter() {
            let index = op.qubit().index();

            if index >= num_qubits {
                return Err(QubitOutOfRangeError { index, num_qubits });
            }

            if last_index == Some(index) {
                return Err(DuplicateQubitError { index });
            }

            last_index = Some(index);
        }

        Ok(Self { num_qubits, ops })
    }

    /// 创建给定量子比特数上的恒等 Pauli 字符串。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零时返回错误。
    pub fn identity(num_qubits: usize) -> ObservableResult<Self> {
        Self::new(num_qubits, Vec::new())
    }

    /// 创建仅在指定量子比特上取 `pauli` 的 Pauli 字符串。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零，或非恒等 `pauli` 的量子比特越界时返回错误。
    pub fn single(
        num_qubits: usize,
        qubit: impl Into<Qubit>,
        pauli: Pauli,
    ) -> ObservableResult<Self> {
        Self::new(num_qubits, vec![PauliOp::new(qubit, pauli)])
    }

    /// 创建 `X` 作用于指定量子比特、其他位置为恒等的 Pauli 字符串。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零或 `qubit` 越界时返回错误。
    pub fn x(num_qubits: usize, qubit: impl Into<Qubit>) -> ObservableResult<Self> {
        Self::single(num_qubits, qubit, Pauli::X)
    }

    /// 创建 `Y` 作用于指定量子比特、其他位置为恒等的 Pauli 字符串。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零或 `qubit` 越界时返回错误。
    pub fn y(num_qubits: usize, qubit: impl Into<Qubit>) -> ObservableResult<Self> {
        Self::single(num_qubits, qubit, Pauli::Y)
    }

    /// 创建 `Z` 作用于指定量子比特、其他位置为恒等的 Pauli 字符串。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零或 `qubit` 越界时返回错误。
    pub fn z(num_qubits: usize, qubit: impl Into<Qubit>) -> ObservableResult<Self> {
        Self::single(num_qubits, qubit, Pauli::Z)
    }

    /// 返回张量积包含的量子比特数量，包括隐含的恒等位置。
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// 返回显式保存的非恒等 Pauli 操作数量。
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// 判断是否不含非 I Pauli 操作。
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// 判断是否为单位 PauliString。
    pub fn is_identity(&self) -> bool {
        self.is_empty()
    }

    /// 按量子比特下标升序返回显式保存的非恒等操作。
    pub fn ops(&self) -> &[PauliOp] {
        &self.ops
    }

    /// 返回指定量子比特上的 Pauli；未显式保存的位置返回 [`Pauli::I`]。
    ///
    /// # Errors
    ///
    /// 当 `qubit` 超出当前字符串的量子比特范围时返回错误。
    pub fn pauli_on(&self, qubit: impl Into<Qubit>) -> ObservableResult<Pauli> {
        let qubit = qubit.into();
        let index = qubit.index();

        if index >= self.num_qubits {
            return Err(QubitOutOfRangeError {
                index,
                num_qubits: self.num_qubits,
            });
        }

        Ok(self
            .ops
            .iter()
            .find(|op| op.qubit().index() == index)
            .map(|op| op.pauli())
            .unwrap_or(Pauli::I))
    }

    /// 判断两个 Pauli 字符串是否对易。
    ///
    /// 在同一位置均非恒等且 Pauli 不同的量子比特数为偶数时，两者对易。
    ///
    /// # Errors
    ///
    /// 当两个字符串的量子比特数不一致时返回错误。
    pub fn commutes_with(&self, rhs: &PauliString) -> ObservableResult<bool> {
        self.ensure_same_num_qubits(rhs)?;

        let mut anti_commuting_count = 0usize;

        for qubit in 0..self.num_qubits {
            let lhs_pauli = self.pauli_on(qubit)?;
            let rhs_pauli = rhs.pauli_on(qubit)?;

            if !lhs_pauli.is_identity() && !rhs_pauli.is_identity() && lhs_pauli != rhs_pauli {
                anti_commuting_count += 1;
            }
        }

        Ok(anti_commuting_count.is_multiple_of(2))
    }

    /// 按 `self × rhs` 的顺序计算乘积，返回 `(复相位, Pauli 字符串)`。
    ///
    /// # Errors
    ///
    /// 当两个字符串的量子比特数不一致时返回错误。
    pub fn multiply(&self, rhs: &PauliString) -> ObservableResult<(Complex64, PauliString)> {
        self.ensure_same_num_qubits(rhs)?;

        let mut phase = Complex64::new(1.0, 0.0);
        let mut map = BTreeMap::new();

        for op in self.ops.iter() {
            map.insert(op.qubit().index(), op.pauli());
        }

        for op in rhs.ops.iter() {
            let index = op.qubit().index();
            let lhs_pauli = map.remove(&index).unwrap_or(Pauli::I);
            let (local_phase, pauli) = lhs_pauli.multiply(op.pauli());

            phase *= local_phase;

            if !pauli.is_identity() {
                map.insert(index, pauli);
            }
        }

        let ops = map
            .into_iter()
            .map(|(index, pauli)| PauliOp::new(index, pauli))
            .collect();

        Ok((phase, PauliString::new(self.num_qubits, ops)?))
    }

    fn ensure_same_num_qubits(&self, rhs: &PauliString) -> ObservableResult<()> {
        if self.num_qubits != rhs.num_qubits {
            return Err(QubitCountMismatchError {
                expected: self.num_qubits,
                actual: rhs.num_qubits,
            });
        }

        Ok(())
    }
}
