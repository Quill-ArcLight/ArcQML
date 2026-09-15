use crate::{ObservableError, ObservableResult, Pauli, SparsePauliOp};
use num_complex::Complex64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// 稀疏 Pauli Hamiltonian 的内部编译表示。
///
/// 该类型预先合并全部对角项，并将非对角 Pauli 串转换为位掩码，供状态矢量
/// 模拟器在训练循环中重复复用。它不在用户 prelude 中导出。
#[doc(hidden)]
#[derive(Debug)]
pub struct CompiledHamiltonian {
    dimension: usize,
    diagonal: Vec<f64>,
    off_diagonal_terms: Vec<CompiledPauliTerm>,
}

/// 单个非对角 Pauli 项的紧凑状态矢量执行元数据。
#[derive(Debug, Clone, Copy)]
struct CompiledPauliTerm {
    coefficient: f64,
    flip_mask: usize,
    phase_mask: usize,
    y_count_mod_four: u8,
}

impl CompiledHamiltonian {
    /// 从公开的 SparsePauliOp 构建一次性可复用的执行计划。
    pub(crate) fn compile(observable: &SparsePauliOp) -> ObservableResult<Self> {
        let num_qubits = observable.num_qubits();
        let dimension = state_dimension(num_qubits)?;
        let mut diagonal = vec![0.0; dimension];
        let mut off_diagonal_terms = Vec::new();

        for term in observable.terms() {
            let mut flip_mask = 0usize;
            let mut phase_mask = 0usize;
            let mut y_count_mod_four = 0u8;

            for operation in term.pauli_string().ops() {
                let mask = 1usize
                    .checked_shl(operation.qubit().index() as u32)
                    .ok_or(ObservableError::StateDimensionOverflowError { num_qubits })?;
                match operation.pauli() {
                    Pauli::I => {}
                    Pauli::X => flip_mask |= mask,
                    Pauli::Y => {
                        flip_mask |= mask;
                        phase_mask |= mask;
                        y_count_mod_four = (y_count_mod_four + 1) % 4;
                    }
                    Pauli::Z => phase_mask |= mask,
                }
            }

            let compiled = CompiledPauliTerm {
                coefficient: term.coefficient(),
                flip_mask,
                phase_mask,
                y_count_mod_four,
            };

            if compiled.flip_mask == 0 {
                for (index, value) in diagonal.iter_mut().enumerate() {
                    *value += compiled.coefficient * compiled.phase(index).re;
                }
            } else {
                off_diagonal_terms.push(compiled);
            }
        }

        Ok(Self {
            dimension,
            diagonal,
            off_diagonal_terms,
        })
    }

    /// 直接计算编译 Hamiltonian 的期望值，不分配 H|ψ⟩ 缓冲区。
    pub(crate) fn expectation(&self, amplitudes: &[Complex64]) -> ObservableResult<f64> {
        self.validate_state(amplitudes)?;
        let mut value = 0.0;

        for (index, amplitude) in amplitudes.iter().copied().enumerate() {
            value += self.diagonal[index] * amplitude.norm_sqr();
        }

        for term in &self.off_diagonal_terms {
            for (index, amplitude) in amplitudes.iter().copied().enumerate() {
                let mapped_index = index ^ term.flip_mask;
                value += (amplitudes[mapped_index].conj()
                    * term.phase(index)
                    * term.coefficient
                    * amplitude)
                    .re;
            }
        }

        Ok(value)
    }

    /// 计算 H|ψ⟩，供伴随反向传播和普通自动微分共享。
    pub(crate) fn apply(&self, amplitudes: &[Complex64]) -> ObservableResult<Vec<Complex64>> {
        self.validate_state(amplitudes)?;
        let mut output = amplitudes
            .iter()
            .zip(&self.diagonal)
            .map(|(amplitude, diagonal)| *amplitude * *diagonal)
            .collect::<Vec<_>>();

        #[cfg(feature = "parallel")]
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(target, value)| {
                for term in &self.off_diagonal_terms {
                    let source = target ^ term.flip_mask;
                    *value += term.coefficient * term.phase(source) * amplitudes[source];
                }
            });

        #[cfg(not(feature = "parallel"))]
        for (target, value) in output.iter_mut().enumerate() {
            for term in &self.off_diagonal_terms {
                let source = target ^ term.flip_mask;
                *value += term.coefficient * term.phase(source) * amplitudes[source];
            }
        }

        Ok(output)
    }

    /// 串行计算 H|ψ⟩，供已在外层按 batch 并行的调用方使用。
    pub(crate) fn apply_serial(
        &self,
        amplitudes: &[Complex64],
    ) -> ObservableResult<Vec<Complex64>> {
        self.validate_state(amplitudes)?;
        let mut output = amplitudes
            .iter()
            .zip(&self.diagonal)
            .map(|(amplitude, diagonal)| *amplitude * *diagonal)
            .collect::<Vec<_>>();

        for (target, value) in output.iter_mut().enumerate() {
            for term in &self.off_diagonal_terms {
                let source = target ^ term.flip_mask;
                *value += term.coefficient * term.phase(source) * amplitudes[source];
            }
        }

        Ok(output)
    }

    /// 一次计算 H|ψ⟩ 与 Re⟨ψ|H|ψ⟩，避免前反向之间重复扫描 Hamiltonian。
    pub(crate) fn expectation_and_apply(
        &self,
        amplitudes: &[Complex64],
    ) -> ObservableResult<(f64, Vec<Complex64>)> {
        let hamiltonian_state = self.apply(amplitudes)?;
        let value = amplitudes
            .iter()
            .zip(&hamiltonian_state)
            .map(|(amplitude, transformed)| (amplitude.conj() * *transformed).re)
            .sum();
        Ok((value, hamiltonian_state))
    }

    /// 串行计算 H|ψ⟩ 与 Re⟨ψ|H|ψ⟩，供已在外层按 batch 并行的调用方使用。
    pub(crate) fn expectation_and_apply_serial(
        &self,
        amplitudes: &[Complex64],
    ) -> ObservableResult<(f64, Vec<Complex64>)> {
        let hamiltonian_state = self.apply_serial(amplitudes)?;
        let value = amplitudes
            .iter()
            .zip(&hamiltonian_state)
            .map(|(amplitude, transformed)| (amplitude.conj() * *transformed).re)
            .sum();
        Ok((value, hamiltonian_state))
    }

    /// 验证单个状态矢量长度与编译 Hamiltonian 一致。
    fn validate_state(&self, amplitudes: &[Complex64]) -> ObservableResult<()> {
        if amplitudes.len() != self.dimension {
            return Err(ObservableError::StateVectorLengthMismatchError {
                expected: self.dimension,
                actual: amplitudes.len(),
            });
        }
        Ok(())
    }
}

impl CompiledPauliTerm {
    /// 返回 Pauli 项作用于给定计算基索引时的复相位。
    fn phase(&self, index: usize) -> Complex64 {
        let base = match self.y_count_mod_four {
            0 => Complex64::new(1.0, 0.0),
            1 => Complex64::new(0.0, 1.0),
            2 => Complex64::new(-1.0, 0.0),
            _ => Complex64::new(0.0, -1.0),
        };
        if (index & self.phase_mask).count_ones().is_multiple_of(2) {
            base
        } else {
            -base
        }
    }
}

/// 计算给定量子比特数对应的状态矢量维度。
fn state_dimension(num_qubits: usize) -> ObservableResult<usize> {
    1usize
        .checked_shl(num_qubits as u32)
        .ok_or(ObservableError::StateDimensionOverflowError { num_qubits })
}
