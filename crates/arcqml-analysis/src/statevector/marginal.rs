use crate::{AnalysisError, AnalysisResult, probabilities};
use arcqml_core::Tensor;
use arcqml_linalg::segment_sum;
use arcqml_sim::StateVectorSimulator;
use std::collections::BTreeSet;

/// 返回指定量子比特集合的可微边缘测量概率 Tensor。
///
/// 输出为形状 `[2^k]` 的 `F64` Tensor。量子比特会按编号从大到小排列，输出下标对应该顺序下的
/// 二进制结果；例如选择 `[0, 2]` 时，输出依次表示 `q2q0 = 00, 01, 10, 11`。
///
/// # Errors
///
/// 当 `qubits` 为空、含重复或越界编号，模拟器状态无效，或概率与分段归约失败时返回错误。
pub fn marginal_probabilities(
    simulator: &StateVectorSimulator,
    qubits: &[usize],
) -> AnalysisResult<Tensor> {
    let display_qubits = validate_qubits(simulator, qubits)?;
    let full_probabilities = probabilities(simulator)?;
    let num_segments = 1usize << display_qubits.len();
    let segment_ids = (0..full_probabilities.numel())
        .map(|basis_index| marginal_index(basis_index, &display_qubits))
        .collect::<Vec<_>>();
    Ok(segment_sum(
        &full_probabilities,
        &segment_ids,
        num_segments,
    )?)
}

/// 校验量子位选择并返回用于显示和输出排序的降序量子位编号。
fn validate_qubits(
    simulator: &StateVectorSimulator,
    qubits: &[usize],
) -> AnalysisResult<Vec<usize>> {
    if qubits.is_empty() {
        return Err(AnalysisError::EmptyQubitSelectionError);
    }

    let mut seen = BTreeSet::new();
    for &qubit in qubits {
        if qubit >= simulator.num_qubits() {
            return Err(AnalysisError::QubitOutOfRangeError {
                index: qubit,
                num_qubits: simulator.num_qubits(),
            });
        }
        if !seen.insert(qubit) {
            return Err(AnalysisError::DuplicateQubitError { index: qubit });
        }
    }

    let mut display_qubits = qubits.to_vec();
    display_qubits.sort_unstable_by(|lhs, rhs| rhs.cmp(lhs));
    Ok(display_qubits)
}

/// 将完整计算基索引映射为边缘概率 Tensor 的输出下标。
fn marginal_index(basis_index: usize, display_qubits: &[usize]) -> usize {
    display_qubits.iter().fold(0, |index, qubit| {
        (index << 1) | ((basis_index >> qubit) & 1)
    })
}
