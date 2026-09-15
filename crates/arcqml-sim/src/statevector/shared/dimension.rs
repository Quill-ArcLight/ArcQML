use crate::{SimError::*, SimResult};
use arcqml_core::checked_power_of_two;

/// 返回 n-qubit 状态向量的维度。
pub(crate) fn dimension(num_qubits: usize) -> SimResult<usize> {
    if num_qubits == 0 {
        return Err(EmptyQubitError);
    }

    checked_power_of_two(num_qubits).ok_or(StateDimensionOverflowError { num_qubits })
}
