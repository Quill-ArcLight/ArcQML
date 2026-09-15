use crate::{AnalysisError, AnalysisResult};
use arcqml_core::Tensor;
use arcqml_linalg::{abs, conj, mul, square, sum};
use arcqml_sim::StateVectorSimulator;

/// 返回两个纯态的可微保真度 F64 标量 Tensor。
///
/// 保真度按 `|⟨lhs|rhs⟩|²` 计算。为保持反向规则与前向表达式一致，结果不截断到 `[0, 1]`。
///
/// # Errors
///
/// 当两个状态的量子比特数不同、任一模拟器状态无效，或共轭、乘法、求和及取模
/// 等底层 Tensor 运算失败时返回错误。
pub fn fidelity(lhs: &StateVectorSimulator, rhs: &StateVectorSimulator) -> AnalysisResult<Tensor> {
    if lhs.num_qubits() != rhs.num_qubits() {
        return Err(AnalysisError::QubitCountMismatchError {
            lhs: lhs.num_qubits(),
            rhs: rhs.num_qubits(),
        });
    }

    let lhs_amplitudes = lhs.amplitudes()?;
    let rhs_amplitudes = rhs.amplitudes()?;
    let overlap = sum(&mul(&conj(&lhs_amplitudes)?, &rhs_amplitudes)?)?;
    Ok(square(&abs(&overlap)?)?)
}
