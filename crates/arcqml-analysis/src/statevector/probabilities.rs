use crate::AnalysisResult;
use arcqml_core::Tensor;
use arcqml_linalg::{abs, square};
use arcqml_sim::StateVectorSimulator;

/// 返回当前量子态在计算基上的可微测量概率 Tensor。
///
/// 输出为形状 `[2^n]` 的 `F64` Tensor，第 `index` 项为振幅 `ψ[index]` 的模平方；
/// `q0` 对应 `index` 的最低有效位。
///
/// # Errors
///
/// 当模拟器状态无效，或复数取模与平方的底层 Tensor 运算失败时返回错误。
pub fn probabilities(simulator: &StateVectorSimulator) -> AnalysisResult<Tensor> {
    let amplitudes = simulator.amplitudes()?;
    Ok(square(&abs(&amplitudes)?)?)
}
