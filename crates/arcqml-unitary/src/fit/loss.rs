use crate::{DenseUnitary, UnitaryResult};
use num_complex::Complex64;

/// 计算目标矩阵与内部列批次矩阵的迹重叠 `Tr(U_target† U_current)`。
pub(crate) fn trace_overlap(
    target: &DenseUnitary,
    columns: &[Complex64],
) -> UnitaryResult<Complex64> {
    let dimension = target.dimension();
    let data = target.as_row_major();
    let mut overlap = Complex64::new(0.0, 0.0);
    for input in 0..dimension {
        for output in 0..dimension {
            overlap +=
                data[output * dimension + input].conj() * columns[input * dimension + output];
        }
    }
    Ok(overlap)
}

/// 根据迹重叠计算截断到 `[0, 1]` 的保真度及损失 `1 - fidelity`。
pub(crate) fn fidelity_and_loss(overlap: Complex64, dimension: usize) -> (f64, f64) {
    let scale = (dimension as f64).powi(2);
    let fidelity = (overlap.norm_sqr() / scale).clamp(0.0, 1.0);
    (fidelity, 1.0 - fidelity)
}
