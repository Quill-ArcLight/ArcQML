use crate::{DenseUnitary, UnitaryError, UnitaryResult};
use num_complex::Complex64;

/// 完整验证一个稠密矩阵是否为酉矩阵。
pub(crate) fn validate_unitary(unitary: &DenseUnitary, tolerance: f64) -> UnitaryResult<()> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(UnitaryError::InvalidToleranceError { tolerance });
    }

    let dimension = unitary.dimension();
    let data = unitary.as_row_major();
    let mut max_deviation = 0.0_f64;
    for left in 0..dimension {
        for right in 0..dimension {
            let value = column_inner_product(data, dimension, left, right);
            let expected = if left == right {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            max_deviation = max_deviation.max((value - expected).norm());
        }
    }

    if max_deviation > tolerance {
        return Err(UnitaryError::NonUnitaryMatrixError {
            max_deviation,
            tolerance,
        });
    }
    Ok(())
}

/// 计算行主序矩阵中两列的复内积。
fn column_inner_product(
    values: &[Complex64],
    dimension: usize,
    left: usize,
    right: usize,
) -> Complex64 {
    (0..dimension)
        .map(|row| values[row * dimension + left].conj() * values[row * dimension + right])
        .sum()
}
