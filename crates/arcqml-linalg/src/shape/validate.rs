use crate::error::{LinalgError::*, LinalgResult};

/// 返回形状各维度的乘积。
///
/// 此底层辅助函数不显式检查 `usize` 溢出；调用方必须保证乘积可表示。
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// 校验二维矩阵乘法形状：`[m, k] @ [k, n] -> [m, n]`。
///
/// # Errors
///
/// 当任一形状不是二维，或两个内维 `k` 不相等时返回错误。
pub fn validate_matmul_shapes(lhs: &[usize], rhs: &[usize]) -> LinalgResult<()> {
    if lhs.len() != 2 {
        return Err(InvalidDimensionError {
            op: "matrix multiplication",
            expected: "the left operand to have rank 2".to_string(),
            actual: lhs.len(),
        });
    }

    if rhs.len() != 2 {
        return Err(InvalidDimensionError {
            op: "matrix multiplication",
            expected: "the right operand to have rank 2".to_string(),
            actual: rhs.len(),
        });
    }

    let lhs_k = lhs[1];
    let rhs_k = rhs[0];

    if lhs_k != rhs_k {
        return Err(ShapeMismatchError {
            op: "matrix multiplication",
            expected: format!(
                "matrix shapes [m, k] and [k, n] with matching inner dimensions, got left k={lhs_k}"
            ),
            actual: format!("lhs={lhs:?}, rhs={rhs:?}, right k={rhs_k}"),
        });
    }

    Ok(())
}

/// 校验形状是否恰好为二维。
///
/// # Errors
///
/// 当 `shape` 的秩不是二时返回错误。
pub fn validate_transpose_shape(shape: &[usize]) -> LinalgResult<()> {
    if shape.len() != 2 {
        return Err(InvalidDimensionError {
            op: "transpose",
            expected: "the input to have rank 2".to_string(),
            actual: shape.len(),
        });
    }

    Ok(())
}

/// 校验 `axis` 是否位于 `shape` 的秩范围内。
///
/// # Errors
///
/// 当 `axis >= shape.len()` 时返回错误。
pub fn validate_axis(op: &'static str, shape: &[usize], axis: usize) -> LinalgResult<()> {
    if axis >= shape.len() {
        return Err(InvalidAxisError {
            op,
            axis,
            rank: shape.len(),
        });
    }

    Ok(())
}

/// 校验两个形状的维度乘积是否相同。
///
/// # Errors
///
/// 当两个形状的元素数量不同返回错误。此函数不显式检查维度乘积的 `usize` 溢出。
pub fn validate_same_numel(from_shape: &[usize], to_shape: &[usize]) -> LinalgResult<()> {
    let from_numel = numel(from_shape);
    let to_numel = numel(to_shape);

    if from_numel != to_numel {
        return Err(InvalidReshapeError {
            from: from_shape.to_vec(),
            to: to_shape.to_vec(),
            from_numel,
            to_numel,
        });
    }

    Ok(())
}

/// 校验变形前后的元素数量是否相同。
///
/// # Errors
///
/// 当两个形状的元素数量不同返回错误。此函数不显式检查维度乘积的 `usize` 溢出。
pub fn validate_reshape_shape(from_shape: &[usize], to_shape: &[usize]) -> LinalgResult<()> {
    validate_same_numel(from_shape, to_shape)
}

/// 按 NumPy 尾轴对齐规则校验两个形状是否可广播。
///
/// # Errors
///
/// 当某一对齐维度既不相等、也都不为 `1` 时返回错误。
pub fn validate_broadcast_shapes(lhs: &[usize], rhs: &[usize]) -> LinalgResult<()> {
    let max_rank = lhs.len().max(rhs.len());

    for i in 0..max_rank {
        let lhs_dim = lhs.iter().rev().nth(i).copied().unwrap_or(1);
        let rhs_dim = rhs.iter().rev().nth(i).copied().unwrap_or(1);

        if lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 {
            return Err(ShapeMismatchError {
                op: "broadcast",
                expected: "each right-aligned dimension to be equal or for one dimension to be 1"
                    .to_string(),
                actual: format!("lhs={lhs:?}, rhs={rhs:?}"),
            });
        }
    }

    Ok(())
}

/// 校验缓冲区长度是否等于形状的维度乘积。
///
/// # Errors
///
/// 当 `actual_len` 与形状元素数不相等时返回错误。此函数不显式检查维度乘积溢出。
pub fn validate_buffer_len(
    op: &'static str,
    shape: &[usize],
    actual_len: usize,
) -> LinalgResult<()> {
    let expected = numel(shape);

    if expected != actual_len {
        return Err(InvalidBufferLengthError {
            op,
            expected,
            actual: actual_len,
        });
    }

    Ok(())
}
