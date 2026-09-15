use crate::error::LinalgResult;
use crate::shape::validate::{
    validate_axis, validate_broadcast_shapes, validate_matmul_shapes, validate_transpose_shape,
};

/// 推断二维矩阵乘法输出形状。
///
/// # Errors
///
/// 当任一形状不是二维，或两个矩阵的内维不相等时返回错误。
pub fn infer_matmul_shape(lhs: &[usize], rhs: &[usize]) -> LinalgResult<Vec<usize>> {
    validate_matmul_shapes(lhs, rhs)?;

    Ok(vec![lhs[0], rhs[1]])
}

/// 推断二维转置输出形状。
///
/// # Errors
///
/// 当 `shape` 不是二维时返回错误。
pub fn infer_transpose_shape(shape: &[usize]) -> LinalgResult<Vec<usize>> {
    validate_transpose_shape(shape)?;

    Ok(vec![shape[1], shape[0]])
}

/// 推断全量或单轴归约的输出形状。
///
/// `axis` 为 `None` 时归约全部维度；`keepdim` 为 `true` 时被归约维度保留为长度 `1`。
///
/// # Errors
///
/// 当显式 `axis` 超出 `shape` 的秩范围时返回错误。
pub fn infer_reduction_shape(
    shape: &[usize],
    axis: Option<usize>,
    keepdim: bool,
) -> LinalgResult<Vec<usize>> {
    match axis {
        None => {
            if keepdim {
                Ok(vec![1; shape.len()])
            } else {
                Ok(vec![])
            }
        }

        Some(axis) => {
            validate_axis("reduction", shape, axis)?;

            let mut out = shape.to_vec();

            if keepdim {
                out[axis] = 1;
            } else {
                out.remove(axis);
            }

            Ok(out)
        }
    }
}

/// 按 NumPy 尾轴对齐规则推断广播后的输出形状。
///
/// # Errors
///
/// 当某一对齐维度既不相等、也都不为 `1` 时返回错误。
pub fn infer_broadcast_shape(lhs: &[usize], rhs: &[usize]) -> LinalgResult<Vec<usize>> {
    validate_broadcast_shapes(lhs, rhs)?;

    let max_rank = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(max_rank);

    for i in 0..max_rank {
        let lhs_dim = lhs.iter().rev().nth(i).copied().unwrap_or(1);
        let rhs_dim = rhs.iter().rev().nth(i).copied().unwrap_or(1);
        out.push(lhs_dim.max(rhs_dim));
    }

    out.reverse();
    Ok(out)
}
