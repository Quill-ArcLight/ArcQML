use crate::error::LinalgResult;
use crate::shape::validate_matmul_shapes;

/// 推断矩阵乘法 backward 中两个输入梯度的形状。
///
/// # Errors
///
/// 当上游梯度或原始操作数的形状与对应反向规则不兼容时返回错误。
pub fn matmul_backward_shapes(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
) -> LinalgResult<(Vec<usize>, Vec<usize>)> {
    validate_matmul_shapes(lhs_shape, rhs_shape)?;

    Ok((lhs_shape.to_vec(), rhs_shape.to_vec()))
}
