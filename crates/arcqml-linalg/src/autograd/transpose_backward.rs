use crate::error::LinalgResult;
use crate::shape::validate_transpose_shape;

/// 推断二维 transpose backward 的输出形状。
///
/// # Errors
///
/// 当上游梯度或原始操作数的形状与对应反向规则不兼容时返回错误。
pub fn transpose_backward_shape(input_shape: &[usize]) -> LinalgResult<Vec<usize>> {
    validate_transpose_shape(input_shape)?;

    Ok(vec![input_shape[0], input_shape[1]])
}
