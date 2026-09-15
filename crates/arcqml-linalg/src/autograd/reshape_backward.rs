use crate::error::LinalgResult;
use crate::shape::validate_reshape_shape;

/// 推断 reshape backward 需要还原到的输入形状。
///
/// # Errors
///
/// 当上游梯度或原始操作数的形状与对应反向规则不兼容时返回错误。
pub fn reshape_backward_shape(
    input_shape: &[usize],
    output_shape: &[usize],
) -> LinalgResult<Vec<usize>> {
    validate_reshape_shape(input_shape, output_shape)?;

    Ok(input_shape.to_vec())
}
