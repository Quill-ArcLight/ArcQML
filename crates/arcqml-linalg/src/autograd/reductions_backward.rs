use crate::error::LinalgResult;
use crate::shape::infer_reduction_shape;

/// 推断归约 backward 需要还原到的输入形状。
///
/// # Errors
///
/// 当上游梯度或原始操作数的形状与对应反向规则不兼容时返回错误。
pub fn reduction_backward_shape(
    input_shape: &[usize],
    axis: Option<usize>,
    keepdim: bool,
) -> LinalgResult<Vec<usize>> {
    infer_reduction_shape(input_shape, axis, keepdim)?;

    Ok(input_shape.to_vec())
}
