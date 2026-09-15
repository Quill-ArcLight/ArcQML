use crate::error::{LinalgError::*, LinalgResult};
use crate::ops::ensure_supported_tensor;
use crate::shape::validate_reshape_shape;
use arcqml_core::Tensor;

/// 按行主序逻辑元素顺序改变 Tensor 形状。
///
/// 连续输入共享存储；非连续输入会先复制成连续 Tensor。
///
/// # Errors
///
/// 当目标形状计算溢出、变形前后元素数不同，或非连续输入无法连续化时返回错误。
pub fn reshape(input: &Tensor, shape: Vec<usize>) -> LinalgResult<Tensor> {
    ensure_supported_tensor("reshape", input)?;

    if !input.is_contiguous() {
        return Err(NonContiguousTensorError { op: "reshape" });
    }

    validate_reshape_shape(input.shape(), &shape)?;

    // arcqml-core 的 reshape 已处理连续 view 与非连续输入的连续化反向传播。
    input.reshape(shape).map_err(|_| TensorCreateError {
        op: "reshape",
        message: "arcqml-core could not create the reshaped tensor".to_string(),
    })
}
