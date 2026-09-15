use crate::error::{LinalgError::TensorCreateError, LinalgResult};
use crate::ops::ensure_supported_tensor;
use crate::shape::validate_transpose_shape;
use arcqml_core::Tensor;

/// 返回共享存储的二维 Tensor 转置视图。
///
/// # Errors
///
/// 当输入不是二维 Tensor 或无法创建转置视图时返回错误。
pub fn transpose(input: &Tensor) -> LinalgResult<Tensor> {
    ensure_supported_tensor("transpose", input)?;
    validate_transpose_shape(input.shape())?;

    // arcqml-core 的 transpose 复用 view 的反向散射规则。
    input.transpose().map_err(|_| TensorCreateError {
        op: "transpose",
        message: "arcqml-core could not create the transposed tensor".to_string(),
    })
}
