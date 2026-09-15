use crate::{
    LossResult,
    common::{scalar_like, tensor_operation_error, validate_real_tensor},
};
use arcqml_core::Tensor;
use arcqml_linalg::{mean, mul, square, sub};

/// 计算半均方误差 `0.5 × mean((prediction - target)²)`。
///
/// `prediction` 与 `target` 必须是非空的同类型 `F32` 或 `F64` Tensor；两者形状可以按
/// NumPy 广播规则兼容。返回保留自动微分图的同 dtype 标量 Tensor。
///
/// # Errors
///
/// 当输入为空、数据类型不受支持、形状不能广播或底层 Tensor 运算失败时返回错误。
pub fn mse_loss(prediction: &Tensor, target: &Tensor) -> LossResult<Tensor> {
    validate_real_tensor(prediction)?;
    validate_real_tensor(target)?;

    let error = sub(prediction, target).map_err(tensor_operation_error)?;
    let squared = square(&error).map_err(tensor_operation_error)?;
    let average = mean(&squared).map_err(tensor_operation_error)?;
    let half = scalar_like(prediction, 0.5)?;

    mul(&average, &half).map_err(tensor_operation_error)
}
