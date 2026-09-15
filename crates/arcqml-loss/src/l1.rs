use crate::{
    LossResult,
    common::{tensor_operation_error, validate_real_tensor},
};
use arcqml_core::Tensor;
use arcqml_linalg::{abs, mean, sub};

/// 计算平均绝对误差 `mean(abs(prediction - target))`。
///
/// `prediction` 与 `target` 必须是非空的同类型 `F32` 或 `F64` Tensor；两者形状可以按
/// NumPy 广播规则兼容。返回保留自动微分图的同 dtype 标量 Tensor。
///
/// # Errors
///
/// 当输入为空、数据类型不受支持、形状不能广播或底层 Tensor 运算失败时返回错误。
pub fn l1_loss(prediction: &Tensor, target: &Tensor) -> LossResult<Tensor> {
    validate_real_tensor(prediction)?;
    validate_real_tensor(target)?;

    let error = sub(prediction, target).map_err(tensor_operation_error)?;
    let absolute = abs(&error).map_err(tensor_operation_error)?;

    mean(&absolute).map_err(tensor_operation_error)
}
