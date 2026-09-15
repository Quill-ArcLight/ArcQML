use crate::{LossError, LossResult};
use arcqml_core::{DType, Tensor};

/// 验证输入是非空的实数张量。
pub(crate) fn validate_real_tensor(tensor: &Tensor) -> LossResult<()> {
    if tensor.numel() == 0 {
        return Err(LossError::EmptyTensorError);
    }

    match tensor.dtype() {
        DType::F32 | DType::F64 => Ok(()),
        dtype => Err(LossError::UnsupportedTensorDType { dtype }),
    }
}

/// 按参考张量的数据类型创建标量张量。
pub(crate) fn scalar_like(tensor: &Tensor, value: f64) -> LossResult<Tensor> {
    match tensor.dtype() {
        DType::F32 => Tensor::new(value as f32),
        DType::F64 => Tensor::new(value),
        dtype => return Err(LossError::UnsupportedTensorDType { dtype }),
    }
    .map_err(tensor_create_error)
}

/// 将底层张量运算错误转换为损失模块错误。
pub(crate) fn tensor_operation_error(error: impl ToString) -> LossError {
    LossError::TensorOperationError {
        message: error.to_string(),
    }
}

/// 将底层张量创建错误转换为损失模块错误。
pub(crate) fn tensor_create_error(error: impl ToString) -> LossError {
    LossError::TensorCreateError {
        message: error.to_string(),
    }
}
