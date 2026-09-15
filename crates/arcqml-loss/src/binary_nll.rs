use crate::{
    LossError, LossResult,
    common::{scalar_like, tensor_operation_error, validate_real_tensor},
};
use arcqml_core::Tensor;
use arcqml_linalg::{add, clamp, log, mean, mul, neg, sub};

const PROBABILITY_EPSILON: f64 = 1e-12;

/// 计算 Pauli Z 期望值对应的二元负对数似然损失。
///
/// 每个预测值表示 `z = ⟨Z⟩`，类别概率分别为 `p(0) = (1 + z) / 2` 与
/// `p(1) = (1 - z) / 2`。
/// 输入必须是一维、非空的 `F32` 或 `F64` Tensor，长度与 `labels` 相同；标签只能是 0
/// 或 1。为避免对数奇点，期望值会在概率空间以 `1e-12` 的边界截断。返回批次平均损失。
///
/// # Errors
///
/// 当输入类型、维度、长度或标签非法，或底层 Tensor 运算失败时返回错误。
pub fn binary_nll_loss(prediction: &Tensor, labels: &[u8]) -> LossResult<Tensor> {
    validate_real_tensor(prediction)?;
    if prediction.ndim() != 1 || prediction.numel() != labels.len() {
        return Err(LossError::TensorOperationError {
            message: format!(
                "binary_nll_loss expects a rank-1 prediction with {} entries, got shape {:?}",
                labels.len(),
                prediction.shape()
            ),
        });
    }
    if labels.iter().any(|label| *label > 1) {
        return Err(LossError::TensorOperationError {
            message: "binary_nll_loss labels must be 0 or 1".to_string(),
        });
    }

    let clipped = clamp(
        prediction,
        -1.0 + 2.0 * PROBABILITY_EPSILON,
        1.0 - 2.0 * PROBABILITY_EPSILON,
    )
    .map_err(tensor_operation_error)?;
    let one = scalar_like(prediction, 1.0)?;
    let half = scalar_like(prediction, 0.5)?;
    let probability_zero = mul(&add(&one, &clipped).map_err(tensor_operation_error)?, &half)
        .map_err(tensor_operation_error)?;
    let probability_one = sub(&one, &probability_zero).map_err(tensor_operation_error)?;
    let labels_as_tensor = labels_tensor(prediction, labels)?;
    let inverse_labels = sub(&one, &labels_as_tensor).map_err(tensor_operation_error)?;
    let zero_loss = mul(
        &inverse_labels,
        &log(&probability_zero).map_err(tensor_operation_error)?,
    )
    .map_err(tensor_operation_error)?;
    let one_loss = mul(
        &labels_as_tensor,
        &log(&probability_one).map_err(tensor_operation_error)?,
    )
    .map_err(tensor_operation_error)?;
    let selected_log_probability = add(&zero_loss, &one_loss).map_err(tensor_operation_error)?;
    let average = mean(&selected_log_probability).map_err(tensor_operation_error)?;
    neg(&average).map_err(tensor_operation_error)
}

/// 按预测张量的数据类型创建标签常量张量。
fn labels_tensor(prediction: &Tensor, labels: &[u8]) -> LossResult<Tensor> {
    match prediction.dtype() {
        arcqml_core::DType::F32 => Tensor::new(
            labels
                .iter()
                .map(|label| f32::from(*label))
                .collect::<Vec<_>>(),
        ),
        arcqml_core::DType::F64 => Tensor::new(
            labels
                .iter()
                .map(|label| f64::from(*label))
                .collect::<Vec<_>>(),
        ),
        dtype => return Err(LossError::UnsupportedTensorDType { dtype }),
    }
    .map_err(|error| LossError::TensorCreateError {
        message: error.to_string(),
    })
}
