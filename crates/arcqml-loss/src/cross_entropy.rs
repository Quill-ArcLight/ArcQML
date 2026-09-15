use crate::{
    LossError, LossResult,
    common::{scalar_like, tensor_create_error, tensor_operation_error, validate_real_tensor},
};
use arcqml_core::{ArcQmlError, BackwardFn, DType, Tensor, TensorData, no_grad};
use arcqml_linalg::{
    abs, add, clamp, div, exp, log, log_softmax, mean, mul, neg, sigmoid, sub, sum,
};
use std::sync::Arc;

/// 计算类别索引标签对应的多分类交叉熵损失。
///
/// `logits` 的形状必须是 `[批大小, 类别数]`，`labels` 的每个元素是对应样本的类别索引。
/// 内部使用数值稳定的 `log_softmax`，并对批次维度取平均。
///
/// # Errors
///
/// 当 logits 为空、不是 `F32`/`F64` 二维 Tensor、批大小与标签数不同、类别数为零、
/// 标签越界或底层 Tensor 运算失败时返回错误。
pub fn cross_entropy_loss(logits: &Tensor, labels: &[usize]) -> LossResult<Tensor> {
    validate_real_tensor(logits)?;
    validate_logits_and_labels(logits, labels)?;

    let batch_size = logits.shape()[0];
    let log_probabilities = log_softmax(logits, 1).map_err(tensor_operation_error)?;
    let targets = one_hot_targets(logits, labels)?;
    let selected_log_probabilities =
        mul(&log_probabilities, &targets).map_err(tensor_operation_error)?;
    let total = sum(&selected_log_probabilities).map_err(tensor_operation_error)?;
    let negative_total = neg(&total).map_err(tensor_operation_error)?;
    let batch_size_tensor = scalar_like(logits, batch_size as f64)?;

    div(&negative_total, &batch_size_tensor).map_err(tensor_operation_error)
}

/// 计算二元交叉熵的 logits 形式，并对全部元素取平均。
///
/// `targets` 必须与 `logits` 形状、数据类型一致，且每个目标值在闭区间 `[0, 1]` 内。
/// 公式使用 `max(x, 0) + log(1 + exp(-abs(x))) - y * x`，避免直接计算
/// `log(1 + exp(x))` 在大正 logits 上溢出。
/// 反向使用整体解析导数 `(sigmoid(x) - y) / N`，在零 logits 处同样成立；
/// 若 targets 需要梯度，其导数为 `-x / N`，其中 `N` 为元素总数。
///
/// # Errors
///
/// 当输入为空、形状或数据类型不一致、目标值不在闭区间 `[0, 1]`，或底层 Tensor
/// 运算失败时返回错误。
pub fn binary_cross_entropy_with_logits_loss(
    logits: &Tensor,
    targets: &Tensor,
) -> LossResult<Tensor> {
    validate_real_tensor(logits)?;
    validate_real_tensor(targets)?;
    validate_binary_logit_targets(logits, targets)?;

    let output = {
        // 不记录 clamp/abs 的局部反向，避免其零点约定改变 BCE 的整体导数。
        let _guard = no_grad();
        binary_cross_entropy_with_logits_forward(logits, targets)?
    };
    Tensor::from_operation_named(
        "binary_cross_entropy_with_logits",
        output.storage().clone(),
        output.meta().clone(),
        vec![logits.clone(), targets.clone()],
        Arc::new(BinaryCrossEntropyWithLogitsBackward),
    )
    .map_err(tensor_create_error)
}

/// 在禁用梯度记录的作用域内计算数值稳定的 BCE 前向值。
fn binary_cross_entropy_with_logits_forward(
    logits: &Tensor,
    targets: &Tensor,
) -> LossResult<Tensor> {
    let one = scalar_like(logits, 1.0)?;
    let positive_part = clamp(logits, 0.0, f64::INFINITY).map_err(tensor_operation_error)?;
    let absolute_logits = abs(logits).map_err(tensor_operation_error)?;
    let negative_absolute_logits = neg(&absolute_logits).map_err(tensor_operation_error)?;
    let correction = log(&add(
        &one,
        &exp(&negative_absolute_logits).map_err(tensor_operation_error)?,
    )
    .map_err(tensor_operation_error)?)
    .map_err(tensor_operation_error)?;
    let unscaled_loss = sub(
        &add(&positive_part, &correction).map_err(tensor_operation_error)?,
        &mul(targets, logits).map_err(tensor_operation_error)?,
    )
    .map_err(tensor_operation_error)?;

    mean(&unscaled_loss).map_err(tensor_operation_error)
}

/// 均值归约 BCE-with-logits 的整体反向规则。
#[derive(Debug)]
struct BinaryCrossEntropyWithLogitsBackward;

impl BackwardFn for BinaryCrossEntropyWithLogitsBackward {
    fn backward(
        &self,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> arcqml_core::Result<Vec<Option<Tensor>>> {
        let [logits, targets] = parents else {
            return Err(ArcQmlError::AutogradError(
                "BCE-with-logits backward expects logits and targets".to_string(),
            ));
        };
        let _guard = no_grad();
        let backward_error = |error: arcqml_linalg::LinalgError| {
            ArcQmlError::AutogradError(format!("BCE-with-logits backward: {error}"))
        };
        let count = scalar_like(logits, logits.numel() as f64)
            .map_err(|error| ArcQmlError::AutogradError(error.to_string()))?;
        let scale = div(grad_output, &count).map_err(backward_error)?;

        let logits_gradient = if logits.requires_grad() {
            let probability = sigmoid(logits).map_err(backward_error)?;
            let error = sub(&probability, targets).map_err(backward_error)?;
            Some(mul(&error, &scale).map_err(backward_error)?)
        } else {
            None
        };
        let targets_gradient = if targets.requires_grad() {
            let negative_logits = neg(logits).map_err(backward_error)?;
            Some(mul(&negative_logits, &scale).map_err(backward_error)?)
        } else {
            None
        };

        Ok(vec![logits_gradient, targets_gradient])
    }
}

/// 校验 logits 的二维批量形状与类别标签是否合法。
fn validate_logits_and_labels(logits: &Tensor, labels: &[usize]) -> LossResult<()> {
    if logits.ndim() != 2 {
        return Err(LossError::TensorOperationError {
            message: format!(
                "cross_entropy_loss 要求 logits 为二维 [批大小, 类别数]，实际形状为 {:?}",
                logits.shape()
            ),
        });
    }

    let batch_size = logits.shape()[0];
    let class_count = logits.shape()[1];
    if batch_size == 0 || class_count == 0 {
        return Err(LossError::EmptyTensorError);
    }
    if labels.len() != batch_size {
        return Err(LossError::TensorOperationError {
            message: format!(
                "cross_entropy_loss 需要 {batch_size} 个标签，实际得到 {} 个",
                labels.len()
            ),
        });
    }
    if let Some((index, label)) = labels
        .iter()
        .copied()
        .enumerate()
        .find(|(_, label)| *label >= class_count)
    {
        return Err(LossError::TensorOperationError {
            message: format!(
                "cross_entropy_loss 的第 {index} 个标签为 {label}，但类别索引必须小于 {class_count}"
            ),
        });
    }

    Ok(())
}

/// 校验二元交叉熵输入的形状、数据类型与目标值范围。
fn validate_binary_logit_targets(logits: &Tensor, targets: &Tensor) -> LossResult<()> {
    if logits.dtype() != targets.dtype() {
        return Err(LossError::TensorOperationError {
            message: format!(
                "binary_cross_entropy_with_logits_loss 要求 logits 与 targets 数据类型一致，实际为 {} 和 {}",
                logits.dtype(),
                targets.dtype()
            ),
        });
    }
    if logits.shape() != targets.shape() {
        return Err(LossError::TensorOperationError {
            message: format!(
                "binary_cross_entropy_with_logits_loss 要求 logits 与 targets 形状一致，实际为 {:?} 和 {:?}",
                logits.shape(),
                targets.shape()
            ),
        });
    }

    let invalid_target = match &*targets.storage() {
        arcqml_core::Storage::F32(values) => {
            values.iter().enumerate().find_map(|(index, &value)| {
                (!value.is_finite() || !(0.0_f32..=1.0_f32).contains(&value))
                    .then_some((index, f64::from(value)))
            })
        }
        arcqml_core::Storage::F64(values) => {
            values.iter().enumerate().find_map(|(index, &value)| {
                (!value.is_finite() || !(0.0_f64..=1.0_f64).contains(&value))
                    .then_some((index, value))
            })
        }
        _ => None,
    };
    if let Some((index, value)) = invalid_target {
        return Err(LossError::TensorOperationError {
            message: format!(
                "binary_cross_entropy_with_logits_loss 的第 {index} 个目标值为 {value}，必须位于 [0, 1]"
            ),
        });
    }

    Ok(())
}

/// 根据类别索引生成与 logits 同数据类型的 one-hot 常量张量。
fn one_hot_targets(logits: &Tensor, labels: &[usize]) -> LossResult<Tensor> {
    let batch_size = logits.shape()[0];
    let class_count = logits.shape()[1];

    match logits.dtype() {
        DType::F32 => {
            let mut values = vec![0.0_f32; batch_size * class_count];
            for (row, label) in labels.iter().copied().enumerate() {
                values[row * class_count + label] = 1.0;
            }
            Tensor::new(TensorData::FlatF32 {
                data: values,
                shape: vec![batch_size, class_count],
            })
        }
        DType::F64 => {
            let mut values = vec![0.0_f64; batch_size * class_count];
            for (row, label) in labels.iter().copied().enumerate() {
                values[row * class_count + label] = 1.0;
            }
            Tensor::new(TensorData::FlatF64 {
                data: values,
                shape: vec![batch_size, class_count],
            })
        }
        dtype => return Err(LossError::UnsupportedTensorDType { dtype }),
    }
    .map_err(tensor_create_error)
}
