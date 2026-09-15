use crate::autograd::{ClampBackward, DivBackward, ExtraUnaryBackward, ExtraUnaryKind};
use crate::error::{LinalgError::*, LinalgResult};
use crate::ops::{
    broadcast_index, ensure_binary_compatible, ensure_contiguous_tensor, make_tensor_with_backward,
};
use crate::shape::{infer_broadcast_shape, numel};
use arcqml_core::{Storage, Tensor};
use std::sync::Arc;

/// 计算实数 Tensor 的逐元素相反数。
///
/// # Errors
///
/// 当输入不是连续的 CPU 稠密 `F32`/`F64` Tensor，或无法构造输出时返回错误。
pub fn neg(input: &Tensor) -> LinalgResult<Tensor> {
    unary_real("neg", ExtraUnaryKind::Neg, input, |value| -value)
}

/// 计算实数 Tensor 的逐元素自然指数。
///
/// # Errors
///
/// 当输入不是连续的 CPU 稠密 `F32`/`F64` Tensor，或无法构造输出时返回错误。
pub fn exp(input: &Tensor) -> LinalgResult<Tensor> {
    unary_real("exp", ExtraUnaryKind::Exp, input, f64::exp)
}

/// 计算实数 Tensor 的逐元素自然对数；本函数不预先拒绝非正输入，结果遵循浮点运算语义。
///
/// # Errors
///
/// 当输入不是连续的 CPU 稠密 `F32`/`F64` Tensor，或无法构造输出时返回错误。
pub fn log(input: &Tensor) -> LinalgResult<Tensor> {
    unary_real("log", ExtraUnaryKind::Log, input, f64::ln)
}

/// 计算逐元素 sigmoid 函数。
///
/// # Errors
///
/// 当输入不是连续的 CPU 稠密 `F32`/`F64` Tensor，或无法构造输出时返回错误。
pub fn sigmoid(input: &Tensor) -> LinalgResult<Tensor> {
    unary_real("sigmoid", ExtraUnaryKind::Sigmoid, input, |value| {
        if value >= 0.0 {
            1.0 / (1.0 + (-value).exp())
        } else {
            let exp_value = value.exp();
            exp_value / (1.0 + exp_value)
        }
    })
}

/// 计算逐元素双曲正切函数。
///
/// # Errors
///
/// 当输入不是连续的 CPU 稠密 `F32`/`F64` Tensor，或无法构造输出时返回错误。
pub fn tanh(input: &Tensor) -> LinalgResult<Tensor> {
    unary_real("tanh", ExtraUnaryKind::Tanh, input, f64::tanh)
}

/// 计算支持 NumPy 风格广播的实数逐元素除法。
///
/// 除数为零时遵循 IEEE 754 浮点语义，本函数不会因此返回错误。
///
/// # Errors
///
/// 当输入不是同 dtype 的连续 CPU 稠密 `F32`/`F64` Tensor、形状不能广播，
/// 或无法构造输出时返回错误。
pub fn div(lhs: &Tensor, rhs: &Tensor) -> LinalgResult<Tensor> {
    ensure_binary_compatible("elementwise divide", lhs, rhs)?;
    let output_shape = infer_broadcast_shape(lhs.shape(), rhs.shape())?;
    let output_len = numel(&output_shape);
    let lhs_storage = lhs.storage();
    let rhs_storage = rhs.storage();
    let storage = match (&*lhs_storage, &*rhs_storage) {
        (Storage::F32(left), Storage::F32(right)) => Storage::F32(
            (0..output_len)
                .map(|index| {
                    left[broadcast_index(index, &output_shape, lhs.shape())]
                        / right[broadcast_index(index, &output_shape, rhs.shape())]
                })
                .collect(),
        ),
        (Storage::F64(left), Storage::F64(right)) => Storage::F64(
            (0..output_len)
                .map(|index| {
                    left[broadcast_index(index, &output_shape, lhs.shape())]
                        / right[broadcast_index(index, &output_shape, rhs.shape())]
                })
                .collect(),
        ),
        _ => {
            return Err(UnsupportedDTypeError {
                op: "elementwise divide",
                dtype: lhs.dtype().to_string(),
            });
        }
    };
    make_tensor_with_backward(
        "elementwise divide",
        storage,
        output_shape.clone(),
        lhs,
        vec![lhs.clone(), rhs.clone()],
        Arc::new(DivBackward { output_shape }),
    )
}

/// 将每个元素限制在闭区间 `[min, max]` 内。
///
/// # Errors
///
/// 当 `min > max`，输入不是连续的 CPU 稠密 `F32`/`F64` Tensor，或无法构造输出时返回错误。
pub fn clamp(input: &Tensor, min: f64, max: f64) -> LinalgResult<Tensor> {
    if min > max {
        return Err(ShapeMismatchError {
            op: "clamp",
            expected: "min <= max".to_string(),
            actual: format!("min={min}, max={max}"),
        });
    }
    ensure_contiguous_tensor("clamp", input)?;
    let storage = match &*input.storage() {
        Storage::F32(values) => Storage::F32(
            values
                .iter()
                .map(|value| value.clamp(min as f32, max as f32))
                .collect(),
        ),
        Storage::F64(values) => {
            Storage::F64(values.iter().map(|value| value.clamp(min, max)).collect())
        }
        _ => {
            return Err(UnsupportedDTypeError {
                op: "clamp",
                dtype: input.dtype().to_string(),
            });
        }
    };
    make_tensor_with_backward(
        "clamp",
        storage,
        input.shape().to_vec(),
        input,
        vec![input.clone()],
        Arc::new(ClampBackward { min, max }),
    )
}

/// 执行仅支持实数类型的逐元素一元操作。
fn unary_real(
    operation: &'static str,
    kind: ExtraUnaryKind,
    input: &Tensor,
    transform: impl Fn(f64) -> f64,
) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor(operation, input)?;
    let storage = match &*input.storage() {
        Storage::F32(values) => Storage::F32(
            values
                .iter()
                .map(|value| transform(f64::from(*value)) as f32)
                .collect(),
        ),
        Storage::F64(values) => {
            Storage::F64(values.iter().map(|value| transform(*value)).collect())
        }
        _ => {
            return Err(UnsupportedDTypeError {
                op: operation,
                dtype: input.dtype().to_string(),
            });
        }
    };
    make_tensor_with_backward(
        operation,
        storage,
        input.shape().to_vec(),
        input,
        vec![input.clone()],
        Arc::new(ExtraUnaryBackward { kind }),
    )
}
