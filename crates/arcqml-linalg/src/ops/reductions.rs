use crate::autograd::{ExtremumBackward, ExtremumKind, ReductionBackward, ReductionKind};
use crate::error::{LinalgError::*, LinalgResult};
use crate::kernels::{
    max_f32, max_f64, max_i64, mean_c64, mean_f32, mean_f64, mean_i64, min_f32, min_f64, min_i64,
    sum_c64, sum_f32, sum_f64, sum_i64,
};
use crate::ops::{ensure_contiguous_tensor, make_tensor_with_backward};
use arcqml_core::{Storage, Tensor};
use std::sync::Arc;

/// 对整个 Tensor 求和，输出为标量 Tensor。
///
/// # Errors
///
/// 当输入为空、不是连续 CPU 稠密数值 Tensor、dtype 为 `Bool`、`I64` 累加溢出，
/// 或无法构造输出时返回错误。
pub fn sum(input: &Tensor) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("sum", input)?;

    let storage = match &*input.storage() {
        Storage::F32(v) => Storage::F32(vec![sum_f32(v)?]),
        Storage::F64(v) => Storage::F64(vec![sum_f64(v)?]),
        Storage::C64(v) => Storage::C64(vec![sum_c64(v)?]),
        Storage::I64(v) => Storage::I64(vec![sum_i64(v)?]),
        Storage::Bool(_) => {
            return Err(UnsupportedDTypeError {
                op: "sum",
                dtype: input.dtype().to_string(),
            });
        }
    };

    make_tensor_with_backward(
        "sum",
        storage,
        vec![],
        input,
        vec![input.clone()],
        Arc::new(ReductionBackward {
            kind: ReductionKind::Sum,
        }),
    )
}

/// 对整个 Tensor 求算术平均，输出为同 dtype 标量 Tensor；`I64` 使用截断整数除法。
///
/// # Errors
///
/// 当输入为空、不是连续 CPU 稠密数值 Tensor、dtype 为 `Bool`、`I64` 累加溢出，
/// 或无法构造输出时返回错误。
pub fn mean(input: &Tensor) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("mean", input)?;

    let storage = match &*input.storage() {
        Storage::F32(v) => Storage::F32(vec![mean_f32(v)?]),
        Storage::F64(v) => Storage::F64(vec![mean_f64(v)?]),
        Storage::C64(v) => Storage::C64(vec![mean_c64(v)?]),
        Storage::I64(v) => Storage::I64(vec![mean_i64(v)?]),
        Storage::Bool(_) => {
            return Err(UnsupportedDTypeError {
                op: "mean",
                dtype: input.dtype().to_string(),
            });
        }
    };

    make_tensor_with_backward(
        "mean",
        storage,
        vec![],
        input,
        vec![input.clone()],
        Arc::new(ReductionBackward {
            kind: ReductionKind::Mean,
        }),
    )
}

/// 对整个 Tensor 求最大值，输出为标量 Tensor。
///
/// # Errors
///
/// 当输入为空、不是连续 CPU 稠密 Tensor、dtype 为 `C64`/`Bool`，或无法构造输出时返回错误。
pub fn max(input: &Tensor) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("max", input)?;

    let storage = match &*input.storage() {
        Storage::F32(v) => Storage::F32(vec![max_f32(v)?]),
        Storage::F64(v) => Storage::F64(vec![max_f64(v)?]),
        Storage::I64(v) => Storage::I64(vec![max_i64(v)?]),
        Storage::C64(_) | Storage::Bool(_) => {
            return Err(UnsupportedDTypeError {
                op: "max",
                dtype: input.dtype().to_string(),
            });
        }
    };

    make_tensor_with_backward(
        "max",
        storage,
        vec![],
        input,
        vec![input.clone()],
        Arc::new(ExtremumBackward {
            kind: ExtremumKind::Max,
        }),
    )
}

/// 对整个 Tensor 求最小值，输出为标量 Tensor。
///
/// # Errors
///
/// 当输入为空、不是连续 CPU 稠密 Tensor、dtype 为 `C64`/`Bool`，或无法构造输出时返回错误。
pub fn min(input: &Tensor) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("min", input)?;

    let storage = match &*input.storage() {
        Storage::F32(v) => Storage::F32(vec![min_f32(v)?]),
        Storage::F64(v) => Storage::F64(vec![min_f64(v)?]),
        Storage::I64(v) => Storage::I64(vec![min_i64(v)?]),
        Storage::C64(_) | Storage::Bool(_) => {
            return Err(UnsupportedDTypeError {
                op: "min",
                dtype: input.dtype().to_string(),
            });
        }
    };

    make_tensor_with_backward(
        "min",
        storage,
        vec![],
        input,
        vec![input.clone()],
        Arc::new(ExtremumBackward {
            kind: ExtremumKind::Min,
        }),
    )
}
