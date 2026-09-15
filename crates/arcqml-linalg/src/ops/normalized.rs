use crate::autograd::{LogSoftmaxBackward, LogSumExpBackward, SoftmaxBackward};
use crate::error::{LinalgError::*, LinalgResult};
use crate::ops::reductions_dim::reduction_shape;
use crate::ops::{ensure_contiguous_tensor, make_tensor_with_backward};
use arcqml_core::{Storage, Tensor};
use std::sync::Arc;

/// 沿指定维度计算 softmax，输出形状和数据类型与输入相同。
///
/// 支持 `F32` 和 `F64` Tensor；`axis` 对应的维度必须非空。
///
/// # Errors
///
/// 当输入不是连续的 CPU Dense Tensor、数据类型不是 `F32`/`F64`、`axis`
/// 越界或对应维度为空，或无法构造输出时返回错误。
pub fn softmax(input: &Tensor, axis: usize) -> LinalgResult<Tensor> {
    normalized("softmax", input, axis, NormalizedKind::Softmax)
}

/// 沿指定维度计算 log-softmax，输出形状和数据类型与输入相同。
///
/// 支持 `F32` 和 `F64` Tensor；`axis` 对应的维度必须非空。
///
/// # Errors
///
/// 当输入不是连续的 CPU Dense Tensor、数据类型不是 `F32`/`F64`、`axis`
/// 越界或对应维度为空，或无法构造输出时返回错误。
pub fn log_softmax(input: &Tensor, axis: usize) -> LinalgResult<Tensor> {
    normalized("log_softmax", input, axis, NormalizedKind::LogSoftmax)
}

/// 沿指定维度计算数值稳定的 logsumexp。
///
/// 支持 `F32` 和 `F64` Tensor。`keepdim` 为 `true` 时，归约维保留且长度为
/// `1`；否则从输出形状中移除该维。
///
/// # Errors
///
/// 当输入不是连续的 CPU Dense Tensor、数据类型不是 `F32`/`F64`、`axis`
/// 越界或对应维度为空，或无法构造输出时返回错误。
pub fn logsumexp(input: &Tensor, axis: usize, keepdim: bool) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("logsumexp", input)?;
    if axis >= input.ndim() {
        return Err(InvalidAxisError {
            op: "logsumexp",
            axis,
            rank: input.ndim(),
        });
    }
    if input.shape()[axis] == 0 {
        return Err(EmptyInputError { op: "logsumexp" });
    }
    let output_shape = reduction_shape(input.shape(), axis, keepdim);
    let storage = logsumexp_storage(input, axis)?;
    make_tensor_with_backward(
        "logsumexp",
        storage,
        output_shape,
        input,
        vec![input.clone()],
        Arc::new(LogSumExpBackward { axis, keepdim }),
    )
}

/// 归一化算子的内部种类。
#[derive(Clone, Copy)]
enum NormalizedKind {
    Softmax,
    LogSoftmax,
}

/// 执行 softmax 或 log-softmax 前向计算并创建反向节点。
fn normalized(
    operation: &'static str,
    input: &Tensor,
    axis: usize,
    kind: NormalizedKind,
) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor(operation, input)?;
    if axis >= input.ndim() {
        return Err(InvalidAxisError {
            op: operation,
            axis,
            rank: input.ndim(),
        });
    }
    if input.shape()[axis] == 0 {
        return Err(EmptyInputError { op: operation });
    }
    let storage = match (&*input.storage(), kind) {
        (Storage::F32(values), NormalizedKind::Softmax) => {
            Storage::F32(normalize_f32(values, input.shape(), axis, false))
        }
        (Storage::F64(values), NormalizedKind::Softmax) => {
            Storage::F64(normalize_f64(values, input.shape(), axis, false))
        }
        (Storage::F32(values), NormalizedKind::LogSoftmax) => {
            Storage::F32(normalize_f32(values, input.shape(), axis, true))
        }
        (Storage::F64(values), NormalizedKind::LogSoftmax) => {
            Storage::F64(normalize_f64(values, input.shape(), axis, true))
        }
        _ => {
            return Err(UnsupportedDTypeError {
                op: operation,
                dtype: input.dtype().to_string(),
            });
        }
    };
    let backward: Arc<dyn arcqml_core::BackwardFn> = match kind {
        NormalizedKind::Softmax => Arc::new(SoftmaxBackward { axis }),
        NormalizedKind::LogSoftmax => Arc::new(LogSoftmaxBackward { axis }),
    };
    make_tensor_with_backward(
        operation,
        storage,
        input.shape().to_vec(),
        input,
        vec![input.clone()],
        backward,
    )
}

/// 计算 logsumexp 的前向结果。
fn logsumexp_storage(input: &Tensor, axis: usize) -> LinalgResult<Storage> {
    let shape = input.shape();
    let outer: usize = shape[..axis].iter().product();
    let reduced = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    match &*input.storage() {
        Storage::F32(values) => Ok(Storage::F32(
            (0..outer * inner)
                .map(|index| {
                    let outer_index = index / inner;
                    let inner_index = index % inner;
                    let maximum = (0..reduced)
                        .map(|reduced_index| {
                            values[(outer_index * reduced + reduced_index) * inner + inner_index]
                        })
                        .fold(f32::NEG_INFINITY, f32::max);
                    maximum
                        + (0..reduced)
                            .map(|reduced_index| {
                                (values
                                    [(outer_index * reduced + reduced_index) * inner + inner_index]
                                    - maximum)
                                    .exp()
                            })
                            .sum::<f32>()
                            .ln()
                })
                .collect(),
        )),
        Storage::F64(values) => Ok(Storage::F64(
            (0..outer * inner)
                .map(|index| {
                    let outer_index = index / inner;
                    let inner_index = index % inner;
                    let maximum = (0..reduced)
                        .map(|reduced_index| {
                            values[(outer_index * reduced + reduced_index) * inner + inner_index]
                        })
                        .fold(f64::NEG_INFINITY, f64::max);
                    maximum
                        + (0..reduced)
                            .map(|reduced_index| {
                                (values
                                    [(outer_index * reduced + reduced_index) * inner + inner_index]
                                    - maximum)
                                    .exp()
                            })
                            .sum::<f64>()
                            .ln()
                })
                .collect(),
        )),
        _ => Err(UnsupportedDTypeError {
            op: "logsumexp",
            dtype: input.dtype().to_string(),
        }),
    }
}

/// 计算 F32 softmax 或 log-softmax。
fn normalize_f32(values: &[f32], shape: &[usize], axis: usize, logarithmic: bool) -> Vec<f32> {
    let outer: usize = shape[..axis].iter().product();
    let reduced = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let mut output = vec![0.0; values.len()];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let maximum = (0..reduced)
                .map(|index| values[(outer_index * reduced + index) * inner + inner_index])
                .fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = (0..reduced)
                .map(|index| {
                    (values[(outer_index * reduced + index) * inner + inner_index] - maximum).exp()
                })
                .sum();
            for index in 0..reduced {
                let position = (outer_index * reduced + index) * inner + inner_index;
                let shifted = values[position] - maximum;
                output[position] = if logarithmic {
                    shifted - sum_exp.ln()
                } else {
                    shifted.exp() / sum_exp
                };
            }
        }
    }
    output
}

/// 计算 F64 softmax 或 log-softmax。
fn normalize_f64(values: &[f64], shape: &[usize], axis: usize, logarithmic: bool) -> Vec<f64> {
    let outer: usize = shape[..axis].iter().product();
    let reduced = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let mut output = vec![0.0; values.len()];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let maximum = (0..reduced)
                .map(|index| values[(outer_index * reduced + index) * inner + inner_index])
                .fold(f64::NEG_INFINITY, f64::max);
            let sum_exp: f64 = (0..reduced)
                .map(|index| {
                    (values[(outer_index * reduced + index) * inner + inner_index] - maximum).exp()
                })
                .sum();
            for index in 0..reduced {
                let position = (outer_index * reduced + index) * inner + inner_index;
                let shifted = values[position] - maximum;
                output[position] = if logarithmic {
                    shifted - sum_exp.ln()
                } else {
                    shifted.exp() / sum_exp
                };
            }
        }
    }
    output
}
