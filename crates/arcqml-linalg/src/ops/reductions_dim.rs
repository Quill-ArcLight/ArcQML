use crate::autograd::{DimReductionBackward, DimReductionKind};
use crate::error::{LinalgError::*, LinalgResult};
use crate::ops::{ensure_contiguous_tensor, make_tensor_with_backward};
use arcqml_core::{Storage, Tensor};
use std::sync::Arc;

/// 沿指定维度对 `F32` 或 `F64` Tensor 求和。
///
/// `keepdim` 为 `true` 时，归约维保留且长度为 `1`；否则移除该维。
///
/// # Errors
///
/// 当输入不是连续的 CPU Dense Tensor、数据类型不是 `F32`/`F64`、`axis`
/// 越界或对应维度为空，或无法构造输出时返回错误。
pub fn sum_dim(input: &Tensor, axis: usize, keepdim: bool) -> LinalgResult<Tensor> {
    reduce_dim("sum_dim", DimReductionKind::Sum, input, axis, keepdim)
}

/// 沿指定维度对 `F32` 或 `F64` Tensor 求算术平均值。
///
/// `keepdim` 为 `true` 时，归约维保留且长度为 `1`；否则移除该维。
///
/// # Errors
///
/// 当输入不是连续的 CPU Dense Tensor、数据类型不是 `F32`/`F64`、`axis`
/// 越界或对应维度为空，或无法构造输出时返回错误。
pub fn mean_dim(input: &Tensor, axis: usize, keepdim: bool) -> LinalgResult<Tensor> {
    reduce_dim("mean_dim", DimReductionKind::Mean, input, axis, keepdim)
}

/// 执行实数张量的按维度归约。
fn reduce_dim(
    operation: &'static str,
    kind: DimReductionKind,
    input: &Tensor,
    axis: usize,
    keepdim: bool,
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
    let output_shape = reduction_shape(input.shape(), axis, keepdim);
    let outer: usize = input.shape()[..axis].iter().product();
    let reduced = input.shape()[axis];
    let inner: usize = input.shape()[axis + 1..].iter().product();
    let storage = match &*input.storage() {
        Storage::F32(values) => Storage::F32(reduce_f32(values, outer, reduced, inner, kind)),
        Storage::F64(values) => Storage::F64(reduce_f64(values, outer, reduced, inner, kind)),
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
        output_shape,
        input,
        vec![input.clone()],
        Arc::new(DimReductionBackward {
            kind,
            axis,
            keepdim,
        }),
    )
}

/// 推导按维度归约后的输出形状。
pub(crate) fn reduction_shape(shape: &[usize], axis: usize, keepdim: bool) -> Vec<usize> {
    let mut output = shape.to_vec();
    if keepdim {
        output[axis] = 1;
    } else {
        output.remove(axis);
    }
    output
}

/// 按 `outer * reduced * inner` 布局执行归约计算。
///
/// - `outer`：归约轴之前所有维度的乘积。
/// - `reduced`：归约维的长度，即 `shape[axis]`。
/// - `inner`：归约轴之后所有维度的乘积。
fn reduce_f32(
    values: &[f32],
    outer: usize,
    reduced: usize,
    inner: usize,
    kind: DimReductionKind,
) -> Vec<f32> {
    let mut output = vec![0.0; outer * inner];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let output_index = outer_index * inner + inner_index;
            for reduced_index in 0..reduced {
                output[output_index] +=
                    values[(outer_index * reduced + reduced_index) * inner + inner_index];
            }
            if matches!(kind, DimReductionKind::Mean) {
                output[output_index] /= reduced as f32;
            }
        }
    }
    output
}

/// 使用 F64 数据按指定维度执行归约计算。
fn reduce_f64(
    values: &[f64],
    outer: usize,
    reduced: usize,
    inner: usize,
    kind: DimReductionKind,
) -> Vec<f64> {
    let mut output = vec![0.0; outer * inner];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let output_index = outer_index * inner + inner_index;
            for reduced_index in 0..reduced {
                output[output_index] +=
                    values[(outer_index * reduced + reduced_index) * inner + inner_index];
            }
            if matches!(kind, DimReductionKind::Mean) {
                output[output_index] /= reduced as f64;
            }
        }
    }
    output
}
