use crate::ops::reductions_dim::reduction_shape;
use arcqml_core::{ArcQmlError, BackwardFn, Result, Storage, Tensor};

/// softmax 的反向规则。
#[derive(Debug)]
pub(crate) struct SoftmaxBackward {
    pub(crate) axis: usize,
}

impl BackwardFn for SoftmaxBackward {
    /// 使用 `softmax * (upstream - sum(upstream * softmax))` 计算 VJP。
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        normalized_backward(
            parents,
            grad_output,
            self.axis,
            NormalizedBackwardKind::Softmax,
        )
    }
}

/// log-softmax 的反向规则。
#[derive(Debug)]
pub(crate) struct LogSoftmaxBackward {
    pub(crate) axis: usize,
}

impl BackwardFn for LogSoftmaxBackward {
    /// 使用 `upstream - exp(log_softmax) * sum(upstream)` 计算 VJP。
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        normalized_backward(
            parents,
            grad_output,
            self.axis,
            NormalizedBackwardKind::LogSoftmax,
        )
    }
}

/// logsumexp 的反向规则。
#[derive(Debug)]
pub(crate) struct LogSumExpBackward {
    pub(crate) axis: usize,
    pub(crate) keepdim: bool,
}

impl BackwardFn for LogSumExpBackward {
    /// 将上游梯度乘以沿归约维度的 softmax 权重。
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(ArcQmlError::AutogradError(
                "logsumexp backward expects one parent".to_string(),
            ));
        }
        let input = &parents[0];
        let expected_shape = reduction_shape(input.shape(), self.axis, self.keepdim);
        if grad_output.shape() != expected_shape {
            return Err(ArcQmlError::AutogradError(format!(
                "logsumexp backward expected upstream shape {:?}, got {:?}",
                expected_shape,
                grad_output.shape()
            )));
        }
        let output = match (&*input.storage(), &*grad_output.storage()) {
            (Storage::F32(values), Storage::F32(upstream)) => Storage::F32(logsumexp_gradient_f32(
                values,
                upstream,
                input.shape(),
                self.axis,
                self.keepdim,
            )),
            (Storage::F64(values), Storage::F64(upstream)) => Storage::F64(logsumexp_gradient_f64(
                values,
                upstream,
                input.shape(),
                self.axis,
                self.keepdim,
            )),
            _ => {
                return Err(ArcQmlError::AutogradError(format!(
                    "dtype {} does not support logsumexp backward",
                    input.dtype()
                )));
            }
        };
        Ok(vec![Some(Tensor::from_storage_meta(
            output,
            input.meta().clone(),
        )?)])
    }
}

/// 归一化反向规则的类型。
enum NormalizedBackwardKind {
    Softmax,
    LogSoftmax,
}

/// 计算 softmax 与 log-softmax 的反向传播。
fn normalized_backward(
    parents: &[Tensor],
    grad_output: &Tensor,
    axis: usize,
    kind: NormalizedBackwardKind,
) -> Result<Vec<Option<Tensor>>> {
    if parents.len() != 1 {
        return Err(ArcQmlError::AutogradError(
            "normalized backward expects one parent".to_string(),
        ));
    }
    let input = &parents[0];
    let output =
        match (&*input.storage(), &*grad_output.storage()) {
            (Storage::F32(values), Storage::F32(upstream)) => Storage::F32(
                normalized_gradient_f32(values, upstream, input.shape(), axis, kind),
            ),
            (Storage::F64(values), Storage::F64(upstream)) => Storage::F64(
                normalized_gradient_f64(values, upstream, input.shape(), axis, kind),
            ),
            _ => {
                return Err(ArcQmlError::AutogradError(format!(
                    "dtype {} does not support normalized backward",
                    input.dtype()
                )));
            }
        };
    Ok(vec![Some(Tensor::from_storage_meta(
        output,
        input.meta().clone(),
    )?)])
}

/// 计算 F32 softmax 或 log-softmax 梯度。
fn normalized_gradient_f32(
    values: &[f32],
    upstream: &[f32],
    shape: &[usize],
    axis: usize,
    kind: NormalizedBackwardKind,
) -> Vec<f32> {
    let outer: usize = shape[..axis].iter().product();
    let reduced = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let mut output = vec![0.0; values.len()];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let maximum = (0..reduced)
                .map(|index| values[(outer_index * reduced + index) * inner + inner_index])
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = (0..reduced)
                .map(|index| {
                    (values[(outer_index * reduced + index) * inner + inner_index] - maximum).exp()
                })
                .sum();
            let sum: f32 = match kind {
                NormalizedBackwardKind::Softmax => (0..reduced)
                    .map(|index| {
                        let position = (outer_index * reduced + index) * inner + inner_index;
                        upstream[position] * ((values[position] - maximum).exp() / denominator)
                    })
                    .sum(),
                NormalizedBackwardKind::LogSoftmax => (0..reduced)
                    .map(|index| upstream[(outer_index * reduced + index) * inner + inner_index])
                    .sum(),
            };
            for index in 0..reduced {
                let position = (outer_index * reduced + index) * inner + inner_index;
                let softmax = (values[position] - maximum).exp() / denominator;
                output[position] = match kind {
                    NormalizedBackwardKind::Softmax => softmax * (upstream[position] - sum),
                    NormalizedBackwardKind::LogSoftmax => upstream[position] - softmax * sum,
                };
            }
        }
    }
    output
}

/// 计算 F64 softmax 或 log-softmax 梯度。
fn normalized_gradient_f64(
    values: &[f64],
    upstream: &[f64],
    shape: &[usize],
    axis: usize,
    kind: NormalizedBackwardKind,
) -> Vec<f64> {
    let outer: usize = shape[..axis].iter().product();
    let reduced = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let mut output = vec![0.0; values.len()];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let maximum = (0..reduced)
                .map(|index| values[(outer_index * reduced + index) * inner + inner_index])
                .fold(f64::NEG_INFINITY, f64::max);
            let denominator: f64 = (0..reduced)
                .map(|index| {
                    (values[(outer_index * reduced + index) * inner + inner_index] - maximum).exp()
                })
                .sum();
            let sum: f64 = match kind {
                NormalizedBackwardKind::Softmax => (0..reduced)
                    .map(|index| {
                        let position = (outer_index * reduced + index) * inner + inner_index;
                        upstream[position] * ((values[position] - maximum).exp() / denominator)
                    })
                    .sum(),
                NormalizedBackwardKind::LogSoftmax => (0..reduced)
                    .map(|index| upstream[(outer_index * reduced + index) * inner + inner_index])
                    .sum(),
            };
            for index in 0..reduced {
                let position = (outer_index * reduced + index) * inner + inner_index;
                let softmax = (values[position] - maximum).exp() / denominator;
                output[position] = match kind {
                    NormalizedBackwardKind::Softmax => softmax * (upstream[position] - sum),
                    NormalizedBackwardKind::LogSoftmax => upstream[position] - softmax * sum,
                };
            }
        }
    }
    output
}

/// 计算 F32 logsumexp 梯度。
fn logsumexp_gradient_f32(
    values: &[f32],
    upstream: &[f32],
    shape: &[usize],
    axis: usize,
    keepdim: bool,
) -> Vec<f32> {
    logsumexp_gradient(values, upstream, shape, axis, keepdim, |value| value.exp())
}

/// 计算 F64 logsumexp 梯度。
fn logsumexp_gradient_f64(
    values: &[f64],
    upstream: &[f64],
    shape: &[usize],
    axis: usize,
    keepdim: bool,
) -> Vec<f64> {
    logsumexp_gradient(values, upstream, shape, axis, keepdim, |value| value.exp())
}

/// 将稳定 softmax 权重与对应的归约上游梯度相乘。
fn logsumexp_gradient<
    T: Copy
        + Default
        + PartialOrd
        + std::ops::Sub<Output = T>
        + std::ops::Mul<Output = T>
        + std::ops::AddAssign
        + std::ops::Div<Output = T>
        + From<f32>,
>(
    values: &[T],
    upstream: &[T],
    shape: &[usize],
    axis: usize,
    keepdim: bool,
    exponential: impl Fn(T) -> T,
) -> Vec<T> {
    let outer: usize = shape[..axis].iter().product();
    let reduced = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let output_shape = reduction_shape(shape, axis, keepdim);
    let mut output = vec![T::default(); values.len()];
    for outer_index in 0..outer {
        for inner_index in 0..inner {
            let maximum = (0..reduced)
                .map(|reduced_index| {
                    values[(outer_index * reduced + reduced_index) * inner + inner_index]
                })
                .reduce(|left, right| if left > right { left } else { right })
                .unwrap();
            let denominator = (0..reduced).fold(T::default(), |mut total, reduced_index| {
                total += exponential(
                    values[(outer_index * reduced + reduced_index) * inner + inner_index] - maximum,
                );
                total
            });
            for reduced_index in 0..reduced {
                let index = (outer_index * reduced + reduced_index) * inner + inner_index;
                let upstream_index = if keepdim {
                    let mut coordinates = Vec::new();
                    let mut remaining = index;
                    for dimension in shape.iter().rev() {
                        coordinates.push(remaining % *dimension);
                        remaining /= *dimension;
                    }
                    coordinates.reverse();
                    coordinates[axis] = 0;
                    coordinates
                        .into_iter()
                        .zip(output_shape.iter())
                        .fold(0, |result, (coordinate, dimension)| {
                            result * dimension + coordinate
                        })
                } else {
                    outer_index * inner + inner_index
                };
                output[index] =
                    exponential(values[index] - maximum) / denominator * upstream[upstream_index];
            }
        }
    }
    output
}
