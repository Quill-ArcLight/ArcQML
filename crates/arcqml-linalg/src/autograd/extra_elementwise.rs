use crate::ops::broadcast_index;
use arcqml_core::{ArcQmlError, BackwardFn, DType, Result, Storage, Tensor, TensorMeta};

/// 额外实数一元算子的种类。
#[derive(Debug, Clone, Copy)]
pub(crate) enum ExtraUnaryKind {
    Neg,
    Exp,
    Log,
    Sigmoid,
    Tanh,
}

/// 额外实数一元算子的反向规则。
#[derive(Debug)]
pub(crate) struct ExtraUnaryBackward {
    pub(crate) kind: ExtraUnaryKind,
}

impl BackwardFn for ExtraUnaryBackward {
    /// 根据一元函数的局部导数计算输入梯度。
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(ArcQmlError::AutogradError(
                "unary backward expects one parent".to_string(),
            ));
        }
        let input = &parents[0];
        let gradient = match (&*input.storage(), &*grad_output.storage()) {
            (Storage::F32(values), Storage::F32(upstream)) => Storage::F32(
                values
                    .iter()
                    .zip(upstream)
                    .map(|(value, gradient)| {
                        *gradient * unary_derivative(f64::from(*value), self.kind) as f32
                    })
                    .collect(),
            ),
            (Storage::F64(values), Storage::F64(upstream)) => Storage::F64(
                values
                    .iter()
                    .zip(upstream)
                    .map(|(value, gradient)| *gradient * unary_derivative(*value, self.kind))
                    .collect(),
            ),
            _ => return unsupported_dtype(input.dtype()),
        };
        Ok(vec![Some(Tensor::from_storage_meta(
            gradient,
            input.meta().clone(),
        )?)])
    }
}

/// 计算实数一元函数的局部导数。
fn unary_derivative(value: f64, kind: ExtraUnaryKind) -> f64 {
    match kind {
        ExtraUnaryKind::Neg => -1.0,
        ExtraUnaryKind::Exp => value.exp(),
        ExtraUnaryKind::Log => 1.0 / value,
        ExtraUnaryKind::Sigmoid => {
            let output = if value >= 0.0 {
                1.0 / (1.0 + (-value).exp())
            } else {
                let exp_value = value.exp();
                exp_value / (1.0 + exp_value)
            };
            output * (1.0 - output)
        }
        ExtraUnaryKind::Tanh => {
            let output = value.tanh();
            1.0 - output * output
        }
    }
}

/// 支持广播的实数逐元素除法反向规则。
#[derive(Debug)]
pub(crate) struct DivBackward {
    pub(crate) output_shape: Vec<usize>,
}

impl BackwardFn for DivBackward {
    /// 计算 lhs / rhs 对两个输入的梯度，并在广播维度上累加。
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 2 {
            return Err(ArcQmlError::AutogradError(
                "div backward expects two parents".to_string(),
            ));
        }
        match (
            &*parents[0].storage(),
            &*parents[1].storage(),
            &*grad_output.storage(),
        ) {
            (Storage::F32(lhs), Storage::F32(rhs), Storage::F32(upstream)) => {
                let (left, right) =
                    div_backward_values(lhs, rhs, upstream, parents, &self.output_shape, |value| {
                        value as f64
                    });
                Ok(vec![
                    Some(gradient_f32(left, parents[0].meta().clone())?),
                    Some(gradient_f32(right, parents[1].meta().clone())?),
                ])
            }
            (Storage::F64(lhs), Storage::F64(rhs), Storage::F64(upstream)) => {
                let (left, right) =
                    div_backward_values(lhs, rhs, upstream, parents, &self.output_shape, |value| {
                        value
                    });
                Ok(vec![
                    Some(gradient_f64(left, parents[0].meta().clone())?),
                    Some(gradient_f64(right, parents[1].meta().clone())?),
                ])
            }
            _ => unsupported_dtype(parents[0].dtype()),
        }
    }
}

/// 在广播语义下计算除法两侧的 F64 梯度值。
fn div_backward_values<T>(
    lhs: &[T],
    rhs: &[T],
    upstream: &[T],
    parents: &[Tensor],
    output_shape: &[usize],
    convert: impl Fn(T) -> f64,
) -> (Vec<f64>, Vec<f64>)
where
    T: Copy,
{
    let mut lhs_gradient = vec![0.0; lhs.len()];
    let mut rhs_gradient = vec![0.0; rhs.len()];
    for (index, value) in upstream.iter().copied().enumerate() {
        let lhs_index = broadcast_index(index, output_shape, parents[0].shape());
        let rhs_index = broadcast_index(index, output_shape, parents[1].shape());
        let left = convert(lhs[lhs_index]);
        let right = convert(rhs[rhs_index]);
        let gradient = convert(value);
        lhs_gradient[lhs_index] += gradient / right;
        rhs_gradient[rhs_index] -= gradient * left / (right * right);
    }
    (lhs_gradient, rhs_gradient)
}

/// clamp 的反向规则，区间边界与 PyTorch 一致地取导数一。
#[derive(Debug)]
pub(crate) struct ClampBackward {
    pub(crate) min: f64,
    pub(crate) max: f64,
}

impl BackwardFn for ClampBackward {
    /// 仅在闭区间内传递上游梯度。
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(ArcQmlError::AutogradError(
                "clamp backward expects one parent".to_string(),
            ));
        }
        let input = &parents[0];
        let gradient = match (&*input.storage(), &*grad_output.storage()) {
            (Storage::F32(values), Storage::F32(upstream)) => Storage::F32(
                values
                    .iter()
                    .zip(upstream)
                    .map(|(value, gradient)| {
                        if f64::from(*value) >= self.min && f64::from(*value) <= self.max {
                            *gradient
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            ),
            (Storage::F64(values), Storage::F64(upstream)) => Storage::F64(
                values
                    .iter()
                    .zip(upstream)
                    .map(|(value, gradient)| {
                        if *value >= self.min && *value <= self.max {
                            *gradient
                        } else {
                            0.0
                        }
                    })
                    .collect(),
            ),
            _ => return unsupported_dtype(input.dtype()),
        };
        Ok(vec![Some(Tensor::from_storage_meta(
            gradient,
            input.meta().clone(),
        )?)])
    }
}

/// 使用 F32 数据创建与目标元信息一致的梯度张量。
fn gradient_f32(values: Vec<f64>, meta: TensorMeta) -> Result<Tensor> {
    Tensor::from_storage_meta(
        Storage::F32(values.into_iter().map(|value| value as f32).collect()),
        meta,
    )
}

/// 使用 F64 数据创建与目标元信息一致的梯度张量。
fn gradient_f64(values: Vec<f64>, meta: TensorMeta) -> Result<Tensor> {
    Tensor::from_storage_meta(Storage::F64(values), meta)
}

/// 返回不支持求导的数据类型错误。
fn unsupported_dtype<T>(dtype: DType) -> Result<T> {
    Err(ArcQmlError::AutogradError(format!(
        "dtype {dtype} does not support this backward rule"
    )))
}
