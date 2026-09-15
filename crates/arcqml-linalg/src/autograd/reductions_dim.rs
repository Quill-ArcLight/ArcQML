use crate::ops::reductions_dim::reduction_shape;
use arcqml_core::{ArcQmlError, BackwardFn, Result, Storage, Tensor};

/// 按维度归约算子的种类。
#[derive(Debug, Clone, Copy)]
pub(crate) enum DimReductionKind {
    Sum,
    Mean,
}

/// 按维度归约算子的反向规则。
#[derive(Debug)]
pub(crate) struct DimReductionBackward {
    pub(crate) kind: DimReductionKind,
    pub(crate) axis: usize,
    pub(crate) keepdim: bool,
}

impl BackwardFn for DimReductionBackward {
    /// 将上游梯度扩展回被归约前的形状。
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(ArcQmlError::AutogradError(
                "dim reduction backward expects one parent".to_string(),
            ));
        }
        let input = &parents[0];
        let expected = reduction_shape(input.shape(), self.axis, self.keepdim);
        if grad_output.shape() != expected {
            return Err(ArcQmlError::AutogradError(format!(
                "dim reduction backward expected upstream shape {:?}, got {:?}",
                expected,
                grad_output.shape()
            )));
        }
        let outer: usize = input.shape()[..self.axis].iter().product();
        let reduced = input.shape()[self.axis];
        let inner: usize = input.shape()[self.axis + 1..].iter().product();
        let storage = match &*grad_output.storage() {
            Storage::F32(values) => Storage::F32(expand_gradient_f32(
                values,
                outer,
                reduced,
                inner,
                matches!(self.kind, DimReductionKind::Mean),
            )),
            Storage::F64(values) => Storage::F64(expand_gradient_f64(
                values,
                outer,
                reduced,
                inner,
                matches!(self.kind, DimReductionKind::Mean),
            )),
            _ => {
                return Err(ArcQmlError::AutogradError(format!(
                    "dtype {} does not support dim reduction backward",
                    input.dtype()
                )));
            }
        };
        Ok(vec![Some(Tensor::from_storage_meta(
            storage,
            input.meta().clone(),
        )?)])
    }
}

/// 将归约输出梯度广播到每一个被归约元素。
fn expand_gradient_f32(
    upstream: &[f32],
    outer: usize,
    reduced: usize,
    inner: usize,
    mean: bool,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(outer * reduced * inner);
    for outer_index in 0..outer {
        for _ in 0..reduced {
            for inner_index in 0..inner {
                let value = upstream[outer_index * inner + inner_index];
                output.push(if mean { value / reduced as f32 } else { value });
            }
        }
    }
    output
}

/// 将 F64 归约输出梯度广播到每一个被归约元素。
fn expand_gradient_f64(
    upstream: &[f64],
    outer: usize,
    reduced: usize,
    inner: usize,
    mean: bool,
) -> Vec<f64> {
    let mut output = Vec::with_capacity(outer * reduced * inner);
    for outer_index in 0..outer {
        for _ in 0..reduced {
            for inner_index in 0..inner {
                let value = upstream[outer_index * inner + inner_index];
                output.push(if mean { value / reduced as f64 } else { value });
            }
        }
    }
    output
}
