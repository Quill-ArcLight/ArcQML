/// 广播形状推导与广播二元运算。
pub mod broadcast;
/// 复数共轭运算。
pub mod complex;
/// 一维向量点积。
pub mod dot;
/// 逐元素算术和非线性函数。
pub mod elementwise;
/// 二维矩阵乘法。
pub mod matmul;
/// Softmax、log-softmax 和 logsumexp。
pub mod normalized;
/// 向量范数。
pub mod norms;
/// 对全部元素进行归约。
pub mod reductions;
/// 沿指定轴进行归约。
pub mod reductions_dim;
/// Tensor 形状变换。
pub mod reshape;
/// 按分段编号聚合元素。
pub mod segment;
/// 二维 Tensor 转置。
pub mod transpose;

pub use broadcast::{abs, add, mul, sqrt, square, sub};
pub use complex::conj;
pub use dot::dot;
pub use elementwise::{clamp, div, exp, log, neg, sigmoid, tanh};
pub use matmul::matmul;
pub use normalized::{log_softmax, logsumexp, softmax};
pub use norms::l2_norm;
pub use reductions::{max, mean, min, sum};
pub use reductions_dim::{mean_dim, sum_dim};
pub use reshape::reshape;
pub use segment::segment_sum;
pub use transpose::transpose;

use crate::error::{LinalgError::*, LinalgResult};
use arcqml_core::{BackwardFn, Device, Layout, Storage, Tensor, TensorMeta};
use std::sync::Arc;

/// 校验 Tensor 是否是当前 CPU Dense 后端支持的输入。
pub(crate) fn ensure_supported_tensor(op: &'static str, tensor: &Tensor) -> LinalgResult<()> {
    if tensor.device() != &Device::Cpu {
        return Err(UnsupportedDeviceError {
            op,
            device: tensor.device().to_string(),
        });
    }

    if tensor.layout() != Layout::Dense {
        return Err(UnsupportedDTypeError {
            op,
            dtype: tensor.layout().to_string(),
        });
    }

    Ok(())
}

/// 校验 Tensor 是否是连续存储。
pub(crate) fn ensure_contiguous_tensor(op: &'static str, tensor: &Tensor) -> LinalgResult<()> {
    ensure_supported_tensor(op, tensor)?;

    if !tensor.is_contiguous() {
        return Err(NonContiguousTensorError { op });
    }

    Ok(())
}

/// 校验两个 Tensor 的 dtype、device、layout 是否一致。
pub(crate) fn ensure_binary_compatible(
    op: &'static str,
    lhs: &Tensor,
    rhs: &Tensor,
) -> LinalgResult<()> {
    ensure_contiguous_tensor(op, lhs)?;
    ensure_contiguous_tensor(op, rhs)?;

    if lhs.dtype() != rhs.dtype() {
        return Err(ShapeMismatchError {
            op,
            expected: format!("matching Tensor dtypes, lhs.dtype={}", lhs.dtype()),
            actual: format!("rhs.dtype={}", rhs.dtype()),
        });
    }

    if lhs.device() != rhs.device() {
        return Err(UnsupportedDeviceError {
            op,
            device: format!("lhs.device={}, rhs.device={}", lhs.device(), rhs.device()),
        });
    }

    if lhs.layout() != rhs.layout() {
        return Err(UnsupportedDTypeError {
            op,
            dtype: format!("lhs.layout={}, rhs.layout={}", lhs.layout(), rhs.layout()),
        });
    }

    Ok(())
}

/// 使用前向结果和反向规则创建可微输出 Tensor。只有当任一父 Tensor 需要梯度时才会记录计算图。
pub(crate) fn make_tensor_with_backward(
    op: &'static str,
    storage: Storage,
    shape: Vec<usize>,
    like: &Tensor,
    parents: Vec<Tensor>,
    backward: Arc<dyn BackwardFn>,
) -> LinalgResult<Tensor> {
    let meta = TensorMeta::new(shape, storage.dtype(), like.device().clone(), like.layout())
        .map_err(|_| TensorCreateError {
            op,
            message: "arcqml-core could not create the output tensor".to_string(),
        })?;

    Tensor::from_operation_named(op, storage, meta, parents, backward).map_err(|_| {
        TensorCreateError {
            op,
            message: "arcqml-core could not create the output tensor".to_string(),
        }
    })
}

/// 根据输出线性位置计算广播输入线性位置。
pub(crate) fn broadcast_index(out_index: usize, out_shape: &[usize], in_shape: &[usize]) -> usize {
    if in_shape.is_empty() {
        return 0;
    }

    let mut remaining = out_index;
    let mut in_index = 0usize;
    let mut in_stride = 1usize;

    for axis_from_right in 0..out_shape.len() {
        let out_axis = out_shape.len() - 1 - axis_from_right;
        let out_dim = out_shape[out_axis];
        let coord = remaining % out_dim;
        remaining /= out_dim;

        if axis_from_right < in_shape.len() {
            let in_axis = in_shape.len() - 1 - axis_from_right;
            let in_dim = in_shape[in_axis];
            let in_coord = if in_dim == 1 { 0 } else { coord };

            in_index += in_coord * in_stride;
            in_stride *= in_dim;
        }
    }

    in_index
}
