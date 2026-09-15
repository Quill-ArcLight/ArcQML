use super::is_grad_enabled;
use crate::{ArcQmlError, DType, Result, Storage, Tensor};

use num_complex::Complex64;
use std::collections::{HashMap, HashSet};
use std::ops::Add;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// 判断一次前向操作是否应该创建自动微分图节点。
pub(crate) fn should_record(parents: &[Tensor]) -> bool {
    // 仅在梯度模式开启且至少一个父 Tensor 需要梯度时记录计算图。
    is_grad_enabled() && parents.iter().any(Tensor::requires_grad)
}

/// 从输出 Tensor 开始执行反向传播。
pub(crate) fn backward(
    output: &Tensor,
    gradient: Option<Tensor>,
    retain_graph: bool,
) -> Result<()> {
    if !output.requires_grad() {
        return Err(ArcQmlError::AutogradError(
            "cannot call backward on a tensor that does not require gradients".to_string(),
        ));
    }

    let output_autograd = output.autograd();
    let output_meta = output_autograd.read().unwrap();

    // 非叶子节点缺少 grad_fn，表示上一次反向传播已经释放了计算图。
    if !output_meta.is_leaf() && output_meta.grad_fn().is_none() {
        return Err(ArcQmlError::AutogradError(
            "cannot call backward because the computation graph was already freed".to_string(),
        ));
    }

    // 后续遍历还会访问自动微分元数据，因此先释放读锁。
    drop(output_meta);

    let gradient = match gradient {
        Some(gradient) => gradient,
        // 未显式传入上游梯度时，仅允许标量输出，并以 1 作为初始梯度。
        None => ones_for_scalar(output)?,
    };

    validate_gradient(output, &gradient)?;

    let mut topology = Vec::new();
    let mut visited = HashSet::new();
    collect_topology(output, &mut visited, &mut topology);

    let mut pending_gradients = HashMap::new();
    accumulate_pending(&mut pending_gradients, output, gradient)?;

    for tensor in topology.into_iter().rev() {
        let Some(grad_output) = pending_gradients.remove(&tensor.node_id()) else {
            continue;
        };

        let (is_leaf, retain_grad, node) = {
            let autograd_handle = tensor.autograd();
            let autograd = autograd_handle.read().unwrap();
            (
                autograd.is_leaf(),
                autograd.retain_grad_enabled(),
                autograd.grad_fn(),
            )
        };

        if is_leaf || retain_grad {
            accumulate_persistent_grad(&tensor, grad_output.clone())?;
        }

        let Some(node) = node else {
            continue;
        };

        for (parent, parent_grad) in node.backward(&grad_output)? {
            if parent.requires_grad() {
                validate_gradient(&parent, &parent_grad)?;
                accumulate_pending(&mut pending_gradients, &parent, parent_grad)?;
            }
        }

        if !retain_graph {
            tensor.autograd().write().unwrap().clear_grad_fn();
        }
    }

    Ok(())
}

/// 迭代 DFS 收集输出 Tensor 可达的图节点，避免深计算图耗尽线程调用栈。
fn collect_topology(tensor: &Tensor, visited: &mut HashSet<usize>, topology: &mut Vec<Tensor>) {
    // 标记为 true 表示父节点已经入栈，可将当前节点按后序顺序写入 topology。
    let mut stack = vec![(tensor.clone(), false)];

    while let Some((current, parents_queued)) = stack.pop() {
        if parents_queued {
            topology.push(current);
            continue;
        }

        // 每个节点只访问一次，避免共享子图导致重复遍历。
        if !visited.insert(current.node_id()) {
            continue;
        }

        // 先压入当前节点的完成标记，再压入父节点，以保持后序遍历顺序。
        stack.push((current.clone(), true));

        let autograd = current.autograd();
        let node = autograd.read().unwrap().grad_fn();
        if let Some(node) = node {
            for parent in node.parents().into_iter().rev() {
                if parent.requires_grad() && !visited.contains(&parent.node_id()) {
                    stack.push((parent, false));
                }
            }
        }
    }
}

/// 为标量输出构造默认的上游梯度 1。
fn ones_for_scalar(output: &Tensor) -> Result<Tensor> {
    // Rust 标量 Tensor 的形状为空切片。
    if !output.shape().is_empty() {
        return Err(ArcQmlError::AutogradError(
            "backward without an explicit gradient requires a scalar tensor".to_string(),
        ));
    }

    let storage = match output.dtype() {
        DType::F32 => Storage::F32(vec![1.0]),
        DType::F64 => Storage::F64(vec![1.0]),
        DType::C64 => Storage::C64(vec![Complex64::new(1.0, 0.0)]),
        DType::I64 | DType::Bool => {
            return Err(ArcQmlError::AutogradError(format!(
                "dtype {} does not support gradients",
                output.dtype()
            )));
        }
    };

    Tensor::from_storage_meta(storage, output.meta().clone())
}

/// 验证一个梯度是否可以附加到目标 Tensor。
fn validate_gradient(tensor: &Tensor, gradient: &Tensor) -> Result<()> {
    if tensor.shape() != gradient.shape() {
        return Err(ArcQmlError::AutogradError(format!(
            "gradient shape mismatch: tensor shape {:?}, gradient shape {:?}",
            tensor.shape(),
            gradient.shape()
        )));
    }

    if tensor.dtype() != gradient.dtype() {
        return Err(ArcQmlError::DTypeMismatchError {
            expected: tensor.dtype(),
            actual: gradient.dtype(),
        });
    }

    if tensor.device() != gradient.device() {
        return Err(ArcQmlError::AutogradError(
            "gradient device does not match tensor".to_string(),
        ));
    }

    if tensor.layout() != gradient.layout() {
        return Err(ArcQmlError::AutogradError(
            "gradient layout does not match tensor".to_string(),
        ));
    }

    Ok(())
}

/// 将一条路径传来的梯度累计到临时梯度表中。
fn accumulate_pending(
    pending: &mut HashMap<usize, Tensor>,
    tensor: &Tensor,
    gradient: Tensor,
) -> Result<()> {
    let id = tensor.node_id();

    if let Some(existing) = pending.remove(&id) {
        pending.insert(id, add_gradients(&existing, &gradient)?);
    } else {
        pending.insert(id, gradient);
    }

    Ok(())
}

/// 将叶子节点或要求保留的非叶子节点的梯度写回 AutogradMeta。
fn accumulate_persistent_grad(tensor: &Tensor, gradient: Tensor) -> Result<()> {
    let current = tensor.grad();
    let accumulated = match current {
        Some(current) => add_gradients(&current, &gradient)?,
        None => gradient,
    };

    tensor.set_grad(accumulated);

    Ok(())
}

/// 对两个形状、dtype、设备和布局相同的梯度做逐元素相加。
pub(crate) fn add_gradients(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    validate_gradient(lhs, rhs)?;

    let lhs_storage = lhs.storage();
    let rhs_storage = rhs.storage();

    let storage = match (&*lhs_storage, &*rhs_storage) {
        (Storage::F32(a), Storage::F32(b)) => Storage::F32(add_gradient_values(a, b)),
        (Storage::F64(a), Storage::F64(b)) => Storage::F64(add_gradient_values(a, b)),
        (Storage::C64(a), Storage::C64(b)) => Storage::C64(add_gradient_values(a, b)),
        _ => {
            return Err(ArcQmlError::AutogradError(format!(
                "dtype {} does not support gradient accumulation",
                lhs.dtype()
            )));
        }
    };

    Tensor::from_storage_meta(storage, lhs.meta().clone())
}

/// 添加两个大小相同的梯度缓冲区。小型缓冲区保持串行，以避免在单个标量和门梯度上的 Rayon 调度成本。
#[cfg(feature = "parallel")]
fn add_gradient_values<T>(lhs: &[T], rhs: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T> + Send + Sync,
{
    const PARALLEL_MIN_ELEMENTS: usize = 8 * 1024;

    if lhs.len() < PARALLEL_MIN_ELEMENTS {
        return lhs
            .iter()
            .zip(rhs)
            .map(|(left, right)| *left + *right)
            .collect();
    }

    lhs.par_iter()
        .zip(rhs.par_iter())
        .map(|(left, right)| *left + *right)
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn add_gradient_values<T>(lhs: &[T], rhs: &[T]) -> Vec<T>
where
    T: Copy + Add<Output = T>,
{
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| *left + *right)
        .collect()
}
