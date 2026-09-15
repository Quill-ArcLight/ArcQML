use super::Tensor;
use crate::ArcQmlError::*;
use crate::autograd::{self, AutogradNode};
use crate::{AutogradMeta, BackwardFn, DType, Result, Storage, TensorMeta};

use num_complex::Complex64;
use std::sync::{Arc, RwLock};

impl Tensor {
    /// 使用指定元数据创建共享当前存储和版本计数的新逻辑视图。
    ///
    /// 若当前 Tensor 需要梯度，新视图会记录可微的视图节点；`meta` 的 dtype 必须
    /// 与存储一致，且描述的全部逻辑元素必须落在共享存储范围内。
    ///
    /// # Errors
    ///
    /// 当视图元数据与共享存储不兼容或访问范围越界时返回错误。
    pub fn view_with_meta(&self, meta: TensorMeta) -> Result<Self> {
        let storage = self.storage();

        if storage.dtype() != meta.dtype() {
            return Err(DTypeMismatchError {
                expected: meta.dtype(),
                actual: storage.dtype(),
            });
        }

        if storage.len() < meta.storage_len_required() {
            return Err(ShapeError(format!(
                "storage length insufficient for tensor elements: expected {}, got {}",
                storage.len(),
                meta.storage_len_required()
            )));
        }

        drop(storage);

        let parents = vec![self.clone()]; // 当前节点是 view 的唯一父节点
        let autograd = if autograd::should_record(&parents) {
            AutogradMeta::from_node(AutogradNode::new(
                "view",
                parents,
                Arc::new(ViewBackward {
                    parent_meta: self.meta.clone(),
                    view_meta: meta.clone(),
                }),
            ))
        } else {
            AutogradMeta::default()
        };

        Ok(Self {
            storage: Arc::clone(&self.storage),
            version: Arc::clone(&self.version),
            meta,
            autograd: Arc::new(RwLock::new(autograd)),
        })
    }
}

/// 视图操作的反向规则。
#[derive(Debug)]
struct ViewBackward {
    parent_meta: TensorMeta,
    view_meta: TensorMeta,
}

impl BackwardFn for ViewBackward {
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(AutogradError(
                "view backward expects exactly one parent".to_string(),
            ));
        }

        let parent = &parents[0];

        if grad_output.shape() != self.view_meta.shape() {
            return Err(ShapeError(format!(
                "view gradient shape mismatch: expected {:?}, got {:?}",
                self.view_meta.shape(),
                grad_output.shape()
            )));
        }

        if grad_output.dtype() != parent.dtype() {
            return Err(DTypeMismatchError {
                expected: parent.dtype(),
                actual: grad_output.dtype(),
            });
        }

        let storage = scatter_gradient_to_meta(grad_output, &self.parent_meta, &self.view_meta)?;
        let gradient = Tensor::from_storage_meta(storage, self.parent_meta.clone())?;

        Ok(vec![Some(gradient)])
    }
}

/// 将视图的逻辑梯度散射到与父 Tensor 相同的存储布局中。
pub(crate) fn scatter_gradient_to_meta(
    gradient: &Tensor,
    storage_meta: &TensorMeta,
    destination_meta: &TensorMeta,
) -> Result<Storage> {
    // Storage 的长度覆盖父 Tensor 的完整访问范围，目标元信息决定写入位置。
    let mut storage = zeros_storage(storage_meta.storage_len_required(), gradient.dtype())?;
    let gradient_storage = gradient.storage();

    match (&*gradient_storage, &mut storage) {
        (Storage::F32(source), Storage::F32(destination)) => {
            scatter_values(source, destination, gradient.meta(), destination_meta)?;
        }
        (Storage::F64(source), Storage::F64(destination)) => {
            scatter_values(source, destination, gradient.meta(), destination_meta)?;
        }
        (Storage::C64(source), Storage::C64(destination)) => {
            scatter_values(source, destination, gradient.meta(), destination_meta)?;
        }
        _ => {
            return Err(AutogradError(format!(
                "dtype {} does not support view gradients",
                gradient.dtype()
            )));
        }
    }

    Ok(storage)
}

/// 为可微 dtype 分配指定长度的零值梯度存储。
pub(crate) fn zeros_storage(len: usize, dtype: DType) -> Result<Storage> {
    match dtype {
        DType::F32 => Ok(Storage::F32(vec![0.0; len])),
        DType::F64 => Ok(Storage::F64(vec![0.0; len])),
        DType::C64 => Ok(Storage::C64(vec![Complex64::new(0.0, 0.0); len])),
        DType::I64 | DType::Bool => Err(AutogradError(format!(
            "dtype {} does not support gradients",
            dtype
        ))),
    }
}

/// 遍历视图逻辑索引，并将每个梯度值累加到父存储的对应位置。
fn scatter_values<T>(
    source: &[T],
    destination: &mut [T],
    source_meta: &TensorMeta,
    view_meta: &TensorMeta,
) -> Result<()>
where
    T: Copy + std::ops::AddAssign,
{
    for linear_index in 0..view_meta.numel() {
        let source_index = storage_index_for_linear(source_meta, linear_index)?;
        let destination_index = storage_index_for_linear(view_meta, linear_index)?;

        destination[destination_index] += source[source_index];
    }

    Ok(())
}

/// 将行主序逻辑线性下标转换为 [`TensorMeta`] 描述的底层存储下标。
pub(crate) fn storage_index_for_linear(meta: &TensorMeta, linear_index: usize) -> Result<usize> {
    if meta.shape().len() != meta.strides().len() {
        return Err(AutogradError(
            "tensor meta shape and strides must have the same rank".to_string(),
        ));
    }

    if linear_index >= meta.numel() {
        return Err(AutogradError(format!(
            "logical tensor index {} is out of range for shape {:?}",
            linear_index,
            meta.shape()
        )));
    }

    let mut remaining = linear_index;
    let mut index = meta.offset() as isize;

    for axis in (0..meta.shape().len()).rev() {
        let dim = meta.shape()[axis];
        let coordinate = remaining % dim;
        remaining /= dim;
        index += coordinate as isize * meta.strides()[axis];
    }

    if index < 0 {
        return Err(AutogradError(
            "view points to a negative storage index".to_string(),
        ));
    }

    Ok(index as usize)
}
