use super::Tensor;
use super::view::{scatter_gradient_to_meta, storage_index_for_linear};
use crate::{BackwardFn, Result, Storage, TensorMeta};

use std::sync::Arc;

impl Tensor {
    /// 返回逻辑元素顺序相同、存储为连续行主序布局的 Tensor。
    ///
    /// 已连续时直接返回共享句柄的克隆；否则复制逻辑元素，并在需要时记录可微节点。
    ///
    /// # Errors
    ///
    /// 当 Tensor 元数据无效、存储范围不匹配或无法构造连续副本时返回错误。
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let output_meta = TensorMeta::new(
            self.shape().to_vec(),
            self.dtype(),
            self.device().clone(),
            self.layout(),
        )?;
        let storage = copy_logical_storage(self)?;

        // 连续化会复制数据，因此必须通过可微操作记录父节点。
        Tensor::from_operation_named(
            "contiguous",
            storage,
            output_meta,
            vec![self.clone()],
            Arc::new(ContiguousBackward {
                parent_meta: self.meta().clone(),
            }),
        )
    }
}

/// 连续化的反向规则：将连续梯度散射回原 Tensor 的逻辑布局。
#[derive(Debug)]
struct ContiguousBackward {
    parent_meta: TensorMeta,
}

impl BackwardFn for ContiguousBackward {
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(crate::ArcQmlError::AutogradError(
                "contiguous backward expects one parent".to_string(),
            ));
        }

        let parent = &parents[0];

        if grad_output.shape() != parent.shape() {
            return Err(crate::ArcQmlError::ShapeError(format!(
                "contiguous gradient shape mismatch: expected {:?}, got {:?}",
                parent.shape(),
                grad_output.shape()
            )));
        }

        // 输出梯度是连续布局，将其按父 Tensor 的 stride 与 offset 写回。
        let storage = scatter_gradient_to_meta(grad_output, &self.parent_meta, &self.parent_meta)?;
        let gradient = Tensor::from_storage_meta(storage, self.parent_meta.clone())?;

        Ok(vec![Some(gradient)])
    }
}

/// 按 Tensor 的逻辑行主序读取元素，生成独立且连续的存储。
fn copy_logical_storage(input: &Tensor) -> Result<Storage> {
    let source = input.storage();

    // 根据传入的数据 dtype 自动匹配
    macro_rules! copy_values {
        ($values:expr, $variant:ident) => {{
            let mut output = Vec::with_capacity(input.numel());

            for linear_index in 0..input.numel() {
                let storage_index = storage_index_for_linear(input.meta(), linear_index)?;
                output.push($values[storage_index].clone());
            }

            Ok(Storage::$variant(output))
        }};
    }

    match &*source {
        Storage::F32(values) => copy_values!(values, F32),
        Storage::F64(values) => copy_values!(values, F64),
        Storage::C64(values) => copy_values!(values, C64),
        Storage::I64(values) => copy_values!(values, I64),
        Storage::Bool(values) => copy_values!(values, Bool),
    }
}
