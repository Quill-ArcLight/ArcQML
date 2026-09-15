use super::Tensor;
use crate::ArcQmlError::ShapeError;
use crate::tensordata::checked_numel;
use crate::{Result, TensorMeta};

impl Tensor {
    /// 返回具有目标形状且保持行主序逻辑元素顺序的 Tensor。
    ///
    /// 连续输入共享原存储；非连续输入会先通过 [`Tensor::contiguous`] 复制为连续存储。
    ///
    /// # Errors
    ///
    /// 当目标形状计算溢出、变形前后元素数不一致，或非连续输入无法连续化时返回错误。
    pub fn reshape(&self, shape: Vec<usize>) -> Result<Self> {
        let requested_numel = checked_numel(&shape)?;

        if requested_numel != self.numel() {
            return Err(ShapeError(format!(
                "cannot reshape tensor with {} elements to shape {:?}",
                self.numel(),
                shape
            )));
        }

        let input = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()? // 非连续 Tensor 先构造为连续副本
        };
        let output_meta =
            TensorMeta::new(shape, input.dtype(), input.device().clone(), input.layout())?;

        // reshape 只调整 TensorMeta，直接复用 view 的共享 Storage 与反向规则
        input.view_with_meta(output_meta)
    }
}
