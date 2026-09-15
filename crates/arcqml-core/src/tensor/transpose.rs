use super::Tensor;
use crate::{Result, TensorMeta};

impl Tensor {
    /// 交换二维 Tensor 的两个轴，返回共享存储的转置视图。
    ///
    /// # Errors
    ///
    /// 当 Tensor 的秩不是二时返回错误。
    pub fn transpose(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(crate::ArcQmlError::ShapeError(format!(
                "transpose expects a 2-D tensor, got shape {:?}",
                self.shape()
            )));
        }

        self.transpose_dims(0, 1)
    }

    /// 交换任意两个轴，返回共享存储的视图；其他轴顺序保持不变。
    ///
    /// # Errors
    ///
    /// 当 `dim0` 或 `dim1` 不在 Tensor 秩的范围内时返回错误。
    pub fn transpose_dims(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.ndim() || dim1 >= self.ndim() {
            return Err(crate::ArcQmlError::ShapeError(format!(
                "transpose dimensions ({dim0}, {dim1}) are out of range for shape {:?}",
                self.shape()
            )));
        }

        if dim0 == dim1 {
            return Ok(self.clone());
        }

        let mut shape = self.shape().to_vec();
        let mut strides = self.strides().to_vec();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);

        let output_meta = TensorMeta::from_parts(
            shape,
            strides,
            self.offset(),
            self.dtype(),
            self.device().clone(),
            self.layout(),
        )?;
        // transpose 只调整 shape 与 stride，直接复用 view 的共享 Storage 与反向规则
        self.view_with_meta(output_meta)
    }
}
