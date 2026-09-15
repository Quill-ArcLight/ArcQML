use super::Tensor;
use crate::{ArcQmlError, Result, Storage};

impl Tensor {
    /// 读取仅含一个 F32 或 F64 元素的 Tensor，并统一转换为 f64。
    ///
    /// 此方法仅用于记录或输出数值；若随后继续参与可微计算，应保持使用原 Tensor。
    ///
    /// # Errors
    ///
    /// 当 Tensor 不是单元素 `F32` 或 `F64` 标量时返回错误。
    pub fn value(&self) -> Result<f64> {
        if self.numel() != 1 {
            return Err(ArcQmlError::InvalidOperationError(format!(
                "Tensor::value requires exactly one element, got shape {:?}",
                self.shape()
            )));
        }

        let offset = self.offset();
        match &*self.storage() {
            Storage::F32(values) => values
                .get(offset)
                .map(|value| f64::from(*value))
                .ok_or_else(|| {
                    ArcQmlError::InvalidOperationError(
                        "Tensor::value could not read the scalar from F32 storage".to_string(),
                    )
                }),
            Storage::F64(values) => values.get(offset).copied().ok_or_else(|| {
                ArcQmlError::InvalidOperationError(
                    "Tensor::value could not read the scalar from F64 storage".to_string(),
                )
            }),
            storage => Err(ArcQmlError::InvalidOperationError(format!(
                "Tensor::value supports only F32 or F64 tensors, got {}",
                storage.dtype()
            ))),
        }
    }
}
