use arcqml_core::DType;
use thiserror::Error;

/// 损失函数使用的结果类型。
pub type LossResult<T> = Result<T, LossError>;

/// 损失函数的输入验证或 Tensor 运算错误。
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LossError {
    /// 无法构造损失计算所需的常量 Tensor。
    #[error("tensor creation error: {message}")]
    TensorCreateError {
        /// 底层 Tensor 构造错误信息。
        message: String
    },

    /// 输入 Tensor 不是损失函数支持的实数类型。
    #[error("tensor loss supports only f32 and f64, got {dtype}")]
    UnsupportedTensorDType {
        /// 实际收到的数据类型。
        dtype: DType
    },

    /// 输入 Tensor 不包含任何元素。
    #[error("tensor loss does not support empty tensors")]
    EmptyTensorError,

    /// 底层可微 Tensor 运算失败。
    #[error("tensor operation error: {message}")]
    TensorOperationError {
        /// 底层运算或输入约束错误信息。
        message: String
    },
}
