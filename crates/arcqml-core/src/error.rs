use crate::DType;

use thiserror::Error;

/// ArcQML 核心 Tensor 和自动微分操作使用的结果类型。
pub type Result<T> = std::result::Result<T, ArcQmlError>;

/// Tensor 元数据、存储、操作或自动微分状态无效。
#[derive(Debug, Error)]
pub enum ArcQmlError {
    /// 操作要求的数据类型与实际 Tensor 数据类型不同。
    #[error("dtype mismatch, expected {expected:?}, got {actual:?}")]
    DTypeMismatchError {
        /// 操作要求的数据类型。
        expected: DType,
        /// 实际收到的数据类型。
        actual: DType
    },

    /// 请求的设备、布局、数据类型或操作尚未实现。
    #[error("feature not implemented: {0}")]
    NotImplementedError(String),

    /// 操作与 Tensor 当前状态或输入约束不兼容。
    #[error("invalid operation: {0}")]
    InvalidOperationError(String),

    /// 形状、步长、元素数量或广播关系无效。
    #[error("shape error: {0}")]
    ShapeError(String),

    /// 自动微分图、上游梯度或反向传播状态无效。
    #[error("autograd error: {0}")]
    AutogradError(String),
}
