use thiserror::Error;

/// 优化器构造、状态恢复和更新操作使用的结果类型。
pub type OptimResult<T> = Result<T, OptimError>;

/// 优化器超参数、梯度或内部状态无效。
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OptimError {
    /// 优化器超参数不是有限值或超出允许范围。
    #[error("invalid hyperparameter: {name} has invalid value {value}")]
    InvalidHyperParameterError {
        /// 无效超参数的名称。
        name: &'static str,
        /// 调用方提供的值。
        value: f64
    },

    /// 梯度元素数量与参数不一致。
    #[error(
        "gradient shape mismatch: parameter {name} has {parameter_numel} elements, gradient has {grad_numel}"
    )]
    GradShapeMismatchError {
        /// 参数名称；未命名 Tensor 使用实现生成的标识。
        name: String,
        /// 参数元素数量。
        parameter_numel: usize,
        /// 梯度元素数量。
        grad_numel: usize,
    },

    /// 梯度和参数的数据类型不同。
    #[error(
        "gradient dtype mismatch: parameter {name} has dtype {parameter_dtype}, gradient has dtype {grad_dtype}"
    )]
    GradDTypeMismatchError {
        /// 数据类型不匹配的参数名称。
        name: String,
        /// 参数数据类型名称。
        parameter_dtype: String,
        /// 梯度数据类型名称。
        grad_dtype: String,
    },

    /// 参数数据类型不支持原地优化。
    #[error("unsupported parameter dtype: parameter {name} has dtype {dtype}")]
    UnsupportedParameterDTypeError {
        /// 数据类型不受支持的参数名称。
        name: String,
        /// 参数数据类型名称。
        dtype: String
    },

    /// 无法根据更新后的数值构造 Tensor。
    #[error(
        "tensor creation error: failed to create updated tensor for parameter {name}: {message}"
    )]
    TensorCreateError {
        /// 更新失败的参数名称。
        name: String,
        /// 底层 Tensor 构造错误信息。
        message: String
    },

    /// 导入的 Adam 状态与当前参数集合、形状或数据类型不兼容。
    #[error("Adam state does not match current parameters: {message}")]
    AdamStateMismatchError {
        /// 不兼容状态的具体说明。
        message: String
    },
}
