use thiserror::Error;

/// 线性代数和数值算子使用的结果类型。
pub type LinalgResult<T> = Result<T, LinalgError>;

/// 算子输入形状、数据类型、设备、存储或数值范围无效。
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LinalgError {
    /// 一个或多个输入形状不满足算子约束。
    #[error("shape mismatch in {op}: expected {expected}, got {actual}")]
    ShapeMismatchError {
        /// 报错的算子名称。
        op: &'static str,
        /// 算子要求的形状关系。
        expected: String,
        /// 实际收到的形状。
        actual: String,
    },

    /// Tensor 秩不满足算子约束。
    #[error("invalid dimension for {op}: expected {expected}, got rank {actual}")]
    InvalidDimensionError {
        /// 报错的算子名称。
        op: &'static str,
        /// 算子要求的秩。
        expected: String,
        /// 实际 Tensor 秩。
        actual: usize,
    },

    /// 归约或归一化轴超出 Tensor 秩。
    #[error("invalid axis for {op}: axis={axis} is out of range for rank={rank}")]
    InvalidAxisError {
        /// 报错的算子名称。
        op: &'static str,
        /// 调用方提供的轴下标。
        axis: usize,
        /// 输入 Tensor 秩。
        rank: usize,
    },

    /// reshape 前后元素数量不同或发生溢出。
    #[error(
        "invalid reshape: cannot reshape {from_numel} elements from shape {from:?} to {to_numel} elements with shape {to:?}"
    )]
    InvalidReshapeError {
        /// 输入形状。
        from: Vec<usize>,
        /// 请求的输出形状。
        to: Vec<usize>,
        /// 输入元素数量。
        from_numel: usize,
        /// 输出形状要求的元素数量。
        to_numel: usize,
    },

    /// 输入数据类型不受指定算子支持。
    #[error("unsupported dtype for {op}: dtype={dtype}")]
    UnsupportedDTypeError {
        /// 报错的算子名称。
        op: &'static str,
        /// 实际数据类型名称。
        dtype: String
    },

    /// 输入所在设备不受指定算子支持。
    #[error("unsupported device for {op}: device={device}")]
    UnsupportedDeviceError {
        /// 报错的算子名称。
        op: &'static str,
        /// 实际设备名称。
        device: String
    },

    /// 算子要求连续存储，但输入是带步长的非连续视图。
    #[error("non-contiguous tensor is not supported for {op}")]
    NonContiguousTensorError {
        /// 报错的算子名称。
        op: &'static str
    },

    /// 低层内核缓冲区长度与形状不一致。
    #[error("invalid buffer length for {op}: expected {expected}, got {actual}")]
    InvalidBufferLengthError {
        /// 报错的内核名称。
        op: &'static str,
        /// 根据形状计算出的缓冲区长度。
        expected: usize,
        /// 实际缓冲区长度。
        actual: usize,
    },

    /// 归约或统计算子收到空输入。
    #[error("empty input is not supported for {op}")]
    EmptyInputError {
        /// 报错的算子名称。
        op: &'static str
    },

    /// `i64` 算术运算发生溢出。
    #[error("i64 overflow in {op} at index {index}")]
    IntegerOverflowError {
        /// 报错的算子名称。
        op: &'static str,
        /// 检测到溢出的输出元素下标。
        index: usize
    },

    /// 计算形状元素数量时发生 `usize` 乘法溢出。
    #[error("usize overflow in {op} while calculating {lhs} * {rhs}")]
    ShapeSizeOverflowError {
        /// 报错的算子名称。
        op: &'static str,
        /// 溢出乘法的左操作数。
        lhs: usize,
        /// 溢出乘法的右操作数。
        rhs: usize,
    },

    /// 根据算子结果构造 Tensor 失败。
    #[error("failed to create output tensor for {op}: {message}")]
    TensorCreateError {
        /// 报错的算子名称。
        op: &'static str,
        /// 底层 Tensor 构造错误信息。
        message: String
    },
}
