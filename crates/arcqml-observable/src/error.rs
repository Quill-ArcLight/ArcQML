use thiserror::Error;

/// Pauli 可观测量构造、化简和求值操作使用的结果类型。
pub type ObservableResult<T> = Result<T, ObservableError>;

/// Pauli 可观测量、量子比特选择或状态向量维度无效。
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ObservableError {
    /// 可观测量的量子比特数为零。
    #[error("num_qubits must be greater than 0")]
    EmptyQubitError,

    /// Pauli 操作引用了范围之外的量子比特。
    #[error("qubit index out of range: num_qubits={num_qubits}, index={index}")]
    QubitOutOfRangeError {
        /// 无效的零起始量子比特下标。
        index: usize,
        /// 可观测量实际包含的量子比特数。
        num_qubits: usize
    },

    /// 同一个 Pauli 字符串重复指定了某个量子比特。
    #[error("duplicate Pauli operation on qubit {index}")]
    DuplicateQubitError {
        /// 重复的量子比特下标。
        index: usize
    },

    /// 两个 Pauli 对象或状态的量子比特数不同。
    #[error("num_qubits mismatch: expected {expected}, got {actual}")]
    QubitCountMismatchError {
        /// 操作要求的量子比特数。
        expected: usize,
        /// 实际收到的量子比特数。
        actual: usize
    },

    /// Pauli 项系数是 NaN 或无穷值。
    #[error("Pauli operator coefficient must be finite, got {value}")]
    NonFiniteCoefficientError {
        /// 无效系数的显示值。
        value: String
    },

    /// 化简容差为负数、NaN 或无穷值。
    #[error("simplify tolerance must be finite and non-negative, got {value}")]
    InvalidToleranceError {
        /// 无效容差的显示值。
        value: String
    },

    /// JSON 内容无法解析为有效的稀疏 Pauli 和。
    #[error("cannot deserialize SparsePauliOp from JSON: {message}")]
    JsonDeserializationError {
        /// JSON 解析或结构校验错误信息。
        message: String
    },

    /// `2^num_qubits` 无法用 `usize` 表示。
    #[error("state-vector dimension overflow for {num_qubits} qubits")]
    StateDimensionOverflowError {
        /// 导致维度溢出的量子比特数量。
        num_qubits: usize
    },

    /// 状态向量长度与可观测量的量子比特数不匹配。
    #[error("state-vector length mismatch: expected {expected}, got {actual}")]
    StateVectorLengthMismatchError {
        /// 根据量子比特数计算出的长度。
        expected: usize,
        /// 实际振幅切片长度。
        actual: usize
    },

    /// batch 求值的批大小为零。
    #[error("batch size must be greater than zero")]
    EmptyBatchError,
}
