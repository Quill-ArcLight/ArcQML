use thiserror::Error;

/// 参数绑定和状态向量内核使用的结果类型。
pub type KernelResult<T> = Result<T, KernelError>;

/// 参数绑定、量子门或状态向量维度无效。
#[derive(Debug, Clone, PartialEq, Error)]
pub enum KernelError {
    /// 底层电路参数读取或验证失败。
    #[error("circuit error: {message}")]
    CircuitError {
        /// 底层电路错误信息。
        message: String
    },

    /// 执行前仍存在没有解析为标量值的门参数。
    #[error("unbound parameter: gate {gate} contains parameter {parameter}")]
    UnboundParameterError {
        /// 包含未绑定参数的门名称。
        gate: String,
        /// 未绑定参数的显示值。
        parameter: String
    },

    /// 门引用了系统范围之外的量子比特。
    #[error("qubit index out of range: system has {num_qubits} qubits, got {index}")]
    QubitOutOfRangeError {
        /// 无效量子比特下标。
        index: usize,
        /// 系统实际包含的量子比特数。
        num_qubits: usize
    },

    /// 操作提供的量子比特数量与门的 arity 不同。
    #[error("gate arity mismatch: gate {gate} expects {expected} qubits, got {actual}")]
    GateArityError {
        /// 量子门名称。
        gate: String,
        /// 门要求的量子比特数量。
        expected: usize,
        /// 实际提供的数量。
        actual: usize,
    },

    /// 同一门操作重复引用某个量子比特。
    #[error("duplicate qubit in gate operation: qubit {index}")]
    DuplicateQubitError {
        /// 重复的量子比特下标。
        index: usize
    },

    /// `2^num_qubits` 无法用 `usize` 表示。
    #[error("state dimension overflow for {num_qubits} qubits")]
    StateDimensionOverflowError {
        /// 导致维度溢出的量子比特数量。
        num_qubits: usize
    },

    /// 单态振幅缓冲区长度不是 `2^num_qubits`。
    #[error(
        "invalid state length: expected {expected} amplitudes for {num_qubits} qubits, got {actual}"
    )]
    StateLengthError {
        /// 目标系统的量子比特数。
        num_qubits: usize,
        /// 计算出的正确振幅数量。
        expected: usize,
        /// 实际缓冲区长度。
        actual: usize,
    },

    /// batch 大小为零。
    #[error("invalid batch size: batch_size must be greater than zero")]
    EmptyBatchError,

    /// batch 振幅缓冲区长度不是 `batch_size * 2^num_qubits`。
    #[error(
        "invalid batched state length: expected {expected} amplitudes for batch size {batch_size} and {num_qubits} qubits, got {actual}"
    )]
    BatchStateLengthError {
        /// batch 中的状态数量。
        batch_size: usize,
        /// 每个状态的量子比特数。
        num_qubits: usize,
        /// 计算出的正确振幅总数。
        expected: usize,
        /// 实际缓冲区长度。
        actual: usize,
    },
}
