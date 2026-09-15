use arcqml_kernel::KernelError;
use arcqml_observable::ObservableError;
use thiserror::Error;

/// 状态向量模拟和测量操作使用的结果类型。
pub type SimResult<T> = Result<T, SimError>;

/// 模拟器状态、量子比特、门、可观测量或抽样参数无效。
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SimError {
    /// 模拟器的量子比特数为零。
    #[error("invalid system size: num_qubits must be greater than zero")]
    EmptyQubitError,

    /// 单态振幅数量不是 `2^num_qubits`。
    #[error(
        "invalid tensor state length: expected {expected} amplitudes for {num_qubits} qubits, got {actual}"
    )]
    InvalidStateLengthError {
        /// 目标系统的量子比特数。
        num_qubits: usize,
        /// 计算出的正确振幅数量。
        expected: usize,
        /// 实际振幅数量。
        actual: usize,
    },

    /// 单态振幅的模平方和不在归一化容差内。
    #[error("tensor state is not normalized: expected norm squared close to 1, got {norm_sqr}")]
    NonNormalizedStateError {
        /// 实际计算得到的模平方和。
        norm_sqr: f64
    },

    /// batch 大小为零。
    #[error("invalid batch size: batch_size must be greater than zero")]
    EmptyBatchError,

    /// 抽样次数为零。
    #[error("invalid shot count: shots must be greater than zero")]
    InvalidShotsError,

    /// batch Tensor 不是 `[batch_size, 2^num_qubits]` 形状。
    #[error("invalid batch tensor state shape: expected [batch_size, {expected}], got {actual:?}")]
    BatchTensorStateShapeError {
        /// 每个 batch 状态应包含的振幅数量。
        expected: usize,
        /// 实际 Tensor 形状。
        actual: Vec<usize>
    },

    /// batch 中某个状态没有归一化。
    #[error(
        "batch tensor state at index {batch_index} is not normalized: expected norm squared close to 1, got {norm_sqr}"
    )]
    NonNormalizedBatchStateError {
        /// 未归一化状态的 batch 下标。
        batch_index: usize,
        /// 该状态实际计算得到的模平方和。
        norm_sqr: f64
    },

    /// 门或分析请求引用了范围之外的量子比特。
    #[error("qubit index out of range: system has {num_qubits} qubits, got {index}")]
    QubitOutOfRangeError {
        /// 无效的零起始量子比特下标。
        index: usize,
        /// 模拟器实际包含的量子比特数。
        num_qubits: usize
    },

    /// 电路与模拟器状态的量子比特数不同。
    #[error(
        "qubit count mismatch: tensor state has {state_qubits} qubits, circuit has {circuit_qubits}"
    )]
    CircuitQubitMismatchError {
        /// 模拟器状态的量子比特数。
        state_qubits: usize,
        /// 电路的量子比特数。
        circuit_qubits: usize,
    },

    /// 可观测量与模拟器状态的量子比特数不同。
    #[error(
        "qubit count mismatch: tensor state has {state_qubits} qubits, observable has {observable_qubits}"
    )]
    ObservableQubitMismatchError {
        /// 模拟器状态的量子比特数。
        state_qubits: usize,
        /// 可观测量的量子比特数。
        observable_qubits: usize,
    },

    /// 执行前仍存在没有解析为标量值的门参数。
    #[error("unbound parameter: gate {gate} contains parameter {parameter}")]
    UnboundParameterError {
        /// 包含未绑定参数的门名称。
        gate: String,
        /// 未绑定参数的显示值。
        parameter: String
    },

    /// 电路验证或参数读取失败。
    #[error("circuit error: {message}")]
    CircuitError {
        /// 底层电路错误信息。
        message: String
    },

    /// 初始状态 Tensor 不是 `C64`。
    #[error("invalid tensor state dtype: expected C64, got {dtype}")]
    TensorStateDTypeError {
        /// 实际数据类型名称。
        dtype: String
    },

    /// 单态 Tensor 不是一维 `[2^num_qubits]` 形状。
    #[error("invalid tensor state shape: expected [{expected}], got {actual:?}")]
    TensorStateShapeError {
        /// 正确的一维振幅数量。
        expected: usize,
        /// 实际 Tensor 形状。
        actual: Vec<usize>
    },

    /// `2^num_qubits` 无法用 `usize` 表示。
    #[error("tensor state dimension overflow for {num_qubits} qubits")]
    StateDimensionOverflowError {
        /// 导致维度溢出的量子比特数量。
        num_qubits: usize
    },

    /// 初始状态 Tensor 是非连续视图。
    #[error("tensor state must use contiguous storage")]
    NonContiguousTensorStateError,

    /// 初始状态 Tensor 不是 CPU 稠密布局。
    #[error("tensor state must use CPU Dense layout")]
    UnsupportedTensorStateLayoutError,

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

    /// 同一个门操作重复引用某个量子比特。
    #[error("duplicate qubit in gate operation: qubit {index}")]
    DuplicateQubitError {
        /// 重复的量子比特下标。
        index: usize
    },

    /// 底层 Tensor 构造或自动微分操作失败。
    #[error("tensor error: {message}")]
    TensorError {
        /// 底层 Tensor 错误信息。
        message: String
    },

    /// 可观测量构造或求值失败。
    #[error("observable error: {message}")]
    ObservableError {
        /// 底层可观测量错误信息。
        message: String
    },
}

/// 将共享门 kernel 的错误映射为模拟器兼容错误。
impl From<KernelError> for SimError {
    fn from(error: KernelError) -> Self {
        match error {
            KernelError::CircuitError { message } => Self::CircuitError { message },
            KernelError::UnboundParameterError { gate, parameter } => {
                Self::UnboundParameterError { gate, parameter }
            }
            KernelError::QubitOutOfRangeError { index, num_qubits } => {
                Self::QubitOutOfRangeError { index, num_qubits }
            }
            KernelError::GateArityError {
                gate,
                expected,
                actual,
            } => Self::GateArityError {
                gate,
                expected,
                actual,
            },
            KernelError::DuplicateQubitError { index } => Self::DuplicateQubitError { index },
            KernelError::StateDimensionOverflowError { num_qubits } => {
                Self::StateDimensionOverflowError { num_qubits }
            }
            KernelError::StateLengthError {
                num_qubits,
                expected,
                actual,
            } => Self::InvalidStateLengthError {
                num_qubits,
                expected,
                actual,
            },
            KernelError::EmptyBatchError => Self::EmptyBatchError,
            KernelError::BatchStateLengthError { .. } => Self::TensorError {
                message: error.to_string(),
            },
        }
    }
}

/// 将可观测量模块的错误映射为模拟器公开错误。
impl From<ObservableError> for SimError {
    fn from(error: ObservableError) -> Self {
        Self::ObservableError {
            message: error.to_string(),
        }
    }
}
