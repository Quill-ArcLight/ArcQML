use thiserror::Error;

/// 酉矩阵构造、验证和拟合操作使用的结果类型。
pub type UnitaryResult<T> = Result<T, UnitaryError>;

/// 电路演化、稠密矩阵、容差或拟合输入无效。
#[derive(Debug, Clone, PartialEq, Error)]
pub enum UnitaryError {
    /// 底层电路读取或验证失败。
    #[error("circuit error: {message}")]
    CircuitError {
        /// 底层电路错误信息。
        message: String
    },

    /// 底层状态向量内核失败。
    #[error("kernel error: {message}")]
    KernelError {
        /// 底层内核错误信息。
        message: String
    },

    /// `2^num_qubits` 或 `4^num_qubits` 无法用 `usize` 表示。
    #[error("dense unitary dimension overflow for {num_qubits} qubits")]
    DimensionOverflowError {
        /// 导致维度溢出的量子比特数量。
        num_qubits: usize
    },

    /// 无法为整体稠密酉矩阵预留内存。
    #[error(
        "could not allocate {elements} complex amplitudes for the dense unitary of {num_qubits} qubits"
    )]
    AllocationError {
        /// 电路的量子比特数量。
        num_qubits: usize,
        /// 计划分配的复数元素数量。
        elements: usize
    },

    /// 构造或读取 Tensor 失败。
    #[error("tensor error: {message}")]
    TensorError {
        /// 底层 Tensor 错误信息。
        message: String
    },

    /// 稠密酉矩阵维度为零或不是二的幂。
    #[error("unitary dimension must be a nonzero power of two, got {dimension}")]
    InvalidUnitaryDimensionError {
        /// 无效的矩阵行列维度。
        dimension: usize
    },

    /// 输入 Tensor 不是方形二维矩阵。
    #[error("unitary matrix must be a square rank-2 tensor, got shape {shape:?}")]
    InvalidMatrixShapeError {
        /// 输入 Tensor 的实际形状。
        shape: Vec<usize>
    },

    /// 输入矩阵不是 `C64` Tensor。
    #[error("unitary matrix must use C64 values, got {actual}")]
    InvalidMatrixDTypeError {
        /// 实际数据类型名称。
        actual: String
    },

    /// row-major 缓冲区长度与矩阵维度不一致。
    #[error("unitary matrix length mismatch: expected {expected}, got {actual}")]
    InvalidMatrixLengthError {
        /// 方阵要求的元素数量。
        expected: usize,
        /// 实际缓冲区元素数量。
        actual: usize
    },

    /// 矩阵包含 NaN 或无穷复数分量。
    #[error("unitary matrix contains non-finite values")]
    NonFiniteMatrixError,

    /// 矩阵的 `U†U` 与单位矩阵的偏差超过容差。
    #[error(
        "matrix is not unitary: maximum deviation {max_deviation} exceeds tolerance {tolerance}"
    )]
    NonUnitaryMatrixError {
        /// 所有矩阵元素中的最大绝对偏差。
        max_deviation: f64,
        /// 调用方指定的酉性容差。
        tolerance: f64
    },

    /// 酉性校验容差为负数、NaN 或无穷值。
    #[error("tolerance must be finite and nonnegative, got {tolerance}")]
    InvalidToleranceError {
        /// 调用方提供的无效容差。
        tolerance: f64
    },

    /// 拟合过程引用了电路范围之外的操作。
    #[error("operation index {index} is outside a circuit with {operation_count} operations")]
    InvalidOperationIndexError {
        /// 无效的零起始操作下标。
        index: usize,
        /// 电路实际包含的操作数量。
        operation_count: usize,
    },

    /// 目标酉矩阵与拟合电路的量子比特数不同。
    #[error("target has {target} qubits but circuit has {circuit} qubits")]
    QubitCountMismatchError {
        /// 目标酉矩阵的量子比特数。
        target: usize,
        /// 拟合电路的量子比特数。
        circuit: usize
    },

    /// 电路参数数据类型不支持酉矩阵梯度。
    #[error("parameter {index} has unsupported dtype {actual}")]
    UnsupportedParameterDTypeError {
        /// 参数表中的零起始下标。
        index: usize,
        /// 参数的实际数据类型名称。
        actual: String
    },
}
