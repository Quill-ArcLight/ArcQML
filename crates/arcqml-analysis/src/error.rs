use arcqml_linalg::LinalgError;
use arcqml_sim::SimError;
use thiserror::Error;

/// 状态分析操作使用的结果类型。
pub type AnalysisResult<T> = Result<T, AnalysisError>;

/// 状态分析失败时返回的错误。
#[derive(Debug, Clone, PartialEq, Error)]
pub enum AnalysisError {
    /// 底层状态向量模拟器拒绝了状态读取或校验。
    #[error("statevector simulator error: {0}")]
    SimulatorError(#[from] SimError),

    /// 构造分析结果 Tensor 时发生线性代数错误。
    #[error("linear algebra error: {0}")]
    LinalgError(#[from] LinalgError),

    /// 两个待比较纯态的量子比特数不同。
    #[error("cannot compare states with different qubit counts: {lhs} and {rhs}")]
    QubitCountMismatchError {
        /// 左侧状态的量子比特数。
        lhs: usize,
        /// 右侧状态的量子比特数。
        rhs: usize
    },

    /// 边缘概率计算没有选择任何量子比特。
    #[error("at least one qubit must be selected")]
    EmptyQubitSelectionError,

    /// 指定的量子比特下标超出模拟器范围。
    #[error("qubit index out of range: simulator has {num_qubits} qubits, got {index}")]
    QubitOutOfRangeError {
        /// 无效的零起始量子比特下标。
        index: usize,
        /// 模拟器实际包含的量子比特数。
        num_qubits: usize
    },

    /// 边缘概率的量子比特选择中包含重复下标。
    #[error("duplicate qubit in selection: {index}")]
    DuplicateQubitError {
        /// 重复出现的量子比特下标。
        index: usize
    },
}
