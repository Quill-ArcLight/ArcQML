use std::io;
use std::path::PathBuf;
use thiserror::Error;

/// 检查点读写和校验操作使用的结果类型。
pub type CheckpointResult<T> = Result<T, CheckpointError>;

/// 权重检查点读写、解析或兼容性校验失败。
#[derive(Debug, Error)]
pub enum CheckpointError {
    /// 无法从指定路径读取检查点文件。
    #[error("failed to read checkpoint {path}: {source}")]
    ReadError {
        /// 尝试读取的文件路径。
        path: PathBuf,
        #[source]
        /// 底层 I/O 错误。
        source: io::Error,
    },

    /// 无法把检查点写入指定路径。
    #[error("failed to write checkpoint {path}: {source}")]
    WriteError {
        /// 尝试写入的文件路径。
        path: PathBuf,
        #[source]
        /// 底层 I/O 错误。
        source: io::Error,
    },

    /// 文件内容不是有效的检查点 JSON。
    #[error("failed to parse checkpoint {path}: {source}")]
    ParseError {
        /// 被解析的文件路径。
        path: PathBuf,
        #[source]
        /// 底层 JSON 解析错误。
        source: serde_json::Error,
    },

    /// 检查点无法序列化为 JSON。
    #[error("failed to serialize checkpoint {path}: {source}")]
    SerializeError {
        /// 计划写入的文件路径。
        path: PathBuf,
        #[source]
        /// 底层 JSON 序列化错误。
        source: serde_json::Error,
    },

    /// 文件中的格式标识不是当前实现支持的格式。
    #[error("unsupported checkpoint format {actual:?}; expected {expected:?}")]
    UnsupportedFormatError {
        /// 加载器要求的格式标识。
        expected: &'static str,
        /// 文件中实际记录的格式标识。
        actual: String,
    },

    /// 检查点和目标电路的量子比特数不同。
    #[error("checkpoint qubit count mismatch: expected {expected}, got {actual}")]
    QubitCountMismatchError {
        /// 目标电路的量子比特数。
        expected: usize,
        /// 检查点记录的量子比特数。
        actual: usize
    },

    /// 检查点中同一参数名称出现多次。
    #[error("checkpoint contains duplicate parameter name {name:?}")]
    DuplicateParameterNameError {
        /// 重复的参数名称。
        name: String
    },

    /// 目标电路中的参数在检查点中缺失。
    #[error("checkpoint is missing parameter {name:?}")]
    MissingParameterError {
        /// 缺失的参数名称。
        name: String
    },

    /// 检查点包含目标电路没有的参数。
    #[error("checkpoint contains unexpected parameter {name:?}")]
    UnexpectedParameterError {
        /// 多余的参数名称。
        name: String
    },

    /// 同名参数在检查点和目标电路中的数据类型不同。
    #[error("parameter {name:?} has dtype mismatch: expected {expected}, got {actual}")]
    DtypeMismatchError {
        /// 数据类型不匹配的参数名称。
        name: String,
        /// 目标电路要求的数据类型名称。
        expected: &'static str,
        /// 检查点中记录的数据类型名称。
        actual: String,
    },

    /// 目标参数不是检查点格式支持的标量。
    #[error("parameter {name:?} must be scalar, got {numel} values")]
    InvalidParameterShapeError {
        /// 非标量参数的名称。
        name: String,
        /// 参数包含的元素数量。
        numel: usize
    },

    /// 参数数据类型不受当前权重格式支持。
    #[error("parameter {name:?} has unsupported dtype {dtype}")]
    UnsupportedParameterDtypeError {
        /// 数据类型不受支持的参数名称。
        name: String,
        /// 参数的数据类型名称。
        dtype: String
    },

    /// 检查点包含 NaN 或无穷参数值。
    #[error("parameter {name:?} has a non-finite value")]
    NonFiniteValueError {
        /// 包含非有限值的参数名称。
        name: String
    },

    /// 从检查点值重建参数 Tensor 失败。
    #[error("failed to construct parameter {name:?}: {message}")]
    TensorConstructionError {
        /// 无法重建的参数名称。
        name: String,
        /// 底层 Tensor 构造错误信息。
        message: String
    },
}
