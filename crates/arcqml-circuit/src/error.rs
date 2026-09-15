use thiserror::Error;

/// 电路构建、验证、拼接和序列化操作使用的结果类型。
pub type CircuitResult<T> = Result<T, CircuitError>;

/// 电路结构、量子门、参数绑定或序列化无效。
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CircuitError {
    /// 操作引用了电路范围之外的量子比特。
    #[error("qubit out of range: circuit has {num_qubits} qubits, got qubit={index}")]
    QubitOutOfRangeError {
        /// 无效的零起始量子比特下标。
        index: usize,
        /// 电路实际包含的量子比特数。
        num_qubits: usize
    },

    /// 操作提供的量子比特数量与门的 arity 不同。
    #[error("gate arity mismatch: gate {gate} expects {expected} qubits, got {actual}")]
    GateArityError {
        /// 量子门名称。
        gate: String,
        /// 门要求的量子比特数量。
        expected: usize,
        /// 操作实际提供的数量。
        actual: usize,
    },

    /// 同一个操作重复引用某个量子比特。
    #[error("duplicate qubit: qubit={index} is used more than once in the same operation")]
    DuplicateQubitError {
        /// 重复的量子比特下标。
        index: usize
    },

    /// 电路的量子比特数为零。
    #[error("invalid circuit size: num_qubits must be greater than 0")]
    EmptyCircuitError,

    /// 自定义酉门矩阵的元素数量与 arity 不匹配。
    #[error(
        "invalid custom unitary shape: gate {name} with arity={arity} expects {expected} matrix elements, got {actual}"
    )]
    InvalidUnitaryShapeError {
        /// 自定义门名称。
        name: String,
        /// 自定义门作用的量子比特数量。
        arity: usize,
        /// 该 arity 所需的矩阵元素数量。
        expected: usize,
        /// 调用方实际提供的元素数量。
        actual: usize,
    },

    /// 自定义矩阵未通过酉性校验。
    #[error("invalid custom unitary: gate {name} with arity={arity} is not unitary")]
    NonUnitaryMatrixError {
        /// 自定义门名称。
        name: String,
        /// 自定义门作用的量子比特数量。
        arity: usize
    },

    /// 无法为指定 arity 表示或分配方阵维度。
    #[error("custom unitary dimension overflow for arity={arity}")]
    UnitaryDimensionOverflowError {
        /// 导致维度溢出的量子比特数量。
        arity: usize
    },

    /// 参数名称为空字符串。
    #[error("invalid parameter: parameter name cannot be empty")]
    EmptyParameterNameError,

    /// 参数标识没有指向电路参数表中的有效元素。
    #[error("parameter id out of range: id={index}, parameter_count={parameter_count}")]
    InvalidParameterIdError {
        /// 无效参数标识的零起始下标。
        index: usize,
        /// 当前电路参数表长度。
        parameter_count: usize,
    },

    /// 无法构造参数所需的 Tensor。
    #[error("tensor create error: {message}")]
    TensorCreateError {
        /// 底层 Tensor 构造错误信息。
        message: String
    },

    /// 电路参数不是单元素标量 Tensor。
    #[error("parameter {index} must be scalar, got numel={numel}")]
    InvalidParameterShapeError {
        /// 参数表中的零起始下标。
        index: usize,
        /// 参数实际包含的元素数量。
        numel: usize
    },

    /// 电路参数的数据类型不支持门角度绑定。
    #[error("parameter {index} dtype is not supported: {dtype}")]
    UnsupportedParameterDTypeError {
        /// 参数表中的零起始下标。
        index: usize,
        /// 参数的实际数据类型名称。
        dtype: String
    },

    /// 参数没有可用于查找和序列化的名称。
    #[error("parameter {index} has no usable name")]
    MissingParameterNameError {
        /// 参数表中的零起始下标。
        index: usize
    },

    /// 电路参数表中出现重复名称。
    #[error("duplicate parameter name: {name}")]
    DuplicateParameterNameError {
        /// 重复的参数名称。
        name: String
    },

    /// 尝试拼接量子比特数不同的两个电路。
    #[error("cannot append circuits with different qubit counts: left={left}, right={right}")]
    AppendQubitCountMismatchError {
        /// 左侧目标电路的量子比特数。
        left: usize,
        /// 右侧被追加电路的量子比特数。
        right: usize
    },

    /// 参数绑定引用了右侧电路范围之外的源参数。
    #[error(
        "parameter binding source id out of range: id={index}, parameter_count={parameter_count}"
    )]
    InvalidBindingSourceParameterIdError {
        /// 无效源参数标识的下标。
        index: usize,
        /// 右侧电路参数表长度。
        parameter_count: usize,
    },

    /// 参数绑定引用了左侧电路范围之外的目标参数。
    #[error(
        "parameter binding target id out of range: id={index}, parameter_count={parameter_count}"
    )]
    InvalidBindingTargetParameterIdError {
        /// 无效目标参数标识的下标。
        index: usize,
        /// 左侧电路参数表长度。
        parameter_count: usize,
    },

    /// 多条显式绑定使用了同一个右侧源参数。
    #[error("parameter binding source id is repeated: id={index}")]
    DuplicateBindingSourceParameterIdError {
        /// 重复绑定的源参数下标。
        index: usize
    },

    /// 拼接后的参数数量无法用 `usize` 表示。
    #[error("parameter count overflow while appending circuits")]
    AppendParameterCountOverflowError,

    /// 为拼接后的参数或操作预留容量失败。
    #[error("cannot reserve {additional} additional entries for {collection}: {message}")]
    AppendCapacityError {
        /// 无法扩容的集合名称。
        collection: &'static str,
        /// 计划额外预留的元素数量。
        additional: usize,
        /// 底层容量错误信息。
        message: String,
    },

    /// 无法把电路结构序列化为 JSON。
    #[error("cannot serialize circuit as JSON: {message}")]
    JsonSerializationError {
        /// 底层 JSON 序列化错误信息。
        message: String
    },

    /// 无法从 JSON 重建有效电路。
    #[error("cannot deserialize circuit from JSON: {message}")]
    JsonDeserializationError {
        /// 底层解析或电路验证错误信息。
        message: String
    },
}
