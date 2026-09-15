//! 量子电路、量子门、操作与可训练参数引用。
//!
//! [`Circuit`] 保存量子比特数量、按时间顺序排列的 [`Operation`]，以及参数化门共享的
//! 参数表。量子比特索引从零开始；状态向量后端将 `q0` 解释为计算基索引的最低有效位。
//!
//! # 示例
//!
//! ```
//! use arcqml_circuit::Circuit;
//!
//! let mut circuit = Circuit::new(2)?;
//! circuit.h(0usize)?.cnot(0usize, 1usize)?;
//! assert_eq!(circuit.len(), 2);
//! assert_eq!(circuit.depth(), 2);
//! # Ok::<(), arcqml_circuit::CircuitError>(())
//! ```
//!
//! 参数化便捷方法（如 [`Circuit::ry`]）创建可训练参数；对应的 `_fixed` 方法嵌入固定值，
//! `_param` 方法则复用现有 [`ParameterId`]。

mod append;
/// 电路容器及其查询、验证与通用编辑操作。
pub mod circuit;
mod circuit_gates_multi;
mod circuit_gates_parametric_multi;
mod circuit_gates_single;
mod circuit_parameters;
/// 电路构建、验证和序列化错误。
pub mod error;
/// 内置量子门与自定义酉门定义。
pub mod gate;
mod gate_metadata;
mod gate_remap;
/// 门参数值与电路参数标识。
pub mod gateparam;
mod json;
/// 一次量子门操作及其目标量子比特。
pub mod operation;
/// 零起始的量子比特标识。
pub mod qubit;

pub use append::{AppendReport, ParameterBinding};
pub use circuit::Circuit;
pub use error::{CircuitError, CircuitResult};
pub use gate::Gate;
pub use gateparam::{Gateparam, ParameterId};
pub use operation::Operation;
pub use qubit::Qubit;

/// 电路构建常用类型的预导入集合。
pub mod prelude {
    pub use crate::{
        AppendReport, Circuit, CircuitError, CircuitResult, Gate, Gateparam, Operation,
        ParameterBinding, ParameterId, Qubit,
    };
}
