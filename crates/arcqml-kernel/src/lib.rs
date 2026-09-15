//! 状态向量模拟器的低层数值内核。
//!
//! 本 crate 负责将电路参数绑定到门、对单态或 batch 振幅应用门及其伴随，并计算参数
//! 导数。它是实现层 API；普通应用通常应使用 `arcqml-sim` 的模拟器接口。
//!
//! 所有状态向量都使用连续 `C64` 存储，`q0` 对应基态索引的最低有效位。调用低层函数时，
//! 调用方必须遵守各函数记录的长度、量子比特和参数槽位约束。

/// 电路门的运行时参数绑定。
pub mod circuit;
/// 内核校验和执行错误。
pub mod error;
#[doc(hidden)]
pub mod runtime;
/// 单态与 batch 状态向量内核。
pub mod statevector;

pub use circuit::{BoundGate, BoundParameter, bind_gate, resolve_gate};
pub use error::{KernelError, KernelResult};
pub use statevector::{
    apply_adjoint_gate, apply_adjoint_gate_batch, apply_gate, apply_gate_batch,
    parameter_derivative, parameter_derivative_batch, parameter_derivative_into,
};

/// 内核类型和函数的预导入集合。
pub mod prelude {
    pub use crate::{
        BoundGate, BoundParameter, apply_adjoint_gate, apply_adjoint_gate_batch, apply_gate,
        apply_gate_batch, bind_gate, parameter_derivative, parameter_derivative_batch,
        parameter_derivative_into, resolve_gate,
    };
    pub use crate::{KernelError, KernelResult};
}
