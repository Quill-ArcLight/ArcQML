/// 行主序 batch 状态向量的门、伴随门和参数导数内核。
pub mod batch;
/// 单个状态向量的门、伴随门和参数导数内核。
pub mod single;

pub use batch::{
    apply_adjoint_gate as apply_adjoint_gate_batch,
    apply_adjoint_gate_serial as apply_adjoint_gate_batch_serial, apply_gate as apply_gate_batch,
    apply_gate_serial as apply_gate_batch_serial,
    parameter_derivative as parameter_derivative_batch,
    parameter_derivative_serial as parameter_derivative_batch_serial,
};
pub use single::{apply_adjoint_gate, apply_gate, parameter_derivative, parameter_derivative_into};
