//! 状态向量执行层。
//!
//! `adjoint` 实现公开 `run` 的整电路伴随节点；`gatewise` 只实现会保存中间
//! 状态的状态 API，例如 `apply_circuit`、`amplitudes` 与状态分析，二者不可互相替代。

pub(crate) mod adjoint;
pub(crate) mod gatewise;
