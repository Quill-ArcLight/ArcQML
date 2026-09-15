//! 逐门自动微分实现，专供有状态的状态 API 使用。
//!
//! 此模块支撑 `apply_circuit` 后继续读取 `amplitudes`、概率、保真度或抽样的场景；
//! 它不是 `run` 的备选训练路径，整电路训练固定由 `adjoint` 完成。

pub(crate) mod batch;
pub(crate) mod single;
