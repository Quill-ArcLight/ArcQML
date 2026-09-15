//! 理想纯态量子电路的状态向量模拟器。
//!
//! [`StateVectorSimulator`] 存储长度为 `2^n` 的单个状态向量；
//! [`BatchStateVectorSimulator`] 存储形状为 `[batch_size, 2^n]` 的状态批次。两者均使用
//! 连续 `C64` Tensor，且 `q0` 对应基态索引的最低有效位。
//!
//! # 示例
//!
//! ```
//! use arcqml_circuit::Circuit;
//! use arcqml_sim::StateVectorSimulator;
//!
//! let mut circuit = Circuit::new(2)?;
//! circuit.h(0usize)?.cnot(0usize, 1usize)?;
//! let mut simulator = StateVectorSimulator::new(2)?;
//! simulator.apply_circuit(&circuit)?;
//! assert_eq!(simulator.amplitudes()?.shape(), &[4]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! 当前后端仅支持 CPU 稠密纯态，不模拟噪声、密度矩阵、中途测量或经典条件控制。

/// 模拟器构造、状态校验和执行错误。
pub mod error;
/// 单态与 batch 状态向量模拟器。
pub mod statevector;

pub use error::{SimError, SimResult};
pub use statevector::{
    MeasurementCounts, batch::BatchStateVectorSimulator, single::StateVectorSimulator,
};

/// 模拟器常用类型的预导入集合。
pub mod prelude {
    pub use crate::{
        BatchStateVectorSimulator, MeasurementCounts, SimError, SimResult, StateVectorSimulator,
    };
}
