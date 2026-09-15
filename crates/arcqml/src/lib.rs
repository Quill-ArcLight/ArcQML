//! ArcQML 的统一 Rust API 入口。
//!
//! 本 crate 重导出电路构建、状态向量模拟、可观测量、自动微分、线性代数、损失函数、
//! 优化器、检查点、酉矩阵和可视化能力。应用程序通常只需依赖本 crate，并通过
//! [`prelude`] 导入常用类型。
//!
//! # 示例
//!
//! ```
//! use arcqml::prelude::*;
//!
//! let mut circuit = Circuit::new(2)?;
//! circuit.h(0usize)?.cnot(0usize, 1usize)?;
//! let observable = SparsePauliOp::single(2, 0usize, Pauli::Z, 1.0)?;
//! let simulator = StateVectorSimulator::new(circuit.num_qubits())?;
//! let expectation = simulator.run(&circuit, &observable)?;
//! assert!(expectation.shape().is_empty());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # 功能模块
//!
//! - [`core`]：Tensor、参数和自动微分。
//! - [`circuit`]：量子电路、量子门和参数引用。
//! - [`sim`] 与 [`observable`]：状态向量模拟和 Pauli 可观测量。
//! - [`linalg`]、[`loss`] 与 [`optim`]：可微数值计算和训练。
//! - [`checkpoint`]、[`analysis`]、[`unitary`] 与 [`visualization`]：持久化与辅助工具。

pub use arcqml_analysis as analysis;
pub use arcqml_checkpoint as checkpoint;
pub use arcqml_circuit as circuit;
pub use arcqml_core as core;
pub use arcqml_core::{init_rayon, rayon_num_threads};
pub use arcqml_linalg as linalg;
pub use arcqml_loss as loss;
pub use arcqml_observable as observable;
pub use arcqml_optim as optim;
pub use arcqml_sim as sim;
pub use arcqml_unitary as unitary;
pub use arcqml_visualization as visualization;

/// 面向交互式使用的便捷预导入集合。
///
/// 此模块会导入各功能 crate 的公开 prelude，覆盖面较广。库代码若希望明确依赖边界
/// 并减少名称冲突，宜从 [`core`]、[`circuit`]、[`sim`] 等具体模块按需导入。
///
/// 使用 `use arcqml::prelude::*;` 还会导入权重读写和文本电路图接口。
pub mod prelude {
    pub use arcqml_analysis::prelude::*;
    pub use arcqml_circuit::prelude::*;
    pub use arcqml_core::prelude::*;
    pub use arcqml_linalg::prelude::*;
    pub use arcqml_loss::prelude::*;
    pub use arcqml_observable::prelude::*;
    pub use arcqml_optim::prelude::*;
    pub use arcqml_sim::prelude::*;
    pub use arcqml_unitary::prelude::*;

    pub use arcqml_checkpoint::{load_weights, save_weights};
    pub use arcqml_visualization::draw;
}
