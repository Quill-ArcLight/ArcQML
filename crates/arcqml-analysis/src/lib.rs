//! 状态向量的只读分析工具。
//!
//! 本 crate 从模拟器的当前纯态计算概率分布、边缘概率、纯态保真度和单量子比特
//! Bloch 向量。分析不会执行电路、进行随机抽样或修改模拟器状态；返回 Tensor 的接口
//! 会保留自动微分关系。
//!
//! # 示例
//!
//! ```
//! use arcqml_analysis::{bloch_vector, probabilities};
//! use arcqml_sim::StateVectorSimulator;
//!
//! let simulator = StateVectorSimulator::new(1)?;
//! assert_eq!(probabilities(&simulator)?.shape(), &[2]);
//! assert_eq!(bloch_vector(&simulator, 0usize)?.z, 1.0);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// 分析错误和结果类型。
pub mod error;
/// 基于纯态状态向量的确定性分析函数。
pub mod statevector;

pub use error::{AnalysisError, AnalysisResult};
pub use statevector::{BlochVector, bloch_vector, fidelity, marginal_probabilities, probabilities};

/// 状态分析常用类型与函数的预导入集合。
pub mod prelude {
    pub use crate::{
        AnalysisError, AnalysisResult, BlochVector, bloch_vector, fidelity, marginal_probabilities,
        probabilities,
    };
}
