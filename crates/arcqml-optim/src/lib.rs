//! ArcQML 参数和叶子 Tensor 的一阶优化器。
//!
//! 本 crate 提供 [`Sgd`] 和 [`Adam`]。优化器读取已累积梯度并原地更新参数；一次典型
//! 训练迭代依次执行前向计算、`backward`、`step` 和 `zero_grad`。
//!
//! # 示例
//!
//! ```
//! use arcqml_core::{Parameter, Tensor};
//! use arcqml_optim::Sgd;
//!
//! let parameter = Parameter::new(Tensor::new(0.5_f64)?);
//! let optimizer = Sgd::new(0.01, 0.0)?;
//! parameter.set_grad(Tensor::new(2.0_f64)?);
//! let stats = optimizer.step(std::slice::from_ref(&parameter))?;
//! assert_eq!(stats.updated(), 1);
//! optimizer.zero_grad(std::slice::from_ref(&parameter));
//! assert!(parameter.grad().is_none());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Adam 优化器及其可序列化状态。
pub mod adam;
/// 优化器参数、梯度和状态错误。
pub mod error;
/// 随机梯度下降优化器和步骤统计。
pub mod sgd;

pub use adam::Adam;
pub use error::{OptimError, OptimResult};
pub use sgd::{Sgd, StepStats};

/// 优化器常用类型的预导入集合。
pub mod prelude {
    pub use crate::{Adam, OptimError, OptimResult, Sgd, StepStats};
}
