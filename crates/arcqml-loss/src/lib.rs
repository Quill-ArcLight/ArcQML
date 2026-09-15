//! 面向 ArcQML Tensor 的可微损失函数。
//!
//! 损失函数返回标量 Tensor，并将反向规则连接到 `arcqml-core` 的自动微分图。输入必须是
//! 非空的 `F32` 或 `F64` Tensor；形状和标签约束由各函数单独说明。
//!
//! # 示例
//!
//! ```
//! use arcqml_core::Tensor;
//! use arcqml_loss::mse_loss;
//!
//! let prediction = Tensor::new(0.8_f64)?;
//! let target = Tensor::new(1.0_f64)?;
//! let loss = mse_loss(&prediction, &target)?;
//! assert_eq!(loss.shape(), &[]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod binary_nll;
mod common;
mod cross_entropy;
/// 损失函数的输入校验和底层 Tensor 错误。
pub mod error;
mod l1;
mod mse;

pub use binary_nll::binary_nll_loss;
pub use cross_entropy::{binary_cross_entropy_with_logits_loss, cross_entropy_loss};
pub use error::{LossError, LossResult};
pub use l1::l1_loss;
pub use mse::mse_loss;

/// 损失函数和错误类型的预导入集合。
pub mod prelude {
    pub use crate::{
        LossError, LossResult, binary_cross_entropy_with_logits_loss, binary_nll_loss,
        cross_entropy_loss, l1_loss, mse_loss,
    };
}
