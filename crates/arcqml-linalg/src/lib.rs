//! 基于 `arcqml-core` Tensor 的可微数值与线性代数运算。
//!
//! 本 crate 提供逐元素运算、广播、矩阵乘法、归约、归一化函数、形状变换以及 ndarray
//! 互操作。公开算子会在需要时向动态自动微分图登记反向规则。
//!
//! # 示例
//!
//! ```
//! use arcqml_core::Tensor;
//! use arcqml_linalg::{TensorLinalgExt, mean};
//!
//! let lhs = Tensor::new(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]])?;
//! let rhs = Tensor::new(vec![vec![5.0_f64], vec![6.0]])?;
//! let product = lhs.matmul(&rhs)?;
//! assert_eq!(product.shape(), &[2, 1]);
//! assert_eq!(mean(&product)?.shape(), &[]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! 除另有说明外，数值内核要求连续 row-major 存储；非连续视图可先调用
//! `Tensor::contiguous`。复数反向传播采用面向实值损失的共轭 Wirtinger VJP 约定。

/// 线性代数算子的反向传播节点。
pub mod autograd;
/// 形状、数据类型和内核错误。
pub mod error;
/// 面向连续缓冲区的低层数值内核。
pub mod kernels;
/// Tensor 与 `ndarray::ArrayD` 的转换和借用视图。
pub mod ndarray_bridge;
/// 面向 Tensor 的可微运算函数。
pub mod ops;
/// 广播、矩阵和视图形状校验工具。
pub mod shape;
/// 为 Tensor 提供方法式调用的扩展 trait。
pub mod tensor_ext;

pub use error::{LinalgError, LinalgResult};
pub use ops::{
    abs, add, clamp, conj, div, dot, exp, l2_norm, log, log_softmax, logsumexp, matmul, max, mean,
    mean_dim, min, mul, neg, reshape, segment_sum, sigmoid, softmax, sqrt, square, sub, sum,
    sum_dim, tanh, transpose,
};
pub use tensor_ext::TensorLinalgExt;

/// 常用线性代数函数与扩展 trait 的预导入集合。
pub mod prelude {
    pub use crate::error::{LinalgError, LinalgResult};
    pub use crate::ops::{
        abs, add, clamp, conj, div, dot, exp, l2_norm, log, log_softmax, logsumexp, matmul, max,
        mean, mean_dim, min, mul, neg, reshape, segment_sum, sigmoid, softmax, sqrt, square, sub,
        sum, sum_dim, tanh, transpose,
    };
    pub use crate::tensor_ext::TensorLinalgExt;
}
