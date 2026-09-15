//! ArcQML 的 Tensor、存储、参数与自动微分基础设施。
//!
//! [`Tensor`] 是共享所有权的稠密 CPU Tensor。克隆 Tensor 只克隆句柄；需要独立数据时
//! 使用 [`Tensor::deep_clone`]。当前数值存储支持 [`DType::F32`]、[`DType::F64`]、
//! [`DType::C64`]、[`DType::I64`] 和 [`DType::Bool`]。
//!
//! # 示例
//!
//! ```
//! use arcqml_core::Tensor;
//!
//! let tensor = Tensor::new(vec![1.0_f64, 2.0, 3.0])?;
//! assert_eq!(tensor.shape(), &[3]);
//! assert_eq!(tensor.numel(), 3);
//! # Ok::<(), arcqml_core::ArcQmlError>(())
//! ```
//!
//! 自动微分采用动态计算图。叶子 Tensor 通过 [`Tensor::set_requires_grad`] 加入图；标量
//! 输出可调用 [`Tensor::backward`]，非标量输出应调用 [`Tensor::backward_with_grad`]。

/// 动态自动微分元数据、反向函数和梯度模式。
pub mod autograd;
/// 计算设备标识。
pub mod device;
/// 维度相关的安全算术工具。
pub mod dimension;
/// Tensor 元素数据类型。
pub mod dtype;
/// 核心错误和结果类型。
pub mod error;
/// Tensor 存储布局标识。
pub mod layout;
/// Tensor 的形状、步长、偏移和布局元数据。
pub mod meta;
/// Rayon 全局线程池控制。
pub mod parallel;
/// 可训练参数封装。
pub mod parameter;
/// 类型化连续存储缓冲区。
pub mod storage;
/// Tensor 容器、构造、视图和梯度接口。
pub mod tensor;
/// 从 Rust 标量和嵌套向量构造 Tensor 的中间数据表示。
pub mod tensordata;

pub use autograd::{
    AutogradMeta, BackwardFn, CustomOp, NoGradGuard, apply_custom_op, is_grad_enabled, no_grad,
};
pub use device::Device;
pub use dimension::checked_power_of_two;
pub use dtype::DType;
pub use error::{ArcQmlError, Result};
pub use layout::Layout;
pub use meta::TensorMeta;
pub use parallel::{init_rayon, rayon_num_threads};
pub use parameter::Parameter;
pub use storage::Storage;
pub use tensor::{StorageWriteGuard, Tensor};
pub use tensordata::TensorData;

/// 核心数据结构和自动微分接口的预导入集合。
pub mod prelude {

    pub use crate::{
        ArcQmlError, AutogradMeta, BackwardFn, CustomOp, DType, Device, Layout, NoGradGuard,
        Parameter, Result, Storage, StorageWriteGuard, Tensor, TensorData, TensorMeta,
        apply_custom_op, checked_power_of_two, init_rayon, is_grad_enabled, no_grad,
        rayon_num_threads,
    };
}
