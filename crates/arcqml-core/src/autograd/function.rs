use crate::{Result, Tensor};

use std::fmt;

/// 一个可微前向操作的局部反向传播规则。
///
/// 实现必须按 `parents` 顺序返回等长的梯度槽；不需要或不可用的父梯度使用 `None`。
pub trait BackwardFn: fmt::Debug + Send + Sync {
    /// 根据父 Tensor 和输出的上游梯度计算局部向量—雅可比积。
    ///
    /// # Errors
    ///
    /// 当反向函数无法根据父节点和上游梯度计算有效梯度时返回错误。
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>>;
}
