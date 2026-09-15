use super::AutogradNode;
use crate::Tensor;

use std::fmt;
use std::sync::Arc;

/// Tensor 独有的自动微分状态。grad 保存累计梯度，grad_fn 指向产生该非叶子 Tensor 的 AutogradNode。
#[derive(Clone)]
pub struct AutogradMeta {
    requires_grad: bool,
    is_leaf: bool,
    retain_grad: bool,
    grad: Option<Tensor>,
    grad_fn: Option<Arc<AutogradNode>>,
}

impl fmt::Debug for AutogradMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AutogradMeta")
            .field("requires_grad", &self.requires_grad)
            .field("is_leaf", &self.is_leaf)
            .field("retain_grad", &self.retain_grad)
            .field("has_grad", &self.grad.is_some())
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish()
    }
}

impl Default for AutogradMeta {
    /// 创建默认状态：不需要梯度、是叶子节点且没有累计梯度。
    fn default() -> Self {
        Self::new(false, true)
    }
}

impl AutogradMeta {
    /// 创建叶子 Tensor 或手动构造 Tensor 使用的自动微分状态。
    pub fn new(requires_grad: bool, is_leaf: bool) -> Self {
        Self {
            requires_grad,
            is_leaf,
            retain_grad: false,
            grad: None,
            grad_fn: None,
        }
    }

    /// 根据一条前向操作创建非叶子 Tensor 的自动微分状态。
    pub(crate) fn from_node(node: AutogradNode) -> Self {
        Self {
            requires_grad: true,
            is_leaf: false,
            retain_grad: false,
            grad: None,
            grad_fn: Some(Arc::new(node)),
        }
    }

    /// 返回当前 Tensor 是否参与梯度计算。
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// 返回当前 Tensor 是否是计算图的叶子节点。
    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }

    /// 返回非叶子 Tensor 是否会在反向传播后保留其梯度。
    pub fn retain_grad_enabled(&self) -> bool {
        self.retain_grad
    }

    /// 设置当前 Tensor 是否需要梯度。
    pub fn set_requires_grad(&mut self, value: bool) {
        self.requires_grad = value;
    }

    /// 设置当前 Tensor 是否为叶子节点。
    ///
    /// 此接口主要用于内部构造和测试；普通前向操作应通过 `Tensor::from_operation_named` 建图。
    pub fn set_leaf(&mut self, value: bool) {
        self.is_leaf = value;
    }

    /// 让非叶子 Tensor 在反向传播后也保留其累计梯度。
    pub fn retain_grad(&mut self) {
        self.retain_grad = true;
    }

    /// 返回已累计的梯度引用，没有梯度时返回 `None`。
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }

    /// 覆盖当前累计梯度。
    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad = Some(grad);
    }

    /// 清空当前累计梯度。
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// 返回产生当前非叶子 Tensor 的计算图节点。
    pub(crate) fn grad_fn(&self) -> Option<Arc<AutogradNode>> {
        self.grad_fn.clone()
    }

    /// 释放指向前向图节点的引用，使这条图不能再次用于反向传播。
    pub(crate) fn clear_grad_fn(&mut self) {
        self.grad_fn = None;
    }
}
