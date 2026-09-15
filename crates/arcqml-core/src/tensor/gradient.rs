use super::Tensor;
use crate::autograd::{self, AutogradNode};
use crate::{ArcQmlError, AutogradMeta, BackwardFn, DType, Result, Storage, TensorMeta};

use std::sync::{Arc, RwLock};

impl Tensor {
    /// 根据带诊断名称的可微前向结果创建非叶子 Tensor。
    ///
    /// 仅当全局梯度记录已启用且至少一个父 Tensor 需要梯度时保存 `backward` 和父节点；
    /// 否则返回普通叶子 Tensor。`backward` 应为输入顺序中的每个父节点返回一个梯度槽。
    ///
    /// # Errors
    ///
    /// 当 `storage` 与 `meta` 的 dtype 不一致，或存储长度不足时返回错误。
    pub fn from_operation_named(
        operation_name: impl Into<String>,
        storage: Storage,
        meta: TensorMeta,
        parents: Vec<Tensor>,
        backward: Arc<dyn BackwardFn>,
    ) -> Result<Self> {
        let autograd = if autograd::should_record(&parents) {
            AutogradMeta::from_node(AutogradNode::new(operation_name, parents, backward))
            // 需要梯度，不是叶子节点，grad_fn 根据 parents 和 backward 创建
        } else {
            AutogradMeta::default() // 不需要梯度，是叶子节点
        };

        Self::from_parts(storage, meta, autograd)
    }

    /// 根据不可导前向计算创建带有明确断图信息的非叶子张量。
    ///
    /// 当父张量需要梯度时，反向传播会报告操作名称和不可导原因；用户应改用
    /// 可微算子、为该操作实现反向规则，或在调用前显式执行 `detach()`。
    ///
    /// # Errors
    ///
    /// 当 `storage` 与 `meta` 的 dtype 不一致，或存储长度不足时返回错误。
    pub fn from_non_differentiable_operation(
        operation_name: impl Into<String>,
        reason: impl Into<String>,
        storage: Storage,
        meta: TensorMeta,
        parents: Vec<Tensor>,
    ) -> Result<Self> {
        let operation_name = operation_name.into();
        let reason = reason.into();
        Self::from_operation_named(
            operation_name.clone(),
            storage,
            meta,
            parents,
            Arc::new(NonDifferentiableBackward {
                operation_name,
                reason,
            }),
        )
    }

    /// 返回当前 Tensor 是否参与自动微分并接收反向梯度。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn requires_grad(&self) -> bool {
        self.autograd.read().unwrap().requires_grad()
    }

    /// 返回当前 Tensor 是否是计算图叶子节点。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn is_leaf(&self) -> bool {
        self.autograd.read().unwrap().is_leaf()
    }

    /// 设置当前 Tensor 是否需要梯度。
    ///
    /// # Panics
    ///
    /// 当尝试为非叶子 Tensor 启用梯度，或内部自动微分状态的锁已中毒时会 panic；需要可恢复错误时请使用 `try_set_requires_grad`。
    pub fn set_requires_grad(&self, requires_grad: bool) {
        self.try_set_requires_grad(requires_grad).expect(
            "requires_grad can only be enabled for a leaf tensor with dtype F32, F64, or C64",
        );
    }

    /// 尝试设置当前 Tensor 是否需要梯度。
    ///
    /// 禁用梯度不会自动清除已经累积的梯度。
    ///
    /// # Errors
    ///
    /// 当尝试为非叶子 Tensor 启用梯度，或 Tensor dtype 不是 `F32`、`F64`、`C64` 时返回错误。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn try_set_requires_grad(&self, requires_grad: bool) -> Result<()> {
        if requires_grad {
            let autograd = self.autograd.read().unwrap();

            if !autograd.is_leaf() {
                return Err(ArcQmlError::AutogradError(
                    "requires_grad can only be enabled for a leaf tensor".to_string(),
                ));
            }

            if !matches!(self.dtype(), DType::F32 | DType::F64 | DType::C64) {
                return Err(ArcQmlError::AutogradError(format!(
                    "requires_grad is not supported for dtype {}",
                    self.dtype()
                )));
            }
        }

        self.autograd
            .write()
            .unwrap()
            .set_requires_grad(requires_grad);

        Ok(())
    }

    /// 返回当前累计梯度的共享 Tensor 句柄，没有梯度时返回 `None`。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn grad(&self) -> Option<Tensor> {
        self.autograd.read().unwrap().grad().cloned()
    }

    /// 直接覆盖当前累计梯度。
    ///
    /// 此底层接口不校验形状、dtype、设备或布局；调用方必须保证 `grad` 与当前
    /// Tensor 兼容，否则后续反向传播或优化器步骤可能失败。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn set_grad(&self, grad: Tensor) {
        self.autograd.write().unwrap().set_grad(grad);
    }

    /// 清空当前累计梯度，不改变 `requires_grad` 或计算图。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn zero_grad(&self) {
        self.autograd.write().unwrap().zero_grad();
    }

    /// 要求后续反向传播保留该非叶子 Tensor 的梯度；默认仅叶子 Tensor 保存梯度。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn retain_grad(&self) {
        self.autograd.write().unwrap().retain_grad();
    }

    /// 返回共享存储与版本计数、但脱离当前计算图的新叶子 Tensor。
    ///
    /// 对返回值或原值执行原地写入会影响另一方；若需要独立存储，请使用 [`Tensor::deep_clone`]。
    pub fn detach(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            version: Arc::clone(&self.version),
            meta: self.meta.clone(),
            autograd: Arc::new(RwLock::new(AutogradMeta::default())),
        }
    }

    /// 以标量 Tensor 为输出、默认上游梯度 `1` 执行反向传播，并释放已遍历的计算图。
    ///
    /// # Errors
    ///
    /// 当当前 Tensor 不需要梯度、不是标量、dtype 不可微、计算图已释放，前向保存的
    /// 数据被原地修改，或某个反向规则返回不兼容梯度时返回错误。
    pub fn backward(&self) -> Result<()> {
        autograd::backward(self, None, false)
    }

    /// 使用显式上游梯度执行反向传播，并释放已遍历的计算图。
    ///
    /// # Errors
    ///
    /// 除 [`Tensor::backward`] 的计算图错误外，当 `gradient` 的形状、dtype、设备或布局
    /// 与当前 Tensor 不一致时返回错误。
    pub fn backward_with_grad(&self, gradient: Tensor) -> Result<()> {
        autograd::backward(self, Some(gradient), false)
    }

    /// 使用显式上游梯度执行反向传播，并保留计算图以允许再次反向传播。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`Tensor::backward_with_grad`] 相同。
    pub fn backward_with_grad_retain_graph(&self, gradient: Tensor) -> Result<()> {
        autograd::backward(self, Some(gradient), true)
    }

    /// 深复制数值、元数据和已累积梯度，返回不共享存储、版本计数或计算图的新叶子 Tensor。
    ///
    /// # Errors
    ///
    /// 当梯度深复制失败，或复制后的存储与元数据不一致时返回错误。

    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn deep_clone(&self) -> Result<Self> {
        let storage = self.storage().clone();
        let meta = self.meta().clone();
        let mut autograd = AutogradMeta::new(self.requires_grad(), true);

        if let Some(grad) = self.grad() {
            autograd.set_grad(grad.deep_clone()?);
        }

        Self::from_parts(storage, meta, autograd)
    }
}

/// 表示已知但没有反向规则的前向操作。
#[derive(Debug)]
struct NonDifferentiableBackward {
    operation_name: String,
    reason: String,
}

impl BackwardFn for NonDifferentiableBackward {
    /// 返回包含操作名称和原因的反向传播错误。
    fn backward(&self, _parents: &[Tensor], _grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Err(ArcQmlError::AutogradError(format!(
            "operation `{}` is not differentiable: {}. Use differentiable tensor operations, provide a custom backward rule, or call detach() explicitly",
            self.operation_name, self.reason
        )))
    }
}
