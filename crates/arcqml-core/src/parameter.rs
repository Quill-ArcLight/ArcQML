use crate::{ArcQmlError, DType, Result, Tensor};

use std::sync::{Arc, RwLock};

/// 可命名、可冻结的模型参数。
///
/// `Parameter` 是叶子 [`Tensor`] 的共享语义包装；克隆只复制句柄。新参数默认未命名、
/// 可训练，并为底层 Tensor 启用梯度。
#[derive(Debug, Clone)]
pub struct Parameter {
    inner: Arc<RwLock<ParameterInner>>,
}

/// 模型参数内部状态。
#[derive(Debug)]
pub struct ParameterInner {
    tensor: Tensor,
    name: Option<String>,
    trainable: bool,
}

impl Parameter {
    /// 将叶子 Tensor 包装为未命名、可训练的参数，并启用其梯度。
    ///
    /// # Panics
    ///
    /// 当 `tensor` 不是叶子 Tensor 或 dtype 不是 `F32`、`F64`、`C64` 时会 panic；
    /// 需要可恢复错误时请使用 [`Parameter::try_new`]。
    pub fn new(tensor: Tensor) -> Self {
        Self::try_new(tensor).expect("Parameter requires a leaf tensor with dtype F32, F64, or C64")
    }

    /// 尝试将叶子 Tensor 包装为未命名、可训练的参数。
    ///
    /// # Errors
    ///
    /// 当 Tensor 不是叶子节点，或 dtype 不是 `F32`、`F64`、`C64` 时返回错误。
    pub fn try_new(tensor: Tensor) -> Result<Self> {
        validate_parameter_tensor(&tensor)?;
        tensor.try_set_requires_grad(true)?;

        Ok(Self {
            inner: Arc::new(RwLock::new(ParameterInner {
                tensor,
                name: None,
                trainable: true,
            })),
        })
    }

    /// 返回参数名称的副本；未命名时返回 `None`。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn name(&self) -> Option<String> {
        self.inner.read().unwrap().name.clone()
    }

    /// 设置参数名。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn set_name(&self, name: impl Into<String>) {
        self.inner.write().unwrap().name = Some(name.into());
    }

    /// 清除参数名称，使其恢复为 `None`。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn clear_name(&self) {
        self.inner.write().unwrap().name = None;
    }

    /// 返回底层 Tensor 的共享句柄。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn tensor(&self) -> Tensor {
        self.inner.read().unwrap().tensor.clone()
    }

    /// 替换底层 Tensor，同时尽可能保留兼容的已有梯度。
    ///
    /// # Panics
    ///
    /// 当 `tensor` 不是受支持的叶子参数 Tensor，或内部共享状态的锁已中毒时会
    /// panic；需要可恢复错误时请使用 [`Parameter::try_set_tensor`]。
    pub fn set_tensor(&self, tensor: Tensor) {
        self.try_set_tensor(tensor)
            .expect("Parameter requires a leaf tensor with dtype F32, F64, or C64");
    }

    /// 尝试替换底层 Tensor。
    ///
    /// 新 Tensor 会同步当前参数的训练状态。旧梯度只有在形状和 dtype 均匹配时才
    /// 转移，否则会被丢弃；参数名称和训练状态保持不变。
    ///
    /// # Errors
    ///
    /// 当新 Tensor 不是叶子节点、dtype 不可微，或无法同步 `requires_grad` 时返回错误。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn try_set_tensor(&self, tensor: Tensor) -> Result<()> {
        validate_parameter_tensor(&tensor)?;
        let mut inner = self.inner.write().unwrap();
        let grad = inner.tensor.grad();

        tensor.try_set_requires_grad(inner.trainable)?;

        if let Some(grad) = grad
            && grad.shape() == tensor.shape()
            && grad.dtype() == tensor.dtype()
        {
            tensor.set_grad(grad);
        }

        inner.tensor = tensor;
        Ok(())
    }

    /// 返回优化器是否应更新该参数。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn trainable(&self) -> bool {
        self.inner.read().unwrap().trainable
    }

    /// 设置优化器是否应更新该参数。
    ///
    /// 设为 `true` 时同时为底层 Tensor 启用 `requires_grad`；设为 `false` 时只阻止
    /// 优化器更新，不关闭梯度记录，以支持“计算梯度但不更新”的用法。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn set_trainable(&self, value: bool) {
        let mut inner = self.inner.write().unwrap();
        inner.trainable = value;

        if value {
            inner.tensor.set_requires_grad(value);
        }
    }

    /// 返回底层 Tensor 是否需要梯度。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn requires_grad(&self) -> bool {
        self.inner.read().unwrap().tensor.requires_grad()
    }

    /// 返回底层 Tensor 当前累计梯度的共享句柄。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn grad(&self) -> Option<Tensor> {
        self.inner.read().unwrap().tensor.grad()
    }

    /// 直接覆盖底层 Tensor 的累计梯度。
    ///
    /// 此方法与 [`Tensor::set_grad`] 一样不执行兼容性校验。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn set_grad(&self, grad: Tensor) {
        self.inner.read().unwrap().tensor.set_grad(grad);
    }

    /// 清空底层 Tensor 当前累计的梯度。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn zero_grad(&self) {
        self.inner.read().unwrap().tensor.zero_grad();
    }

    /// 深拷贝参数及其 Tensor 状态，不与原参数共享值、梯度或训练状态。
    ///
    /// # Errors
    ///
    /// 当底层 Tensor 或其梯度无法深复制时返回错误。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn deep_clone(&self) -> Result<Self> {
        let inner = self.inner.read().unwrap();

        Ok(Self {
            inner: Arc::new(RwLock::new(ParameterInner {
                tensor: inner.tensor.deep_clone()?,
                name: inner.name.clone(),
                trainable: inner.trainable,
            })),
        })
    }

    /// 冻结参数；等价于 `set_trainable(false)`，但仍保留梯度记录状态。
    pub fn freeze(&self) {
        self.set_trainable(false);
    }

    /// 解冻参数；等价于 `set_trainable(true)`，并确保底层 Tensor 启用梯度。
    pub fn unfreeze(&self) {
        self.set_trainable(true);
    }
}

/// 校验 Tensor 能否作为 Parameter 的底层存储。
fn validate_parameter_tensor(tensor: &Tensor) -> Result<()> {
    if !tensor.is_leaf() {
        return Err(ArcQmlError::AutogradError(
            "Parameter requires a leaf tensor".to_string(),
        ));
    }

    if !matches!(tensor.dtype(), DType::F32 | DType::F64 | DType::C64) {
        return Err(ArcQmlError::AutogradError(format!(
            "Parameter does not support dtype {}",
            tensor.dtype()
        )));
    }

    Ok(())
}
