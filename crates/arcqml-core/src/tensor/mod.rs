mod constructors;
mod contiguous;
mod gradient;
mod reshape;
mod transpose;
mod value;
mod view;

use crate::{AutogradMeta, DType, Device, Layout, Storage, TensorMeta};

use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Tensor 底层存储的独占写入守卫。
///
/// 守卫离开作用域时会递增共享版本计数，使自动微分能够检测前向之后发生的原地修改。
pub struct StorageWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, Storage>,
    version: &'a AtomicU64,
}

impl Deref for StorageWriteGuard<'_> {
    type Target = Storage;

    /// 将守卫借用为只读 [`Storage`]。
    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl DerefMut for StorageWriteGuard<'_> {
    /// 将守卫借用为可写 [`Storage`]。
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}

impl Drop for StorageWriteGuard<'_> {
    /// 在释放写锁前递增版本号，记录一次潜在的原地修改。
    fn drop(&mut self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }
}

/// 具有共享存储、形状元数据和可选自动微分状态的多维数组。
///
/// [`Clone`] 只复制句柄：克隆值共享存储、版本计数和自动微分状态。使用
/// [`Tensor::deep_clone`] 创建完全独立的副本。
#[derive(Debug, Clone)]
pub struct Tensor {
    storage: Arc<RwLock<Storage>>,
    version: Arc<AtomicU64>,
    meta: TensorMeta,
    autograd: Arc<RwLock<AutogradMeta>>,
}

impl Tensor {
    /// 返回当前 Tensor 的元信息。
    pub fn meta(&self) -> &TensorMeta {
        &self.meta
    }

    /// 获取底层存储的共享读锁。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn storage(&self) -> RwLockReadGuard<'_, Storage> {
        self.storage.read().unwrap()
    }

    /// 获取底层存储的独占写锁，用于原地更新元素。
    ///
    /// 守卫释放时版本计数始终递增，即使调用方没有实际改变元素。
    ///
    /// # Panics
    ///
    /// 当此前的 panic 已使内部共享状态锁中毒时会 panic。
    pub fn storage_mut(&self) -> StorageWriteGuard<'_> {
        StorageWriteGuard {
            guard: self.storage.write().unwrap(),
            version: &self.version,
        }
    }

    /// 返回自动微分元数据的共享读写锁。
    ///
    /// 这是供算子实现者使用的底层接口。直接修改其中状态可能破坏计算图不变量；
    /// 普通用户应优先使用 [`Tensor::requires_grad`]、[`Tensor::grad`] 等安全封装。
    pub fn autograd(&self) -> Arc<RwLock<AutogradMeta>> {
        Arc::clone(&self.autograd)
    }

    /// 返回与共享 Storage 对应的原地修改版本号。
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// 返回 Tensor 的形状。
    pub fn shape(&self) -> &[usize] {
        self.meta().shape()
    }

    /// 返回 Tensor 的数据类型。
    pub fn dtype(&self) -> DType {
        self.meta().dtype()
    }

    /// 返回 Tensor 所在设备。
    pub fn device(&self) -> &Device {
        self.meta().device()
    }

    /// 返回 Tensor 的存储布局类型。
    pub fn layout(&self) -> Layout {
        self.meta().layout()
    }

    /// 返回 Tensor 的秩，即形状中的维度数量。
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// 返回逻辑 Tensor 中的元素数量。
    pub fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// 返回每个维度访问下一元素时在 Storage 中跨越的步长。
    pub fn strides(&self) -> &[isize] {
        self.meta().strides()
    }

    /// 返回逻辑 Tensor 第一个元素在 Storage 中的偏移量。
    pub fn offset(&self) -> usize {
        self.meta().offset()
    }

    /// 判断当前逻辑 Tensor 是否是连续布局。
    pub fn is_contiguous(&self) -> bool {
        self.meta().is_contiguous()
    }

    /// 返回自动微分图的内部节点标识。主要用于 backward 遍历计算图时去重。
    pub(crate) fn node_id(&self) -> usize {
        Arc::as_ptr(&self.autograd) as usize
    }
}
