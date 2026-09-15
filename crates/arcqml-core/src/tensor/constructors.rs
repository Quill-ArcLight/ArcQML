use super::Tensor;
use crate::ArcQmlError::*;
use crate::tensordata::checked_numel;
use crate::{AutogradMeta, DType, Device, Layout, Result, Storage, TensorData, TensorMeta};

use num_complex::Complex64;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};

impl Tensor {
    /// 从标量、一维向量、二维矩阵或 [`TensorData`] 推断形状和 dtype，构造 CPU 稠密 Tensor。
    ///
    /// # Errors
    ///
    /// 当二维输入的行长度不一致或为空、显式形状与数据长度不匹配、形状计算溢出，
    /// 或生成的存储与元数据不一致时返回错误。
    pub fn new<T>(data: T) -> Result<Self>
    where
        T: Into<TensorData>,
    {
        let (storage, meta) = data.into().into_storage_meta()?;
        Self::from_storage_meta(storage, meta)
    }

    /// 创建 dtype 为 `F64`、位于 CPU 且采用稠密布局的全零 Tensor。
    ///
    /// # Errors
    ///
    /// 当 `shape` 的元素数量或连续步长计算溢出时返回错误。
    pub fn zeros(shape: Vec<usize>) -> Result<Self> {
        Self::zeros_with_dtype(shape, DType::F64)
    }

    /// 创建指定 dtype、位于 CPU 且采用稠密布局的全零 Tensor。
    ///
    /// # Errors
    ///
    /// 当 `shape` 的元素数量或连续步长计算溢出时返回错误。
    pub fn zeros_with_dtype(shape: Vec<usize>, dtype: DType) -> Result<Self> {
        let numel = checked_numel(&shape)?;

        let storage = match dtype {
            DType::F64 => Storage::F64(vec![0.0; numel]),
            DType::F32 => Storage::F32(vec![0.0; numel]),
            DType::I64 => Storage::I64(vec![0; numel]),
            DType::C64 => Storage::C64(vec![Complex64::new(0.0, 0.0); numel]),
            DType::Bool => Storage::Bool(vec![false; numel]),
        };

        let meta = TensorMeta::new(shape, dtype, Device::Cpu, Layout::Dense)?;
        Self::from_storage_meta(storage, meta)
    }

    /// 使用固定种子 `42` 创建 dtype 为 `F64` 的标准正态 CPU 稠密 Tensor。
    ///
    /// # Errors
    ///
    /// 当 `shape` 的元素数量或连续步长计算溢出时返回错误。
    pub fn randn(shape: Vec<usize>) -> Result<Self> {
        Self::randn_with_seed(shape, 42)
    }

    /// 使用指定种子创建 dtype 为 `F64` 的标准正态 CPU 稠密 Tensor。
    ///
    /// # Errors
    ///
    /// 当 `shape` 的元素数量或连续步长计算溢出时返回错误。
    pub fn randn_with_seed(shape: Vec<usize>, seed: u64) -> Result<Self> {
        Self::randn_with_dtype_and_seed(shape, DType::F64, seed)
    }

    /// 使用指定 dtype 和种子创建正态随机 CPU 稠密 Tensor。
    ///
    /// 对 `F32` 和 `F64`，每个元素独立服从 `N(0, 1)`。对 `C64`，实部和虚部
    /// 独立服从 `N(0, 1/2)`，因此复数元素的期望模平方为 `1`。相同参数和种子
    /// 产生可复现的元素序列。
    ///
    /// # Errors
    ///
    /// 当 `shape` 计算溢出，或 `dtype` 为不支持正态抽样的 `I64`/`Bool` 时返回错误。
    pub fn randn_with_dtype_and_seed(shape: Vec<usize>, dtype: DType, seed: u64) -> Result<Self> {
        let numel = checked_numel(&shape)?;
        let mut rng = StdRng::seed_from_u64(seed);

        let storage = match dtype {
            DType::F64 => Storage::F64(
                (0..numel)
                    .map(|_| {
                        let value: f64 = StandardNormal.sample(&mut rng);
                        value
                    })
                    .collect(),
            ),
            DType::F32 => Storage::F32(
                (0..numel)
                    .map(|_| {
                        let value: f64 = StandardNormal.sample(&mut rng);
                        value as f32
                    })
                    .collect(),
            ),
            DType::C64 => {
                let scale = 1.0_f64 / 2.0_f64.sqrt();

                Storage::C64(
                    (0..numel)
                        .map(|_| {
                            let real: f64 = StandardNormal.sample(&mut rng);
                            let imag: f64 = StandardNormal.sample(&mut rng);
                            Complex64::new(scale * real, scale * imag)
                        })
                        .collect(),
                )
            }
            DType::I64 | DType::Bool => {
                return Err(DTypeMismatchError {
                    expected: DType::F64,
                    actual: dtype,
                });
            }
        };

        let meta = TensorMeta::new(shape, dtype, Device::Cpu, Layout::Dense)?;
        Self::from_storage_meta(storage, meta)
    }

    /// 根据已有存储和元数据创建不需要梯度的叶子 Tensor。
    ///
    /// # Errors
    ///
    /// 当存储 dtype 与元数据不一致，或存储长度不足以覆盖元数据描述的逻辑视图时返回错误。
    pub fn from_storage_meta(storage: Storage, meta: TensorMeta) -> Result<Self> {
        Self::from_parts(storage, meta, AutogradMeta::default())
    }

    /// 根据存储、元数据与显式自动微分状态创建 Tensor。
    ///
    /// 这是底层构造接口；调用方负责保证 `autograd` 与 Tensor 的叶子/计算图语义一致。
    ///
    /// # Errors
    ///
    /// 当存储 dtype 与元数据不一致，或存储长度不足以覆盖元数据描述的逻辑视图时返回错误。
    pub fn from_parts(storage: Storage, meta: TensorMeta, autograd: AutogradMeta) -> Result<Self> {
        if storage.dtype() != meta.dtype() {
            return Err(DTypeMismatchError {
                expected: meta.dtype(),
                actual: storage.dtype(),
            });
        }

        if storage.len() < meta.storage_len_required() {
            return Err(ShapeError(format!(
                "storage length insufficient for tensor elements: expected {}, got {}",
                storage.len(),
                meta.storage_len_required()
            )));
        }

        Ok(Self {
            storage: Arc::new(RwLock::new(storage)),
            version: Arc::new(AtomicU64::new(0)),
            meta,
            autograd: Arc::new(RwLock::new(autograd)),
        })
    }
}
