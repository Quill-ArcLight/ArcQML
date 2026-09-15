use crate::ArcQmlError::*;
use crate::{Device, Layout, Result, Storage, TensorMeta};

use num_complex::Complex64;

/// [`Tensor`](crate::Tensor) 构造函数接受的拥有所有权的原始数据。
///
/// 支持标量、一维向量、规则二维矩阵，以及带显式任意维形状的扁平行主序数据。
#[derive(Debug, Clone)]
pub enum TensorData {
    /// 单个 `f32` 标量，构造零维 Tensor。
    ScalarF32(f32),
    /// 一维 `f32` 数据。
    VectorF32(Vec<f32>),
    /// 二维 `f32` 数据；所有行必须等长。
    MatrixF32(Vec<Vec<f32>>),
    /// 带显式形状的扁平 `f32` 数据。
    FlatF32 {
        /// 行主序元素缓冲区。
        data: Vec<f32>,
        /// Tensor 形状；各维度乘积必须等于 `data.len()`。
        shape: Vec<usize>,
    },

    /// 单个 `f64` 标量，构造零维 Tensor。
    ScalarF64(f64),
    /// 一维 `f64` 数据。
    VectorF64(Vec<f64>),
    /// 二维 `f64` 数据；所有行必须等长。
    MatrixF64(Vec<Vec<f64>>),
    /// 带显式形状的扁平 `f64` 数据。
    FlatF64 {
        /// 行主序元素缓冲区。
        data: Vec<f64>,
        /// Tensor 形状；各维度乘积必须等于 `data.len()`。
        shape: Vec<usize>,
    },

    /// 单个 `Complex64` 标量，构造零维 Tensor。
    ScalarC64(Complex64),
    /// 一维 `Complex64` 数据。
    VectorC64(Vec<Complex64>),
    /// 二维 `Complex64` 数据；所有行必须等长。
    MatrixC64(Vec<Vec<Complex64>>),
    /// 带显式形状的扁平 `Complex64` 数据。
    FlatC64 {
        /// 行主序元素缓冲区。
        data: Vec<Complex64>,
        /// Tensor 形状；各维度乘积必须等于 `data.len()`。
        shape: Vec<usize>,
    },

    /// 单个 `i64` 标量，构造零维 Tensor。
    ScalarI64(i64),
    /// 一维 `i64` 数据。
    VectorI64(Vec<i64>),
    /// 二维 `i64` 数据；所有行必须等长。
    MatrixI64(Vec<Vec<i64>>),
    /// 带显式形状的扁平 `i64` 数据。
    FlatI64 {
        /// 行主序元素缓冲区。
        data: Vec<i64>,
        /// Tensor 形状；各维度乘积必须等于 `data.len()`。
        shape: Vec<usize>,
    },

    /// 单个布尔标量，构造零维 Tensor。
    ScalarBool(bool),
    /// 一维布尔数据。
    VectorBool(Vec<bool>),
    /// 二维布尔数据；所有行必须等长。
    MatrixBool(Vec<Vec<bool>>),
    /// 带显式形状的扁平布尔数据。
    FlatBool {
        /// 行主序元素缓冲区。
        data: Vec<bool>,
        /// Tensor 形状；各维度乘积必须等于 `data.len()`。
        shape: Vec<usize>,
    },
}

impl TensorData {
    /// 将拥有所有权的数据转换为匹配的存储和 CPU 稠密元数据。
    ///
    /// # Errors
    ///
    /// 当嵌套输入行长度不一致、显式形状与数据长度不匹配或元素数量溢出时返回错误。
    pub fn into_storage_meta(self) -> Result<(Storage, TensorMeta)> {
        match self {
            TensorData::ScalarF32(value) => scalar(Storage::F32(vec![value])),
            TensorData::VectorF32(data) => vector(Storage::F32(data)),
            TensorData::MatrixF32(data) => matrix(data, Storage::F32),
            TensorData::FlatF32 { data, shape } => flat(data, shape, Storage::F32),

            TensorData::ScalarF64(value) => scalar(Storage::F64(vec![value])),
            TensorData::VectorF64(data) => vector(Storage::F64(data)),
            TensorData::MatrixF64(data) => matrix(data, Storage::F64),
            TensorData::FlatF64 { data, shape } => flat(data, shape, Storage::F64),

            TensorData::ScalarC64(value) => scalar(Storage::C64(vec![value])),
            TensorData::VectorC64(data) => vector(Storage::C64(data)),
            TensorData::MatrixC64(data) => matrix(data, Storage::C64),
            TensorData::FlatC64 { data, shape } => flat(data, shape, Storage::C64),

            TensorData::ScalarI64(value) => scalar(Storage::I64(vec![value])),
            TensorData::VectorI64(data) => vector(Storage::I64(data)),
            TensorData::MatrixI64(data) => matrix(data, Storage::I64),
            TensorData::FlatI64 { data, shape } => flat(data, shape, Storage::I64),

            TensorData::ScalarBool(value) => scalar(Storage::Bool(vec![value])),
            TensorData::VectorBool(data) => vector(Storage::Bool(data)),
            TensorData::MatrixBool(data) => matrix(data, Storage::Bool),
            TensorData::FlatBool { data, shape } => flat(data, shape, Storage::Bool),
        }
    }
}

impl From<f32> for TensorData {
    fn from(value: f32) -> Self {
        TensorData::ScalarF32(value)
    }
}

impl From<Vec<f32>> for TensorData {
    fn from(data: Vec<f32>) -> Self {
        TensorData::VectorF32(data)
    }
}

impl From<Vec<Vec<f32>>> for TensorData {
    fn from(data: Vec<Vec<f32>>) -> Self {
        TensorData::MatrixF32(data)
    }
}

impl From<f64> for TensorData {
    fn from(value: f64) -> Self {
        TensorData::ScalarF64(value)
    }
}

impl From<Vec<f64>> for TensorData {
    fn from(data: Vec<f64>) -> Self {
        TensorData::VectorF64(data)
    }
}

impl From<Vec<Vec<f64>>> for TensorData {
    fn from(data: Vec<Vec<f64>>) -> Self {
        TensorData::MatrixF64(data)
    }
}

impl From<Complex64> for TensorData {
    fn from(value: Complex64) -> Self {
        TensorData::ScalarC64(value)
    }
}

impl From<Vec<Complex64>> for TensorData {
    fn from(data: Vec<Complex64>) -> Self {
        TensorData::VectorC64(data)
    }
}

impl From<Vec<Vec<Complex64>>> for TensorData {
    fn from(data: Vec<Vec<Complex64>>) -> Self {
        TensorData::MatrixC64(data)
    }
}

impl From<i64> for TensorData {
    fn from(value: i64) -> Self {
        TensorData::ScalarI64(value)
    }
}

impl From<Vec<i64>> for TensorData {
    fn from(data: Vec<i64>) -> Self {
        TensorData::VectorI64(data)
    }
}

impl From<Vec<Vec<i64>>> for TensorData {
    fn from(data: Vec<Vec<i64>>) -> Self {
        TensorData::MatrixI64(data)
    }
}

impl From<bool> for TensorData {
    fn from(value: bool) -> Self {
        TensorData::ScalarBool(value)
    }
}

impl From<Vec<bool>> for TensorData {
    fn from(data: Vec<bool>) -> Self {
        TensorData::VectorBool(data)
    }
}

impl From<Vec<Vec<bool>>> for TensorData {
    fn from(data: Vec<Vec<bool>>) -> Self {
        TensorData::MatrixBool(data)
    }
}

/// 为标量数据构造形状为 `[]` 的 CPU Dense 元数据。
fn scalar(storage: Storage) -> Result<(Storage, TensorMeta)> {
    let dtype = storage.dtype();
    let meta = TensorMeta::new(vec![], dtype, Device::Cpu, Layout::Dense)?;

    Ok((storage, meta))
}

/// 为一维数据构造形状为 `[storage.len()]` 的 CPU Dense 元数据。
fn vector(storage: Storage) -> Result<(Storage, TensorMeta)> {
    let shape = vec![storage.len()];
    let dtype = storage.dtype();
    let meta = TensorMeta::new(shape, dtype, Device::Cpu, Layout::Dense)?;

    Ok((storage, meta))
}

/// 将规则二维数据展平，并构造 CPU Dense 元数据。
fn matrix<T>(
    data: Vec<Vec<T>>,
    storage_fn: fn(Vec<T>) -> Storage,
) -> Result<(Storage, TensorMeta)> {
    let (shape, flat_data) = flatten_matrix(data)?;
    flat(flat_data, shape, storage_fn)
}

/// 使用显式形状为扁平数据构造 CPU Dense 元数据。
fn flat<T>(
    data: Vec<T>,
    shape: Vec<usize>,
    storage_fn: fn(Vec<T>) -> Storage,
) -> Result<(Storage, TensorMeta)> {
    validate_flat_len(data.len(), &shape)?;

    let storage = storage_fn(data);
    let dtype = storage.dtype();
    let meta = TensorMeta::new(shape, dtype, Device::Cpu, Layout::Dense)?;

    Ok((storage, meta))
}

/// 验证二维数据为规则矩阵，并按行优先顺序展平。
fn flatten_matrix<T>(data: Vec<Vec<T>>) -> Result<(Vec<usize>, Vec<T>)> {
    let rows = data.len();

    if rows == 0 {
        return Err(ShapeError(
            "matrix row count must be greater than 0".to_string(),
        ));
    }

    let cols = data[0].len();

    if cols == 0 {
        return Err(ShapeError(
            "matrix column count must be greater than 0".to_string(),
        ));
    }

    for row in data.iter() {
        if row.len() != cols {
            return Err(ShapeError(
                "matrix rows must have the same length".to_string(),
            ));
        }
    }

    let flat = data.into_iter().flatten().collect();

    Ok((vec![rows, cols], flat))
}

/// 检查输入数据长度是否和 shape 匹配。
fn validate_flat_len(data_len: usize, shape: &[usize]) -> Result<()> {
    let expected = checked_numel(shape)?;

    if data_len != expected {
        return Err(ShapeError(format!(
            "data length does not match shape, data.len = {}, shape numel = {}, shape = {:?}",
            data_len, expected, shape
        )));
    }

    Ok(())
}

/// 根据 shape 计算应有的元素总数。
pub(crate) fn checked_numel(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim)
            .ok_or_else(|| ShapeError(format!("shape numel overflow, shape = {:?}", shape)))
    })
}
