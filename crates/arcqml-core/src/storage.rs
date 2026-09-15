use crate::DType;

use num_complex::Complex64;

/// Tensor 拥有所有权的类型化元素缓冲区。
#[derive(Debug, Clone)]
pub enum Storage {
    /// 连续的单精度浮点缓冲区。
    F32(Vec<f32>),
    /// 连续的双精度浮点缓冲区。
    F64(Vec<f64>),
    /// 连续的双精度复数缓冲区。
    C64(Vec<Complex64>),
    /// 连续的 64 位有符号整数缓冲区。
    I64(Vec<i64>),
    /// 连续的布尔缓冲区。
    Bool(Vec<bool>),
}

impl Storage {
    /// 返回缓冲区元素对应的 [`DType`]。
    pub fn dtype(&self) -> DType {
        match self {
            Storage::F32(_) => DType::F32,
            Storage::F64(_) => DType::F64,
            Storage::C64(_) => DType::C64,
            Storage::I64(_) => DType::I64,
            Storage::Bool(_) => DType::Bool,
        }
    }

    /// 返回底层缓冲区的元素数量。
    pub fn len(&self) -> usize {
        match self {
            Storage::F32(v) => v.len(),
            Storage::F64(v) => v.len(),
            Storage::C64(v) => v.len(),
            Storage::I64(v) => v.len(),
            Storage::Bool(v) => v.len(),
        }
    }

    /// 判断底层缓冲区是否没有元素。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 当存储为 [`Storage::F32`] 时返回其只读切片，否则返回 `None`。
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        match self {
            Storage::F32(v) => Some(v),
            _ => None,
        }
    }

    /// 当存储为 [`Storage::F64`] 时返回其只读切片，否则返回 `None`。
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match self {
            Storage::F64(v) => Some(v),
            _ => None,
        }
    }

    /// 当存储为 [`Storage::C64`] 时返回其只读切片，否则返回 `None`。
    pub fn as_c64_slice(&self) -> Option<&[Complex64]> {
        match self {
            Storage::C64(v) => Some(v),
            _ => None,
        }
    }

    /// 当存储为 [`Storage::I64`] 时返回其只读切片，否则返回 `None`。
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        match self {
            Storage::I64(v) => Some(v),
            _ => None,
        }
    }

    /// 当存储为 [`Storage::Bool`] 时返回其只读切片，否则返回 `None`。
    pub fn as_bool_slice(&self) -> Option<&[bool]> {
        match self {
            Storage::Bool(v) => Some(v),
            _ => None,
        }
    }
}
