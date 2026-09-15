use crate::error::{LinalgError::*, LinalgResult};
use crate::ops::ensure_contiguous_tensor;
use arcqml_core::{DType, Device, Layout, Storage, Tensor, TensorMeta};
use ndarray::{ArrayD, IxDyn};
use num_complex::Complex64;

/// 把 f32 Tensor 转换为 ndarray ArrayD。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局或存储与目标 ndarray 类型不兼容时返回错误。
pub fn to_arrayd_f32(tensor: &Tensor) -> LinalgResult<ArrayD<f32>> {
    tensor_to_array("Tensor to ndarray<f32>", tensor, Storage::as_f32_slice)
}

/// 把 f64 Tensor 转换为 ndarray ArrayD。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局或存储与目标 ndarray 类型不兼容时返回错误。
pub fn to_arrayd_f64(tensor: &Tensor) -> LinalgResult<ArrayD<f64>> {
    tensor_to_array("Tensor to ndarray<f64>", tensor, Storage::as_f64_slice)
}

/// 把 c64 Tensor 转换为 ndarray ArrayD。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局或存储与目标 ndarray 类型不兼容时返回错误。
pub fn to_arrayd_c64(tensor: &Tensor) -> LinalgResult<ArrayD<Complex64>> {
    tensor_to_array("Tensor to ndarray<c64>", tensor, Storage::as_c64_slice)
}

/// 把 i64 Tensor 转换为 ndarray ArrayD。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局或存储与目标 ndarray 类型不兼容时返回错误。
pub fn to_arrayd_i64(tensor: &Tensor) -> LinalgResult<ArrayD<i64>> {
    tensor_to_array("Tensor to ndarray<i64>", tensor, Storage::as_i64_slice)
}

/// 把 bool Tensor 转换为 ndarray ArrayD。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局或存储与目标 ndarray 类型不兼容时返回错误。
pub fn to_arrayd_bool(tensor: &Tensor) -> LinalgResult<ArrayD<bool>> {
    tensor_to_array("Tensor to ndarray<bool>", tensor, Storage::as_bool_slice)
}

/// 从 ndarray ArrayD 创建 f32 Tensor。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局或存储与目标 ndarray 类型不兼容时返回错误。
pub fn from_arrayd_f32(array: ArrayD<f32>) -> LinalgResult<Tensor> {
    array_to_tensor("ndarray<f32> to Tensor", array, DType::F32, Storage::F32)
}

/// 从 ndarray ArrayD 创建 f64 Tensor。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局或存储与目标 ndarray 类型不兼容时返回错误。
pub fn from_arrayd_f64(array: ArrayD<f64>) -> LinalgResult<Tensor> {
    array_to_tensor("ndarray<f64> to Tensor", array, DType::F64, Storage::F64)
}

/// 从 ndarray ArrayD 创建 c64 Tensor。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局或存储与目标 ndarray 类型不兼容时返回错误。
pub fn from_arrayd_c64(array: ArrayD<Complex64>) -> LinalgResult<Tensor> {
    array_to_tensor("ndarray<c64> to Tensor", array, DType::C64, Storage::C64)
}

/// 从 ndarray ArrayD 创建 i64 Tensor。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局或存储与目标 ndarray 类型不兼容时返回错误。
pub fn from_arrayd_i64(array: ArrayD<i64>) -> LinalgResult<Tensor> {
    array_to_tensor("ndarray<i64> to Tensor", array, DType::I64, Storage::I64)
}

/// 从 ndarray ArrayD 创建 bool Tensor。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局或存储与目标 ndarray 类型不兼容时返回错误。
pub fn from_arrayd_bool(array: ArrayD<bool>) -> LinalgResult<Tensor> {
    array_to_tensor("ndarray<bool> to Tensor", array, DType::Bool, Storage::Bool)
}

fn tensor_to_array<T: Clone>(
    op: &'static str,
    tensor: &Tensor,
    accessor: fn(&Storage) -> Option<&[T]>,
) -> LinalgResult<ArrayD<T>> {
    ensure_contiguous_tensor(op, tensor)?;

    let storage = tensor.storage();
    let data = accessor(&storage).ok_or(UnsupportedDTypeError {
        op,
        dtype: tensor.dtype().to_string(),
    })?;

    ArrayD::from_shape_vec(IxDyn(tensor.shape()), data.to_vec()).map_err(|err| TensorCreateError {
        op,
        message: err.to_string(),
    })
}

fn array_to_tensor<T: Clone>(
    op: &'static str,
    array: ArrayD<T>,
    dtype: DType,
    storage_fn: fn(Vec<T>) -> Storage,
) -> LinalgResult<Tensor> {
    let shape = array.shape().to_vec();
    let data = array.iter().cloned().collect();
    let storage = storage_fn(data);
    let meta = TensorMeta::new(shape, dtype, Device::Cpu, Layout::Dense).map_err(|_| {
        TensorCreateError {
            op,
            message: "arcqml-core could not create the tensor from ndarray data".to_string(),
        }
    })?;

    Tensor::from_storage_meta(storage, meta).map_err(|_| TensorCreateError {
        op,
        message: "arcqml-core could not create the tensor from ndarray data".to_string(),
    })
}
