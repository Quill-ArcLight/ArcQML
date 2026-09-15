use crate::error::{LinalgError::*, LinalgResult};
use crate::ops::ensure_contiguous_tensor;
use arcqml_core::{Storage, Tensor};
use ndarray::{ArrayViewD, IxDyn};
use num_complex::Complex64;

/// 在读锁存活期间把 f32 Tensor 作为 ndarray view 使用。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局、步长或存储与借用 ndarray 视图不兼容时返回错误。
pub fn with_array_view_f32<R>(
    tensor: &Tensor,
    f: impl FnOnce(ArrayViewD<'_, f32>) -> R,
) -> LinalgResult<R> {
    tensor_with_view("Tensor view<f32>", tensor, Storage::as_f32_slice, f)
}

/// 在读锁存活期间把 f64 Tensor 作为 ndarray view 使用。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局、步长或存储与借用 ndarray 视图不兼容时返回错误。
pub fn with_array_view_f64<R>(
    tensor: &Tensor,
    f: impl FnOnce(ArrayViewD<'_, f64>) -> R,
) -> LinalgResult<R> {
    tensor_with_view("Tensor view<f64>", tensor, Storage::as_f64_slice, f)
}

/// 在读锁存活期间把 c64 Tensor 作为 ndarray view 使用。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局、步长或存储与借用 ndarray 视图不兼容时返回错误。
pub fn with_array_view_c64<R>(
    tensor: &Tensor,
    f: impl FnOnce(ArrayViewD<'_, Complex64>) -> R,
) -> LinalgResult<R> {
    tensor_with_view("Tensor view<c64>", tensor, Storage::as_c64_slice, f)
}

/// 在读锁存活期间把 i64 Tensor 作为 ndarray view 使用。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局、步长或存储与借用 ndarray 视图不兼容时返回错误。
pub fn with_array_view_i64<R>(
    tensor: &Tensor,
    f: impl FnOnce(ArrayViewD<'_, i64>) -> R,
) -> LinalgResult<R> {
    tensor_with_view("Tensor view<i64>", tensor, Storage::as_i64_slice, f)
}

/// 在读锁存活期间把 bool Tensor 作为 ndarray view 使用。
///
/// # Errors
///
/// 当 Tensor 数据类型、布局、步长或存储与借用 ndarray 视图不兼容时返回错误。
pub fn with_array_view_bool<R>(
    tensor: &Tensor,
    f: impl FnOnce(ArrayViewD<'_, bool>) -> R,
) -> LinalgResult<R> {
    tensor_with_view("Tensor view<bool>", tensor, Storage::as_bool_slice, f)
}

fn tensor_with_view<T, R>(
    op: &'static str,
    tensor: &Tensor,
    accessor: fn(&Storage) -> Option<&[T]>,
    f: impl FnOnce(ArrayViewD<'_, T>) -> R,
) -> LinalgResult<R> {
    ensure_contiguous_tensor(op, tensor)?;

    let storage = tensor.storage();
    let data = accessor(&storage).ok_or(UnsupportedDTypeError {
        op,
        dtype: tensor.dtype().to_string(),
    })?;

    let view =
        ArrayViewD::from_shape(IxDyn(tensor.shape()), data).map_err(|err| TensorCreateError {
            op,
            message: err.to_string(),
        })?;

    Ok(f(view))
}
