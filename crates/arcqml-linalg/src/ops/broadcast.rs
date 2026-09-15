use crate::autograd::{BinaryBackward, BinaryKind, UnaryBackward, UnaryKind};
use crate::error::{LinalgError::*, LinalgResult};
use crate::kernels::{
    abs_c64, abs_f32, abs_f64, add_c64, add_f32, add_f64, add_i64, mul_c64, mul_f32, mul_f64,
    mul_i64, sqrt_c64, sqrt_f32, sqrt_f64, square_c64, square_f32, square_f64, square_i64, sub_c64,
    sub_f32, sub_f64, sub_i64,
};
use crate::ops::{
    broadcast_index, ensure_binary_compatible, ensure_contiguous_tensor, make_tensor_with_backward,
};
use crate::shape::{infer_broadcast_shape, numel};
use arcqml_core::{DType, Storage, Tensor};
use num_complex::Complex64;
use std::sync::Arc;

/// Tensor 逐元素加法，支持 NumPy 风格广播。
///
/// # Errors
///
/// 当输入的数据类型、设备或形状不兼容，或广播后的元素数量溢出时返回错误。
pub fn add(lhs: &Tensor, rhs: &Tensor) -> LinalgResult<Tensor> {
    binary_op(
        "elementwise add",
        BinaryKind::Add,
        lhs,
        rhs,
        add_same_shape,
        add_broadcast,
    )
}

/// Tensor 逐元素减法，支持 NumPy 风格广播。
///
/// # Errors
///
/// 当输入的数据类型、设备或形状不兼容，或广播后的元素数量溢出时返回错误。
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> LinalgResult<Tensor> {
    binary_op(
        "elementwise subtract",
        BinaryKind::Sub,
        lhs,
        rhs,
        sub_same_shape,
        sub_broadcast,
    )
}

/// Tensor 逐元素乘法，支持 NumPy 风格广播。
///
/// # Errors
///
/// 当输入的数据类型、设备或形状不兼容，或广播后的元素数量溢出时返回错误。
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> LinalgResult<Tensor> {
    binary_op(
        "elementwise multiply",
        BinaryKind::Mul,
        lhs,
        rhs,
        mul_same_shape,
        mul_broadcast,
    )
}

/// 计算逐元素平方；支持 `F32`、`F64`、`C64` 和 `I64`。
///
/// # Errors
///
/// 当输入不是连续的 CPU 稠密 Tensor、dtype 为 `Bool`、`I64` 平方溢出，
/// 或无法构造输出时返回错误。
pub fn square(input: &Tensor) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("elementwise square", input)?;

    let storage = match &*input.storage() {
        Storage::F32(v) => {
            let mut out = vec![0.0_f32; v.len()];
            square_f32(v, &mut out)?;
            Storage::F32(out)
        }
        Storage::F64(v) => {
            let mut out = vec![0.0_f64; v.len()];
            square_f64(v, &mut out)?;
            Storage::F64(out)
        }
        Storage::C64(v) => {
            let mut out = vec![Complex64::new(0.0, 0.0); v.len()];
            square_c64(v, &mut out)?;
            Storage::C64(out)
        }
        Storage::I64(v) => {
            let mut out = vec![0_i64; v.len()];
            square_i64(v, &mut out)?;
            Storage::I64(out)
        }
        Storage::Bool(_) => {
            return Err(UnsupportedDTypeError {
                op: "elementwise square",
                dtype: input.dtype().to_string(),
            });
        }
    };

    make_tensor_with_backward(
        "elementwise square",
        storage,
        input.shape().to_vec(),
        input,
        vec![input.clone()],
        Arc::new(UnaryBackward {
            kind: UnaryKind::Square,
        }),
    )
}

/// 计算逐元素平方根；复数输入使用主值平方根。
///
/// # Errors
///
/// 当输入不是连续的 CPU 稠密 Tensor、dtype 不是 `F32`/`F64`/`C64`，
/// 或无法构造输出时返回错误。
pub fn sqrt(input: &Tensor) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("elementwise sqrt", input)?;

    let storage = match &*input.storage() {
        Storage::F32(v) => {
            let mut out = vec![0.0_f32; v.len()];
            sqrt_f32(v, &mut out)?;
            Storage::F32(out)
        }
        Storage::F64(v) => {
            let mut out = vec![0.0_f64; v.len()];
            sqrt_f64(v, &mut out)?;
            Storage::F64(out)
        }
        Storage::C64(v) => {
            let mut out = vec![Complex64::new(0.0, 0.0); v.len()];
            sqrt_c64(v, &mut out)?;
            Storage::C64(out)
        }
        Storage::I64(_) | Storage::Bool(_) => {
            return Err(UnsupportedDTypeError {
                op: "elementwise sqrt",
                dtype: input.dtype().to_string(),
            });
        }
    };

    make_tensor_with_backward(
        "elementwise sqrt",
        storage,
        input.shape().to_vec(),
        input,
        vec![input.clone()],
        Arc::new(UnaryBackward {
            kind: UnaryKind::Sqrt,
        }),
    )
}

/// 计算逐元素绝对值；`C64` 输入返回各元素模长组成的 `F64` Tensor。
///
/// # Errors
///
/// 当输入不是连续的 CPU 稠密 Tensor、dtype 不是 `F32`/`F64`/`C64`，
/// 或无法构造输出时返回错误。
pub fn abs(input: &Tensor) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("elementwise abs", input)?;

    let storage = match &*input.storage() {
        Storage::F32(v) => {
            let mut out = vec![0.0_f32; v.len()];
            abs_f32(v, &mut out)?;
            Storage::F32(out)
        }
        Storage::F64(v) => {
            let mut out = vec![0.0_f64; v.len()];
            abs_f64(v, &mut out)?;
            Storage::F64(out)
        }
        Storage::C64(v) => {
            let mut out = vec![0.0_f64; v.len()];
            abs_c64(v, &mut out)?;
            Storage::F64(out)
        }
        Storage::I64(_) | Storage::Bool(_) => {
            return Err(UnsupportedDTypeError {
                op: "elementwise abs",
                dtype: input.dtype().to_string(),
            });
        }
    };

    make_tensor_with_backward(
        "elementwise abs",
        storage,
        input.shape().to_vec(),
        input,
        vec![input.clone()],
        Arc::new(UnaryBackward {
            kind: UnaryKind::Abs,
        }),
    )
}

/// 二元逐元素操作。
fn binary_op(
    op: &'static str,
    kind: BinaryKind,
    lhs: &Tensor,
    rhs: &Tensor,
    same_shape: fn(&Tensor, &Tensor) -> LinalgResult<Storage>,
    broadcast: fn(&Tensor, &Tensor, &[usize]) -> LinalgResult<Storage>,
) -> LinalgResult<Tensor> {
    ensure_binary_compatible(op, lhs, rhs)?;

    let out_shape = infer_broadcast_shape(lhs.shape(), rhs.shape())?;
    let storage = if lhs.shape() == rhs.shape() {
        same_shape(lhs, rhs)?
    } else {
        broadcast(lhs, rhs, &out_shape)?
    };

    make_tensor_with_backward(
        op,
        storage,
        out_shape.clone(),
        lhs,
        vec![lhs.clone(), rhs.clone()],
        Arc::new(BinaryBackward {
            kind,
            output_shape: out_shape,
        }),
    )
}

fn add_same_shape(lhs: &Tensor, rhs: &Tensor) -> LinalgResult<Storage> {
    let lhs_storage = lhs.storage();
    let rhs_storage = rhs.storage();

    match (&*lhs_storage, &*rhs_storage) {
        (Storage::F32(a), Storage::F32(b)) => {
            let mut out = vec![0.0_f32; a.len()];
            add_f32(a, b, &mut out)?;
            Ok(Storage::F32(out))
        }
        (Storage::F64(a), Storage::F64(b)) => {
            let mut out = vec![0.0_f64; a.len()];
            add_f64(a, b, &mut out)?;
            Ok(Storage::F64(out))
        }
        (Storage::C64(a), Storage::C64(b)) => {
            let mut out = vec![Complex64::new(0.0, 0.0); a.len()];
            add_c64(a, b, &mut out)?;
            Ok(Storage::C64(out))
        }
        (Storage::I64(a), Storage::I64(b)) => {
            let mut out = vec![0_i64; a.len()];
            add_i64(a, b, &mut out)?;
            Ok(Storage::I64(out))
        }
        _ => unsupported_binary_dtype("elementwise add", lhs.dtype()),
    }
}

fn sub_same_shape(lhs: &Tensor, rhs: &Tensor) -> LinalgResult<Storage> {
    let lhs_storage = lhs.storage();
    let rhs_storage = rhs.storage();

    match (&*lhs_storage, &*rhs_storage) {
        (Storage::F32(a), Storage::F32(b)) => {
            let mut out = vec![0.0_f32; a.len()];
            sub_f32(a, b, &mut out)?;
            Ok(Storage::F32(out))
        }
        (Storage::F64(a), Storage::F64(b)) => {
            let mut out = vec![0.0_f64; a.len()];
            sub_f64(a, b, &mut out)?;
            Ok(Storage::F64(out))
        }
        (Storage::C64(a), Storage::C64(b)) => {
            let mut out = vec![Complex64::new(0.0, 0.0); a.len()];
            sub_c64(a, b, &mut out)?;
            Ok(Storage::C64(out))
        }
        (Storage::I64(a), Storage::I64(b)) => {
            let mut out = vec![0_i64; a.len()];
            sub_i64(a, b, &mut out)?;
            Ok(Storage::I64(out))
        }
        _ => unsupported_binary_dtype("elementwise subtract", lhs.dtype()),
    }
}

fn mul_same_shape(lhs: &Tensor, rhs: &Tensor) -> LinalgResult<Storage> {
    let lhs_storage = lhs.storage();
    let rhs_storage = rhs.storage();

    match (&*lhs_storage, &*rhs_storage) {
        (Storage::F32(a), Storage::F32(b)) => {
            let mut out = vec![0.0_f32; a.len()];
            mul_f32(a, b, &mut out)?;
            Ok(Storage::F32(out))
        }
        (Storage::F64(a), Storage::F64(b)) => {
            let mut out = vec![0.0_f64; a.len()];
            mul_f64(a, b, &mut out)?;
            Ok(Storage::F64(out))
        }
        (Storage::C64(a), Storage::C64(b)) => {
            let mut out = vec![Complex64::new(0.0, 0.0); a.len()];
            mul_c64(a, b, &mut out)?;
            Ok(Storage::C64(out))
        }
        (Storage::I64(a), Storage::I64(b)) => {
            let mut out = vec![0_i64; a.len()];
            mul_i64(a, b, &mut out)?;
            Ok(Storage::I64(out))
        }
        _ => unsupported_binary_dtype("elementwise multiply", lhs.dtype()),
    }
}

fn add_broadcast(lhs: &Tensor, rhs: &Tensor, out_shape: &[usize]) -> LinalgResult<Storage> {
    binary_broadcast(
        lhs,
        rhs,
        out_shape,
        |a, b| a + b,
        |a, b| a + b,
        |a, b| a + b,
        |a, b, index| {
            a.checked_add(b).ok_or(IntegerOverflowError {
                op: "elementwise add",
                index,
            })
        },
    )
}

fn sub_broadcast(lhs: &Tensor, rhs: &Tensor, out_shape: &[usize]) -> LinalgResult<Storage> {
    binary_broadcast(
        lhs,
        rhs,
        out_shape,
        |a, b| a - b,
        |a, b| a - b,
        |a, b| a - b,
        |a, b, index| {
            a.checked_sub(b).ok_or(IntegerOverflowError {
                op: "elementwise subtract",
                index,
            })
        },
    )
}

fn mul_broadcast(lhs: &Tensor, rhs: &Tensor, out_shape: &[usize]) -> LinalgResult<Storage> {
    binary_broadcast(
        lhs,
        rhs,
        out_shape,
        |a, b| a * b,
        |a, b| a * b,
        |a, b| a * b,
        |a, b, index| {
            a.checked_mul(b).ok_or(IntegerOverflowError {
                op: "elementwise multiply",
                index,
            })
        },
    )
}

fn binary_broadcast(
    lhs: &Tensor,
    rhs: &Tensor,
    out_shape: &[usize],
    f32_op: impl Fn(f32, f32) -> f32,
    f64_op: impl Fn(f64, f64) -> f64,
    c64_op: impl Fn(Complex64, Complex64) -> Complex64,
    i64_op: impl Fn(i64, i64, usize) -> LinalgResult<i64>,
) -> LinalgResult<Storage> {
    let out_len = numel(out_shape);

    let lhs_storage = lhs.storage();
    let rhs_storage = rhs.storage();

    match (&*lhs_storage, &*rhs_storage) {
        (Storage::F32(a), Storage::F32(b)) => {
            let mut out = Vec::with_capacity(out_len);

            for index in 0..out_len {
                let lhs_index = broadcast_index(index, out_shape, lhs.shape());
                let rhs_index = broadcast_index(index, out_shape, rhs.shape());
                out.push(f32_op(a[lhs_index], b[rhs_index]));
            }

            Ok(Storage::F32(out))
        }
        (Storage::F64(a), Storage::F64(b)) => {
            let mut out = Vec::with_capacity(out_len);

            for index in 0..out_len {
                let lhs_index = broadcast_index(index, out_shape, lhs.shape());
                let rhs_index = broadcast_index(index, out_shape, rhs.shape());
                out.push(f64_op(a[lhs_index], b[rhs_index]));
            }

            Ok(Storage::F64(out))
        }
        (Storage::C64(a), Storage::C64(b)) => {
            let mut out = Vec::with_capacity(out_len);

            for index in 0..out_len {
                let lhs_index = broadcast_index(index, out_shape, lhs.shape());
                let rhs_index = broadcast_index(index, out_shape, rhs.shape());
                out.push(c64_op(a[lhs_index], b[rhs_index]));
            }

            Ok(Storage::C64(out))
        }
        (Storage::I64(a), Storage::I64(b)) => {
            let mut out = Vec::with_capacity(out_len);

            for index in 0..out_len {
                let lhs_index = broadcast_index(index, out_shape, lhs.shape());
                let rhs_index = broadcast_index(index, out_shape, rhs.shape());
                out.push(i64_op(a[lhs_index], b[rhs_index], index)?);
            }

            Ok(Storage::I64(out))
        }
        _ => unsupported_binary_dtype("broadcast elementwise operation", lhs.dtype()),
    }
}

fn unsupported_binary_dtype<T>(op: &'static str, dtype: DType) -> LinalgResult<T> {
    Err(UnsupportedDTypeError {
        op,
        dtype: dtype.to_string(),
    })
}
