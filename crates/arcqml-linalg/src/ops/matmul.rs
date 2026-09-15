use crate::autograd::MatmulBackward;
use crate::error::{LinalgError::*, LinalgResult};
use crate::kernels::{gemm_c64, gemm_f32, gemm_f64, gemm_i64};
use crate::ops::{ensure_binary_compatible, make_tensor_with_backward};
use crate::shape::infer_matmul_shape;
use arcqml_core::{Storage, Tensor};
use num_complex::Complex64;
use std::sync::Arc;

/// 执行二维 Tensor 矩阵乘法 `[m, k] @ [k, n]`。
///
/// 两个输入必须是同 dtype 的连续 CPU 稠密 Tensor；支持 `F32`、`F64`、`C64` 和 `I64`。
///
/// # Errors
///
/// 当输入不是同类型、连续且内维匹配的二维数值 Tensor，或底层矩阵内核校验失败时返回错误。
///
/// # Panics
///
/// 当前实现分配输出前直接计算 `m * n`；该乘积发生 `usize` 溢出时，调试构建会 panic。
/// 输出缓冲区无法分配时也可能 panic 或由分配器终止进程。
pub fn matmul(lhs: &Tensor, rhs: &Tensor) -> LinalgResult<Tensor> {
    ensure_binary_compatible("matrix multiplication", lhs, rhs)?;

    let out_shape = infer_matmul_shape(lhs.shape(), rhs.shape())?;
    let m = lhs.shape()[0];
    let k = lhs.shape()[1];
    let n = rhs.shape()[1];

    let lhs_storage = lhs.storage();
    let rhs_storage = rhs.storage();
    let storage = match (&*lhs_storage, &*rhs_storage) {
        (Storage::F32(a), Storage::F32(b)) => {
            let mut out = vec![0.0_f32; m * n];
            gemm_f32(a, b, &mut out, m, k, n)?;
            Storage::F32(out)
        }
        (Storage::F64(a), Storage::F64(b)) => {
            let mut out = vec![0.0_f64; m * n];
            gemm_f64(a, b, &mut out, m, k, n)?;
            Storage::F64(out)
        }
        (Storage::C64(a), Storage::C64(b)) => {
            let mut out = vec![Complex64::new(0.0, 0.0); m * n];
            gemm_c64(a, b, &mut out, m, k, n)?;
            Storage::C64(out)
        }
        (Storage::I64(a), Storage::I64(b)) => {
            let mut out = vec![0_i64; m * n];
            gemm_i64(a, b, &mut out, m, k, n)?;
            Storage::I64(out)
        }
        _ => {
            return Err(UnsupportedDTypeError {
                op: "matrix multiplication",
                dtype: lhs.dtype().to_string(),
            });
        }
    };

    make_tensor_with_backward(
        "matrix multiplication",
        storage,
        out_shape,
        lhs,
        vec![lhs.clone(), rhs.clone()],
        Arc::new(MatmulBackward),
    )
}
