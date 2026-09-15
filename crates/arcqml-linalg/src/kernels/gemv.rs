use crate::error::{LinalgError::*, LinalgResult};
use num_complex::Complex64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// `f32` 行主序连续矩阵—向量乘法：`[m, n] @ [n] -> [m]`。
///
/// # Errors
///
/// 当矩阵元素数计算溢出或任一缓冲区长度不匹配时返回错误；仅 `i64` 内核还会在乘加溢出时返回错误。
pub fn gemv_f32(a: &[f32], x: &[f32], out: &mut [f32], m: usize, n: usize) -> LinalgResult<()> {
    validate_gemv_buffers(
        "f32 matrix-vector multiplication",
        a.len(),
        x.len(),
        out.len(),
        m,
        n,
    )?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut().enumerate().for_each(|(row, dst)| {
            let mut acc = 0.0_f32;

            for col in 0..n {
                acc += a[row * n + col] * x[col];
            }

            *dst = acc;
        });

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..m {
            let mut acc = 0.0_f32;

            for col in 0..n {
                acc += a[row * n + col] * x[col];
            }

            out[row] = acc;
        }

        Ok(())
    }
}

/// `f64` 行主序连续矩阵—向量乘法：`[m, n] @ [n] -> [m]`。
///
/// # Errors
///
/// 当矩阵元素数计算溢出或任一缓冲区长度不匹配时返回错误；仅 `i64` 内核还会在乘加溢出时返回错误。
pub fn gemv_f64(a: &[f64], x: &[f64], out: &mut [f64], m: usize, n: usize) -> LinalgResult<()> {
    validate_gemv_buffers(
        "f64 matrix-vector multiplication",
        a.len(),
        x.len(),
        out.len(),
        m,
        n,
    )?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut().enumerate().for_each(|(row, dst)| {
            let mut acc = 0.0_f64;

            for col in 0..n {
                acc += a[row * n + col] * x[col];
            }

            *dst = acc;
        });

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..m {
            let mut acc = 0.0_f64;

            for col in 0..n {
                acc += a[row * n + col] * x[col];
            }

            out[row] = acc;
        }

        Ok(())
    }
}

/// `Complex64` 行主序连续矩阵—向量乘法。
///
/// # Errors
///
/// 当矩阵元素数计算溢出或任一缓冲区长度不匹配时返回错误；仅 `i64` 内核还会在乘加溢出时返回错误。
pub fn gemv_c64(
    a: &[Complex64],
    x: &[Complex64],
    out: &mut [Complex64],
    m: usize,
    n: usize,
) -> LinalgResult<()> {
    validate_gemv_buffers(
        "c64 matrix-vector multiplication",
        a.len(),
        x.len(),
        out.len(),
        m,
        n,
    )?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut().enumerate().for_each(|(row, dst)| {
            let mut acc = Complex64::new(0.0, 0.0);

            for col in 0..n {
                acc += a[row * n + col] * x[col];
            }

            *dst = acc;
        });

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..m {
            let mut acc = Complex64::new(0.0, 0.0);

            for col in 0..n {
                acc += a[row * n + col] * x[col];
            }

            out[row] = acc;
        }

        Ok(())
    }
}

/// `i64` 行主序连续矩阵—向量乘法；启用 `parallel` 特性时按行并行计算。
///
/// 每次乘法与累加均执行溢出检查。
///
/// # Errors
///
/// 当矩阵元素数计算溢出或任一缓冲区长度不匹配时返回错误；仅 `i64` 内核还会在乘加溢出时返回错误。
pub fn gemv_i64(a: &[i64], x: &[i64], out: &mut [i64], m: usize, n: usize) -> LinalgResult<()> {
    validate_gemv_buffers(
        "i64 matrix-vector multiplication",
        a.len(),
        x.len(),
        out.len(),
        m,
        n,
    )?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut()
            .enumerate()
            .try_for_each(|(row, dst)| -> LinalgResult<()> {
                let mut acc = 0_i64;

                for col in 0..n {
                    let product =
                        a[row * n + col]
                            .checked_mul(x[col])
                            .ok_or(IntegerOverflowError {
                                op: "i64 matrix-vector multiplication",
                                index: row,
                            })?;
                    acc = acc.checked_add(product).ok_or(IntegerOverflowError {
                        op: "i64 matrix-vector multiplication",
                        index: row,
                    })?;
                }

                *dst = acc;
                Ok(())
            })?;

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..m {
            let mut acc = 0_i64;

            for col in 0..n {
                let product = a[row * n + col]
                    .checked_mul(x[col])
                    .ok_or(IntegerOverflowError {
                        op: "i64 matrix-vector multiplication",
                        index: row,
                    })?;
                acc = acc.checked_add(product).ok_or(IntegerOverflowError {
                    op: "i64 matrix-vector multiplication",
                    index: row,
                })?;
            }

            out[row] = acc;
        }

        Ok(())
    }
}

/// 校验 GEMV 输入输出 buffer 长度。
fn validate_gemv_buffers(
    op: &'static str,
    a_len: usize,
    x_len: usize,
    out_len: usize,
    m: usize,
    n: usize,
) -> LinalgResult<()> {
    let expected_a = checked_mul_shape_size(op, m, n)?;
    let expected_x = n;
    let expected_out = m;

    if a_len != expected_a {
        return Err(InvalidBufferLengthError {
            op,
            expected: expected_a,
            actual: a_len,
        });
    }

    if x_len != expected_x {
        return Err(InvalidBufferLengthError {
            op,
            expected: expected_x,
            actual: x_len,
        });
    }

    if out_len != expected_out {
        return Err(InvalidBufferLengthError {
            op,
            expected: expected_out,
            actual: out_len,
        });
    }

    Ok(())
}

/// 计算 shape 元素数量时防止 usize 溢出。
fn checked_mul_shape_size(op: &'static str, lhs: usize, rhs: usize) -> LinalgResult<usize> {
    lhs.checked_mul(rhs)
        .ok_or(ShapeSizeOverflowError { op, lhs, rhs })
}
