use crate::error::{LinalgError::*, LinalgResult};
use matrixmultiply::CGemmOption;
use matrixmultiply::{dgemm, sgemm, zgemm};
use num_complex::Complex64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// `f32` 行主序连续矩阵乘法：`[m, k] @ [k, n] -> [m, n]`。
///
/// # Errors
///
/// 当矩阵元素数计算溢出或任一缓冲区长度不匹配时返回错误；仅 `i64` 内核还会在乘加溢出时返回错误。
pub fn gemm_f32(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> LinalgResult<()> {
    validate_gemm_buffers(
        "f32 matrix multiplication",
        a.len(),
        b.len(),
        out.len(),
        m,
        k,
        n,
    )?;

    unsafe {
        sgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }

    Ok(())
}

/// `f64` 行主序连续矩阵乘法：`[m, k] @ [k, n] -> [m, n]`。
///
/// # Errors
///
/// 当矩阵元素数计算溢出或任一缓冲区长度不匹配时返回错误；仅 `i64` 内核还会在乘加溢出时返回错误。
pub fn gemm_f64(
    a: &[f64],
    b: &[f64],
    out: &mut [f64],
    m: usize,
    k: usize,
    n: usize,
) -> LinalgResult<()> {
    validate_gemm_buffers(
        "f64 matrix multiplication",
        a.len(),
        b.len(),
        out.len(),
        m,
        k,
        n,
    )?;

    unsafe {
        dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }

    Ok(())
}

/// `Complex64` 行主序连续矩阵乘法，底层调用 `matrixmultiply::zgemm`。
///
/// # Errors
///
/// 当矩阵元素数计算溢出或任一缓冲区长度不匹配时返回错误；仅 `i64` 内核还会在乘加溢出时返回错误。
pub fn gemm_c64(
    a: &[Complex64],
    b: &[Complex64],
    out: &mut [Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> LinalgResult<()> {
    validate_gemm_buffers(
        "c64 matrix multiplication",
        a.len(),
        b.len(),
        out.len(),
        m,
        k,
        n,
    )?;

    unsafe {
        zgemm(
            CGemmOption::Standard,
            CGemmOption::Standard,
            m,
            k,
            n,
            [1.0, 0.0],
            a.as_ptr() as *const [f64; 2],
            k as isize,
            1,
            b.as_ptr() as *const [f64; 2],
            n as isize,
            1,
            [0.0, 0.0],
            out.as_mut_ptr() as *mut [f64; 2],
            n as isize,
            1,
        );
    }

    Ok(())
}

/// `i64` 行主序连续矩阵乘法；启用 `parallel` 特性时按输出元素并行计算。
///
/// 每次乘法与累加均执行溢出检查。
///
/// # Errors
///
/// 当矩阵元素数计算溢出或任一缓冲区长度不匹配时返回错误；仅 `i64` 内核还会在乘加溢出时返回错误。
pub fn gemm_i64(
    a: &[i64],
    b: &[i64],
    out: &mut [i64],
    m: usize,
    k: usize,
    n: usize,
) -> LinalgResult<()> {
    validate_gemm_buffers(
        "i64 matrix multiplication",
        a.len(),
        b.len(),
        out.len(),
        m,
        k,
        n,
    )?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut()
            .enumerate()
            .try_for_each(|(out_index, dst)| -> LinalgResult<()> {
                let row = out_index / n;
                let col = out_index % n;
                let mut acc = 0_i64;

                for inner in 0..k {
                    let product = a[row * k + inner].checked_mul(b[inner * n + col]).ok_or(
                        IntegerOverflowError {
                            op: "i64 matrix multiplication",
                            index: out_index,
                        },
                    )?;
                    acc = acc.checked_add(product).ok_or(IntegerOverflowError {
                        op: "i64 matrix multiplication",
                        index: out_index,
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
            for col in 0..n {
                let out_index = row * n + col;
                let mut acc = 0_i64;

                for inner in 0..k {
                    let product = a[row * k + inner].checked_mul(b[inner * n + col]).ok_or(
                        IntegerOverflowError {
                            op: "i64 matrix multiplication",
                            index: out_index,
                        },
                    )?;
                    acc = acc.checked_add(product).ok_or(IntegerOverflowError {
                        op: "i64 matrix multiplication",
                        index: out_index,
                    })?;
                }

                out[out_index] = acc;
            }
        }

        Ok(())
    }
}

/// 校验 GEMM 输入输出 buffer 长度。
fn validate_gemm_buffers(
    op: &'static str,
    a_len: usize,
    b_len: usize,
    out_len: usize,
    m: usize,
    k: usize,
    n: usize,
) -> LinalgResult<()> {
    let expected_a = checked_mul_shape_size(op, m, k)?;
    let expected_b = checked_mul_shape_size(op, k, n)?;
    let expected_out = checked_mul_shape_size(op, m, n)?;

    if a_len != expected_a {
        return Err(InvalidBufferLengthError {
            op,
            expected: expected_a,
            actual: a_len,
        });
    }

    if b_len != expected_b {
        return Err(InvalidBufferLengthError {
            op,
            expected: expected_b,
            actual: b_len,
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
