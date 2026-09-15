use crate::error::{LinalgError::*, LinalgResult};
use num_complex::Complex64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// f32 逐元素加法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn add_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) -> LinalgResult<()> {
    binary_f32("f32 elementwise add", lhs, rhs, out, |a, b| a + b)
}

/// f64 逐元素加法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn add_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) -> LinalgResult<()> {
    binary_f64("f64 elementwise add", lhs, rhs, out, |a, b| a + b)
}

/// c64 逐元素加法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn add_c64(lhs: &[Complex64], rhs: &[Complex64], out: &mut [Complex64]) -> LinalgResult<()> {
    binary_c64("c64 elementwise add", lhs, rhs, out, |a, b| a + b)
}

/// i64 逐元素加法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn add_i64(lhs: &[i64], rhs: &[i64], out: &mut [i64]) -> LinalgResult<()> {
    binary_i64("i64 elementwise add", lhs, rhs, out, |a, b, index| {
        a.checked_add(b).ok_or(IntegerOverflowError {
            op: "i64 elementwise add",
            index,
        })
    })
}

/// f32 逐元素减法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn sub_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) -> LinalgResult<()> {
    binary_f32("f32 elementwise subtract", lhs, rhs, out, |a, b| a - b)
}

/// f64 逐元素减法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn sub_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) -> LinalgResult<()> {
    binary_f64("f64 elementwise subtract", lhs, rhs, out, |a, b| a - b)
}

/// c64 逐元素减法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn sub_c64(lhs: &[Complex64], rhs: &[Complex64], out: &mut [Complex64]) -> LinalgResult<()> {
    binary_c64("c64 elementwise subtract", lhs, rhs, out, |a, b| a - b)
}

/// i64 逐元素减法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn sub_i64(lhs: &[i64], rhs: &[i64], out: &mut [i64]) -> LinalgResult<()> {
    binary_i64("i64 elementwise subtract", lhs, rhs, out, |a, b, index| {
        a.checked_sub(b).ok_or(IntegerOverflowError {
            op: "i64 elementwise subtract",
            index,
        })
    })
}

/// f32 逐元素乘法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn mul_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) -> LinalgResult<()> {
    binary_f32("f32 elementwise multiply", lhs, rhs, out, |a, b| a * b)
}

/// f64 逐元素乘法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn mul_f64(lhs: &[f64], rhs: &[f64], out: &mut [f64]) -> LinalgResult<()> {
    binary_f64("f64 elementwise multiply", lhs, rhs, out, |a, b| a * b)
}

/// c64 逐元素乘法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn mul_c64(lhs: &[Complex64], rhs: &[Complex64], out: &mut [Complex64]) -> LinalgResult<()> {
    binary_c64("c64 elementwise multiply", lhs, rhs, out, |a, b| a * b)
}

/// i64 逐元素乘法。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn mul_i64(lhs: &[i64], rhs: &[i64], out: &mut [i64]) -> LinalgResult<()> {
    binary_i64("i64 elementwise multiply", lhs, rhs, out, |a, b, index| {
        a.checked_mul(b).ok_or(IntegerOverflowError {
            op: "i64 elementwise multiply",
            index,
        })
    })
}

/// f32 逐元素平方。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn square_f32(input: &[f32], out: &mut [f32]) -> LinalgResult<()> {
    unary_f32("f32 elementwise square", input, out, |x| x * x)
}

/// f64 逐元素平方。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn square_f64(input: &[f64], out: &mut [f64]) -> LinalgResult<()> {
    unary_f64("f64 elementwise square", input, out, |x| x * x)
}

/// c64 逐元素平方。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn square_c64(input: &[Complex64], out: &mut [Complex64]) -> LinalgResult<()> {
    unary_c64("c64 elementwise square", input, out, |x| x * x)
}

/// i64 逐元素平方。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn square_i64(input: &[i64], out: &mut [i64]) -> LinalgResult<()> {
    unary_i64("i64 elementwise square", input, out, |x, index| {
        x.checked_mul(x).ok_or(IntegerOverflowError {
            op: "i64 elementwise square",
            index,
        })
    })
}

/// f32 逐元素开方。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn sqrt_f32(input: &[f32], out: &mut [f32]) -> LinalgResult<()> {
    unary_f32("f32 elementwise sqrt", input, out, f32::sqrt)
}

/// f64 逐元素开方。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn sqrt_f64(input: &[f64], out: &mut [f64]) -> LinalgResult<()> {
    unary_f64("f64 elementwise sqrt", input, out, f64::sqrt)
}

/// c64 逐元素开方。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn sqrt_c64(input: &[Complex64], out: &mut [Complex64]) -> LinalgResult<()> {
    unary_c64("c64 elementwise sqrt", input, out, |x| x.sqrt())
}

/// f32 逐元素绝对值。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn abs_f32(input: &[f32], out: &mut [f32]) -> LinalgResult<()> {
    unary_f32("f32 elementwise abs", input, out, f32::abs)
}

/// f64 逐元素绝对值。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn abs_f64(input: &[f64], out: &mut [f64]) -> LinalgResult<()> {
    unary_f64("f64 elementwise abs", input, out, f64::abs)
}

/// c64 逐元素模长，输出为 f64。
///
/// # Errors
///
/// 当输入与输出缓冲区长度不一致时返回错误；仅 `i64` 算术内核还会在运算溢出时返回错误。
pub fn abs_c64(input: &[Complex64], out: &mut [f64]) -> LinalgResult<()> {
    validate_unary_len("c64 elementwise abs", input.len(), out.len())?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut().enumerate().for_each(|(index, dst)| {
            *dst = input[index].norm();
        });

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (dst, src) in out.iter_mut().zip(input.iter().copied()) {
            *dst = src.norm();
        }

        Ok(())
    }
}

/// f32 一元逐元素 kernel。
fn unary_f32(
    op: &'static str,
    input: &[f32],
    out: &mut [f32],
    f: impl Fn(f32) -> f32 + Sync + Send,
) -> LinalgResult<()> {
    validate_unary_len(op, input.len(), out.len())?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut().enumerate().for_each(|(index, dst)| {
            *dst = f(input[index]);
        });

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (dst, src) in out.iter_mut().zip(input.iter().copied()) {
            *dst = f(src);
        }

        Ok(())
    }
}

/// f64 一元逐元素 kernel。
fn unary_f64(
    op: &'static str,
    input: &[f64],
    out: &mut [f64],
    f: impl Fn(f64) -> f64 + Sync + Send,
) -> LinalgResult<()> {
    validate_unary_len(op, input.len(), out.len())?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut().enumerate().for_each(|(index, dst)| {
            *dst = f(input[index]);
        });

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (dst, src) in out.iter_mut().zip(input.iter().copied()) {
            *dst = f(src);
        }

        Ok(())
    }
}

/// c64 一元逐元素 kernel。
fn unary_c64(
    op: &'static str,
    input: &[Complex64],
    out: &mut [Complex64],
    f: impl Fn(Complex64) -> Complex64 + Sync + Send,
) -> LinalgResult<()> {
    validate_unary_len(op, input.len(), out.len())?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut().enumerate().for_each(|(index, dst)| {
            *dst = f(input[index]);
        });

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (dst, src) in out.iter_mut().zip(input.iter().copied()) {
            *dst = f(src);
        }

        Ok(())
    }
}

/// i64 一元逐元素 kernel。
fn unary_i64(
    op: &'static str,
    input: &[i64],
    out: &mut [i64],
    f: impl Fn(i64, usize) -> LinalgResult<i64> + Sync + Send,
) -> LinalgResult<()> {
    validate_unary_len(op, input.len(), out.len())?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut()
            .enumerate()
            .try_for_each(|(index, dst)| -> LinalgResult<()> {
                *dst = f(input[index], index)?;
                Ok(())
            })?;

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (index, (dst, src)) in out.iter_mut().zip(input.iter().copied()).enumerate() {
            *dst = f(src, index)?;
        }

        Ok(())
    }
}

/// f32 二元逐元素 kernel。
fn binary_f32(
    op: &'static str,
    lhs: &[f32],
    rhs: &[f32],
    out: &mut [f32],
    f: impl Fn(f32, f32) -> f32 + Sync + Send,
) -> LinalgResult<()> {
    validate_binary_len(op, lhs.len(), rhs.len(), out.len())?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut().enumerate().for_each(|(index, dst)| {
            *dst = f(lhs[index], rhs[index]);
        });

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for ((dst, a), b) in out
            .iter_mut()
            .zip(lhs.iter().copied())
            .zip(rhs.iter().copied())
        {
            *dst = f(a, b);
        }

        Ok(())
    }
}

/// f64 二元逐元素 kernel。
fn binary_f64(
    op: &'static str,
    lhs: &[f64],
    rhs: &[f64],
    out: &mut [f64],
    f: impl Fn(f64, f64) -> f64 + Sync + Send,
) -> LinalgResult<()> {
    validate_binary_len(op, lhs.len(), rhs.len(), out.len())?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut().enumerate().for_each(|(index, dst)| {
            *dst = f(lhs[index], rhs[index]);
        });

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for ((dst, a), b) in out
            .iter_mut()
            .zip(lhs.iter().copied())
            .zip(rhs.iter().copied())
        {
            *dst = f(a, b);
        }

        Ok(())
    }
}

/// c64 二元逐元素 kernel。
fn binary_c64(
    op: &'static str,
    lhs: &[Complex64],
    rhs: &[Complex64],
    out: &mut [Complex64],
    f: impl Fn(Complex64, Complex64) -> Complex64 + Sync + Send,
) -> LinalgResult<()> {
    validate_binary_len(op, lhs.len(), rhs.len(), out.len())?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut().enumerate().for_each(|(index, dst)| {
            *dst = f(lhs[index], rhs[index]);
        });

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for ((dst, a), b) in out
            .iter_mut()
            .zip(lhs.iter().copied())
            .zip(rhs.iter().copied())
        {
            *dst = f(a, b);
        }

        Ok(())
    }
}

/// i64 二元逐元素 kernel。
fn binary_i64(
    op: &'static str,
    lhs: &[i64],
    rhs: &[i64],
    out: &mut [i64],
    f: impl Fn(i64, i64, usize) -> LinalgResult<i64> + Sync + Send,
) -> LinalgResult<()> {
    validate_binary_len(op, lhs.len(), rhs.len(), out.len())?;

    #[cfg(feature = "parallel")]
    {
        out.par_iter_mut()
            .enumerate()
            .try_for_each(|(index, dst)| -> LinalgResult<()> {
                *dst = f(lhs[index], rhs[index], index)?;
                Ok(())
            })?;

        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (index, ((dst, a), b)) in out
            .iter_mut()
            .zip(lhs.iter().copied())
            .zip(rhs.iter().copied())
            .enumerate()
        {
            *dst = f(a, b, index)?;
        }

        Ok(())
    }
}

/// 校验一元逐元素 kernel 的输入输出长度。
fn validate_unary_len(op: &'static str, input_len: usize, out_len: usize) -> LinalgResult<()> {
    if input_len != out_len {
        return Err(InvalidBufferLengthError {
            op,
            expected: input_len,
            actual: out_len,
        });
    }

    Ok(())
}

/// 校验二元逐元素 kernel 的输入输出长度。
fn validate_binary_len(
    op: &'static str,
    lhs_len: usize,
    rhs_len: usize,
    out_len: usize,
) -> LinalgResult<()> {
    if lhs_len != rhs_len {
        return Err(ShapeMismatchError {
            op,
            expected: format!(
                "left input length to equal right input length, got left length={lhs_len}"
            ),
            actual: format!("right input length={rhs_len}"),
        });
    }

    if lhs_len != out_len {
        return Err(InvalidBufferLengthError {
            op,
            expected: lhs_len,
            actual: out_len,
        });
    }

    Ok(())
}
