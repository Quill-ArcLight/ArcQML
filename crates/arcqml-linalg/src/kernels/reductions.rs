use crate::error::{LinalgError::*, LinalgResult};
use num_complex::Complex64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// f32 求和。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn sum_f32(input: &[f32]) -> LinalgResult<f32> {
    ensure_non_empty("f32 sum", input)?;

    #[cfg(feature = "parallel")]
    {
        Ok(input.par_iter().copied().sum())
    }

    #[cfg(not(feature = "parallel"))]
    {
        Ok(input.iter().copied().sum())
    }
}

/// f64 求和。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn sum_f64(input: &[f64]) -> LinalgResult<f64> {
    ensure_non_empty("f64 sum", input)?;

    #[cfg(feature = "parallel")]
    {
        Ok(input.par_iter().copied().sum())
    }

    #[cfg(not(feature = "parallel"))]
    {
        Ok(input.iter().copied().sum())
    }
}

/// c64 求和。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn sum_c64(input: &[Complex64]) -> LinalgResult<Complex64> {
    ensure_non_empty("c64 sum", input)?;

    #[cfg(feature = "parallel")]
    {
        Ok(input.par_iter().copied().sum())
    }

    #[cfg(not(feature = "parallel"))]
    {
        Ok(input.iter().copied().sum())
    }
}

/// i64 求和，默认启用 rayon 后按 chunk 并行计算，并用 checked_add 防止溢出。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn sum_i64(input: &[i64]) -> LinalgResult<i64> {
    ensure_non_empty("i64 sum", input)?;

    #[cfg(feature = "parallel")]
    {
        const CHUNK_SIZE: usize = 4096;

        input
            .par_chunks(CHUNK_SIZE)
            .enumerate()
            .map(|(chunk_index, chunk)| -> LinalgResult<i64> {
                let mut acc = 0_i64;
                let base = chunk_index * CHUNK_SIZE;

                for (offset, value) in chunk.iter().copied().enumerate() {
                    acc = acc.checked_add(value).ok_or(IntegerOverflowError {
                        op: "i64 sum",
                        index: base + offset,
                    })?;
                }

                Ok(acc)
            })
            .try_reduce(
                || 0_i64,
                |a, b| {
                    a.checked_add(b).ok_or(IntegerOverflowError {
                        op: "i64 sum",
                        index: 0,
                    })
                },
            )
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut acc = 0_i64;

        for (index, value) in input.iter().copied().enumerate() {
            acc = acc.checked_add(value).ok_or(IntegerOverflowError {
                op: "i64 sum",
                index,
            })?;
        }

        Ok(acc)
    }
}

/// f32 求均值。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn mean_f32(input: &[f32]) -> LinalgResult<f32> {
    ensure_non_empty("f32 mean", input)?;
    Ok(sum_f32(input)? / input.len() as f32)
}

/// f64 求均值。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn mean_f64(input: &[f64]) -> LinalgResult<f64> {
    ensure_non_empty("f64 mean", input)?;
    Ok(sum_f64(input)? / input.len() as f64)
}

/// c64 求均值。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn mean_c64(input: &[Complex64]) -> LinalgResult<Complex64> {
    ensure_non_empty("c64 mean", input)?;
    Ok(sum_c64(input)? / Complex64::new(input.len() as f64, 0.0))
}

/// i64 求均值，使用整数除法，适合整数统计，不适合作为浮点训练 loss。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn mean_i64(input: &[i64]) -> LinalgResult<i64> {
    ensure_non_empty("i64 mean", input)?;
    let len = i64::try_from(input.len()).map_err(|_| IntegerOverflowError {
        op: "i64 mean",
        index: input.len(),
    })?;
    Ok(sum_i64(input)? / len)
}

/// f32 求最大值。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn max_f32(input: &[f32]) -> LinalgResult<f32> {
    ensure_non_empty("f32 max", input)?;

    #[cfg(feature = "parallel")]
    {
        input
            .par_iter()
            .copied()
            .reduce_with(f32::max)
            .ok_or(EmptyInputError { op: "f32 max" })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut value = input[0];

        for x in input.iter().copied().skip(1) {
            value = value.max(x);
        }

        Ok(value)
    }
}

/// f64 求最大值。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn max_f64(input: &[f64]) -> LinalgResult<f64> {
    ensure_non_empty("f64 max", input)?;

    #[cfg(feature = "parallel")]
    {
        input
            .par_iter()
            .copied()
            .reduce_with(f64::max)
            .ok_or(EmptyInputError { op: "f64 max" })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut value = input[0];

        for x in input.iter().copied().skip(1) {
            value = value.max(x);
        }

        Ok(value)
    }
}

/// i64 求最大值。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn max_i64(input: &[i64]) -> LinalgResult<i64> {
    ensure_non_empty("i64 max", input)?;

    #[cfg(feature = "parallel")]
    {
        input
            .par_iter()
            .copied()
            .reduce_with(i64::max)
            .ok_or(EmptyInputError { op: "i64 max" })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut value = input[0];

        for x in input.iter().copied().skip(1) {
            value = value.max(x);
        }

        Ok(value)
    }
}

/// f32 求最小值。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn min_f32(input: &[f32]) -> LinalgResult<f32> {
    ensure_non_empty("f32 min", input)?;

    #[cfg(feature = "parallel")]
    {
        input
            .par_iter()
            .copied()
            .reduce_with(f32::min)
            .ok_or(EmptyInputError { op: "f32 min" })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut value = input[0];

        for x in input.iter().copied().skip(1) {
            value = value.min(x);
        }

        Ok(value)
    }
}

/// f64 求最小值。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn min_f64(input: &[f64]) -> LinalgResult<f64> {
    ensure_non_empty("f64 min", input)?;

    #[cfg(feature = "parallel")]
    {
        input
            .par_iter()
            .copied()
            .reduce_with(f64::min)
            .ok_or(EmptyInputError { op: "f64 min" })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut value = input[0];

        for x in input.iter().copied().skip(1) {
            value = value.min(x);
        }

        Ok(value)
    }
}

/// i64 求最小值。
///
/// # Errors
///
/// 当输入为空时返回错误；`i64` 求和与均值还会在累加溢出时返回错误。
pub fn min_i64(input: &[i64]) -> LinalgResult<i64> {
    ensure_non_empty("i64 min", input)?;

    #[cfg(feature = "parallel")]
    {
        input
            .par_iter()
            .copied()
            .reduce_with(i64::min)
            .ok_or(EmptyInputError { op: "i64 min" })
    }

    #[cfg(not(feature = "parallel"))]
    {
        let mut value = input[0];

        for x in input.iter().copied().skip(1) {
            value = value.min(x);
        }

        Ok(value)
    }
}

/// 校验输入是否为空。
fn ensure_non_empty<T>(op: &'static str, input: &[T]) -> LinalgResult<()> {
    if input.is_empty() {
        return Err(EmptyInputError { op });
    }

    Ok(())
}
