use crate::autograd::DotBackward;
use crate::error::{LinalgError::*, LinalgResult};
use crate::ops::{ensure_binary_compatible, make_tensor_with_backward};
use arcqml_core::{Storage, Tensor};
use num_complex::Complex64;
use std::sync::Arc;

/// 计算两个一维 Tensor 的点积；`C64` 使用 `Σ lhsᵢ × rhsᵢ`，不对任一输入取共轭。
///
/// # Errors
///
/// 当输入不是同类型、等长、连续的一维数值 Tensor，或整数计算溢出时返回错误。
pub fn dot(lhs: &Tensor, rhs: &Tensor) -> LinalgResult<Tensor> {
    ensure_binary_compatible("dot product", lhs, rhs)?;

    if lhs.shape().len() != 1 {
        return Err(InvalidDimensionError {
            op: "dot product",
            expected: "the left operand to have rank 1".to_string(),
            actual: lhs.shape().len(),
        });
    }

    if rhs.shape().len() != 1 {
        return Err(InvalidDimensionError {
            op: "dot product",
            expected: "the right operand to have rank 1".to_string(),
            actual: rhs.shape().len(),
        });
    }

    if lhs.shape()[0] != rhs.shape()[0] {
        return Err(ShapeMismatchError {
            op: "dot product",
            expected: format!(
                "vectors with the same length, got lhs.len={}",
                lhs.shape()[0]
            ),
            actual: format!("rhs.len={}", rhs.shape()[0]),
        });
    }

    let lhs_storage = lhs.storage();
    let rhs_storage = rhs.storage();
    let storage = match (&*lhs_storage, &*rhs_storage) {
        (Storage::F32(a), Storage::F32(b)) => {
            let value = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            Storage::F32(vec![value])
        }
        (Storage::F64(a), Storage::F64(b)) => {
            let value = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            Storage::F64(vec![value])
        }
        (Storage::C64(a), Storage::C64(b)) => {
            let value = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| x * y)
                .sum::<Complex64>();
            Storage::C64(vec![value])
        }
        (Storage::I64(a), Storage::I64(b)) => {
            let mut value = 0_i64;

            for (index, (x, y)) in a.iter().copied().zip(b.iter().copied()).enumerate() {
                let product = x.checked_mul(y).ok_or(IntegerOverflowError {
                    op: "dot product",
                    index,
                })?;
                value = value.checked_add(product).ok_or(IntegerOverflowError {
                    op: "dot product",
                    index,
                })?;
            }

            Storage::I64(vec![value])
        }
        _ => {
            return Err(UnsupportedDTypeError {
                op: "dot product",
                dtype: lhs.dtype().to_string(),
            });
        }
    };

    make_tensor_with_backward(
        "dot product",
        storage,
        vec![],
        lhs,
        vec![lhs.clone(), rhs.clone()],
        Arc::new(DotBackward),
    )
}
