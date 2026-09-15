use crate::autograd::L2NormBackward;
use crate::error::{LinalgError::*, LinalgResult};
use crate::ops::{ensure_contiguous_tensor, make_tensor_with_backward};
use arcqml_core::{Storage, Tensor};
use num_complex::Complex64;
use std::sync::Arc;

/// 计算所有元素的欧几里得范数，输出为 `F64` 标量 Tensor。
///
/// 对复数输入计算 `sqrt(Σ |xᵢ|²)`；对实数和整数输入计算 `sqrt(Σ xᵢ²)`。
///
/// # Errors
///
/// 当输入为空、数据类型或布局不受支持，或无法构造标量输出时返回错误。
pub fn l2_norm(input: &Tensor) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("L2 norm", input)?;

    let value = match &*input.storage() {
        Storage::F32(v) => v
            .iter()
            .map(|x| {
                let x = *x as f64;
                x * x
            })
            .sum::<f64>()
            .sqrt(),
        Storage::F64(v) => v.iter().map(|x| x * x).sum::<f64>().sqrt(),
        Storage::C64(v) => v.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt(),
        Storage::I64(v) => v
            .iter()
            .map(|x| {
                let x = *x as f64;
                x * x
            })
            .sum::<f64>()
            .sqrt(),
        Storage::Bool(_) => {
            return Err(UnsupportedDTypeError {
                op: "L2 norm",
                dtype: input.dtype().to_string(),
            });
        }
    };

    make_tensor_with_backward(
        "L2 norm",
        Storage::F64(vec![value]),
        vec![],
        input,
        vec![input.clone()],
        Arc::new(L2NormBackward),
    )
}
