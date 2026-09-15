use crate::autograd::ConjBackward;
use crate::error::{LinalgError::UnsupportedDTypeError, LinalgResult};
use crate::ops::{ensure_contiguous_tensor, make_tensor_with_backward};
use arcqml_core::{Storage, Tensor};
use num_complex::Complex64;
use std::sync::Arc;

/// 计算 `C64` Tensor 的逐元素复共轭，输出形状与输入相同。
///
/// # Errors
///
/// 当输入不是连续的 CPU Dense `C64` Tensor，或无法构造输出时返回错误。
pub fn conj(input: &Tensor) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("conj", input)?;
    let storage = match &*input.storage() {
        Storage::C64(values) => Storage::C64(values.iter().map(Complex64::conj).collect()),
        _ => {
            return Err(UnsupportedDTypeError {
                op: "conj",
                dtype: input.dtype().to_string(),
            });
        }
    };
    make_tensor_with_backward(
        "conj",
        storage,
        input.shape().to_vec(),
        input,
        vec![input.clone()],
        Arc::new(ConjBackward),
    )
}
