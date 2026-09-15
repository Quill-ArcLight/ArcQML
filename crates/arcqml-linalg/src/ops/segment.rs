use crate::autograd::SegmentSumBackward;
use crate::error::{LinalgError::*, LinalgResult};
use crate::ops::{ensure_contiguous_tensor, make_tensor_with_backward};
use arcqml_core::{Storage, Tensor};
use std::sync::Arc;

/// 按固定分段索引汇总一维 Tensor 的元素。
///
/// `segment_ids[index]` 指定输入第 `index` 个元素累加到的输出分段；索引本身不参与求导。
/// 输出形状为 `[num_segments]`，数据类型与输入相同。支持 `F32`、`F64` 和 `C64`。
///
/// # Errors
///
/// 当输入不是连续的一维 CPU Dense Tensor、数据类型不受支持、`segment_ids`
/// 的长度与输入元素数不同、`num_segments` 为零、分段编号越界，或无法构造输出时返回错误。
pub fn segment_sum(
    input: &Tensor,
    segment_ids: &[usize],
    num_segments: usize,
) -> LinalgResult<Tensor> {
    ensure_contiguous_tensor("segment_sum", input)?;
    if input.ndim() != 1 {
        return Err(InvalidDimensionError {
            op: "segment_sum",
            expected: "a 1-D tensor".to_string(),
            actual: input.ndim(),
        });
    }
    if input.numel() != segment_ids.len() {
        return Err(ShapeMismatchError {
            op: "segment_sum",
            expected: format!("{} segment ids", input.numel()),
            actual: format!("{} segment ids", segment_ids.len()),
        });
    }
    if num_segments == 0 {
        return Err(ShapeMismatchError {
            op: "segment_sum",
            expected: "num_segments > 0".to_string(),
            actual: "num_segments=0".to_string(),
        });
    }
    if let Some((index, segment)) = segment_ids
        .iter()
        .copied()
        .enumerate()
        .find(|(_, segment)| *segment >= num_segments)
    {
        return Err(ShapeMismatchError {
            op: "segment_sum",
            expected: format!("segment ids smaller than {num_segments}"),
            actual: format!("segment_ids[{index}]={segment}"),
        });
    }

    let storage = match &*input.storage() {
        Storage::F32(values) => Storage::F32(segment_sum_values(values, segment_ids, num_segments)),
        Storage::F64(values) => Storage::F64(segment_sum_values(values, segment_ids, num_segments)),
        Storage::C64(values) => Storage::C64(segment_sum_values(values, segment_ids, num_segments)),
        _ => {
            return Err(UnsupportedDTypeError {
                op: "segment_sum",
                dtype: input.dtype().to_string(),
            });
        }
    };
    make_tensor_with_backward(
        "segment_sum",
        storage,
        vec![num_segments],
        input,
        vec![input.clone()],
        Arc::new(SegmentSumBackward {
            segment_ids: segment_ids.to_vec(),
            num_segments,
        }),
    )
}

/// 将输入元素累加到各自所属的分段中。
fn segment_sum_values<T>(values: &[T], segment_ids: &[usize], num_segments: usize) -> Vec<T>
where
    T: Copy + Default + std::ops::AddAssign,
{
    let mut output = vec![T::default(); num_segments];
    for (value, segment) in values.iter().copied().zip(segment_ids.iter().copied()) {
        output[segment] += value;
    }
    output
}
