/// 按行主序连续规则计算元素步长。
///
/// 此底层辅助函数不显式检查 `usize` 乘法溢出。
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }

    let mut strides = vec![1; shape.len()];

    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

/// 判断给定步长是否恰好等于行主序连续步长。
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.len() != strides.len() {
        return false;
    }

    contiguous_strides(shape) == strides
}

/// 交换二维步长；当输入步长数量不是二时返回 `None`。
pub fn transposed_2d_strides(strides: &[usize]) -> Option<Vec<usize>> {
    if strides.len() != 2 {
        return None;
    }

    Some(vec![strides[1], strides[0]])
}
