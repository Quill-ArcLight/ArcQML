/// 计算 `2^exponent`；当结果无法用当前平台的 `usize` 表示时返回 `None`。
pub fn checked_power_of_two(exponent: usize) -> Option<usize> {
    (exponent < usize::BITS as usize).then(|| 1usize << exponent)
}
