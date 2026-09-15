use num_complex::Complex64;

/// 转置方阵数据，用于在公开行主序矩阵与内部状态批次布局之间转换。
pub(crate) fn transpose_square(data: &[Complex64], dimension: usize) -> Vec<Complex64> {
    let mut transposed = vec![Complex64::new(0.0, 0.0); data.len()];
    for row in 0..dimension {
        for column in 0..dimension {
            transposed[column * dimension + row] = data[row * dimension + column];
        }
    }
    transposed
}
