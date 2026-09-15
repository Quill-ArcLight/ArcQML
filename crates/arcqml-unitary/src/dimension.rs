use crate::{UnitaryError, UnitaryResult};

/// 根据量子比特数计算稠密酉矩阵的维度和元素数量。
pub(crate) fn matrix_dimensions(num_qubits: usize) -> UnitaryResult<(usize, usize)> {
    let dimension = 1usize
        .checked_shl(num_qubits as u32)
        .ok_or(UnitaryError::DimensionOverflowError { num_qubits })?;
    let elements = dimension
        .checked_mul(dimension)
        .ok_or(UnitaryError::DimensionOverflowError { num_qubits })?;
    Ok((dimension, elements))
}

/// 根据矩阵边长恢复量子比特数。
pub(crate) fn qubits_from_dimension(dimension: usize) -> UnitaryResult<usize> {
    if dimension == 0 || !dimension.is_power_of_two() {
        return Err(UnitaryError::InvalidUnitaryDimensionError { dimension });
    }
    Ok(dimension.ilog2() as usize)
}

/// 为复数缓冲区预留精确容量，并统一转换分配失败错误。
pub(crate) fn reserve_complex(
    num_qubits: usize,
    elements: usize,
) -> UnitaryResult<Vec<num_complex::Complex64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(|_| UnitaryError::AllocationError {
            num_qubits,
            elements,
        })?;
    values.resize(elements, num_complex::Complex64::new(0.0, 0.0));
    Ok(values)
}
