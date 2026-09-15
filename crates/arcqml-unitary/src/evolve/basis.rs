use crate::{
    UnitaryResult,
    dimension::{matrix_dimensions, reserve_complex},
    evolve::StateBatch,
};
use num_complex::Complex64;

/// 创建由全部计算基态组成的初始状态批次，也就是单位矩阵。
pub(crate) fn basis_batch(num_qubits: usize) -> UnitaryResult<StateBatch> {
    let (dimension, elements) = matrix_dimensions(num_qubits)?;
    let mut data = reserve_complex(num_qubits, elements)?;
    for basis in 0..dimension {
        data[basis * dimension + basis] = Complex64::new(1.0, 0.0);
    }
    Ok(StateBatch { dimension, data })
}
