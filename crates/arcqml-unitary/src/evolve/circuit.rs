use crate::{
    UnitaryError, UnitaryResult,
    evolve::{apply_operation, basis_batch},
};
use arcqml_circuit::Circuit;
use num_complex::Complex64;

/// 保存全部计算基态经过线路后的状态批次。
#[derive(Debug, Clone)]
pub(crate) struct StateBatch {
    pub(crate) dimension: usize,
    pub(crate) data: Vec<Complex64>,
}

/// 验证线路，避免后续执行路径重复进行相同检查。
fn validate_circuit(circuit: &Circuit) -> UnitaryResult<()> {
    circuit
        .validate()
        .map_err(|error| UnitaryError::CircuitError {
            message: error.to_string(),
        })
}

/// 执行完整线路并返回全部计算基态对应的输出状态。
pub(crate) fn circuit_state(circuit: &Circuit) -> UnitaryResult<StateBatch> {
    validate_circuit(circuit)?;
    circuit_state_through(circuit, circuit.operations().len())
}

/// 执行前 `end` 条已验证操作并返回状态批次。
fn circuit_state_through(circuit: &Circuit, end: usize) -> UnitaryResult<StateBatch> {
    if end > circuit.operations().len() {
        return Err(UnitaryError::InvalidOperationIndexError {
            index: end,
            operation_count: circuit.operations().len(),
        });
    }
    let mut state = basis_batch(circuit.num_qubits())?;
    for operation in &circuit.operations()[..end] {
        apply_operation(circuit, operation, &mut state)?;
    }
    Ok(state)
}
