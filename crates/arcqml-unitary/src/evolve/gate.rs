use crate::{UnitaryError, UnitaryResult};
use arcqml_circuit::{Circuit, Operation};

/// 将一条线路操作作用到内部的全部计算基态批次。
pub(crate) fn apply_operation(
    circuit: &Circuit,
    operation: &Operation,
    state: &mut crate::evolve::StateBatch,
) -> UnitaryResult<()> {
    let (gate, _) = arcqml_kernel::resolve_gate(circuit, operation.gate()).map_err(|error| {
        UnitaryError::KernelError {
            message: error.to_string(),
        }
    })?;
    arcqml_kernel::apply_gate_batch(
        &mut state.data,
        state.dimension,
        circuit.num_qubits(),
        &gate,
        operation.qubits(),
    )
    .map_err(|error| UnitaryError::KernelError {
        message: error.to_string(),
    })
}
