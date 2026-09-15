use crate::{SimError::*, SimResult};
use arcqml_circuit::Circuit;

/// 校验模拟器状态与电路声明的量子比特数是否一致。
pub(crate) fn validate_circuit_qubits(num_qubits: usize, circuit: &Circuit) -> SimResult<()> {
    if num_qubits != circuit.num_qubits() {
        return Err(CircuitQubitMismatchError {
            state_qubits: num_qubits,
            circuit_qubits: circuit.num_qubits(),
        });
    }
    Ok(())
}
