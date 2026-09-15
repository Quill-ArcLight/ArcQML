use crate::{SimError::*, SimResult};
use arcqml_observable::SparsePauliOp;
use num_complex::Complex64;

/// 计算 SparsePauliOp 的期望值，并委托 arcqml-observable 的缓存执行计划。
pub(crate) fn expectation_observable(
    amplitudes: &[Complex64],
    num_qubits: usize,
    observable: &SparsePauliOp,
) -> SimResult<f64> {
    validate_observable_qubits(num_qubits, observable.num_qubits())?;
    observable
        .statevector_expectation(amplitudes)
        .map_err(Into::into)
}

/// 一次计算标量期望值与 O|ψ⟩，供单态自动微分前反向共享。
pub(crate) fn expectation_and_apply_observable(
    amplitudes: &[Complex64],
    num_qubits: usize,
    observable: &SparsePauliOp,
) -> SimResult<(f64, Vec<Complex64>)> {
    validate_observable_qubits(num_qubits, observable.num_qubits())?;
    observable
        .statevector_expectation_and_apply(amplitudes)
        .map_err(Into::into)
}

/// 串行计算标量期望值与 O|ψ⟩，供已在 batch 维度并行的调用方使用。
pub(crate) fn expectation_and_apply_observable_serial(
    amplitudes: &[Complex64],
    num_qubits: usize,
    observable: &SparsePauliOp,
) -> SimResult<(f64, Vec<Complex64>)> {
    validate_observable_qubits(num_qubits, observable.num_qubits())?;
    observable
        .statevector_expectation_and_apply_serial(amplitudes)
        .map_err(Into::into)
}

/// 检验 qubit 数是否一致。
fn validate_observable_qubits(state_qubits: usize, observable_qubits: usize) -> SimResult<()> {
    if state_qubits != observable_qubits {
        return Err(ObservableQubitMismatchError {
            state_qubits,
            observable_qubits,
        });
    }
    Ok(())
}
