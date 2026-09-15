use super::single;
use crate::{KernelError::*, KernelResult};
use arcqml_circuit::{Gate, Qubit};
use arcqml_core::checked_power_of_two;
use num_complex::Complex64;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// 关闭 parallel feature 时为行主序并行接口提供串行兼容实现。
#[cfg(not(feature = "parallel"))]
trait ParallelSliceMutCompat<T> {
    /// 返回串行的精确分块可变迭代器。
    fn par_chunks_exact_mut(&mut self, chunk_size: usize) -> std::slice::ChunksExactMut<'_, T>;
}

#[cfg(not(feature = "parallel"))]
impl<T> ParallelSliceMutCompat<T> for [T] {
    /// 返回串行的精确分块可变迭代器。
    fn par_chunks_exact_mut(&mut self, chunk_size: usize) -> std::slice::ChunksExactMut<'_, T> {
        self.chunks_exact_mut(chunk_size)
    }
}

/// 对行主序 `[batch_size, 2^num_qubits]` 状态执行一个量子门。
///
/// # Errors
///
/// 当 `batch_size` 为零、状态维度或总长度溢出、缓冲区长度错误、门无效或含未绑定
/// 参数，或量子比特的数量、范围及唯一性不满足门约束时返回错误。
pub fn apply_gate(
    amplitudes: &mut [Complex64],
    batch_size: usize,
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
) -> KernelResult<()> {
    let dimension = validate_row_major(amplitudes, batch_size, num_qubits)?;
    single::validate_gate(gate)?;
    single::validate_gate_qubits(num_qubits, gate, qubits)?;
    amplitudes
        .par_chunks_exact_mut(dimension)
        .try_for_each(|row| single::apply_gate_unchecked(row, num_qubits, gate, qubits))
}

/// 对行主序 `[batch_size, 2^num_qubits]` 状态串行执行一个量子门。
///
/// 此内核供已经在 batch 维度并行的调用方使用，避免重复进入 Rayon。
///
/// # Errors
///
/// 当 `batch_size` 为零、状态维度或总长度溢出、缓冲区长度错误、门无效或含未绑定
/// 参数，或量子比特的数量、范围及唯一性不满足门约束时返回错误。
pub fn apply_gate_serial(
    amplitudes: &mut [Complex64],
    batch_size: usize,
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
) -> KernelResult<()> {
    let dimension = validate_row_major(amplitudes, batch_size, num_qubits)?;
    single::validate_gate(gate)?;
    single::validate_gate_qubits(num_qubits, gate, qubits)?;
    for row in amplitudes.chunks_exact_mut(dimension) {
        single::apply_gate_unchecked(row, num_qubits, gate, qubits)?;
    }
    Ok(())
}

/// 对行主序 `[batch_size, 2^num_qubits]` 状态执行量子门的伴随操作。
///
/// # Errors
///
/// 当 `batch_size` 为零、状态维度或总长度溢出、缓冲区长度错误、门无效或含未绑定
/// 参数，或量子比特的数量、范围及唯一性不满足门约束时返回错误。
pub fn apply_adjoint_gate(
    amplitudes: &mut [Complex64],
    batch_size: usize,
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
) -> KernelResult<()> {
    let dimension = validate_row_major(amplitudes, batch_size, num_qubits)?;
    single::validate_gate(gate)?;
    single::validate_gate_qubits(num_qubits, gate, qubits)?;
    amplitudes
        .par_chunks_exact_mut(dimension)
        .try_for_each(|row| single::apply_adjoint_gate_unchecked(row, num_qubits, gate, qubits))
}

/// 对行主序 `[batch_size, 2^num_qubits]` 状态串行执行量子门的伴随操作。
///
/// 此内核供已经在 batch 维度并行的调用方使用，避免重复进入 Rayon。
///
/// # Errors
///
/// 当 `batch_size` 为零、状态维度或总长度溢出、缓冲区长度错误、门无效或含未绑定
/// 参数，或量子比特的数量、范围及唯一性不满足门约束时返回错误。
pub fn apply_adjoint_gate_serial(
    amplitudes: &mut [Complex64],
    batch_size: usize,
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
) -> KernelResult<()> {
    let dimension = validate_row_major(amplitudes, batch_size, num_qubits)?;
    single::validate_gate(gate)?;
    single::validate_gate_qubits(num_qubits, gate, qubits)?;
    for row in amplitudes.chunks_exact_mut(dimension) {
        single::apply_adjoint_gate_unchecked(row, num_qubits, gate, qubits)?;
    }
    Ok(())
}

/// 返回行主序批次中每一行的参数导数态 `∂(U|ψ⟩)/∂p`。
///
/// # Errors
///
/// 除批状态、门和量子比特校验错误外，当 `parameter_slot` 不对应门的可微参数时返回错误。
pub fn parameter_derivative(
    input: &[Complex64],
    batch_size: usize,
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
    parameter_slot: usize,
) -> KernelResult<Vec<Complex64>> {
    let dimension = validate_row_major(input, batch_size, num_qubits)?;
    single::validate_gate(gate)?;
    single::validate_gate_qubits(num_qubits, gate, qubits)?;
    let derivative = single::derivative_matrix_for_gate(gate, parameter_slot)?;
    let mut output = input.to_vec();
    output
        .par_chunks_exact_mut(dimension)
        .try_for_each(|output_row| single::apply_matrix(output_row, &derivative, qubits))?;
    Ok(output)
}

/// 串行返回行主序批次中每一行的参数导数态 `∂(U|ψ⟩)/∂p`。
///
/// 此内核供已经在 batch 维度并行的调用方使用，避免重复进入 Rayon。
///
/// # Errors
///
/// 除批状态、门和量子比特校验错误外，当 `parameter_slot` 不对应门的可微参数时返回错误。
pub fn parameter_derivative_serial(
    input: &[Complex64],
    batch_size: usize,
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
    parameter_slot: usize,
) -> KernelResult<Vec<Complex64>> {
    let dimension = validate_row_major(input, batch_size, num_qubits)?;
    single::validate_gate(gate)?;
    single::validate_gate_qubits(num_qubits, gate, qubits)?;
    let derivative = single::derivative_matrix_for_gate(gate, parameter_slot)?;
    let mut output = input.to_vec();
    for row in output.chunks_exact_mut(dimension) {
        single::apply_matrix(row, &derivative, qubits)?;
    }
    Ok(output)
}

/// 校验行主序 batch 状态的长度并返回单个状态向量维度。
fn validate_row_major(
    amplitudes: &[Complex64],
    batch_size: usize,
    num_qubits: usize,
) -> KernelResult<usize> {
    if batch_size == 0 {
        return Err(EmptyBatchError);
    }
    let dimension =
        checked_power_of_two(num_qubits).ok_or(StateDimensionOverflowError { num_qubits })?;
    let expected = dimension.checked_mul(batch_size).ok_or(CircuitError {
        message: format!("batched state length overflow for batch size {batch_size}"),
    })?;
    if amplitudes.len() != expected {
        return Err(BatchStateLengthError {
            batch_size,
            num_qubits,
            expected,
            actual: amplitudes.len(),
        });
    }
    Ok(dimension)
}
