use crate::{KernelError::*, KernelResult};
use arcqml_circuit::{Gate, Gateparam, Qubit};
use arcqml_core::checked_power_of_two;
use arcqml_runtime_abi::{GateDescriptor, opcode};
use arcqml_runtime_sys::DerivativeMatrix;
use num_complex::Complex64;
use std::{collections::BTreeSet, slice};

pub(crate) type Matrix = DerivativeMatrix;

/// 通过专有数值运行时应用一个量子门。
pub fn apply_gate(
    amplitudes: &mut [Complex64],
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
) -> KernelResult<()> {
    validate_gate(gate)?;
    validate_gate_qubits(num_qubits, gate, qubits)?;
    validate_state_length(amplitudes, num_qubits)?;
    apply_gate_unchecked(amplitudes, num_qubits, gate, qubits)
}

pub(crate) fn apply_gate_unchecked(
    amplitudes: &mut [Complex64],
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
) -> KernelResult<()> {
    let descriptor = gate_descriptor(gate)?;
    arcqml_runtime_sys::apply_gate(amplitudes, num_qubits, &descriptor, qubit_indices(qubits))
        .map_err(runtime_error)
}

/// 通过原生运行时应用一个量子门的共轭转置。
pub fn apply_adjoint_gate(
    gradients: &mut [Complex64],
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
) -> KernelResult<()> {
    validate_gate(gate)?;
    validate_gate_qubits(num_qubits, gate, qubits)?;
    validate_state_length(gradients, num_qubits)?;
    apply_adjoint_gate_unchecked(gradients, num_qubits, gate, qubits)
}

pub(crate) fn apply_adjoint_gate_unchecked(
    gradients: &mut [Complex64],
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
) -> KernelResult<()> {
    let descriptor = gate_descriptor(gate)?;
    arcqml_runtime_sys::apply_adjoint_gate(
        gradients,
        num_qubits,
        &descriptor,
        qubit_indices(qubits),
    )
    .map_err(runtime_error)
}

/// 计算一个实数门参数对应的解析导数态。
pub fn parameter_derivative(
    input: &[Complex64],
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
    parameter_slot: usize,
) -> KernelResult<Vec<Complex64>> {
    let mut output = vec![Complex64::new(0.0, 0.0); input.len()];
    parameter_derivative_into(input, &mut output, num_qubits, gate, qubits, parameter_slot)?;
    Ok(output)
}

/// 将解析导数态直接写入调用方拥有的存储空间。
pub fn parameter_derivative_into(
    input: &[Complex64],
    output: &mut [Complex64],
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
    parameter_slot: usize,
) -> KernelResult<()> {
    validate_gate(gate)?;
    validate_gate_qubits(num_qubits, gate, qubits)?;
    validate_state_length(input, num_qubits)?;
    validate_state_length(output, num_qubits)?;
    validate_parameter_slot(gate, parameter_slot)?;
    let descriptor = gate_descriptor(gate)?;
    arcqml_runtime_sys::parameter_derivative_into(
        input,
        output,
        &descriptor,
        qubit_indices(qubits),
        parameter_slot,
    )
    .map_err(runtime_error)
}

pub(crate) fn validate_gate(gate: &Gate) -> KernelResult<()> {
    gate.validate().map_err(|error| CircuitError {
        message: error.to_string(),
    })
}

pub(crate) fn validate_gate_qubits(
    num_qubits: usize,
    gate: &Gate,
    qubits: &[Qubit],
) -> KernelResult<()> {
    if gate.arity() != qubits.len() {
        return Err(GateArityError {
            gate: gate.name().to_string(),
            expected: gate.arity(),
            actual: qubits.len(),
        });
    }
    let mut seen = BTreeSet::new();
    for qubit in qubits {
        if qubit.index() >= num_qubits {
            return Err(QubitOutOfRangeError {
                index: qubit.index(),
                num_qubits,
            });
        }
        if !seen.insert(qubit.index()) {
            return Err(DuplicateQubitError {
                index: qubit.index(),
            });
        }
    }
    Ok(())
}

fn validate_state_length(amplitudes: &[Complex64], num_qubits: usize) -> KernelResult<()> {
    let expected =
        checked_power_of_two(num_qubits).ok_or(StateDimensionOverflowError { num_qubits })?;
    if amplitudes.len() != expected {
        return Err(StateLengthError {
            num_qubits,
            expected,
            actual: amplitudes.len(),
        });
    }
    Ok(())
}

pub(crate) fn derivative_matrix_for_gate(
    gate: &Gate,
    parameter_slot: usize,
) -> KernelResult<Matrix> {
    validate_parameter_slot(gate, parameter_slot)?;
    let descriptor = gate_descriptor(gate)?;
    DerivativeMatrix::create(&descriptor, parameter_slot).map_err(runtime_error)
}

pub(crate) fn apply_matrix(
    amplitudes: &mut [Complex64],
    matrix: &Matrix,
    qubits: &[Qubit],
) -> KernelResult<()> {
    matrix
        .apply(amplitudes, qubit_indices(qubits))
        .map_err(runtime_error)
}

fn validate_parameter_slot(gate: &Gate, parameter_slot: usize) -> KernelResult<()> {
    let valid = match gate {
        Gate::Rx(_)
        | Gate::Ry(_)
        | Gate::Rz(_)
        | Gate::Phase(_)
        | Gate::CPhase(_)
        | Gate::CRx(_)
        | Gate::CRy(_)
        | Gate::CRz(_)
        | Gate::Rxx(_)
        | Gate::Ryy(_)
        | Gate::Rzz(_)
        | Gate::Rzx(_) => parameter_slot == 0,
        Gate::U3(_, _, _) => parameter_slot < 3,
        Gate::FSim(_, _) => parameter_slot < 2,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(CircuitError {
            message: format!(
                "gate {} does not have parameter slot {parameter_slot}",
                gate.name()
            ),
        })
    }
}

pub(crate) fn gate_descriptor(gate: &Gate) -> KernelResult<GateDescriptor> {
    let descriptor = match gate {
        Gate::I => builtin(opcode::I, gate, [0.0; 3]),
        Gate::X => builtin(opcode::X, gate, [0.0; 3]),
        Gate::Y => builtin(opcode::Y, gate, [0.0; 3]),
        Gate::Z => builtin(opcode::Z, gate, [0.0; 3]),
        Gate::H => builtin(opcode::H, gate, [0.0; 3]),
        Gate::S => builtin(opcode::S, gate, [0.0; 3]),
        Gate::Sdg => builtin(opcode::SDG, gate, [0.0; 3]),
        Gate::T => builtin(opcode::T, gate, [0.0; 3]),
        Gate::Tdg => builtin(opcode::TDG, gate, [0.0; 3]),
        Gate::Sx => builtin(opcode::SX, gate, [0.0; 3]),
        Gate::Sxdg => builtin(opcode::SXDG, gate, [0.0; 3]),
        Gate::Rx(theta) => builtin(opcode::RX, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::Ry(theta) => builtin(opcode::RY, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::Rz(theta) => builtin(opcode::RZ, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::Phase(theta) => builtin(opcode::PHASE, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::U3(theta, phi, lambda) => builtin(
            opcode::U3,
            gate,
            [fixed(theta)?, fixed(phi)?, fixed(lambda)?],
        ),
        Gate::CNot => builtin(opcode::CNOT, gate, [0.0; 3]),
        Gate::CY => builtin(opcode::CY, gate, [0.0; 3]),
        Gate::CZ => builtin(opcode::CZ, gate, [0.0; 3]),
        Gate::CH => builtin(opcode::CH, gate, [0.0; 3]),
        Gate::CS => builtin(opcode::CS, gate, [0.0; 3]),
        Gate::CT => builtin(opcode::CT, gate, [0.0; 3]),
        Gate::CPhase(theta) => builtin(opcode::CPHASE, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::CRx(theta) => builtin(opcode::CRX, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::CRy(theta) => builtin(opcode::CRY, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::CRz(theta) => builtin(opcode::CRZ, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::Swap => builtin(opcode::SWAP, gate, [0.0; 3]),
        Gate::ISwap => builtin(opcode::ISWAP, gate, [0.0; 3]),
        Gate::DCX => builtin(opcode::DCX, gate, [0.0; 3]),
        Gate::Ecr => builtin(opcode::ECR, gate, [0.0; 3]),
        Gate::Rxx(theta) => builtin(opcode::RXX, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::Ryy(theta) => builtin(opcode::RYY, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::Rzz(theta) => builtin(opcode::RZZ, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::Rzx(theta) => builtin(opcode::RZX, gate, [fixed(theta)?, 0.0, 0.0]),
        Gate::FSim(theta, phi) => builtin(opcode::FSIM, gate, [fixed(theta)?, fixed(phi)?, 0.0]),
        Gate::Toffoli => builtin(opcode::TOFFOLI, gate, [0.0; 3]),
        Gate::CSwap => builtin(opcode::CSWAP, gate, [0.0; 3]),
        Gate::MCX { .. } => builtin(opcode::MCX, gate, [0.0; 3]),
        Gate::CustomUnitary {
            name,
            arity,
            matrix,
        } => GateDescriptor::custom(
            *arity,
            matrix.as_ptr(),
            matrix.len(),
            name.as_ptr(),
            name.len(),
        ),
    };
    Ok(descriptor)
}

fn builtin(opcode: u32, gate: &Gate, parameters: [f64; 3]) -> GateDescriptor {
    GateDescriptor::builtin(opcode, gate.arity(), parameters)
}

fn fixed(parameter: &Gateparam) -> KernelResult<f64> {
    parameter.as_fixed().ok_or(UnboundParameterError {
        gate: "parameterized gate".to_string(),
        parameter: "unresolved parameter".to_string(),
    })
}

fn qubit_indices(qubits: &[Qubit]) -> &[usize] {
    // Qubit 是 usize 的 repr(transparent) 包装，因此不会发生分配或数值转换。
    unsafe { slice::from_raw_parts(qubits.as_ptr().cast::<usize>(), qubits.len()) }
}

fn runtime_error(error: arcqml_runtime_sys::RuntimeError) -> crate::KernelError {
    CircuitError {
        message: error.to_string(),
    }
}
