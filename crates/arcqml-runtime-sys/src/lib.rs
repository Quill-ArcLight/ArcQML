//! 面向 ArcQML 原生 C ABI 的安全、无分配调度层。

use arcqml_runtime_abi::{
    ABI_VERSION, CircuitPlanHandle, DerivativeMatrixHandle, GateDescriptor, OperationDescriptor,
};
use num_complex::Complex64;
use std::{error::Error, fmt};

#[cfg(feature = "private-source")]
mod raw {
    pub use arcqml_runtime_private::{
        arcqml_rt_abi_version, arcqml_rt_last_error, arcqml_rt_v1_adjoint_backward,
        arcqml_rt_v1_apply_adjoint_gate, arcqml_rt_v1_apply_gate, arcqml_rt_v1_circuit_plan_apply,
        arcqml_rt_v1_circuit_plan_create, arcqml_rt_v1_circuit_plan_create_empty,
        arcqml_rt_v1_circuit_plan_destroy, arcqml_rt_v1_circuit_plan_push,
        arcqml_rt_v1_derivative_matrix_apply, arcqml_rt_v1_derivative_matrix_create,
        arcqml_rt_v1_derivative_matrix_destroy, arcqml_rt_v1_parameter_derivative_into,
    };
}

#[cfg(not(feature = "private-source"))]
mod raw {
    use arcqml_runtime_abi::{
        CircuitPlanHandle, DerivativeMatrixHandle, GateDescriptor, OperationDescriptor,
    };
    use num_complex::Complex64;

    unsafe extern "C" {
        pub fn arcqml_rt_abi_version() -> u32;
        pub fn arcqml_rt_v1_apply_gate(
            amplitudes: *mut Complex64,
            amplitudes_len: usize,
            num_qubits: usize,
            gate: *const GateDescriptor,
            qubits: *const usize,
            qubits_len: usize,
        ) -> i32;
        pub fn arcqml_rt_v1_apply_adjoint_gate(
            amplitudes: *mut Complex64,
            amplitudes_len: usize,
            num_qubits: usize,
            gate: *const GateDescriptor,
            qubits: *const usize,
            qubits_len: usize,
        ) -> i32;
        pub fn arcqml_rt_v1_parameter_derivative_into(
            input: *const Complex64,
            input_len: usize,
            output: *mut Complex64,
            output_len: usize,
            gate: *const GateDescriptor,
            qubits: *const usize,
            qubits_len: usize,
            parameter_slot: usize,
        ) -> i32;
        pub fn arcqml_rt_v1_derivative_matrix_create(
            gate: *const GateDescriptor,
            parameter_slot: usize,
            output: *mut DerivativeMatrixHandle,
        ) -> i32;
        pub fn arcqml_rt_v1_derivative_matrix_apply(
            amplitudes: *mut Complex64,
            amplitudes_len: usize,
            matrix: *const DerivativeMatrixHandle,
            qubits: *const usize,
            qubits_len: usize,
        ) -> i32;
        pub fn arcqml_rt_v1_derivative_matrix_destroy(matrix: *mut DerivativeMatrixHandle);
        pub fn arcqml_rt_v1_circuit_plan_create(
            operations: *const OperationDescriptor,
            operations_len: usize,
            output: *mut CircuitPlanHandle,
        ) -> i32;
        pub fn arcqml_rt_v1_circuit_plan_create_empty(
            operation_capacity: usize,
            output: *mut CircuitPlanHandle,
        ) -> i32;
        pub fn arcqml_rt_v1_circuit_plan_push(
            plan: *mut CircuitPlanHandle,
            operation: *const OperationDescriptor,
        ) -> i32;
        pub fn arcqml_rt_v1_circuit_plan_apply(
            plan: *const CircuitPlanHandle,
            amplitudes: *mut Complex64,
            amplitudes_len: usize,
            num_qubits: usize,
        ) -> i32;
        pub fn arcqml_rt_v1_adjoint_backward(
            plan: *const CircuitPlanHandle,
            forward_state: *mut Complex64,
            adjoint_state: *mut Complex64,
            state_len: usize,
            derivative_state: *mut Complex64,
            num_qubits: usize,
            upstream: f64,
            parameter_requires_grad: *const u8,
            parameter_requires_grad_len: usize,
            gradients: *mut f64,
            gradients_len: usize,
        ) -> i32;
        pub fn arcqml_rt_v1_circuit_plan_destroy(plan: *mut CircuitPlanHandle);
        pub fn arcqml_rt_last_error(buffer: *mut u8, capacity: usize) -> usize;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeError(String);

impl RuntimeError {
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for RuntimeError {}

pub fn ensure_compatible() -> Result<(), RuntimeError> {
    #[cfg(feature = "private-source")]
    let actual = raw::arcqml_rt_abi_version();
    #[cfg(not(feature = "private-source"))]
    let actual = unsafe { raw::arcqml_rt_abi_version() };
    if actual == ABI_VERSION {
        Ok(())
    } else {
        Err(RuntimeError(format!(
            "ArcQML runtime ABI mismatch: expected {ABI_VERSION}, got {actual}"
        )))
    }
}

pub fn apply_gate(
    amplitudes: &mut [Complex64],
    num_qubits: usize,
    gate: &GateDescriptor,
    qubits: &[usize],
) -> Result<(), RuntimeError> {
    status(unsafe {
        raw::arcqml_rt_v1_apply_gate(
            amplitudes.as_mut_ptr(),
            amplitudes.len(),
            num_qubits,
            gate,
            qubits.as_ptr(),
            qubits.len(),
        )
    })
}

pub fn apply_adjoint_gate(
    amplitudes: &mut [Complex64],
    num_qubits: usize,
    gate: &GateDescriptor,
    qubits: &[usize],
) -> Result<(), RuntimeError> {
    status(unsafe {
        raw::arcqml_rt_v1_apply_adjoint_gate(
            amplitudes.as_mut_ptr(),
            amplitudes.len(),
            num_qubits,
            gate,
            qubits.as_ptr(),
            qubits.len(),
        )
    })
}

pub fn parameter_derivative_into(
    input: &[Complex64],
    output: &mut [Complex64],
    gate: &GateDescriptor,
    qubits: &[usize],
    parameter_slot: usize,
) -> Result<(), RuntimeError> {
    status(unsafe {
        raw::arcqml_rt_v1_parameter_derivative_into(
            input.as_ptr(),
            input.len(),
            output.as_mut_ptr(),
            output.len(),
            gate,
            qubits.as_ptr(),
            qubits.len(),
            parameter_slot,
        )
    })
}

#[derive(Debug)]
pub struct DerivativeMatrix {
    raw: DerivativeMatrixHandle,
}

unsafe impl Send for DerivativeMatrix {}
unsafe impl Sync for DerivativeMatrix {}

impl DerivativeMatrix {
    pub fn create(gate: &GateDescriptor, parameter_slot: usize) -> Result<Self, RuntimeError> {
        let mut raw = DerivativeMatrixHandle::empty();
        status(unsafe {
            raw::arcqml_rt_v1_derivative_matrix_create(gate, parameter_slot, &mut raw)
        })?;
        Ok(Self { raw })
    }

    pub fn apply(
        &self,
        amplitudes: &mut [Complex64],
        qubits: &[usize],
    ) -> Result<(), RuntimeError> {
        status(unsafe {
            raw::arcqml_rt_v1_derivative_matrix_apply(
                amplitudes.as_mut_ptr(),
                amplitudes.len(),
                &self.raw,
                qubits.as_ptr(),
                qubits.len(),
            )
        })
    }
}

impl Drop for DerivativeMatrix {
    fn drop(&mut self) {
        unsafe { raw::arcqml_rt_v1_derivative_matrix_destroy(&mut self.raw) };
    }
}

#[derive(Debug)]
pub struct CircuitPlan {
    raw: CircuitPlanHandle,
}

unsafe impl Send for CircuitPlan {}
unsafe impl Sync for CircuitPlan {}

impl CircuitPlan {
    pub fn create(operations: &[OperationDescriptor]) -> Result<Self, RuntimeError> {
        ensure_compatible()?;
        let mut raw = CircuitPlanHandle::empty();
        status(unsafe {
            raw::arcqml_rt_v1_circuit_plan_create(operations.as_ptr(), operations.len(), &mut raw)
        })?;
        Ok(Self { raw })
    }

    pub fn with_capacity(operation_capacity: usize) -> Result<Self, RuntimeError> {
        ensure_compatible()?;
        let mut raw = CircuitPlanHandle::empty();
        status(unsafe {
            raw::arcqml_rt_v1_circuit_plan_create_empty(operation_capacity, &mut raw)
        })?;
        Ok(Self { raw })
    }

    pub fn push(&mut self, operation: &OperationDescriptor) -> Result<(), RuntimeError> {
        status(unsafe { raw::arcqml_rt_v1_circuit_plan_push(&mut self.raw, operation) })
    }

    pub fn apply(
        &self,
        amplitudes: &mut [Complex64],
        num_qubits: usize,
    ) -> Result<(), RuntimeError> {
        status(unsafe {
            raw::arcqml_rt_v1_circuit_plan_apply(
                &self.raw,
                amplitudes.as_mut_ptr(),
                amplitudes.len(),
                num_qubits,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adjoint_backward(
        &self,
        forward_state: &mut [Complex64],
        adjoint_state: &mut [Complex64],
        derivative_state: &mut [Complex64],
        num_qubits: usize,
        upstream: f64,
        parameter_requires_grad: &[u8],
        gradients: &mut [f64],
    ) -> Result<(), RuntimeError> {
        if forward_state.len() != adjoint_state.len()
            || forward_state.len() != derivative_state.len()
        {
            return Err(RuntimeError(
                "adjoint buffers must have identical lengths".to_string(),
            ));
        }
        status(unsafe {
            raw::arcqml_rt_v1_adjoint_backward(
                &self.raw,
                forward_state.as_mut_ptr(),
                adjoint_state.as_mut_ptr(),
                forward_state.len(),
                derivative_state.as_mut_ptr(),
                num_qubits,
                upstream,
                parameter_requires_grad.as_ptr(),
                parameter_requires_grad.len(),
                gradients.as_mut_ptr(),
                gradients.len(),
            )
        })
    }
}

impl Drop for CircuitPlan {
    fn drop(&mut self) {
        unsafe { raw::arcqml_rt_v1_circuit_plan_destroy(&mut self.raw) };
    }
}

fn status(code: i32) -> Result<(), RuntimeError> {
    if code == 0 {
        Ok(())
    } else {
        Err(RuntimeError(last_error()))
    }
}

fn last_error() -> String {
    let len = unsafe { raw::arcqml_rt_last_error(std::ptr::null_mut(), 0) };
    let mut bytes = vec![0; len.saturating_add(1)];
    unsafe { raw::arcqml_rt_last_error(bytes.as_mut_ptr(), bytes.len()) };
    bytes.truncate(len);
    String::from_utf8_lossy(&bytes).into_owned()
}
