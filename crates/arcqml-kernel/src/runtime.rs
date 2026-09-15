//! 为不透明原生执行计划准备源码可见的参数和描述符。

use crate::{KernelError, KernelResult, bind_gate, statevector::single};
use arcqml_circuit::{Circuit, Gate};
use arcqml_runtime_abi::{OperationDescriptor, ParameterBindingDescriptor};
use num_complex::Complex64;

/// 由专有运行时拥有的不透明、不可变电路计划。
#[doc(hidden)]
#[derive(Debug)]
pub struct RuntimeCircuitPlan {
    inner: arcqml_runtime_sys::CircuitPlan,
}

impl RuntimeCircuitPlan {
    pub fn compile(circuit: &Circuit, parameter_parent_offset: usize) -> KernelResult<Self> {
        let mut inner = arcqml_runtime_sys::CircuitPlan::with_capacity(circuit.operations().len())
            .map_err(runtime_error)?;
        for operation in circuit.operations() {
            if matches!(operation.gate(), Gate::CustomUnitary { .. }) {
                let description = OperationDescriptor {
                    gate: single::gate_descriptor(operation.gate())?,
                    qubits: operation.qubits().as_ptr().cast::<usize>(),
                    qubits_len: operation.qubits().len(),
                    parameters: std::ptr::null(),
                    parameters_len: 0,
                };
                inner.push(&description).map_err(runtime_error)?;
                continue;
            }
            let bound = bind_gate(circuit, operation.gate())?;
            let source_parameters = operation.gate().parameters();
            let mut operation_bindings = [ParameterBindingDescriptor {
                parameter_slot: 0,
                parent_index: 0,
            }; 3];
            if bound.parameters.len() > operation_bindings.len() {
                return Err(KernelError::CircuitError {
                    message: "a gate has more parameter slots than the runtime ABI supports"
                        .to_string(),
                });
            }
            for (destination, parameter) in operation_bindings.iter_mut().zip(&bound.parameters) {
                let id = source_parameters
                    .get(parameter.parameter_slot)
                    .and_then(|source| source.as_parameter())
                    .ok_or_else(|| KernelError::CircuitError {
                        message: format!(
                            "bound parameter slot {} no longer refers to a circuit parameter",
                            parameter.parameter_slot
                        ),
                    })?;
                *destination = ParameterBindingDescriptor {
                    parameter_slot: parameter.parameter_slot,
                    parent_index: id.index() + parameter_parent_offset,
                };
            }
            let description = OperationDescriptor {
                gate: single::gate_descriptor(&bound.gate)?,
                qubits: operation.qubits().as_ptr().cast::<usize>(),
                qubits_len: operation.qubits().len(),
                parameters: operation_bindings.as_ptr(),
                parameters_len: bound.parameters.len(),
            };
            inner.push(&description).map_err(runtime_error)?;
        }
        Ok(Self { inner })
    }

    pub fn apply(&self, amplitudes: &mut [Complex64], num_qubits: usize) -> KernelResult<()> {
        self.inner
            .apply(amplitudes, num_qubits)
            .map_err(|error| KernelError::CircuitError {
                message: error.to_string(),
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
    ) -> KernelResult<()> {
        self.inner
            .adjoint_backward(
                forward_state,
                adjoint_state,
                derivative_state,
                num_qubits,
                upstream,
                parameter_requires_grad,
                gradients,
            )
            .map_err(|error| KernelError::CircuitError {
                message: error.to_string(),
            })
    }
}

fn runtime_error(error: arcqml_runtime_sys::RuntimeError) -> KernelError {
    KernelError::CircuitError {
        message: error.to_string(),
    }
}
