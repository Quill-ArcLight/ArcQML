use crate::statevector::shared::observables;
use crate::statevector::state::single as state;
use crate::{SimError::*, SimResult};
use arcqml_circuit::{Gate, Qubit};
use arcqml_core::{
    ArcQmlError, BackwardFn, DType, Device, Layout, Result as CoreResult, Storage, Tensor,
    TensorMeta,
};
use arcqml_kernel::BoundParameter;
use arcqml_observable::SparsePauliOp;
use num_complex::Complex64;
use std::sync::Arc;

/// 将单态 Gate 封装为独立 Tensor Autograd 节点并执行前向计算。
pub(crate) fn apply_gate(
    num_qubits: usize,
    input: Tensor,
    gate: Gate,
    qubits: Vec<Qubit>,
    parameters: Vec<BoundParameter>,
) -> SimResult<Tensor> {
    let input_values = state::tensor_amplitudes(num_qubits, &input)?;
    let mut output_values = input_values.clone();
    arcqml_kernel::statevector::apply_gate(&mut output_values, num_qubits, &gate, &qubits)?;
    let storage = Storage::C64(output_values);
    let meta = input.meta().clone();
    let mut parents = vec![input];
    let parameter_slots = parameters
        .iter()
        .map(|parameter| parameter.parameter_slot)
        .collect();
    parents.extend(parameters.into_iter().map(|parameter| parameter.tensor));
    let backward = GateBackward {
        num_qubits,
        gate,
        qubits,
        parameter_slots,
    };

    Tensor::from_operation_named(
        "statevector_gate",
        storage,
        meta,
        parents,
        Arc::new(backward),
    )
    .map_err(core_tensor_error)
}

/// 将单态 observable 期望值封装为独立 Tensor Autograd 节点。
pub(crate) fn expectation(
    num_qubits: usize,
    state_tensor: Tensor,
    observable: &SparsePauliOp,
) -> SimResult<Tensor> {
    let values = state::tensor_amplitudes(num_qubits, &state_tensor)?;
    let (value, hamiltonian_state) = if state_tensor.requires_grad() {
        let (value, hamiltonian_state) =
            observables::expectation_and_apply_observable(&values, num_qubits, observable)?;
        (value, Some(hamiltonian_state))
    } else {
        (
            observables::expectation_observable(&values, num_qubits, observable)?,
            None,
        )
    };
    let meta = TensorMeta::new(Vec::new(), DType::F64, Device::Cpu, Layout::Dense)
        .map_err(core_tensor_error)?;
    let backward = ExpectationBackward { hamiltonian_state };

    Tensor::from_operation_named(
        "statevector_expectation",
        Storage::F64(vec![value]),
        meta,
        vec![state_tensor],
        Arc::new(backward),
    )
    .map_err(core_tensor_error)
}

#[derive(Debug)]
struct GateBackward {
    num_qubits: usize,
    gate: Gate,
    qubits: Vec<Qubit>,
    parameter_slots: Vec<usize>,
}

impl BackwardFn for GateBackward {
    /// 分别构造输入状态梯度和每个门参数的梯度。
    fn backward(
        &self,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> CoreResult<Vec<Option<Tensor>>> {
        let input = &parents[0];
        let input_requires_grad = input.requires_grad();
        let trainable_parameters = parents[1..]
            .iter()
            .map(Tensor::requires_grad)
            .collect::<Vec<_>>();
        let needs_parameter_gradients = trainable_parameters.iter().any(|required| *required);
        let upstream =
            state::tensor_values(self.num_qubits, grad_output).map_err(sim_autograd_error)?;

        let input_gradient = if input_requires_grad {
            let mut state_gradient = upstream.clone();
            arcqml_kernel::statevector::apply_adjoint_gate(
                &mut state_gradient,
                self.num_qubits,
                &self.gate,
                &self.qubits,
            )
            .map_err(kernel_autograd_error)?;
            Some(state::gradient_tensor_like(input, state_gradient)?)
        } else {
            None
        };
        let input_values = if needs_parameter_gradients {
            Some(state::tensor_values(self.num_qubits, input).map_err(sim_autograd_error)?)
        } else {
            None
        };

        let mut gradients = vec![input_gradient];
        for (parent_index, parameter_slot) in self.parameter_slots.iter().copied().enumerate() {
            if !trainable_parameters[parent_index] {
                gradients.push(None);
                continue;
            }

            let derivative = arcqml_kernel::statevector::parameter_derivative(
                input_values
                    .as_deref()
                    .expect("trainable parameter requires input values"),
                self.num_qubits,
                &self.gate,
                &self.qubits,
                parameter_slot,
            )
            .map_err(kernel_autograd_error)?;
            let parameter_gradient = upstream
                .iter()
                .zip(derivative.iter())
                .map(|(gradient, derivative)| gradient.conj() * derivative)
                .sum::<Complex64>()
                .re;
            gradients.push(Some(scalar_gradient_like(
                &parents[parent_index + 1],
                parameter_gradient,
            )?));
        }

        Ok(gradients)
    }
}

#[derive(Debug)]
struct ExpectationBackward {
    hamiltonian_state: Option<Vec<Complex64>>,
}

impl BackwardFn for ExpectationBackward {
    /// 将标量 expectation 的上游梯度转换为量子态梯度。
    fn backward(
        &self,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> CoreResult<Vec<Option<Tensor>>> {
        let state_parent = &parents[0];
        if !state_parent.requires_grad() {
            return Ok(vec![None]);
        }

        let upstream = scalar_f64_value(grad_output)?;
        let mut gradient = self.hamiltonian_state.clone().ok_or_else(|| {
            ArcQmlError::AutogradError(
                "state-vector expectation backward is missing its saved Hamiltonian state"
                    .to_string(),
            )
        })?;
        for value in &mut gradient {
            *value *= 2.0 * upstream;
        }
        Ok(vec![Some(state::gradient_tensor_like(
            state_parent,
            gradient,
        )?)])
    }
}

/// 读取标量 F64 上游梯度。
fn scalar_f64_value(tensor: &Tensor) -> CoreResult<f64> {
    if !tensor.shape().is_empty() || tensor.dtype() != DType::F64 {
        return Err(ArcQmlError::AutogradError(
            "state-vector expectation backward requires an F64 scalar upstream gradient"
                .to_string(),
        ));
    }
    match &*tensor.storage() {
        Storage::F64(values) => Ok(values[0]),
        _ => Err(ArcQmlError::AutogradError(
            "state-vector expectation backward received invalid upstream storage".to_string(),
        )),
    }
}

/// 按参数 Tensor 的数据类型构造标量梯度。
fn scalar_gradient_like(parent: &Tensor, value: f64) -> CoreResult<Tensor> {
    match parent.dtype() {
        DType::F64 => Tensor::new(value),
        DType::F32 => Tensor::new(value as f32),
        dtype => Err(ArcQmlError::AutogradError(format!(
            "state-vector analytic gate backward does not support parameter dtype {dtype}"
        ))),
    }
}

/// 将 arcqml-core Tensor 构造错误转换为模拟器错误。
fn core_tensor_error(error: impl std::fmt::Display) -> crate::SimError {
    TensorError {
        message: error.to_string(),
    }
}

/// 将模拟器错误转换为 arcqml-core Autograd 错误。
fn sim_autograd_error(error: crate::SimError) -> ArcQmlError {
    ArcQmlError::AutogradError(error.to_string())
}

/// 将 kernel 错误转换为 arcqml-core Autograd 错误。
fn kernel_autograd_error(error: arcqml_kernel::KernelError) -> ArcQmlError {
    ArcQmlError::AutogradError(error.to_string())
}
