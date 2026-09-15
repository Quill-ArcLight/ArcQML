use crate::statevector::shared::{dimension::dimension, observables};
use crate::statevector::state::batch as state;
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

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// 将一个 batch Gate 封装为独立 Tensor Autograd 节点并执行前向计算。
pub(crate) fn apply_gate(
    num_qubits: usize,
    input: Tensor,
    gate: Gate,
    qubits: Vec<Qubit>,
    parameters: Vec<BoundParameter>,
) -> SimResult<Tensor> {
    let (batch_size, input_values) = state::tensor_amplitudes(num_qubits, &input)?;
    let mut output_values = input_values.clone();
    arcqml_kernel::statevector::apply_gate_batch(
        &mut output_values,
        batch_size,
        num_qubits,
        &gate,
        &qubits,
    )?;

    let parameter_slots = parameters
        .iter()
        .map(|parameter| parameter.parameter_slot)
        .collect();
    let mut parents = vec![input];
    parents.extend(parameters.into_iter().map(|parameter| parameter.tensor));
    let backward = BatchGateBackward {
        num_qubits,
        batch_size,
        gate,
        qubits,
        parameter_slots,
    };

    Tensor::from_operation_named(
        "batch_statevector_gate",
        Storage::C64(output_values),
        parents[0].meta().clone(),
        parents,
        Arc::new(backward),
    )
    .map_err(core_tensor_error)
}

/// 将 batch observable 期望值封装为独立 Tensor Autograd 节点。
pub(crate) fn expectation(
    num_qubits: usize,
    state_tensor: Tensor,
    observable: &SparsePauliOp,
) -> SimResult<Tensor> {
    let (batch_size, state_values) = state::tensor_amplitudes(num_qubits, &state_tensor)?;
    let dimension = dimension(num_qubits)?;
    let (values, hamiltonian_state) = if state_tensor.requires_grad() {
        let (values, hamiltonian_state) =
            expectation_and_apply_rows(num_qubits, dimension, &state_values, observable)?;
        (values, Some(hamiltonian_state))
    } else {
        (
            expectation_values(num_qubits, dimension, &state_values, observable)?,
            None,
        )
    };
    let meta = TensorMeta::new(vec![batch_size], DType::F64, Device::Cpu, Layout::Dense)
        .map_err(core_tensor_error)?;
    let backward = BatchExpectationBackward {
        batch_size,
        hamiltonian_state,
    };

    Tensor::from_operation_named(
        "batch_statevector_expectation",
        Storage::F64(values),
        meta,
        vec![state_tensor],
        Arc::new(backward),
    )
    .map_err(core_tensor_error)
}

#[derive(Debug)]
struct BatchGateBackward {
    num_qubits: usize,
    batch_size: usize,
    gate: Gate,
    qubits: Vec<Qubit>,
    parameter_slots: Vec<usize>,
}

impl BackwardFn for BatchGateBackward {
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
        let (gradient_batch_size, upstream_values) =
            state::tensor_values(self.num_qubits, grad_output).map_err(sim_autograd_error)?;
        ensure_batch_size(self.batch_size, gradient_batch_size)?;
        let input_gradient = if input_requires_grad {
            let mut state_gradient = upstream_values.clone();
            arcqml_kernel::statevector::apply_adjoint_gate_batch(
                &mut state_gradient,
                self.batch_size,
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
            let (input_batch_size, values) =
                state::tensor_values(self.num_qubits, input).map_err(sim_autograd_error)?;
            ensure_batch_size(self.batch_size, input_batch_size)?;
            Some(values)
        } else {
            None
        };

        let mut gradients = vec![input_gradient];
        for (parent_index, parameter_slot) in self.parameter_slots.iter().copied().enumerate() {
            if !trainable_parameters[parent_index] {
                gradients.push(None);
                continue;
            }

            // 导数态为 (dRY/dθ)|ψᵢ⟩。
            let derivative = arcqml_kernel::statevector::parameter_derivative_batch(
                input_values
                    .as_deref()
                    .expect("trainable parameter requires input values"),
                self.batch_size,
                self.num_qubits,
                &self.gate,
                &self.qubits,
                parameter_slot,
            )
            .map_err(kernel_autograd_error)?;
            // 上游值为 |λᵢ₊₁⟩，导数态为 (dRY/dθ)|ψᵢ⟩。
            // 梯度为 2 Re[⟨λᵢ₊₁| (dRY/dθ)|ψᵢ⟩]。
            let parameter_gradient = batch_inner_product(&upstream_values, &derivative);
            gradients.push(Some(scalar_gradient_like(
                &parents[parent_index + 1],
                parameter_gradient,
            )?));
        }
        Ok(gradients)
    }
}

#[derive(Debug)]
struct BatchExpectationBackward {
    batch_size: usize,
    hamiltonian_state: Option<Vec<Complex64>>,
}

impl BackwardFn for BatchExpectationBackward {
    /// 将 batch expectation 的上游梯度转换为量子态梯度。
    fn backward(
        &self,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> CoreResult<Vec<Option<Tensor>>> {
        let state_parent = &parents[0];
        if !state_parent.requires_grad() {
            return Ok(vec![None]);
        }

        let upstream = vector_f64_values(self.batch_size, grad_output)?;
        let mut gradient = self.hamiltonian_state.clone().ok_or_else(|| {
            ArcQmlError::AutogradError(
                "batch expectation backward is missing its saved Hamiltonian state".to_string(),
            )
        })?;
        let dimension = gradient.len() / self.batch_size;
        scale_hamiltonian_rows(&mut gradient, dimension, &upstream);

        Ok(vec![Some(state::gradient_tensor_like(
            state_parent,
            gradient,
        )?)])
    }
}

/// 读取形状为 [B] 的 F64 上游梯度。
fn vector_f64_values(batch_size: usize, tensor: &Tensor) -> CoreResult<Vec<f64>> {
    if tensor.shape() != [batch_size] || tensor.dtype() != DType::F64 {
        return Err(ArcQmlError::AutogradError(format!(
            "batch expectation backward requires an F64 upstream gradient with shape [{batch_size}]"
        )));
    }
    match &*tensor.storage() {
        Storage::F64(values) => Ok(values.clone()),
        _ => Err(ArcQmlError::AutogradError(
            "batch expectation backward received invalid upstream storage".to_string(),
        )),
    }
}

/// 逐行委托缓存 Hamiltonian 计划计算无梯度 batch 期望值。
fn expectation_values(
    num_qubits: usize,
    dimension: usize,
    values: &[Complex64],
    observable: &SparsePauliOp,
) -> SimResult<Vec<f64>> {
    debug_assert_eq!(values.len() % dimension, 0);

    #[cfg(feature = "parallel")]
    {
        values
            .par_chunks_exact(dimension)
            .map(|row| observables::expectation_observable(row, num_qubits, observable))
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        values
            .chunks_exact(dimension)
            .map(|row| observables::expectation_observable(row, num_qubits, observable))
            .collect()
    }
}

/// 逐行一次计算期望值与 H|ψ⟩，供 batch 自动微分前反向共享。
fn expectation_and_apply_rows(
    num_qubits: usize,
    dimension: usize,
    values: &[Complex64],
    observable: &SparsePauliOp,
) -> SimResult<(Vec<f64>, Vec<Complex64>)> {
    debug_assert_eq!(values.len() % dimension, 0);

    #[cfg(feature = "parallel")]
    let rows: SimResult<Vec<(f64, Vec<Complex64>)>> = values
        .par_chunks_exact(dimension)
        .map(|row| {
            observables::expectation_and_apply_observable_serial(row, num_qubits, observable)
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let rows: SimResult<Vec<(f64, Vec<Complex64>)>> = values
        .chunks_exact(dimension)
        .map(|row| {
            observables::expectation_and_apply_observable_serial(row, num_qubits, observable)
        })
        .collect();

    let rows = rows?;
    let mut expectations = Vec::with_capacity(rows.len());
    let mut hamiltonian_state = Vec::with_capacity(values.len());
    for (value, row) in rows {
        expectations.push(value);
        hamiltonian_state.extend(row);
    }
    Ok((expectations, hamiltonian_state))
}

/// 按 batch 样本的上游梯度缩放保存的行主序 H|ψ⟩。
fn scale_hamiltonian_rows(values: &mut [Complex64], dimension: usize, upstream: &[f64]) {
    debug_assert_eq!(values.len(), dimension * upstream.len());

    #[cfg(feature = "parallel")]
    values
        .par_chunks_exact_mut(dimension)
        .zip(upstream.par_iter())
        .for_each(|(row, upstream_value)| {
            for value in row {
                *value *= 2.0 * *upstream_value;
            }
        });

    #[cfg(not(feature = "parallel"))]
    for (row, upstream_value) in values.chunks_exact_mut(dimension).zip(upstream) {
        for value in row {
            *value *= 2.0 * *upstream_value;
        }
    }
}
/// 计算整行主序 batch 上的 Re[<adjoint | derivative>]。
fn batch_inner_product(adjoint: &[Complex64], derivative: &[Complex64]) -> f64 {
    debug_assert_eq!(adjoint.len(), derivative.len());

    #[cfg(feature = "parallel")]
    {
        adjoint
            .par_iter()
            .zip(derivative.par_iter())
            .map(|(lambda, value)| (lambda.conj() * value).re)
            .sum::<f64>()
    }

    #[cfg(not(feature = "parallel"))]
    {
        adjoint
            .iter()
            .zip(derivative)
            .map(|(lambda, value)| (lambda.conj() * value).re)
            .sum::<f64>()
    }
}

/// 确保反向过程中 batch 大小未发生变化。
fn ensure_batch_size(expected: usize, actual: usize) -> CoreResult<()> {
    if expected != actual {
        return Err(ArcQmlError::AutogradError(format!(
            "batch size changed inside the state-vector autograd graph: expected {expected}, got {actual}"
        )));
    }
    Ok(())
}

/// 按参数 Tensor 的数据类型构造标量梯度。
fn scalar_gradient_like(parent: &Tensor, value: f64) -> CoreResult<Tensor> {
    match parent.dtype() {
        DType::F64 => Tensor::new(value),
        DType::F32 => Tensor::new(value as f32),
        dtype => Err(ArcQmlError::AutogradError(format!(
            "batch state-vector analytic gate backward does not support parameter dtype {dtype}"
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
