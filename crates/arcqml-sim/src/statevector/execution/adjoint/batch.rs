#[path = "row_major.rs"]
mod row_major;

use crate::{
    SimError::TensorError,
    SimResult,
    statevector::{shared::circuit_validation, state::batch as state},
};
use arcqml_circuit::Circuit;
use arcqml_core::{
    ArcQmlError, BackwardFn, DType, Result as CoreResult, Storage, Tensor, is_grad_enabled,
};
use arcqml_kernel::runtime::RuntimeCircuitPlan;
use arcqml_observable::SparsePauliOp;
use num_complex::Complex64;

/// 批量伴随反向传播需要保留的行主序前向上下文。
#[derive(Debug)]
struct AdjointContext {
    num_qubits: usize,
    batch_size: usize,
    dimension: usize,
    final_state: Vec<Complex64>,
    hamiltonian_state: Vec<Complex64>,
    plan: RuntimeCircuitPlan,
    has_initial_state_parent: bool,
}

/// 以唯一的行主序 batch 分块策略执行批处理电路。
pub(crate) fn run(
    num_qubits: usize,
    initial_state: &Tensor,
    circuit: &Circuit,
    observable: &SparsePauliOp,
) -> SimResult<Tensor> {
    circuit_validation::validate_circuit_qubits(num_qubits, circuit)?;
    let (batch_size, row_major_input) = state::tensor_amplitudes(num_qubits, initial_state)?;
    if !records_gradients(initial_state, circuit) {
        return row_major::run_without_grad(
            num_qubits,
            batch_size,
            &row_major_input,
            circuit,
            observable,
        );
    }
    row_major::run_from_validated_row_major_amplitudes(
        num_qubits,
        batch_size,
        &row_major_input,
        circuit,
        observable,
        Some(initial_state),
    )
}

/// 判断常规 Tensor 初态或电路参数是否需要记录伴随反向图。
fn records_gradients(initial_state: &Tensor, circuit: &Circuit) -> bool {
    is_grad_enabled() && (initial_state.requires_grad() || records_parameter_gradients(circuit))
}

/// 判断电路参数是否需要记录伴随反向图。
fn records_parameter_gradients(circuit: &Circuit) -> bool {
    is_grad_enabled()
        && circuit
            .parameters()
            .iter()
            .any(|parameter| parameter.tensor().requires_grad())
}

/// 批量伴随执行器的自动微分反向函数。
#[derive(Debug)]
struct AdjointBackward {
    context: AdjointContext,
}

impl BackwardFn for AdjointBackward {
    /// 逆序扫描行主序电路，并分别返回初态和电路参数的梯度。
    fn backward(
        &self,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> CoreResult<Vec<Option<Tensor>>> {
        let upstream = vector_f64_values(self.context.batch_size, grad_output)?;
        let parameter_requires_grad = parents
            .iter()
            .map(|parent| u8::from(parent.requires_grad()))
            .collect::<Vec<_>>();
        let (adjoint_state, gradients) =
            row_major::backward(&self.context, &upstream, &parameter_requires_grad)
                .map_err(sim_autograd_error)?;

        let parameter_parent_offset = usize::from(self.context.has_initial_state_parent);
        let mut outputs = Vec::with_capacity(parents.len());
        if self.context.has_initial_state_parent {
            let initial_state = parents.first().ok_or_else(|| {
                ArcQmlError::AutogradError(
                    "adjoint batch context is missing its initial-state parent".to_string(),
                )
            })?;
            let initial_gradient = initial_state
                .requires_grad()
                .then(|| {
                    let values = adjoint_state
                        .iter()
                        .map(|value| 2.0 * *value)
                        .collect::<Vec<_>>();
                    state::gradient_tensor_like(initial_state, values)
                })
                .transpose()?;
            outputs.push(initial_gradient);
        }
        outputs.extend(
            parents
                .iter()
                .skip(parameter_parent_offset)
                .zip(gradients.into_iter().skip(parameter_parent_offset))
                .map(|(parent, gradient)| {
                    parent
                        .requires_grad()
                        .then(|| scalar_gradient_tensor(parent, gradient))
                        .transpose()
                })
                .collect::<CoreResult<Vec<_>>>()?,
        );
        Ok(outputs)
    }
}

/// 读取伴随反向所需的逐样本 F64 上游梯度。
fn vector_f64_values(expected_len: usize, tensor: &Tensor) -> CoreResult<Vec<f64>> {
    if tensor.shape() != [expected_len] || tensor.dtype() != DType::F64 {
        return Err(ArcQmlError::AutogradError(format!(
            "adjoint batch backward requires an F64 upstream gradient with shape [{expected_len}]"
        )));
    }
    match &*tensor.storage() {
        Storage::F64(values) if values.len() == expected_len => Ok(values.clone()),
        _ => Err(ArcQmlError::AutogradError(
            "adjoint batch backward received invalid upstream storage".to_string(),
        )),
    }
}

/// 将标量参数梯度转换为与参数数据类型一致的 Tensor。
fn scalar_gradient_tensor(parameter: &Tensor, gradient: f64) -> CoreResult<Tensor> {
    match parameter.dtype() {
        DType::F32 => Tensor::new(gradient as f32),
        DType::F64 => Tensor::new(gradient),
        dtype => Err(ArcQmlError::AutogradError(format!(
            "run supports only F32 or F64 circuit parameters, got {dtype}"
        ))),
    }
}

/// 将核心 Tensor 错误转换为模拟器错误。
fn core_tensor_error(error: impl std::fmt::Display) -> crate::SimError {
    TensorError {
        message: error.to_string(),
    }
}

/// 将模拟器错误转换为自动微分错误。
fn sim_autograd_error(error: crate::SimError) -> ArcQmlError {
    ArcQmlError::AutogradError(error.to_string())
}
