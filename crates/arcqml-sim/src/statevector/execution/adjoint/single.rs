use crate::{
    SimError::TensorError,
    SimResult,
    statevector::{
        shared::{circuit_validation, observables},
        state::single as state,
    },
};
use arcqml_circuit::Circuit;
use arcqml_core::{
    ArcQmlError, BackwardFn, DType, Device, Layout, Result as CoreResult, Storage, Tensor,
    TensorMeta, is_grad_enabled,
};
use arcqml_kernel::runtime::RuntimeCircuitPlan;
use arcqml_observable::SparsePauliOp;
use num_complex::Complex64;
use std::sync::Arc;

#[derive(Debug)]
struct AdjointContext {
    num_qubits: usize,
    final_state: Vec<Complex64>,
    hamiltonian_state: Vec<Complex64>,
    plan: RuntimeCircuitPlan,
}

/// 执行一次电路并返回标量期望值 Tensor。
/// 其反向规则使用量子伴随算法，并同时计算初始态与电路参数的梯度。
pub(crate) fn run(
    num_qubits: usize,
    initial_state: &Tensor,
    circuit: &Circuit,
    observable: &SparsePauliOp,
) -> SimResult<Tensor> {
    circuit_validation::validate_circuit_qubits(num_qubits, circuit)?;
    if !records_gradients(initial_state, circuit) {
        return run_without_grad(num_qubits, initial_state, circuit, observable);
    }

    let mut final_state = state::tensor_amplitudes(num_qubits, initial_state)?;
    let plan = RuntimeCircuitPlan::compile(circuit, 1)?;
    plan.apply(&mut final_state, num_qubits)?;

    let (value, hamiltonian_state) =
        observables::expectation_and_apply_observable(&final_state, num_qubits, observable)?;
    let mut parents = Vec::with_capacity(circuit.num_parameters() + 1);
    parents.push(initial_state.clone());
    parents.extend(
        circuit
            .parameters()
            .iter()
            .map(|parameter| parameter.tensor()),
    );
    let context = AdjointContext {
        num_qubits,
        final_state,
        hamiltonian_state,
        plan,
    };
    let meta = TensorMeta::new(Vec::new(), DType::F64, Device::Cpu, Layout::Dense)
        .map_err(core_tensor_error)?;
    Tensor::from_operation_named(
        "adjoint_expectation",
        Storage::F64(vec![value]),
        meta,
        parents,
        Arc::new(AdjointBackward { context }),
    )
    .map_err(core_tensor_error)
}

/// 判断初态或电路参数是否需要记录伴随反向图。
fn records_gradients(initial_state: &Tensor, circuit: &Circuit) -> bool {
    is_grad_enabled()
        && (initial_state.requires_grad()
            || circuit
                .parameters()
                .iter()
                .any(|parameter| parameter.tensor().requires_grad()))
}

/// 在不记录自动微分图时执行单态前向计算，不保留伴随上下文。
fn run_without_grad(
    num_qubits: usize,
    initial_state: &Tensor,
    circuit: &Circuit,
    observable: &SparsePauliOp,
) -> SimResult<Tensor> {
    let mut final_state = state::tensor_amplitudes(num_qubits, initial_state)?;
    let plan = RuntimeCircuitPlan::compile(circuit, 0)?;
    plan.apply(&mut final_state, num_qubits)?;
    let (value, _) =
        observables::expectation_and_apply_observable(&final_state, num_qubits, observable)?;
    let meta = TensorMeta::new(Vec::new(), DType::F64, Device::Cpu, Layout::Dense)
        .map_err(core_tensor_error)?;
    Tensor::from_storage_meta(Storage::F64(vec![value]), meta).map_err(core_tensor_error)
}

#[derive(Debug)]
struct AdjointBackward {
    context: AdjointContext,
}

impl BackwardFn for AdjointBackward {
    /// 逆序扫描电路，并分别返回初态和电路参数的梯度。
    fn backward(
        &self,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> CoreResult<Vec<Option<Tensor>>> {
        let upstream = scalar_f64_value(grad_output)?;
        let mut forward_state = self.context.final_state.clone();
        let mut adjoint_state = self.context.hamiltonian_state.clone();
        let mut derivative_state = vec![Complex64::new(0.0, 0.0); forward_state.len()];
        let mut gradients = vec![0.0; parents.len()];
        let required = parents
            .iter()
            .map(|parent| u8::from(parent.requires_grad()))
            .collect::<Vec<_>>();
        self.context
            .plan
            .adjoint_backward(
                &mut forward_state,
                &mut adjoint_state,
                &mut derivative_state,
                self.context.num_qubits,
                upstream,
                &required,
                &mut gradients,
            )
            .map_err(kernel_autograd_error)?;

        let initial_state = parents.first().ok_or_else(|| {
            ArcQmlError::AutogradError(
                "adjoint context is missing its initial-state parent".to_string(),
            )
        })?;
        let initial_gradient = initial_state
            .requires_grad()
            .then(|| {
                state::gradient_tensor_like(
                    initial_state,
                    adjoint_state.into_iter().map(|value| 2.0 * value).collect(),
                )
            })
            .transpose()?;
        let mut outputs = Vec::with_capacity(parents.len());
        outputs.push(initial_gradient);
        outputs.extend(
            parents[1..]
                .iter()
                .zip(gradients.into_iter().skip(1))
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

/// 读取标量 F64 上游梯度。
fn scalar_f64_value(tensor: &Tensor) -> CoreResult<f64> {
    if !tensor.shape().is_empty() || tensor.dtype() != DType::F64 {
        return Err(ArcQmlError::AutogradError(
            "adjoint expectation backward requires an F64 scalar upstream gradient".to_string(),
        ));
    }
    match &*tensor.storage() {
        Storage::F64(values) if values.len() == 1 => Ok(values[0]),
        _ => Err(ArcQmlError::AutogradError(
            "adjoint expectation backward received invalid upstream storage".to_string(),
        )),
    }
}

/// 按电路参数的数据类型构造标量梯度 Tensor。
fn scalar_gradient_tensor(parameter: &Tensor, gradient: f64) -> CoreResult<Tensor> {
    match parameter.dtype() {
        DType::F32 => Tensor::new(gradient as f32),
        DType::F64 => Tensor::new(gradient),
        dtype => Err(ArcQmlError::AutogradError(format!(
            "run supports only F32 or F64 circuit parameters, got {dtype}"
        ))),
    }
}

/// 将 arcqml-core Tensor 错误转换为模拟器错误。
fn core_tensor_error(error: impl std::fmt::Display) -> crate::SimError {
    TensorError {
        message: error.to_string(),
    }
}

/// 将量子内核错误转换为自动微分错误。
fn kernel_autograd_error(error: arcqml_kernel::KernelError) -> ArcQmlError {
    ArcQmlError::AutogradError(error.to_string())
}
