use super::{AdjointBackward, AdjointContext, core_tensor_error};
use crate::{
    SimError::TensorError,
    SimResult,
    statevector::shared::{dimension::dimension, observables},
};
use arcqml_circuit::Circuit;
use arcqml_core::{DType, Device, Layout, Storage, Tensor, TensorMeta};
use arcqml_kernel::runtime::RuntimeCircuitPlan;
use arcqml_observable::SparsePauliOp;
use num_complex::Complex64;
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// 每个行主序顶层 `Rayon` 任务处理的连续 batch 行数；任务内必须保持串行。
const ROW_MAJOR_BATCH_ROWS_PER_TASK: usize = 1;

/// 单个行主序前向分块的结果。
struct RowMajorForwardChunk {
    values: Vec<f64>,
    final_state: Vec<Complex64>,
    hamiltonian_state: Option<Vec<Complex64>>,
}

/// 单个行主序反向分块的结果。
struct RowMajorBackwardChunk {
    initial_gradient: Vec<Complex64>,
    parameter_gradients: Vec<f64>,
}

/// 在 `no_grad` 模式下，以行主序 batch 分块方式执行 `run` 前向计算。
pub(super) fn run_without_grad(
    num_qubits: usize,
    batch_size: usize,
    row_major_input: &[Complex64],
    circuit: &Circuit,
    observable: &SparsePauliOp,
) -> SimResult<Tensor> {
    let dimension = dimension(num_qubits)?;
    let plan = RuntimeCircuitPlan::compile(circuit, 0)?;
    let chunks = execute_forward(
        num_qubits,
        dimension,
        row_major_input,
        &plan,
        observable,
        false,
    )?;
    output_tensor(batch_size, collect_forward_values(chunks, batch_size))
}

/// 以已校验的行主序振幅构建 batch `run` 的整电路伴随反向节点。
pub(super) fn run_from_validated_row_major_amplitudes(
    num_qubits: usize,
    batch_size: usize,
    row_major_input: &[Complex64],
    circuit: &Circuit,
    observable: &SparsePauliOp,
    initial_state: Option<&Tensor>,
) -> SimResult<Tensor> {
    let dimension = dimension(num_qubits)?;
    let parent_offset = usize::from(initial_state.is_some());
    let plan = RuntimeCircuitPlan::compile(circuit, parent_offset)?;
    let chunks = execute_forward(
        num_qubits,
        dimension,
        row_major_input,
        &plan,
        observable,
        true,
    )?;
    let (values, final_state, hamiltonian_state) = collect_forward_context(chunks, batch_size)?;
    let mut parents = Vec::with_capacity(circuit.num_parameters() + parent_offset);
    if let Some(initial_state) = initial_state {
        parents.push(initial_state.clone());
    }
    parents.extend(
        circuit
            .parameters()
            .iter()
            .map(|parameter| parameter.tensor()),
    );
    let meta = TensorMeta::new(vec![batch_size], DType::F64, Device::Cpu, Layout::Dense)
        .map_err(core_tensor_error)?;
    Tensor::from_operation_named(
        "batch_adjoint_expectation",
        Storage::F64(values),
        meta,
        parents,
        Arc::new(AdjointBackward {
            context: AdjointContext {
                num_qubits,
                batch_size,
                dimension,
                final_state,
                hamiltonian_state,
                plan,
                has_initial_state_parent: parent_offset == 1,
            },
        }),
    )
    .map_err(core_tensor_error)
}

/// 仅以 batch 顶层并行方式反向扫描行主序伴随上下文；内层内核严格串行。
pub(super) fn backward(
    context: &AdjointContext,
    upstream: &[f64],
    parameter_requires_grad: &[u8],
) -> SimResult<(Vec<Complex64>, Vec<f64>)> {
    let chunk_size = ROW_MAJOR_BATCH_ROWS_PER_TASK * context.dimension;

    #[cfg(feature = "parallel")]
    let chunks: SimResult<Vec<RowMajorBackwardChunk>> = context
        .final_state
        .par_chunks(chunk_size)
        .zip(context.hamiltonian_state.par_chunks(chunk_size))
        .zip(upstream.par_chunks(ROW_MAJOR_BATCH_ROWS_PER_TASK))
        .map(|((final_state, hamiltonian_state), upstream)| {
            backward_chunk(
                final_state,
                hamiltonian_state,
                upstream,
                context,
                parameter_requires_grad,
            )
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let chunks: SimResult<Vec<RowMajorBackwardChunk>> = context
        .final_state
        .chunks(chunk_size)
        .zip(context.hamiltonian_state.chunks(chunk_size))
        .zip(upstream.chunks(ROW_MAJOR_BATCH_ROWS_PER_TASK))
        .map(|((final_state, hamiltonian_state), upstream)| {
            backward_chunk(
                final_state,
                hamiltonian_state,
                upstream,
                context,
                parameter_requires_grad,
            )
        })
        .collect();

    let chunks = chunks?;
    let mut initial_gradient = Vec::with_capacity(context.final_state.len());
    let mut parameter_gradients = vec![0.0; parameter_requires_grad.len()];
    for chunk in chunks {
        initial_gradient.extend(chunk.initial_gradient);
        for (total, local) in parameter_gradients
            .iter_mut()
            .zip(chunk.parameter_gradients)
        {
            *total += local;
        }
    }
    Ok((initial_gradient, parameter_gradients))
}

/// 在每个连续行主序 batch 分块内串行执行完整前向线路。
fn execute_forward(
    num_qubits: usize,
    dimension: usize,
    row_major_input: &[Complex64],
    plan: &RuntimeCircuitPlan,
    observable: &SparsePauliOp,
    save_hamiltonian_state: bool,
) -> SimResult<Vec<RowMajorForwardChunk>> {
    let chunk_size = ROW_MAJOR_BATCH_ROWS_PER_TASK * dimension;

    #[cfg(feature = "parallel")]
    {
        row_major_input
            .par_chunks(chunk_size)
            .map(|input| {
                forward_chunk(
                    num_qubits,
                    dimension,
                    input,
                    plan,
                    observable,
                    save_hamiltonian_state,
                )
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    {
        row_major_input
            .chunks(chunk_size)
            .map(|input| {
                forward_chunk(
                    num_qubits,
                    dimension,
                    input,
                    plan,
                    observable,
                    save_hamiltonian_state,
                )
            })
            .collect()
    }
}

/// 在单个行主序 batch 分块内串行执行前向线路和可观测量计算；禁止二次并行。
fn forward_chunk(
    num_qubits: usize,
    dimension: usize,
    input: &[Complex64],
    plan: &RuntimeCircuitPlan,
    observable: &SparsePauliOp,
    save_hamiltonian_state: bool,
) -> SimResult<RowMajorForwardChunk> {
    let local_batch_size = input.len() / dimension;
    let mut final_state = input.to_vec();
    debug_assert_eq!(local_batch_size, ROW_MAJOR_BATCH_ROWS_PER_TASK);
    plan.apply(&mut final_state, num_qubits)?;

    let mut values = Vec::with_capacity(local_batch_size);
    let mut hamiltonian_state = save_hamiltonian_state.then(|| Vec::with_capacity(input.len()));
    for row in final_state.chunks_exact(dimension) {
        if let Some(saved_state) = hamiltonian_state.as_mut() {
            let (value, transformed) =
                observables::expectation_and_apply_observable_serial(row, num_qubits, observable)?;
            values.push(value);
            saved_state.extend(transformed);
        } else {
            values.push(observables::expectation_observable(
                row, num_qubits, observable,
            )?);
        }
    }
    Ok(RowMajorForwardChunk {
        values,
        final_state,
        hamiltonian_state,
    })
}

/// 按原始 batch 顺序拼接不需要伴随上下文的前向数值。
fn collect_forward_values(chunks: Vec<RowMajorForwardChunk>, batch_size: usize) -> Vec<f64> {
    let mut values = Vec::with_capacity(batch_size);
    for chunk in chunks {
        values.extend(chunk.values);
    }
    values
}

/// 按原始 batch 顺序拼接前向数值和反向所需状态。
fn collect_forward_context(
    chunks: Vec<RowMajorForwardChunk>,
    batch_size: usize,
) -> SimResult<(Vec<f64>, Vec<Complex64>, Vec<Complex64>)> {
    let mut values = Vec::with_capacity(batch_size);
    let mut final_state = Vec::new();
    let mut hamiltonian_state = Vec::new();
    for chunk in chunks {
        values.extend(chunk.values);
        final_state.extend(chunk.final_state);
        let saved_state = chunk.hamiltonian_state.ok_or_else(|| TensorError {
            message: "row-major adjoint forward did not preserve Hamiltonian state".to_string(),
        })?;
        hamiltonian_state.extend(saved_state);
    }
    Ok((values, final_state, hamiltonian_state))
}

/// 将逐样本期望值构造成形状为 `[batch_size]` 的输出 `Tensor`。
fn output_tensor(batch_size: usize, values: Vec<f64>) -> SimResult<Tensor> {
    let meta = TensorMeta::new(vec![batch_size], DType::F64, Device::Cpu, Layout::Dense)
        .map_err(core_tensor_error)?;
    Tensor::from_storage_meta(Storage::F64(values), meta).map_err(core_tensor_error)
}

/// 在单个 batch 分块内串行执行完整的伴随反向扫描；禁止二次并行。
fn backward_chunk(
    final_state: &[Complex64],
    hamiltonian_state: &[Complex64],
    upstream: &[f64],
    context: &AdjointContext,
    parameter_requires_grad: &[u8],
) -> SimResult<RowMajorBackwardChunk> {
    let local_batch_size = final_state.len() / context.dimension;
    let mut forward_state = final_state.to_vec();
    let mut adjoint_state = hamiltonian_state.to_vec();
    let mut parameter_gradients = vec![0.0; parameter_requires_grad.len()];
    debug_assert_eq!(local_batch_size, ROW_MAJOR_BATCH_ROWS_PER_TASK);
    let mut derivative_state = vec![Complex64::new(0.0, 0.0); forward_state.len()];
    context.plan.adjoint_backward(
        &mut forward_state,
        &mut adjoint_state,
        &mut derivative_state,
        context.num_qubits,
        upstream[0],
        parameter_requires_grad,
        &mut parameter_gradients,
    )?;
    Ok(RowMajorBackwardChunk {
        initial_gradient: adjoint_state,
        parameter_gradients,
    })
}
