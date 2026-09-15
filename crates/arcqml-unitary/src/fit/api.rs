use crate::{
    DenseUnitary, UnitaryError, UnitaryResult,
    evolve::{StateBatch, circuit_state},
    fit::{
        adjoint::gradients,
        loss::{fidelity_and_loss, trace_overlap},
    },
};
use arcqml_circuit::Circuit;
use arcqml_core::{
    ArcQmlError, BackwardFn, DType, Device, Layout, Result as CoreResult, Storage, Tensor,
    TensorMeta,
};
use num_complex::Complex64;
use std::sync::Arc;

/// 计算目标矩阵与电路矩阵的全局相位不变保真度。
///
/// 对维度 `d`，返回 `|Tr(U_target† U_circuit)|² / d²`，并将浮点误差截断到 `[0, 1]`。
///
/// # Errors
///
/// 当目标矩阵与电路维度不同、参数数据类型不受支持或电路执行失败时返回错误。
pub fn unitary_fidelity(target: &DenseUnitary, circuit: &Circuit) -> UnitaryResult<f64> {
    Ok(loss_values(target, circuit)?.0)
}

/// 计算仅用于评估的酉矩阵拟合损失 `1 - fidelity`，不保留反向传播上下文。
///
/// # Errors
///
/// 当目标矩阵与电路维度不同、参数数据类型不受支持或电路执行失败时返回错误。
pub fn unitary_loss_value(target: &DenseUnitary, circuit: &Circuit) -> UnitaryResult<f64> {
    Ok(loss_values(target, circuit)?.1)
}

/// 计算可参与自动微分的 `F64` 标量损失 `1 - fidelity`。
///
/// 对返回 Tensor 调用 `backward()` 会以解析伴随算法把梯度累积到电路参数。
///
/// # Errors
///
/// 当目标矩阵与电路维度不同、参数数据类型不受支持或电路执行失败时返回错误。
pub fn unitary_loss(target: &DenseUnitary, circuit: &Circuit) -> UnitaryResult<Tensor> {
    validate_compatible(target, circuit)?;
    let state = circuit_state(circuit)?;
    let overlap = trace_overlap(target, &state.data)?;
    let (_, loss) = fidelity_and_loss(overlap, state.dimension);
    let parents = circuit
        .parameters()
        .iter()
        .map(|parameter| parameter.tensor())
        .collect();
    let meta = TensorMeta::new(Vec::new(), DType::F64, Device::Cpu, Layout::Dense)
        .map_err(tensor_error)?;
    Tensor::from_operation_named(
        "unitary_loss",
        Storage::F64(vec![loss]),
        meta,
        parents,
        Arc::new(UnitaryLossBackward {
            target: target.clone(),
            circuit: circuit.clone(),
            state,
            overlap,
        }),
    )
    .map_err(tensor_error)
}

/// 将整体酉矩阵伴随算法适配为 Tensor 自动微分反向规则。
#[derive(Debug)]
struct UnitaryLossBackward {
    target: DenseUnitary,
    circuit: Circuit,
    state: StateBatch,
    overlap: Complex64,
}

impl BackwardFn for UnitaryLossBackward {
    /// 复用前向状态计算参数梯度，并乘以标量上游梯度。
    fn backward(
        &self,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> CoreResult<Vec<Option<Tensor>>> {
        let upstream = scalar_f64(grad_output)?;
        if parents.len() != self.circuit.num_parameters() {
            return Err(ArcQmlError::AutogradError(format!(
                "unitary_loss backward expected {} parameter parents, got {}",
                self.circuit.num_parameters(),
                parents.len()
            )));
        }
        let mut state = self.state.clone();
        let values = gradients(&self.target, &self.circuit, &mut state, self.overlap)
            .map_err(|error| ArcQmlError::AutogradError(error.to_string()))?;
        parents
            .iter()
            .zip(values)
            .map(|(parent, value)| {
                if !parent.requires_grad() {
                    return Ok(None);
                }
                scalar_gradient(parent, value * upstream).map(Some)
            })
            .collect()
    }
}

/// 读取 F64 标量上游梯度。
fn scalar_f64(gradient: &Tensor) -> CoreResult<f64> {
    if !gradient.shape().is_empty() || gradient.dtype() != DType::F64 {
        return Err(ArcQmlError::AutogradError(
            "unitary_loss backward requires an F64 scalar upstream gradient".to_string(),
        ));
    }
    match &*gradient.storage() {
        Storage::F64(values) if values.len() == 1 => Ok(values[0]),
        _ => Err(ArcQmlError::AutogradError(
            "unitary_loss backward received invalid upstream storage".to_string(),
        )),
    }
}

/// 按父参数的数据类型创建标量梯度。
fn scalar_gradient(parent: &Tensor, value: f64) -> CoreResult<Tensor> {
    match parent.dtype() {
        DType::F32 => Tensor::new(value as f32),
        DType::F64 => Tensor::new(value),
        dtype => Err(ArcQmlError::AutogradError(format!(
            "unitary_loss supports only F32 or F64 parameter gradients, got {dtype}"
        ))),
    }
}

/// 执行线路并计算保真度与损失数值。
fn loss_values(target: &DenseUnitary, circuit: &Circuit) -> UnitaryResult<(f64, f64)> {
    validate_compatible(target, circuit)?;
    let state = circuit_state(circuit)?;
    let overlap = trace_overlap(target, &state.data)?;
    Ok(fidelity_and_loss(overlap, state.dimension))
}

/// 检查目标酉矩阵与线路的量子比特数量相同。
fn validate_compatible(target: &DenseUnitary, circuit: &Circuit) -> UnitaryResult<()> {
    if target.num_qubits() != circuit.num_qubits() {
        return Err(UnitaryError::QubitCountMismatchError {
            target: target.num_qubits(),
            circuit: circuit.num_qubits(),
        });
    }
    Ok(())
}

/// 将 arcqml-core 错误转换为酉矩阵模块错误。
fn tensor_error(error: impl ToString) -> UnitaryError {
    UnitaryError::TensorError {
        message: error.to_string(),
    }
}
