use crate::{
    DenseUnitary, UnitaryError, UnitaryResult, dense::transpose_square, evolve::StateBatch,
};
use arcqml_circuit::Circuit;
use num_complex::Complex64;

/// 使用逆门恢复前向状态，并计算所有线路参数的解析梯度值。
pub(crate) fn gradients(
    target: &DenseUnitary,
    circuit: &Circuit,
    forward_state: &mut StateBatch,
    overlap: Complex64,
) -> UnitaryResult<Vec<f64>> {
    let dimension = target.dimension();
    let scale = (dimension as f64).powi(2);
    let mut gradients = vec![0.0; circuit.num_parameters()];
    let mut costate = transpose_square(target.as_row_major(), dimension);

    for operation in circuit.operations().iter().rev() {
        let (gate, _) =
            arcqml_kernel::resolve_gate(circuit, operation.gate()).map_err(|error| {
                UnitaryError::KernelError {
                    message: error.to_string(),
                }
            })?;
        arcqml_kernel::apply_adjoint_gate_batch(
            &mut forward_state.data,
            dimension,
            circuit.num_qubits(),
            &gate,
            operation.qubits(),
        )
        .map_err(|error| UnitaryError::KernelError {
            message: error.to_string(),
        })?;

        for (slot, parameter) in operation.gate().parameters().iter().enumerate() {
            let Some(id) = parameter.as_parameter() else {
                continue;
            };
            let derivative = arcqml_kernel::parameter_derivative_batch(
                &forward_state.data,
                dimension,
                circuit.num_qubits(),
                &gate,
                operation.qubits(),
                slot,
            )
            .map_err(|error| UnitaryError::KernelError {
                message: error.to_string(),
            })?;
            let derivative_overlap = inner_product(&costate, &derivative);
            gradients[id.index()] += -2.0 * (overlap.conj() * derivative_overlap).re / scale;
        }

        arcqml_kernel::apply_adjoint_gate_batch(
            &mut costate,
            dimension,
            circuit.num_qubits(),
            &gate,
            operation.qubits(),
        )
        .map_err(|error| UnitaryError::KernelError {
            message: error.to_string(),
        })?;
    }

    Ok(gradients)
}

/// 计算两个同布局复数批次的复内积。
fn inner_product(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * right)
        .sum()
}
