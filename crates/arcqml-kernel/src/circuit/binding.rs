use crate::{KernelError::*, KernelResult};
use arcqml_circuit::{Circuit, Gate, Gateparam, ParameterId};
use arcqml_core::Tensor;

/// 已绑定当前标量值的量子门，以及提供这些数值的可训练张量。
#[derive(Debug, Clone)]
pub struct BoundGate {
    /// 参数引用已替换为固定标量值的可执行门。
    pub gate: Gate,
    /// 门中仍需参与反向传播的原始参数 Tensor。
    pub parameters: Vec<BoundParameter>,
}

/// 一个可训练门参数，以及它在原始门中的槽位。
#[derive(Debug, Clone)]
pub struct BoundParameter {
    /// 参数在原始门参数列表中的零起始槽位；固定参数与可训练参数混用时不会压缩该编号。
    pub parameter_slot: usize,
    /// 提供当前标量值并接收梯度的 Tensor 句柄。
    pub tensor: Tensor,
}

/// 解析量子门，并保留状态向量解析反向传播所需的原始参数槽位。
///
/// # Errors
///
/// 当门引用无效参数标识，或对应参数不是单元素 `F32`/`F64` Tensor 时返回错误。
pub fn bind_gate(circuit: &Circuit, gate: &Gate) -> KernelResult<BoundGate> {
    let (resolved_gate, tensors) = resolve_gate(circuit, gate)?;
    let slots = parameter_slots(gate);

    debug_assert_eq!(slots.len(), tensors.len());
    let parameters = slots
        .into_iter()
        .zip(tensors)
        .map(|(parameter_slot, tensor)| BoundParameter {
            parameter_slot,
            tensor,
        })
        .collect();

    Ok(BoundGate {
        gate: resolved_gate,
        parameters,
    })
}

/// 将电路中一个门的参数解析为当前固定数值，并保留对应参数张量。
///
/// # Errors
///
/// 当门引用无效参数标识，或对应参数不是单元素 `F32`/`F64` Tensor 时返回错误。
pub fn resolve_gate(circuit: &Circuit, gate: &Gate) -> KernelResult<(Gate, Vec<Tensor>)> {
    let mut tensors = Vec::new();
    let resolved = match gate {
        Gate::Rx(p) => Gate::rx(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::Ry(p) => Gate::ry(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::Rz(p) => Gate::rz(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::Phase(p) => Gate::phase(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::U3(theta, phi, lambda) => Gate::u3(
            resolve_parameter(circuit, theta, &mut tensors)?,
            resolve_parameter(circuit, phi, &mut tensors)?,
            resolve_parameter(circuit, lambda, &mut tensors)?,
        ),
        Gate::CPhase(p) => Gate::cp(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::CRx(p) => Gate::crx(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::CRy(p) => Gate::cry(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::CRz(p) => Gate::crz(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::Rxx(p) => Gate::rxx(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::Ryy(p) => Gate::ryy(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::Rzz(p) => Gate::rzz(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::Rzx(p) => Gate::rzx(resolve_parameter(circuit, p, &mut tensors)?),
        Gate::FSim(theta, phi) => Gate::fsim(
            resolve_parameter(circuit, theta, &mut tensors)?,
            resolve_parameter(circuit, phi, &mut tensors)?,
        ),
        _ => gate.clone(),
    };
    Ok((resolved, tensors))
}

/// 将一个固定或参数化门参数转换为固定参数。
fn resolve_parameter(
    circuit: &Circuit,
    parameter: &Gateparam,
    tensors: &mut Vec<Tensor>,
) -> KernelResult<Gateparam> {
    match parameter {
        Gateparam::Fixed(value) => Ok(Gateparam::fixed(*value)),
        Gateparam::Param(id) => {
            let tensor = circuit
                .parameter(*id)
                .map_err(|error| CircuitError {
                    message: error.to_string(),
                })?
                .tensor();
            let value = scalar_parameter_value(circuit, *id)?;
            tensors.push(tensor);
            Ok(Gateparam::fixed(value))
        }
    }
}

/// 读取电路参数表中的标量参数值。
fn scalar_parameter_value(circuit: &Circuit, id: ParameterId) -> KernelResult<f64> {
    circuit
        .parameter_scalar_value(id)
        .map_err(|error| CircuitError {
            message: error.to_string(),
        })
}

/// 返回原始量子门中由可训练参数占据的槽位。
fn parameter_slots(gate: &Gate) -> Vec<usize> {
    let parameters: Vec<&Gateparam> = match gate {
        Gate::Rx(parameter)
        | Gate::Ry(parameter)
        | Gate::Rz(parameter)
        | Gate::Phase(parameter)
        | Gate::CPhase(parameter)
        | Gate::CRx(parameter)
        | Gate::CRy(parameter)
        | Gate::CRz(parameter)
        | Gate::Rxx(parameter)
        | Gate::Ryy(parameter)
        | Gate::Rzz(parameter)
        | Gate::Rzx(parameter) => vec![parameter],
        Gate::U3(theta, phi, lambda) => vec![theta, phi, lambda],
        Gate::FSim(theta, phi) => vec![theta, phi],
        _ => Vec::new(),
    };

    parameters
        .into_iter()
        .enumerate()
        .filter_map(|(slot, parameter)| parameter.as_parameter().map(|_| slot))
        .collect()
}
