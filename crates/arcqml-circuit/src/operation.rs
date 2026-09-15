use crate::{CircuitError::*, CircuitResult, Gate, ParameterId, Qubit};
use std::collections::BTreeSet;

/// 电路中的一个门操作，由门定义和按局部位序排列的量子比特组成。
#[derive(Debug, Clone, PartialEq)]
pub struct Operation {
    gate: Gate,
    qubits: Vec<Qubit>,
}

impl Operation {
    /// 创建并验证门操作。
    ///
    /// # Errors
    ///
    /// 当自定义门无效、门的 arity 与 `qubits.len()` 不匹配，或同一量子比特
    /// 在 `qubits` 中重复出现时返回错误。此处不检查量子比特是否属于某条具体电路。
    pub fn new(gate: Gate, qubits: Vec<Qubit>) -> CircuitResult<Self> {
        gate.validate()?;

        if gate.arity() != qubits.len() {
            return Err(GateArityError {
                gate: gate.name().to_string(),
                expected: gate.arity(),
                actual: qubits.len(),
            });
        }

        let mut seen = BTreeSet::new();

        for qubit in qubits.iter() {
            if !seen.insert(qubit.index()) {
                return Err(DuplicateQubitError {
                    index: qubit.index(),
                });
            }
        }

        Ok(Self { gate, qubits })
    }

    /// 返回量子门。
    pub fn gate(&self) -> &Gate {
        &self.gate
    }

    /// 返回按门的局部位序排列的量子比特。
    pub fn qubits(&self) -> &[Qubit] {
        &self.qubits
    }

    /// 使用已经校验过的映射表重写操作门内的参数标识。
    pub(crate) fn remap_parameter_ids(&mut self, mapping: &[ParameterId]) {
        self.gate.remap_parameter_ids(mapping);
    }
}
