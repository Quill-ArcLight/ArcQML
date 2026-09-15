use crate::{CircuitError::*, CircuitResult, Gate, Operation, ParameterId, Qubit};
use arcqml_core::Parameter;
use std::collections::BTreeSet;

/// 保存量子比特数量、顺序操作和可训练参数的量子电路。
#[derive(Debug)]
pub struct Circuit {
    pub(crate) num_qubits: usize,
    pub(crate) operations: Vec<Operation>,
    pub(crate) parameters: Vec<Parameter>,
}

impl Clone for Circuit {
    /// 深复制电路，保证副本不与原电路共享参数值或梯度。
    fn clone(&self) -> Self {
        self.deep_clone()
            .expect("a valid Circuit must be deep-cloneable")
    }
}

impl Circuit {
    /// 创建具有指定量子比特数量的空电路。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零时返回 [`CircuitError::EmptyCircuitError`](crate::CircuitError::EmptyCircuitError)。
    pub fn new(num_qubits: usize) -> CircuitResult<Self> {
        if num_qubits == 0 {
            return Err(EmptyCircuitError);
        }
        Ok(Self {
            num_qubits,
            operations: Vec::new(),
            parameters: Vec::new(),
        })
    }

    /// 深复制电路及其全部参数状态。
    ///
    /// # Errors
    ///
    /// 当任一参数的底层 Tensor 无法深复制时返回错误。
    pub fn deep_clone(&self) -> CircuitResult<Self> {
        let parameters = self
            .parameters
            .iter()
            .map(|parameter| {
                parameter.deep_clone().map_err(|error| TensorCreateError {
                    message: error.to_string(),
                })
            })
            .collect::<CircuitResult<Vec<_>>>()?;
        Ok(Self {
            num_qubits: self.num_qubits,
            operations: self.operations.clone(),
            parameters,
        })
    }

    /// 返回电路中的量子比特数量。
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// 返回操作数量。
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// 判断电路是否不含任何操作。
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// 返回按量子比特资源计算的并行深度。
    pub fn depth(&self) -> usize {
        let mut qubit_layers = vec![0usize; self.num_qubits];
        let mut depth = 0usize;
        for operation in &self.operations {
            let layer = operation
                .qubits()
                .iter()
                .map(|qubit| qubit_layers[qubit.index()])
                .max()
                .unwrap_or(0)
                + 1;
            for qubit in operation.qubits() {
                qubit_layers[qubit.index()] = layer;
            }
            depth = depth.max(layer);
        }
        depth
    }

    /// 返回按执行顺序排列的操作只读切片。
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }

    /// 返回可训练参数只读切片。
    pub fn parameters(&self) -> &[Parameter] {
        &self.parameters
    }

    /// 返回已登记参数数量。
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }

    /// 根据参数标识返回参数引用。
    ///
    /// # Errors
    ///
    /// 当 `id` 不属于当前电路时返回错误。
    pub fn parameter(&self, id: ParameterId) -> CircuitResult<&Parameter> {
        self.parameters
            .get(id.index())
            .ok_or(InvalidParameterIdError {
                index: id.index(),
                parameter_count: self.parameters.len(),
            })
    }

    /// 根据唯一名称查询参数标识。
    pub fn parameter_id(&self, name: &str) -> Option<ParameterId> {
        self.parameters
            .iter()
            .position(|parameter| parameter.name().as_deref() == Some(name))
            .map(ParameterId::new)
    }

    /// 根据参数标识返回参数名称的副本。
    ///
    /// # Errors
    ///
    /// 当 `id` 不属于当前电路，或对应参数没有名称时返回错误。
    pub fn parameter_name(&self, id: ParameterId) -> CircuitResult<String> {
        self.parameter(id)?.name().ok_or(EmptyParameterNameError)
    }

    /// 清空全部操作和参数，但保留电路的量子比特数量。
    pub fn clear(&mut self) {
        self.operations.clear();
        self.parameters.clear();
    }

    /// 在电路末尾添加一个已经构造好的门操作。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复、门的 arity 与 `qubits` 长度不符、自定义门无效，
    /// 或门引用了不属于当前电路的参数时返回错误。
    pub fn add_gate(&mut self, gate: Gate, qubits: Vec<Qubit>) -> CircuitResult<&mut Self> {
        self.validate_qubits(&qubits)?;
        self.validate_gate_parameters(&gate)?;
        self.operations.push(Operation::new(gate, qubits)?);
        Ok(self)
    }

    /// 验证电路的量子比特数、操作、参数引用及参数名称唯一性。
    ///
    /// # Errors
    ///
    /// 当电路没有量子比特，操作含越界量子比特或无效参数引用，或参数名称为空、
    /// 缺失或重复时返回错误。
    pub fn validate(&self) -> CircuitResult<()> {
        if self.num_qubits == 0 {
            return Err(EmptyCircuitError);
        }
        for operation in &self.operations {
            self.validate_qubits(operation.qubits())?;
            self.validate_gate_parameters(operation.gate())?;
        }
        let mut names = BTreeSet::new();
        for (index, parameter) in self.parameters.iter().enumerate() {
            let name = parameter
                .name()
                .filter(|name| !name.trim().is_empty())
                .ok_or(MissingParameterNameError { index })?;
            if !names.insert(name.clone()) {
                return Err(DuplicateParameterNameError { name });
            }
        }
        Ok(())
    }

    /// 判断电路是否至少有一个参数化门。
    pub fn is_parameterized(&self) -> bool {
        self.operations
            .iter()
            .any(|operation| operation.gate().is_parameterized())
    }

    /// 验证门引用的参数标识均属于当前电路。
    pub(crate) fn validate_gate_parameters(&self, gate: &Gate) -> CircuitResult<()> {
        for id in gate.parameter_ids() {
            self.parameter(id)?;
        }
        Ok(())
    }

    /// 验证操作量子比特未越过电路边界。
    pub(crate) fn validate_qubits(&self, qubits: &[Qubit]) -> CircuitResult<()> {
        for qubit in qubits {
            if qubit.index() >= self.num_qubits {
                return Err(QubitOutOfRangeError {
                    index: qubit.index(),
                    num_qubits: self.num_qubits,
                });
            }
        }
        Ok(())
    }
}
