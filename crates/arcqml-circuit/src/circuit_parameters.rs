use crate::{Circuit, CircuitError::*, CircuitResult, ParameterId};
use arcqml_core::{DType, Parameter, Storage, Tensor};

impl Circuit {
    /// 返回标量浮点参数的数值，并转换为 `f64`。
    ///
    /// # Errors
    ///
    /// 当 `id` 无效、参数不是单元素 Tensor，或其 dtype 不是 `F32`/`F64` 时返回错误。
    pub fn parameter_scalar_value(&self, id: ParameterId) -> CircuitResult<f64> {
        let parameter = self.parameter(id)?;
        let tensor = parameter.tensor();
        if tensor.numel() != 1 {
            return Err(InvalidParameterShapeError {
                index: id.index(),
                numel: tensor.numel(),
            });
        }
        match &*tensor.storage() {
            Storage::F64(values) => Ok(values[0]),
            Storage::F32(values) => Ok(values[0] as f64),
            storage => Err(UnsupportedParameterDTypeError {
                index: id.index(),
                dtype: storage.dtype().to_string(),
            }),
        }
    }

    /// 创建匿名 `F64` 标量参数，并返回它在当前电路中的稳定标识。
    ///
    /// # Errors
    ///
    /// 当无法构造底层标量 Tensor 时返回错误。
    pub fn add_parameter(&mut self, value: f64) -> CircuitResult<ParameterId> {
        let tensor = Tensor::new(value).map_err(|error| TensorCreateError {
            message: error.to_string(),
        })?;
        self.add_parameter_tensor(tensor)
    }

    /// 添加外部提供的单元素浮点 Tensor 参数，并自动生成唯一名称。
    ///
    /// # Errors
    ///
    /// 当 `tensor` 的元素数不是一，或 dtype 不是 `F32`/`F64` 时返回错误。
    pub fn add_parameter_tensor(&mut self, tensor: Tensor) -> CircuitResult<ParameterId> {
        let name = generic_parameter_name(self.parameters.len());
        self.add_parameter_tensor_named(tensor, name)
    }

    /// 添加具名参数，并验证其形状和数据类型。
    pub(crate) fn add_parameter_tensor_named(
        &mut self,
        tensor: Tensor,
        name: String,
    ) -> CircuitResult<ParameterId> {
        let id = ParameterId::new(self.parameters.len());
        if tensor.numel() != 1 {
            return Err(InvalidParameterShapeError {
                index: id.index(),
                numel: tensor.numel(),
            });
        }
        if !matches!(tensor.dtype(), DType::F64 | DType::F32) {
            return Err(UnsupportedParameterDTypeError {
                index: id.index(),
                dtype: tensor.dtype().to_string(),
            });
        }
        let parameter = Parameter::new(tensor);
        parameter.set_name(name);
        self.parameters.push(parameter);
        Ok(id)
    }

    /// 根据门、参数角色和量子比特创建自动命名的参数。
    pub(crate) fn add_gate_parameter(
        &mut self,
        gate: &str,
        role: &str,
        qubit: crate::Qubit,
        value: f64,
    ) -> CircuitResult<ParameterId> {
        let tensor = Tensor::new(value).map_err(|error| TensorCreateError {
            message: error.to_string(),
        })?;
        let name = gate_parameter_name(gate, role, qubit, self.parameters.len());
        self.add_parameter_tensor_named(tensor, name)
    }
}

/// 生成手动添加参数的连续名称。
pub(crate) fn generic_parameter_name(index: usize) -> String {
    format!("parameter_{index}")
}

/// 生成自动门参数的连续名称。
pub(crate) fn gate_parameter_name(
    gate: &str,
    role: &str,
    qubit: crate::Qubit,
    index: usize,
) -> String {
    format!("{gate}_q{}_{role}_{index}", qubit.index())
}
