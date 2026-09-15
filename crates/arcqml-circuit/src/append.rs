use crate::{Circuit, CircuitError::*, CircuitResult, ParameterId};
use std::collections::BTreeSet;

/// 追加电路时，从右侧电路参数到左侧既有参数的显式绑定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParameterBinding {
    source: ParameterId,
    target: ParameterId,
}

impl ParameterBinding {
    /// 创建 `source`（右侧）到 `target`（左侧）的绑定；有效性在追加时校验。
    pub fn new(source: ParameterId, target: ParameterId) -> Self {
        Self { source, target }
    }

    /// 返回右侧电路中的参数标识。
    pub fn source(self) -> ParameterId {
        self.source
    }

    /// 返回左侧电路中的目标参数标识。
    pub fn target(self) -> ParameterId {
        self.target
    }
}

/// 电路追加完成后的参数重映射和新增数量。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendReport {
    parameter_mapping: Vec<ParameterId>,
    appended_parameters: usize,
    appended_operations: usize,
}

impl AppendReport {
    /// 返回右侧每个原参数在结果电路中的最终标识，切片下标即右侧原参数标识。
    pub fn parameter_mapping(&self) -> &[ParameterId] {
        &self.parameter_mapping
    }

    /// 返回从右侧实际转移到左侧的参数数量，不包含已绑定参数。
    pub fn appended_parameters(&self) -> usize {
        self.appended_parameters
    }

    /// 返回追加到左侧末尾的操作数量。
    pub fn appended_operations(&self) -> usize {
        self.appended_operations
    }
}

/// 追加前生成的无失败执行计划。
struct AppendPlan {
    mapping: Vec<ParameterId>,
    transfer: Vec<bool>,
    names: Vec<Option<String>>,
    appended_parameters: usize,
    appended_operations: usize,
}

impl Circuit {
    /// 消费右侧电路，按时间顺序追加到当前电路末尾，并转移其全部参数。
    ///
    /// # Errors
    ///
    /// 当两条电路的量子比特数不同、参数绑定无效、参数数量溢出或无法预留容量时返回错误。
    pub fn append(&mut self, right: Circuit) -> CircuitResult<AppendReport> {
        self.append_with_bindings(right, &[])
    }

    /// 消费右侧电路并追加，同时将指定右侧参数复用为当前电路中的既有参数。
    ///
    /// 未绑定参数会被转移并重命名以保持全局名称唯一；所有参数梯度在成功后清空。
    ///
    /// # Errors
    ///
    /// 当两条电路的量子比特数不同、参数绑定无效、参数数量溢出或无法预留容量时返回错误。
    pub fn append_with_bindings(
        &mut self,
        mut right: Circuit,
        bindings: &[ParameterBinding],
    ) -> CircuitResult<AppendReport> {
        let mut plan = self.plan_append(&right, bindings)?;

        right.remap_operation_parameters(&plan.mapping);
        right.rename_unbound_parameters(&mut plan.names);
        self.reserve_append_capacity(&plan)?;
        self.move_unbound_parameters_from(right, &plan)
    }

    /// 在修改任一电路之前校验输入并创建完整的追加计划。
    fn plan_append(
        &self,
        right: &Circuit,
        bindings: &[ParameterBinding],
    ) -> CircuitResult<AppendPlan> {
        self.validate()?;
        right.validate()?;
        if self.num_qubits != right.num_qubits {
            return Err(AppendQubitCountMismatchError {
                left: self.num_qubits,
                right: right.num_qubits,
            });
        }

        let mut mapping = vec![None; right.parameters.len()];
        for binding in bindings {
            let source = binding.source.index();
            if source >= mapping.len() {
                return Err(InvalidBindingSourceParameterIdError {
                    index: source,
                    parameter_count: mapping.len(),
                });
            }
            let target = binding.target.index();
            if target >= self.parameters.len() {
                return Err(InvalidBindingTargetParameterIdError {
                    index: target,
                    parameter_count: self.parameters.len(),
                });
            }
            if mapping[source].replace(binding.target).is_some() {
                return Err(DuplicateBindingSourceParameterIdError { index: source });
            }
        }

        let appended_parameters = mapping.iter().filter(|entry| entry.is_none()).count();
        self.parameters
            .len()
            .checked_add(appended_parameters)
            .ok_or(AppendParameterCountOverflowError)?;

        let mut used_names = self
            .parameters
            .iter()
            .enumerate()
            .map(|(index, parameter)| parameter_name_at(parameter.name(), index))
            .collect::<CircuitResult<BTreeSet<_>>>()?;
        let mut next_id = self.parameters.len();
        let mut transfer = Vec::with_capacity(mapping.len());
        let mut names = Vec::with_capacity(mapping.len());
        for (index, entry) in mapping.iter_mut().enumerate() {
            if let Some(target) = entry {
                transfer.push(false);
                names.push(None);
                *entry = Some(*target);
                continue;
            }
            let name = continued_parameter_name(right.parameters[index].name(), next_id, index)?;
            if !used_names.insert(name.clone()) {
                return Err(DuplicateParameterNameError { name });
            }
            *entry = Some(ParameterId::new(next_id));
            transfer.push(true);
            names.push(Some(name));
            next_id += 1;
        }

        let mapping = mapping
            .into_iter()
            .map(|entry| entry.expect("every source parameter receives a mapping"))
            .collect();
        Ok(AppendPlan {
            mapping,
            transfer,
            names,
            appended_parameters,
            appended_operations: right.operations.len(),
        })
    }

    /// 为追加准备容量，避免逐项扩容造成额外复制。
    fn reserve_append_capacity(&mut self, plan: &AppendPlan) -> CircuitResult<()> {
        self.parameters
            .try_reserve(plan.appended_parameters)
            .map_err(|error| AppendCapacityError {
                collection: "parameters",
                additional: plan.appended_parameters,
                message: error.to_string(),
            })?;
        self.operations
            .try_reserve(plan.appended_operations)
            .map_err(|error| AppendCapacityError {
                collection: "operations",
                additional: plan.appended_operations,
                message: error.to_string(),
            })?;
        Ok(())
    }

    /// 重写右侧全部门的参数标识，此时映射已完成预校验。
    fn remap_operation_parameters(&mut self, mapping: &[ParameterId]) {
        for operation in &mut self.operations {
            operation.remap_parameter_ids(mapping);
        }
    }

    /// 仅重命名需要转移的参数，绑定参数会被直接丢弃。
    fn rename_unbound_parameters(&mut self, names: &mut [Option<String>]) {
        for (parameter, name) in self.parameters.iter().zip(names) {
            if let Some(name) = name.take() {
                parameter.set_name(name);
            }
        }
    }

    /// 将右侧的操作和未绑定参数移动进当前电路，并清空组合后的梯度。
    fn move_unbound_parameters_from(
        &mut self,
        mut right: Circuit,
        plan: &AppendPlan,
    ) -> CircuitResult<AppendReport> {
        for (index, parameter) in right.parameters.drain(..).enumerate() {
            if plan.transfer[index] {
                self.parameters.push(parameter);
            }
        }
        self.operations.append(&mut right.operations);
        for parameter in &self.parameters {
            parameter.zero_grad();
        }
        Ok(AppendReport {
            parameter_mapping: plan.mapping.clone(),
            appended_parameters: plan.appended_parameters,
            appended_operations: plan.appended_operations,
        })
    }
}

/// 读取并验证参数名称，以便在拼接前建立全局唯一名称集合。
fn parameter_name_at(name: Option<String>, index: usize) -> CircuitResult<String> {
    let name = name.ok_or(MissingParameterNameError { index })?;
    if name.trim().is_empty() {
        return Err(MissingParameterNameError { index });
    }
    Ok(name)
}

/// 将右侧参数名称的末尾编号替换为最终参数标识。
fn continued_parameter_name(
    source_name: Option<String>,
    target_id: usize,
    source_index: usize,
) -> CircuitResult<String> {
    let source_name = parameter_name_at(source_name, source_index)?;
    if let Some((prefix, suffix)) = source_name.rsplit_once('_')
        && suffix.parse::<usize>().is_ok()
    {
        return Ok(format!("{prefix}_{target_id}"));
    }
    Ok(format!("{source_name}_{target_id}"))
}
