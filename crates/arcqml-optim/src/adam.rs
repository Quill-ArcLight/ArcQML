use crate::sgd::validate_grad;
use crate::{OptimError::*, OptimResult, StepStats};
use arcqml_core::{Parameter, Storage, Tensor};

/// 带偏差修正的一阶矩、二阶矩状态的 Adam 优化器。
///
/// 对梯度 `g` 先执行耦合式 L2 权重衰减 `g ← g + weight_decay × parameter`，再按
/// `m ← β1 m + (1-β1)g`、`v ← β2 v + (1-β2)g²` 更新矩估计，并使用偏差修正后的
/// `m`、`v` 更新参数。状态槽与传入切片的位置一一对应，因此连续调用 [`Adam::step`]
/// 或 [`Adam::step_tensors`] 时必须保持元素数量、顺序、dtype 和元素数稳定。
#[derive(Debug, Clone, PartialEq)]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    step: u64,
    states: Vec<Option<AdamState>>,
}

#[derive(Debug, Clone, PartialEq)]
enum AdamState {
    F32 { first: Vec<f32>, second: Vec<f32> },
    F64 { first: Vec<f64>, second: Vec<f64> },
}

impl Adam {
    /// 使用显式超参数创建尚未分配状态槽的 Adam 优化器。
    ///
    /// # Errors
    ///
    /// 当 `learning_rate` 或 `weight_decay` 不是有限非负数，`beta1`/`beta2` 不在
    /// `[0, 1)`，或 `epsilon` 不是有限正数时返回错误。
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> OptimResult<Self> {
        validate_non_negative("learning_rate", learning_rate)?;
        validate_beta("beta1", beta1)?;
        validate_beta("beta2", beta2)?;
        validate_positive("epsilon", epsilon)?;
        validate_non_negative("weight_decay", weight_decay)?;

        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step: 0,
            states: Vec::new(),
        })
    }

    /// 返回学习率。
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// 返回已发起的更新步数。
    ///
    /// 每次调用 [`Adam::step`] 或 [`Adam::step_tensors`] 都会递增该值，即使所有元素
    /// 都因冻结或缺少梯度而被跳过。
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// 清空切片中全部参数当前累积的梯度，不修改优化器状态。
    pub fn zero_grad(&self, parameters: &[Parameter]) {
        for parameter in parameters {
            parameter.zero_grad();
        }
    }

    /// 清空切片中全部 Tensor 当前累积的梯度，不修改优化器状态。
    pub fn zero_grad_tensors(&self, tensors: &[Tensor]) {
        for tensor in tensors {
            tensor.zero_grad();
        }
    }

    /// 按输入切片的稳定顺序更新具有梯度的可训练参数。
    ///
    /// 冻结、未启用梯度或尚无梯度的参数会被跳过并计入返回的 [`StepStats`]。
    ///
    /// # Errors
    ///
    /// 当参数切片长度相对首次调用发生变化、参数或梯度不是匹配的 `F32`/`F64`
    /// Tensor，或某一状态槽对应参数的 dtype 或元素数发生变化时返回错误。
    pub fn step(&mut self, parameters: &[Parameter]) -> OptimResult<StepStats> {
        self.ensure_slot_count(parameters.len())?;
        self.step += 1;
        let mut stats = StepStats::default();

        for (index, parameter) in parameters.iter().enumerate() {
            if !parameter.trainable() || !parameter.requires_grad() {
                stats = StepStats::new(
                    stats.updated(),
                    stats.skipped_frozen() + 1,
                    stats.skipped_no_grad(),
                );
                continue;
            }

            let Some(grad) = parameter.grad() else {
                stats = StepStats::new(
                    stats.updated(),
                    stats.skipped_frozen(),
                    stats.skipped_no_grad() + 1,
                );
                continue;
            };

            let tensor = parameter.tensor();
            self.update_tensor(index, parameter_name(parameter), &tensor, &grad)?;
            stats = StepStats::new(
                stats.updated() + 1,
                stats.skipped_frozen(),
                stats.skipped_no_grad(),
            );
        }

        Ok(stats)
    }

    /// 按输入切片的稳定顺序直接更新具有梯度的 Tensor。
    ///
    /// 未启用梯度或尚无梯度的 Tensor 会被跳过并计入返回的 [`StepStats`]。
    ///
    /// # Errors
    ///
    /// 当 Tensor 切片长度相对首次调用发生变化、Tensor 或梯度不是匹配的
    /// `F32`/`F64`，或某一状态槽对应 Tensor 的 dtype 或元素数发生变化时返回错误。
    pub fn step_tensors(&mut self, tensors: &[Tensor]) -> OptimResult<StepStats> {
        self.ensure_slot_count(tensors.len())?;
        self.step += 1;
        let mut stats = StepStats::default();

        for (index, tensor) in tensors.iter().enumerate() {
            if !tensor.requires_grad() {
                stats = StepStats::new(
                    stats.updated(),
                    stats.skipped_frozen() + 1,
                    stats.skipped_no_grad(),
                );
                continue;
            }

            let Some(grad) = tensor.grad() else {
                stats = StepStats::new(
                    stats.updated(),
                    stats.skipped_frozen(),
                    stats.skipped_no_grad() + 1,
                );
                continue;
            };

            self.update_tensor(index, format!("tensor#{index}"), tensor, &grad)?;
            stats = StepStats::new(
                stats.updated() + 1,
                stats.skipped_frozen(),
                stats.skipped_no_grad(),
            );
        }

        Ok(stats)
    }

    /// 初始化或校验与参数切片位置一一对应的 Adam 状态槽位数量。
    fn ensure_slot_count(&mut self, count: usize) -> OptimResult<()> {
        if self.states.is_empty() {
            self.states.resize_with(count, || None);
            return Ok(());
        }

        if self.states.len() != count {
            return Err(AdamStateMismatchError {
                message: format!("expected {} parameters, got {count}", self.states.len()),
            });
        }

        Ok(())
    }

    /// 使用指定位置的一阶矩和二阶矩状态更新单个 Tensor。
    fn update_tensor(
        &mut self,
        index: usize,
        name: String,
        parameter: &Tensor,
        grad: &Tensor,
    ) -> OptimResult<()> {
        validate_grad(name.clone(), parameter, grad)?;
        let beta1_correction = 1.0 - self.beta1.powi(self.step as i32);
        let beta2_correction = 1.0 - self.beta2.powi(self.step as i32);
        let learning_rate = self.learning_rate;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;
        let weight_decay = self.weight_decay;
        let state = self.state_for(index, &name, parameter)?;
        let grad_storage = grad.storage().clone();
        let mut parameter_storage = parameter.storage_mut();

        match (&mut *parameter_storage, &grad_storage, state) {
            (Storage::F32(values), Storage::F32(grads), AdamState::F32 { first, second }) => {
                for ((value, grad), (first, second)) in values
                    .iter_mut()
                    .zip(grads)
                    .zip(first.iter_mut().zip(second.iter_mut()))
                {
                    let adjusted = grad + weight_decay as f32 * *value;
                    *first = beta1 as f32 * *first + (1.0 - beta1 as f32) * adjusted;
                    *second = beta2 as f32 * *second + (1.0 - beta2 as f32) * adjusted * adjusted;
                    let first_hat = *first / beta1_correction as f32;
                    let second_hat = *second / beta2_correction as f32;
                    *value -=
                        learning_rate as f32 * first_hat / (second_hat.sqrt() + epsilon as f32);
                }
                Ok(())
            }
            (Storage::F64(values), Storage::F64(grads), AdamState::F64 { first, second }) => {
                for ((value, grad), (first, second)) in values
                    .iter_mut()
                    .zip(grads)
                    .zip(first.iter_mut().zip(second.iter_mut()))
                {
                    let adjusted = grad + weight_decay * *value;
                    *first = beta1 * *first + (1.0 - beta1) * adjusted;
                    *second = beta2 * *second + (1.0 - beta2) * adjusted * adjusted;
                    let first_hat = *first / beta1_correction;
                    let second_hat = *second / beta2_correction;
                    *value -= learning_rate * first_hat / (second_hat.sqrt() + epsilon);
                }
                Ok(())
            }
            (storage, _, _) => Err(UnsupportedParameterDTypeError {
                name,
                dtype: storage.dtype().to_string(),
            }),
        }
    }

    /// 返回指定位置的 Adam 状态，并在首次更新时按 Tensor 类型初始化状态。
    fn state_for(
        &mut self,
        index: usize,
        name: &str,
        tensor: &Tensor,
    ) -> OptimResult<&mut AdamState> {
        let state = &mut self.states[index];
        if state.is_none() {
            *state = Some(match tensor.dtype() {
                arcqml_core::DType::F32 => AdamState::F32 {
                    first: vec![0.0; tensor.numel()],
                    second: vec![0.0; tensor.numel()],
                },
                arcqml_core::DType::F64 => AdamState::F64 {
                    first: vec![0.0; tensor.numel()],
                    second: vec![0.0; tensor.numel()],
                },
                dtype => {
                    return Err(UnsupportedParameterDTypeError {
                        name: name.to_string(),
                        dtype: dtype.to_string(),
                    });
                }
            });
        }

        let valid = match state.as_ref().unwrap() {
            AdamState::F32 { first, .. } => {
                tensor.dtype() == arcqml_core::DType::F32 && first.len() == tensor.numel()
            }
            AdamState::F64 { first, .. } => {
                tensor.dtype() == arcqml_core::DType::F64 && first.len() == tensor.numel()
            }
        };
        if !valid {
            return Err(AdamStateMismatchError {
                message: format!("parameter {name} changed dtype or element count"),
            });
        }

        Ok(state.as_mut().unwrap())
    }
}

/// 验证超参数为有限非负数。
fn validate_non_negative(name: &'static str, value: f64) -> OptimResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(InvalidHyperParameterError { name, value });
    }
    Ok(())
}

/// 验证超参数为有限正数。
fn validate_positive(name: &'static str, value: f64) -> OptimResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(InvalidHyperParameterError { name, value });
    }
    Ok(())
}

/// 验证 Adam 的 beta 位于半开区间 [0, 1) 内。
fn validate_beta(name: &'static str, value: f64) -> OptimResult<()> {
    if !value.is_finite() || !(0.0..1.0).contains(&value) {
        return Err(InvalidHyperParameterError { name, value });
    }
    Ok(())
}

/// 返回 Parameter 名称，未命名时使用稳定的占位名称。
fn parameter_name(parameter: &Parameter) -> String {
    parameter.name().unwrap_or_else(|| "<unnamed>".to_string())
}
