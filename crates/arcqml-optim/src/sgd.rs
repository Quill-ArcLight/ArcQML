use crate::{OptimError::*, OptimResult};
use arcqml_core::{Parameter, Storage, Tensor};

/// 一次优化器调用的更新与跳过数量。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StepStats {
    updated: usize,
    skipped_frozen: usize,
    skipped_no_grad: usize,
}

impl StepStats {
    /// 使用三个互斥类别的计数创建统计值。
    pub fn new(updated: usize, skipped_frozen: usize, skipped_no_grad: usize) -> Self {
        Self {
            updated,
            skipped_frozen,
            skipped_no_grad,
        }
    }

    /// 返回实际执行原地更新的参数或 Tensor 数量。
    pub fn updated(&self) -> usize {
        self.updated
    }

    /// 返回因参数被冻结或 Tensor 的 `requires_grad` 为 `false` 而跳过的数量。
    pub fn skipped_frozen(&self) -> usize {
        self.skipped_frozen
    }

    /// 返回因当前没有已累积梯度而跳过的数量。
    pub fn skipped_no_grad(&self) -> usize {
        self.skipped_no_grad
    }
}

/// 采用可选耦合式 L2 权重衰减的随机梯度下降优化器。
///
/// 每一步执行 `parameter -= learning_rate × (gradient + weight_decay × parameter)`；
/// `weight_decay` 因此不是解耦式权重衰减。
#[derive(Debug, Clone, PartialEq)]
pub struct Sgd {
    learning_rate: f64,
    weight_decay: f64,
}

impl Sgd {
    /// 创建 SGD 优化器。
    ///
    /// # Errors
    ///
    /// 当学习率或权重衰减不是有限非负数时返回错误。
    pub fn new(learning_rate: f64, weight_decay: f64) -> OptimResult<Self> {
        validate_non_negative("learning_rate", learning_rate)?;
        validate_non_negative("weight_decay", weight_decay)?;

        Ok(Self {
            learning_rate,
            weight_decay,
        })
    }

    /// 返回学习率。
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// 设置后续步骤使用的学习率。
    ///
    /// # Errors
    ///
    /// 当 `learning_rate` 不是有限非负数时返回错误。
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> OptimResult<()> {
        validate_non_negative("learning_rate", learning_rate)?;

        self.learning_rate = learning_rate;

        Ok(())
    }

    /// 返回权重衰减系数。
    pub fn weight_decay(&self) -> f64 {
        self.weight_decay
    }

    /// 设置后续步骤使用的耦合式 L2 权重衰减系数。
    ///
    /// # Errors
    ///
    /// 当 `weight_decay` 不是有限非负数时返回错误。
    pub fn set_weight_decay(&mut self, weight_decay: f64) -> OptimResult<()> {
        validate_non_negative("weight_decay", weight_decay)?;

        self.weight_decay = weight_decay;

        Ok(())
    }

    /// 清空切片中全部参数当前累积的梯度。
    pub fn zero_grad(&self, parameters: &[Parameter]) {
        for parameter in parameters.iter() {
            parameter.zero_grad();
        }
    }

    /// 清空切片中全部 Tensor 当前累积的梯度。
    pub fn zero_grad_tensors(&self, tensors: &[Tensor]) {
        for tensor in tensors {
            tensor.zero_grad();
        }
    }

    /// 更新切片中具有梯度的可训练参数，并返回更新与跳过数量。
    ///
    /// # Errors
    ///
    /// 当参数及其梯度的元素数或 dtype 不一致，或参数不是 `F32`/`F64` 时返回错误。
    pub fn step(&self, parameters: &[Parameter]) -> OptimResult<StepStats> {
        let mut stats = StepStats::default();

        for parameter in parameters.iter() {
            if !parameter.trainable() || !parameter.requires_grad() {
                stats.skipped_frozen += 1;
                continue;
            }

            let Some(grad) = parameter.grad() else {
                stats.skipped_no_grad += 1;
                continue;
            };

            update_tensor(
                parameter_name(parameter),
                &parameter.tensor(),
                &grad,
                self.learning_rate,
                self.weight_decay,
            )?;
            stats.updated += 1;
        }

        Ok(stats)
    }

    /// 直接更新切片中已启用并具有梯度的 Tensor。
    ///
    /// # Errors
    ///
    /// 当 Tensor 及其梯度的元素数或 dtype 不一致，或 Tensor 不是 `F32`/`F64` 时返回错误。
    pub fn step_tensors(&self, tensors: &[Tensor]) -> OptimResult<StepStats> {
        let mut stats = StepStats::default();

        for (index, tensor) in tensors.iter().enumerate() {
            if !tensor.requires_grad() {
                stats.skipped_frozen += 1;
                continue;
            }

            let Some(grad) = tensor.grad() else {
                stats.skipped_no_grad += 1;
                continue;
            };

            update_tensor(
                format!("tensor#{index}"),
                tensor,
                &grad,
                self.learning_rate,
                self.weight_decay,
            )?;
            stats.updated += 1;
        }

        Ok(stats)
    }
}

/// 验证 SGD 超参数为有限非负数。
fn validate_non_negative(name: &'static str, value: f64) -> OptimResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(InvalidHyperParameterError { name, value });
    }

    Ok(())
}

/// 验证梯度与参数具有相同元素数量和数据类型。
pub(crate) fn validate_grad(name: String, parameter: &Tensor, grad: &Tensor) -> OptimResult<()> {
    if parameter.numel() != grad.numel() {
        return Err(GradShapeMismatchError {
            name,
            parameter_numel: parameter.numel(),
            grad_numel: grad.numel(),
        });
    }

    if parameter.dtype() != grad.dtype() {
        return Err(GradDTypeMismatchError {
            name,
            parameter_dtype: parameter.dtype().to_string(),
            grad_dtype: grad.dtype().to_string(),
        });
    }

    Ok(())
}

/// 使用 SGD 规则直接更新单个 Parameter 或叶子 Tensor。
pub(crate) fn update_tensor(
    name: String,
    parameter: &Tensor,
    grad: &Tensor,
    learning_rate: f64,
    weight_decay: f64,
) -> OptimResult<()> {
    validate_grad(name.clone(), parameter, grad)?;
    let grad_storage = grad.storage().clone();
    let mut parameter_storage = parameter.storage_mut();

    update_storage_in_place(
        name,
        &mut parameter_storage,
        &grad_storage,
        learning_rate,
        weight_decay,
    )
}

/// 对匹配类型的底层存储执行原地 SGD 更新。
fn update_storage_in_place(
    name: String,
    parameter: &mut Storage,
    grad: &Storage,
    learning_rate: f64,
    weight_decay: f64,
) -> OptimResult<()> {
    match (parameter, grad) {
        (Storage::F32(values), Storage::F32(grads)) => {
            for (value, grad) in values.iter_mut().zip(grads.iter().copied()) {
                *value -= learning_rate as f32 * (grad + weight_decay as f32 * *value);
            }

            Ok(())
        }
        (Storage::F64(values), Storage::F64(grads)) => {
            for (value, grad) in values.iter_mut().zip(grads.iter().copied()) {
                *value -= learning_rate * (grad + weight_decay * *value);
            }

            Ok(())
        }
        (storage, _) => Err(UnsupportedParameterDTypeError {
            name,
            dtype: storage.dtype().to_string(),
        }),
    }
}

/// 返回 Parameter 名称，未命名时使用稳定的占位名称。
fn parameter_name(parameter: &Parameter) -> String {
    parameter.name().unwrap_or_else(|| "<unnamed>".to_string())
}
