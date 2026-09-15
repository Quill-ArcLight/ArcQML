use crate::{ArcQmlError, BackwardFn, Result, Tensor};
use std::sync::Arc;

/// 用户定义的可微原子操作。
///
/// 当前向包含 ArcQML 算子无法表达的步骤，或需要为整体表达式指定专用反向时，
/// 可以实现此 trait。普通可微算子的组合无需实现它。前向在
/// [`no_grad`](crate::no_grad) 作用域中执行；反向必须按输入顺序返回同样数量的
/// 梯度槽，并以 `None` 表示不需要或不存在的输入梯度。
pub trait CustomOp: std::fmt::Debug + Send + Sync + 'static {
    /// 保存反向传播所需的前向上下文。
    type Context: std::fmt::Debug + Send + Sync + 'static;
    /// 返回用于诊断和计算图展示的稳定操作名称。
    fn name(&self) -> &'static str;
    /// 执行前向计算，返回一个不带计算图的输出及反向传播所需上下文。
    ///
    /// # Errors
    ///
    /// 错误条件由具体自定义操作定义，通常用于拒绝形状、dtype 或设备不受支持的输入。
    fn forward(&self, inputs: &[Tensor]) -> Result<(Tensor, Self::Context)>;
    /// 根据保存的上下文和上游梯度，按输入顺序计算局部向量—雅可比积。
    ///
    /// # Errors
    ///
    /// 错误条件由具体实现定义，例如上游梯度不兼容或保存的上下文无效。
    fn backward(
        &self,
        context: &Self::Context,
        grad_output: &Tensor,
    ) -> Result<Vec<Option<Tensor>>>;
}

/// 执行用户定义的可微原子操作，并将其结果接入自动微分图。
///
/// 前向在临时禁用梯度记录的作用域中执行。若至少一个输入需要梯度，返回值会以
/// 自定义操作为单个计算图节点；否则返回不记录图的叶子 Tensor。
///
/// # Errors
///
/// 当自定义前向返回错误、返回的 Tensor 自身仍需要梯度，或输出存储与元数据
/// 不一致时返回错误。自定义反向中的错误会在之后调用 `backward` 时返回。
pub fn apply_custom_op<O: CustomOp>(operation: O, inputs: &[Tensor]) -> Result<Tensor> {
    let operation_name = operation.name();
    let (output, context) = {
        let _no_grad = crate::no_grad();
        operation.forward(inputs)?
    };

    if output.requires_grad() {
        return Err(ArcQmlError::AutogradError(format!(
            "custom operation `{operation_name}` forward must not return a tensor that requires gradients"
        )));
    }

    Tensor::from_operation_named(
        operation_name,
        output.storage().clone(),
        output.meta().clone(),
        inputs.to_vec(),
        Arc::new(CustomOpBackward { operation, context }),
    )
}

/// 将用户定义操作及其前向上下文适配为内部反向规则。
#[derive(Debug)]
struct CustomOpBackward<O: CustomOp> {
    operation: O,
    context: O::Context,
}

impl<O: CustomOp> BackwardFn for CustomOpBackward<O> {
    /// 委托给用户提供的局部 VJP 实现。
    fn backward(&self, _parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        self.operation.backward(&self.context, grad_output)
    }
}
