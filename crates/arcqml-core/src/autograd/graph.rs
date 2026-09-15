use super::BackwardFn;
use crate::{ArcQmlError, Result, Tensor};

use std::fmt;
use std::sync::Arc;

/// 反向传播时需要保留的父 Tensor 及其前向版本号。版本号用于检测前向结束后、反向开始前发生的原地修改，避免用错误的前向数据计算梯度。
#[derive(Clone)]
pub(crate) struct SavedTensor {
    tensor: Tensor,
    version: u64,
}

impl SavedTensor {
    /// 共享原 Storage、version 和 autograd 状态。
    pub(crate) fn new(tensor: Tensor) -> Self {
        Self {
            version: tensor.version(),
            tensor,
        }
    }
}

impl fmt::Debug for SavedTensor {
    /// 输出时不打印全部数据。
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SavedTensor")
            .field("shape", &self.tensor.shape())
            .field("dtype", &self.tensor.dtype())
            .field("version", &self.version)
            .finish()
    }
}

/// 非叶子 Tensor 在自动微分图中的一个节点。每个节点保存前向操作的父 Tensor 快照，以及把输出梯度传回父节点的反向规则。
#[derive(Clone)]
pub(crate) struct AutogradNode {
    operation_name: String,
    parents: Vec<SavedTensor>,
    backward: Arc<dyn BackwardFn>,
}

impl AutogradNode {
    /// 由前向操作的父 Tensor 和反向规则创建图节点。
    pub(crate) fn new(
        operation_name: impl Into<String>,
        parents: Vec<Tensor>,
        backward: Arc<dyn BackwardFn>,
    ) -> Self {
        Self {
            operation_name: operation_name.into(),
            parents: parents.into_iter().map(SavedTensor::new).collect(),
            backward,
        }
    }

    /// 返回图遍历所需的父 Tensor 引用副本。
    pub(crate) fn parents(&self) -> Vec<Tensor> {
        self.parents
            .iter()
            .map(|saved| saved.tensor.clone())
            .collect()
    }

    /// 确认反向规则依赖的每个前向 Tensor 都未被原地修改。
    fn check_versions(&self) -> Result<()> {
        for saved in &self.parents {
            let actual = saved.tensor.version();

            if actual != saved.version {
                return Err(ArcQmlError::AutogradError(format!(
                    "a tensor needed for backward was modified in place: expected version {}, got {}",
                    saved.version, actual
                )));
            }
        }

        Ok(())
    }

    /// 执行本节点的反向规则，并去掉不需要梯度的父节点。
    pub(crate) fn backward(&self, grad_output: &Tensor) -> Result<Vec<(Tensor, Tensor)>> {
        self.check_versions()?;

        let parents = self.parents();
        let gradients = self
            .backward
            .backward(&parents, grad_output)
            .map_err(|error| {
                ArcQmlError::AutogradError(format!(
                    "backward for operation `{}` failed: {error}",
                    self.operation_name
                ))
            })?;

        if gradients.len() != parents.len() {
            return Err(ArcQmlError::AutogradError(format!(
                "backward returned {} gradients for {} parents",
                gradients.len(),
                parents.len()
            )));
        }

        Ok(parents
            .into_iter()
            .zip(gradients)
            .filter_map(|(parent, gradient)| gradient.map(|gradient| (parent, gradient)))
            .collect())
    }
}

impl fmt::Debug for AutogradNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AutogradNode")
            .field("operation_name", &self.operation_name)
            .field("parents", &self.parents)
            .field("backward", &self.backward)
            .finish()
    }
}
