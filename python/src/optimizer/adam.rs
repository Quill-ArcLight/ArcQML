use crate::circuit::PyCircuit;
use crate::error::value_error;
use arcqml_optim::Adam;
use pyo3::prelude::*;

/// `Adam` 优化器。
#[pyclass(name = "Adam")]
pub struct PyAdam {
    inner: Adam,
}

#[pymethods]
impl PyAdam {
    /// 使用标准 `beta1`、`beta2` 与 `epsilon` 默认值创建 `Adam` 优化器。
    #[new]
    #[pyo3(signature = (learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0))]
    fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Adam::new(learning_rate, beta1, beta2, epsilon, weight_decay)
                .map_err(value_error)?,
        })
    }

    #[getter]
    /// 返回当前 `Adam` 优化器的学习率 `learning_rate`。
    fn learning_rate(&self) -> f64 {
        self.inner.learning_rate()
    }

    #[getter]
    /// 返回 `Adam` 已执行参数更新的步数 `step_count`。
    fn step_count(&self) -> u64 {
        self.inner.step_count()
    }

    /// 使用当前梯度更新线路全部可训练参数，并返回实际更新的参数数量。
    fn step(&mut self, circuit: PyRef<'_, PyCircuit>) -> PyResult<usize> {
        Ok(self
            .inner
            .step(circuit.inner.parameters())
            .map_err(value_error)?
            .updated())
    }

    /// 清空指定线路全部参数的梯度；通常在 `step` 后调用。
    fn zero_grad(&self, circuit: PyRef<'_, PyCircuit>) {
        self.inner.zero_grad(circuit.inner.parameters());
    }
}
