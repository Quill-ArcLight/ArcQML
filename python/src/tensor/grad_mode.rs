use arcqml_core::NoGradGuard;
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Python `no_grad()` 的无梯度上下文管理器。
#[pyclass(unsendable)]
pub(crate) struct PyNoGrad {
    guard: Option<NoGradGuard>,
}

#[pymethods]
impl PyNoGrad {
    /// 进入 `no_grad` 上下文并返回自身。
    fn __enter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    /// 退出上下文时释放 Rust 无梯度 guard，并且不吞掉 Python 异常。
    fn __exit__(
        &mut self,
        _exception_type: Option<&Bound<'_, PyAny>>,
        _exception_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> bool {
        self.guard.take();
        false
    }
}

/// 创建仅作用于当前 Python 调用线程的 `no_grad()` 上下文。
#[pyfunction]
pub(crate) fn no_grad() -> PyNoGrad {
    PyNoGrad {
        guard: Some(arcqml_core::no_grad()),
    }
}
