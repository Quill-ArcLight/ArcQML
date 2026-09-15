use arcqml::arcqml;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;

/// 创建供绑定集成测试使用的原生模块实例。
pub(crate) fn native_module(py: Python<'_>) -> Py<PyModule> {
    wrap_pymodule!(arcqml)(py)
}
