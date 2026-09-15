use pyo3::PyErr;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use std::fmt::Display;

/// 将底层执行、模拟或自动微分错误转换为 Python `RuntimeError`。
pub(crate) fn runtime_error(error: impl Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

/// 将用户输入、参数或线路配置错误转换为 Python `ValueError`。
pub(crate) fn value_error(error: impl Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}
