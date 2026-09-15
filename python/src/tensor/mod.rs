mod convert;
mod grad_mode;

pub(crate) use grad_mode::{PyNoGrad, no_grad};

use crate::error::runtime_error;
use arcqml_core::{DType, Storage, Tensor};
use convert::tensor_from_python;
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyComplex, PyFloat, PyInt};

/// Python 侧通用可微 `Tensor`。
#[pyclass(name = "Tensor", skip_from_py_object)]
#[derive(Clone)]
pub struct PyTensor {
    pub(crate) inner: Tensor,
}

impl PyTensor {
    /// 根据已有 Rust `Tensor` 构造 Python 包装对象。
    pub(crate) fn from_inner(inner: Tensor) -> Self {
        Self { inner }
    }

    /// 从 Python 输入构造新的 Rust `Tensor`；已有 `Tensor` 会先 `detach`，绝不改写源对象。
    pub(crate) fn from_python(value: &Bound<'_, PyAny>, requires_grad: bool) -> PyResult<Self> {
        let tensor = if let Ok(source) = value.extract::<PyRef<'_, PyTensor>>() {
            source.inner.detach()
        } else {
            tensor_from_python(value)?
        };
        tensor
            .try_set_requires_grad(requires_grad)
            .map_err(runtime_error)?;
        Ok(Self::from_inner(tensor))
    }
}

#[pymethods]
impl PyTensor {
    /// 从 Python 数值、列表或 C 连续 `numpy.ndarray` 创建 `Tensor`。
    #[new]
    #[pyo3(signature = (data, *, requires_grad=false))]
    fn new(data: &Bound<'_, PyAny>, requires_grad: bool) -> PyResult<Self> {
        Self::from_python(data, requires_grad)
    }

    #[getter]
    /// 返回 `Tensor` 的形状 `shape`。
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    /// 返回 `Tensor` 的稳定 Python 数据类型名称 `dtype`。
    fn dtype(&self) -> String {
        dtype_name(self.inner.dtype()).to_owned()
    }

    #[getter]
    /// 返回 `Tensor` 是否需要梯度 `requires_grad`。
    fn requires_grad(&self) -> bool {
        self.inner.requires_grad()
    }

    #[getter]
    /// 返回 `Tensor` 是否为自动微分图叶子节点 `is_leaf`。
    fn is_leaf(&self) -> bool {
        self.inner.is_leaf()
    }

    /// 设置叶子 `Tensor` 的 `requires_grad` 标记。
    fn set_requires_grad(&self, requires_grad: bool) -> PyResult<()> {
        self.inner
            .try_set_requires_grad(requires_grad)
            .map_err(runtime_error)
    }

    /// 从标量 `loss` 开始 `backward()`；非标量 `Tensor` 必须提供同形状上游梯度。
    #[pyo3(signature = (gradient=None))]
    fn backward(&self, gradient: Option<PyRef<'_, PyTensor>>) -> PyResult<()> {
        match gradient {
            Some(gradient) => self
                .inner
                .backward_with_grad(gradient.inner.clone())
                .map_err(runtime_error),
            None => self.inner.backward().map_err(runtime_error),
        }
    }

    /// 返回与当前 `Tensor` 共享数值、但不连接自动微分图的新叶子 `Tensor`。
    fn detach(&self) -> Self {
        Self::from_inner(self.inner.detach())
    }

    /// 返回当前累计梯度；尚未产生梯度时返回 `None`。
    fn grad(&self) -> Option<Self> {
        self.inner.grad().map(Self::from_inner)
    }

    /// 将 `Tensor` 的当前数值复制为独立的 `numpy.ndarray`。
    fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let shape = self.inner.shape().to_vec();
        let storage = self.inner.storage();
        let array = match &*storage {
            Storage::F32(values) => PyArray1::from_vec(py, values.clone()).into_any(),
            Storage::F64(values) => PyArray1::from_vec(py, values.clone()).into_any(),
            Storage::C64(values) => PyArray1::from_vec(py, values.clone()).into_any(),
            Storage::I64(values) => PyArray1::from_vec(py, values.clone()).into_any(),
            Storage::Bool(values) => PyArray1::from_vec(py, values.clone()).into_any(),
        };
        array
            .call_method1("reshape", (shape,))
            .map(|value| value.unbind())
    }

    /// 将只含一个元素的 `Tensor` 转换为 Python 标量 `item()`。
    fn item(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.inner.numel() != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "item 仅支持单元素 Tensor，实际 shape 为 {:?}",
                self.inner.shape()
            )));
        }
        let storage = self.inner.storage();
        let value = match &*storage {
            Storage::F32(values) => PyFloat::new(py, values[0] as f64).into_any().unbind(),
            Storage::F64(values) => PyFloat::new(py, values[0]).into_any().unbind(),
            Storage::C64(values) => PyComplex::from_doubles(py, values[0].re, values[0].im)
                .into_any()
                .unbind(),
            Storage::I64(values) => PyInt::new(py, values[0]).into_any().unbind(),
            Storage::Bool(values) => PyBool::new(py, values[0]).to_owned().into_any().unbind(),
        };
        Ok(value)
    }

    /// 返回便于调试的 `Tensor` 描述。
    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, requires_grad={})",
            self.inner.shape(),
            dtype_name(self.inner.dtype()),
            self.inner.requires_grad()
        )
    }
}

/// 从 Python 数据创建新的通用 `Tensor`。
#[pyfunction(signature = (data, *, requires_grad=false))]
pub(crate) fn tensor(data: &Bound<'_, PyAny>, requires_grad: bool) -> PyResult<PyTensor> {
    PyTensor::from_python(data, requires_grad)
}

/// 返回数据类型的稳定 Python `dtype` 名称。
fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float32",
        DType::F64 => "float64",
        DType::C64 => "complex128",
        DType::I64 => "int64",
        DType::Bool => "bool",
    }
}
