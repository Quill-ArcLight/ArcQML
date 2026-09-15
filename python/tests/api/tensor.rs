use crate::common::native_module;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};

/// 验证通用 Tensor 能从 Python 数值列表创建并保留基本元数据。
#[test]
fn creates_tensor_from_python_values() {
    Python::initialize();
    Python::attach(|py| {
        let module_handle = native_module(py);
        let module = module_handle.bind(py);
        let value = module
            .call_method1("tensor", (vec![1.0_f64, 2.0, 3.0],))
            .unwrap();

        let shape: Vec<usize> = value.getattr("shape").unwrap().extract().unwrap();
        let dtype: String = value.getattr("dtype").unwrap().extract().unwrap();
        let requires_grad: bool = value.getattr("requires_grad").unwrap().extract().unwrap();

        assert_eq!(shape, vec![3]);
        assert_eq!(dtype, "float64");
        assert!(!requires_grad);
    });
}

/// 验证 Tensor 的 detach 会断开自动微分图并保留数值。
#[test]
fn detaches_tensor_from_autograd_graph() {
    Python::initialize();
    Python::attach(|py| {
        let module_handle = native_module(py);
        let module = module_handle.bind(py);
        let source = module
            .call_method(
                "tensor",
                (0.5_f64,),
                Some(&{
                    let kwargs = pyo3::types::PyDict::new(py);
                    kwargs.set_item("requires_grad", true).unwrap();
                    kwargs
                }),
            )
            .unwrap();
        let detached = source.call_method0("detach").unwrap();

        let source_requires_grad: bool =
            source.getattr("requires_grad").unwrap().extract().unwrap();
        let detached_requires_grad: bool = detached
            .getattr("requires_grad")
            .unwrap()
            .extract()
            .unwrap();
        let scalar: f64 = detached.call_method0("item").unwrap().extract().unwrap();

        assert!(source_requires_grad);
        assert!(!detached_requires_grad);
        assert!((scalar - 0.5).abs() < 1e-12);
    });
}

/// 验证用 `tensor(已有 Tensor)` 创建新对象不会改写源对象的梯度标记。
#[test]
fn constructs_detached_tensor_without_mutating_source() {
    Python::initialize();
    Python::attach(|py| {
        let module_handle = native_module(py);
        let module = module_handle.bind(py);
        let kwargs = PyDict::new(py);
        kwargs.set_item("requires_grad", true).unwrap();
        let source = module
            .call_method("tensor", (0.5_f64,), Some(&kwargs))
            .unwrap();
        let copied = module.call_method1("tensor", (&source,)).unwrap();

        let source_requires_grad: bool =
            source.getattr("requires_grad").unwrap().extract().unwrap();
        let copied_requires_grad: bool =
            copied.getattr("requires_grad").unwrap().extract().unwrap();
        let copied_is_leaf: bool = copied.getattr("is_leaf").unwrap().extract().unwrap();

        assert!(source_requires_grad);
        assert!(!copied_requires_grad);
        assert!(copied_is_leaf);
    });
}
