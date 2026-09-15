use crate::common::native_module;
use num_complex::Complex64;
use numpy::{PyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use std::collections::BTreeMap;

const TOLERANCE: f64 = 1e-12;

/// 创建两个计算基初态 |0> 与 |1> 组成的连续 complex128 batch。
fn computational_basis_batch(py: Python<'_>) -> Bound<'_, PyArray2<Complex64>> {
    PyArray::from_vec2(
        py,
        &[
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ],
    )
    .unwrap()
}

/// 验证行主序 batch 训练遵循 run、loss.backward、optimizer.step 的通用流程。
#[test]
fn runs_batch_training_with_generic_tensor_api() {
    Python::initialize();
    Python::attach(|py| {
        let module_handle = native_module(py);
        let module = module_handle.bind(py);
        let circuit = module
            .getattr("Circuit")
            .unwrap()
            .call1((1_usize,))
            .unwrap();
        circuit
            .call_method1("ry", (std::f64::consts::PI / 3.0, 0_usize))
            .unwrap();
        let observable = module
            .getattr("PauliSum")
            .unwrap()
            .call_method1("z", (1_usize, 0_usize))
            .unwrap();
        let states = computational_basis_batch(py);
        let simulator = module
            .getattr("BatchStateVectorSimulator")
            .unwrap()
            .call_method1("from_amplitudes", (1_usize, &states))
            .unwrap();
        let targets = module
            .call_method1("tensor", (vec![1.0_f64, 0.0],))
            .unwrap();
        let optimizer = module.getattr("Adam").unwrap().call1((0.1_f64,)).unwrap();

        optimizer.call_method1("zero_grad", (&circuit,)).unwrap();
        let logits = simulator
            .call_method1("run", (&circuit, &observable))
            .unwrap();
        let loss = module
            .call_method1("binary_cross_entropy_with_logits", (&logits, &targets))
            .unwrap();
        let loss_value: f64 = loss.call_method0("item").unwrap().extract().unwrap();
        loss.call_method0("backward").unwrap();
        let gradients: BTreeMap<String, f64> = circuit
            .call_method0("gradients")
            .unwrap()
            .extract()
            .unwrap();
        let updated: usize = optimizer
            .call_method1("step", (&circuit,))
            .unwrap()
            .extract()
            .unwrap();

        assert!((loss_value - (1.0_f64 + (-0.5_f64).exp()).ln()).abs() < TOLERANCE);
        assert!(gradients["ry_q0_theta_0"].is_finite());
        assert_eq!(updated, 1);
    });
}

/// 验证 no_grad 上下文中的 batch run 不会改变已有线路梯度。
#[test]
fn evaluates_batch_without_creating_or_mutating_gradients() {
    Python::initialize();
    Python::attach(|py| {
        let module_handle = native_module(py);
        let module = module_handle.bind(py);
        let circuit = module
            .getattr("Circuit")
            .unwrap()
            .call1((1_usize,))
            .unwrap();
        circuit.call_method1("ry", (0.37_f64, 0_usize)).unwrap();
        let observable = module
            .getattr("PauliSum")
            .unwrap()
            .call_method1("z", (1_usize, 0_usize))
            .unwrap();
        let states = computational_basis_batch(py);
        let simulator = module
            .getattr("BatchStateVectorSimulator")
            .unwrap()
            .call_method1("from_amplitudes", (1_usize, &states))
            .unwrap();
        let targets = module
            .call_method1("tensor", (vec![0.0_f64, 0.0],))
            .unwrap();
        let logits = simulator
            .call_method1("run", (&circuit, &observable))
            .unwrap();
        let loss = module
            .call_method1("mse_loss", (&logits, &targets))
            .unwrap();
        loss.call_method0("backward").unwrap();
        let gradients_before: BTreeMap<String, f64> = circuit
            .call_method0("gradients")
            .unwrap()
            .extract()
            .unwrap();
        let context = module.call_method0("no_grad").unwrap();
        let none = py.None();

        context.call_method0("__enter__").unwrap();
        let output = simulator
            .call_method1("run", (&circuit, &observable))
            .unwrap();
        context
            .call_method1("__exit__", (&none, &none, &none))
            .unwrap();
        let requires_grad: bool = output.getattr("requires_grad").unwrap().extract().unwrap();
        let gradients_after: BTreeMap<String, f64> = circuit
            .call_method0("gradients")
            .unwrap()
            .extract()
            .unwrap();

        assert!(!requires_grad);
        assert_eq!(gradients_after, gradients_before);
    });
}
