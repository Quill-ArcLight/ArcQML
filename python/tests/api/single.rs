use crate::common::native_module;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods};
use std::collections::BTreeMap;

const TOLERANCE: f64 = 1e-12;

/// 验证单态训练遵循 run、loss.backward、optimizer.step 的通用流程。
#[test]
fn runs_single_state_training_with_generic_tensor_api() {
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
        let simulator = module
            .getattr("StateVectorSimulator")
            .unwrap()
            .call1((1_usize,))
            .unwrap();
        let optimizer = module.getattr("Adam").unwrap().call1((0.1_f64,)).unwrap();
        let target = module.call_method1("tensor", (0.0_f64,)).unwrap();

        optimizer.call_method1("zero_grad", (&circuit,)).unwrap();
        let prediction = simulator
            .call_method1("run", (&circuit, &observable))
            .unwrap();
        let loss = module
            .call_method1("mse_loss", (&prediction, &target))
            .unwrap();
        let prediction_value: f64 = prediction.call_method0("item").unwrap().extract().unwrap();
        let loss_value: f64 = loss.call_method0("item").unwrap().extract().unwrap();
        loss.call_method0("backward").unwrap();

        let gradients: BTreeMap<String, f64> = circuit
            .call_method0("gradients")
            .unwrap()
            .extract()
            .unwrap();
        let gradient = gradients["ry_q0_theta_0"];
        let values_before: BTreeMap<String, f64> = circuit
            .call_method0("parameter_values")
            .unwrap()
            .extract()
            .unwrap();
        let updated: usize = optimizer
            .call_method1("step", (&circuit,))
            .unwrap()
            .extract()
            .unwrap();
        let values_after: BTreeMap<String, f64> = circuit
            .call_method0("parameter_values")
            .unwrap()
            .extract()
            .unwrap();
        optimizer.call_method1("zero_grad", (&circuit,)).unwrap();

        assert!((prediction_value - 0.5).abs() < TOLERANCE);
        assert!((loss_value - 0.125).abs() < TOLERANCE);
        assert!((gradient + 3.0_f64.sqrt() / 4.0).abs() < TOLERANCE);
        assert_eq!(updated, 1);
        assert_ne!(values_before, values_after);
    });
}

/// 验证 no_grad 上下文让单态 run 只执行前向且不连接自动微分图。
#[test]
fn runs_single_state_evaluation_without_autograd_graph() {
    Python::initialize();
    Python::attach(|py| {
        let module_handle = native_module(py);
        let module = module_handle.bind(py);
        let circuit = module
            .getattr("Circuit")
            .unwrap()
            .call1((1_usize,))
            .unwrap();
        circuit.call_method1("ry", (0.2_f64, 0_usize)).unwrap();
        let observable = module
            .getattr("PauliSum")
            .unwrap()
            .call_method1("z", (1_usize, 0_usize))
            .unwrap();
        let simulator = module
            .getattr("StateVectorSimulator")
            .unwrap()
            .call1((1_usize,))
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

        assert!(!requires_grad);
    });
}

/// 验证单态状态演化、分析和抽样接口均以 Tensor 形式提供数值结果。
#[test]
fn exposes_single_state_and_analysis_results_as_tensors() {
    Python::initialize();
    Python::attach(|py| {
        let module_handle = native_module(py);
        let module = module_handle.bind(py);
        let circuit = module
            .getattr("Circuit")
            .unwrap()
            .call1((2_usize,))
            .unwrap();
        circuit.call_method1("h", (0_usize,)).unwrap();
        circuit.call_method1("cnot", (0_usize, 1_usize)).unwrap();
        let simulator = module
            .getattr("StateVectorSimulator")
            .unwrap()
            .call1((2_usize,))
            .unwrap();
        simulator
            .call_method1("apply_circuit", (&circuit,))
            .unwrap();

        let amplitudes = simulator.call_method0("amplitudes").unwrap();
        let probabilities = module
            .getattr("analysis")
            .unwrap()
            .call_method1("probabilities", (&simulator,))
            .unwrap();
        let amplitude_array = amplitudes.call_method0("numpy").unwrap();
        let probability_array = probabilities.call_method0("numpy").unwrap();
        let amplitude_values: Vec<num_complex::Complex64> = amplitude_array.extract().unwrap();
        let probability_values: Vec<f64> = probability_array.extract().unwrap();
        let kwargs = PyDict::new(py);
        kwargs.set_item("seed", 42_u64).unwrap();
        let counts: BTreeMap<String, usize> = simulator
            .call_method("sample_counts", (1_000_usize,), Some(&kwargs))
            .unwrap()
            .extract()
            .unwrap();

        assert!((amplitude_values[0].re - 2.0_f64.sqrt().recip()).abs() < TOLERANCE);
        assert!((amplitude_values[3].re - 2.0_f64.sqrt().recip()).abs() < TOLERANCE);
        assert!((probability_values[0] - 0.5).abs() < TOLERANCE);
        assert!((probability_values[3] - 0.5).abs() < TOLERANCE);
        assert_eq!(counts.values().sum::<usize>(), 1_000);
        assert!(
            counts
                .keys()
                .all(|bitstring| bitstring == "00" || bitstring == "11")
        );
    });
}
