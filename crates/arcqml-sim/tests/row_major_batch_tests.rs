use arcqml_circuit::prelude::*;
use arcqml_core::{Tensor, TensorData, no_grad};
use arcqml_observable::prelude::*;
use arcqml_sim::prelude::*;
use num_complex::Complex64;

const TOLERANCE: f64 = 1.0e-10;

/// 断言两个实数在测试容差内相等。
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < TOLERANCE,
        "actual={actual}, expected={expected}"
    );
}

/// 断言两段复数振幅在测试容差内相等。
fn assert_complex_slice_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (*actual - *expected).norm() < TOLERANCE,
            "actual={actual}, expected={expected}"
        );
    }
}

/// 创建三个两量子比特的归一化行主序初态。
fn initial_state() -> Tensor {
    Tensor::new(TensorData::FlatC64 {
        data: vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.8, 0.0),
            Complex64::new(0.0, 0.6),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        shape: vec![3, 4],
    })
    .unwrap()
}

/// 创建包含受控旋转和交换门的可训练两量子比特线路。
fn circuit(theta_value: f64, phi_value: f64) -> (Circuit, Tensor, Tensor) {
    let theta = Tensor::new(theta_value).unwrap();
    let phi = Tensor::new(phi_value).unwrap();
    let mut circuit = Circuit::new(2).unwrap();
    let theta_id = circuit.add_parameter_tensor(theta.clone()).unwrap();
    let phi_id = circuit.add_parameter_tensor(phi.clone()).unwrap();
    circuit
        .h(0usize)
        .unwrap()
        .ry_param(theta_id, 0usize)
        .unwrap()
        .cry_param(phi_id, 0usize, 1usize)
        .unwrap()
        .swap(0usize, 1usize)
        .unwrap()
        .ry_param(theta_id, 1usize)
        .unwrap();
    (circuit, theta, phi)
}

/// 验证唯一的 batch 伴随路径与逐门状态 API 的前向及梯度一致。
#[test]
fn batch_adjoint_should_match_gatewise_for_forward_and_backward() {
    let observable = SparsePauliOp::single(2, 1usize, Pauli::X, 1.0).unwrap();
    let weights = Tensor::new(vec![0.43_f64, -0.29, 0.17]).unwrap();

    let adjoint_input = initial_state();
    adjoint_input.set_requires_grad(true);
    let (adjoint_circuit, adjoint_theta, adjoint_phi) = circuit(0.23, -0.41);
    let adjoint_output = BatchStateVectorSimulator::from_state_tensor(2, adjoint_input.clone())
        .unwrap()
        .run(&adjoint_circuit, &observable)
        .unwrap();
    adjoint_output.backward_with_grad(weights.clone()).unwrap();

    let gatewise_input = initial_state();
    gatewise_input.set_requires_grad(true);
    let (gatewise_circuit, gatewise_theta, gatewise_phi) = circuit(0.23, -0.41);
    let mut gatewise_simulator =
        BatchStateVectorSimulator::from_state_tensor(2, gatewise_input.clone()).unwrap();
    gatewise_simulator.apply_circuit(&gatewise_circuit).unwrap();
    let gatewise_output = gatewise_simulator
        .expectation_observable(&observable)
        .unwrap();
    gatewise_output.backward_with_grad(weights).unwrap();

    let adjoint_storage = adjoint_output.storage();
    let gatewise_storage = gatewise_output.storage();
    for (adjoint_value, gatewise_value) in adjoint_storage
        .as_f64_slice()
        .unwrap()
        .iter()
        .zip(gatewise_storage.as_f64_slice().unwrap())
    {
        assert_close(*adjoint_value, *gatewise_value);
    }
    assert_close(
        adjoint_theta
            .grad()
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap()[0],
        gatewise_theta
            .grad()
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap()[0],
    );
    assert_close(
        adjoint_phi
            .grad()
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap()[0],
        gatewise_phi
            .grad()
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap()[0],
    );
    assert_complex_slice_close(
        adjoint_input
            .grad()
            .unwrap()
            .storage()
            .as_c64_slice()
            .unwrap(),
        gatewise_input
            .grad()
            .unwrap()
            .storage()
            .as_c64_slice()
            .unwrap(),
    );
}

/// 验证关闭自动微分记录时，batch 伴随路径与逐门状态 API 数值一致。
#[test]
fn batch_adjoint_should_match_gatewise_without_grad() {
    let (circuit, _, _) = circuit(0.37, -0.19);
    let observable = SparsePauliOp::single(2, 0usize, Pauli::Z, 1.0).unwrap();
    let input = initial_state();
    let _guard = no_grad();
    let adjoint_output = BatchStateVectorSimulator::from_state_tensor(2, input.clone())
        .unwrap()
        .run(&circuit, &observable)
        .unwrap();
    let mut gatewise_simulator = BatchStateVectorSimulator::from_state_tensor(2, input).unwrap();
    gatewise_simulator.apply_circuit(&circuit).unwrap();
    let gatewise_output = gatewise_simulator
        .expectation_observable(&observable)
        .unwrap();
    let adjoint_values = adjoint_output.storage();
    let gatewise_values = gatewise_output.storage();
    for (adjoint_value, gatewise_value) in adjoint_values
        .as_f64_slice()
        .unwrap()
        .iter()
        .zip(gatewise_values.as_f64_slice().unwrap())
    {
        assert_close(*adjoint_value, *gatewise_value);
    }
}
