use arcqml_circuit::prelude::*;
use arcqml_core::{DType, Tensor, TensorData, no_grad};
use arcqml_observable::prelude::*;
use arcqml_sim::prelude::*;
use num_complex::Complex64;

/// 断言两个实数在测试容差内相等。
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-6,
        "actual={actual}, expected={expected}"
    );
}

/// 断言两个复数在测试容差内相等。
fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1e-6,
        "actual={actual}, expected={expected}"
    );
}

/// 创建参数为 angle 的单量子比特实振幅归一化初态。
fn one_qubit_state(angle: f64) -> Tensor {
    Tensor::new(TensorData::FlatC64 {
        data: vec![
            Complex64::new(angle.cos(), 0.0),
            Complex64::new(angle.sin(), 0.0),
        ],
        shape: vec![2],
    })
    .unwrap()
}

/// 从标量 Tensor 中读取 F64 值。
fn scalar_value(tensor: &Tensor) -> f64 {
    tensor.value().unwrap()
}

/// 计算单态线路的期望值，用于有限差分校验。
fn single_expectation(initial_angle: f64, theta: f64) -> f64 {
    let mut circuit = Circuit::new(1).unwrap();
    let parameter = circuit
        .add_parameter_tensor(Tensor::new(theta).unwrap())
        .unwrap();
    circuit.ry_param(parameter, 0usize).unwrap();
    let observable = SparsePauliOp::single(1, 0usize, Pauli::Z, 1.0).unwrap();
    let simulator =
        StateVectorSimulator::from_state_tensor(1, one_qubit_state(initial_angle)).unwrap();
    scalar_value(&simulator.run(&circuit, &observable).unwrap())
}

/// 验证单态 run 路径同时回传共享参数和初始 C64 态梯度。
#[test]
fn single_run_should_backpropagate_initial_state_and_shared_parameter() {
    let initial_angle = 0.31_f64;
    let theta_value = -0.27_f64;
    let initial_state = one_qubit_state(initial_angle);
    initial_state.set_requires_grad(true);
    let theta = Tensor::new(theta_value).unwrap();
    let mut circuit = Circuit::new(1).unwrap();
    let parameter = circuit.add_parameter_tensor(theta.clone()).unwrap();
    circuit
        .ry_param(parameter, 0usize)
        .unwrap()
        .ry_param(parameter, 0usize)
        .unwrap();
    let observable = SparsePauliOp::single(1, 0usize, Pauli::Z, 1.0).unwrap();
    let output = StateVectorSimulator::from_state_tensor(1, initial_state.clone())
        .unwrap()
        .run(&circuit, &observable)
        .unwrap();
    output.backward().unwrap();

    assert_close(
        scalar_value(&output),
        (2.0 * initial_angle + 2.0 * theta_value).cos(),
    );
    assert_close(
        scalar_value(&theta.grad().unwrap()),
        -2.0 * (2.0 * initial_angle + 2.0 * theta_value).sin(),
    );

    let gradient = initial_state.grad().unwrap();
    let values = gradient.storage().as_c64_slice().unwrap().to_vec();
    let direction = [
        Complex64::new(-initial_angle.sin(), 0.0),
        Complex64::new(initial_angle.cos(), 0.0),
    ];
    let analytic_direction = values
        .iter()
        .zip(direction)
        .map(|(gradient, direction)| (gradient.conj() * direction).re)
        .sum::<f64>();
    let epsilon = 1e-6;
    let numerical_direction = (single_expectation(initial_angle + epsilon, 2.0 * theta_value)
        - single_expectation(initial_angle - epsilon, 2.0 * theta_value))
        / (2.0 * epsilon);
    assert_close(analytic_direction, numerical_direction);
}

/// 计算包含通用门的批量加权损失，用于有限差分校验。
fn generic_batch_loss(angles: [f64; 2], theta_value: f64, phi_value: f64) -> f64 {
    let theta = Tensor::new(theta_value).unwrap();
    let phi = Tensor::new(phi_value).unwrap();
    let mut circuit = Circuit::new(2).unwrap();
    let theta_id = circuit.add_parameter_tensor(theta).unwrap();
    let phi_id = circuit.add_parameter_tensor(phi).unwrap();
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
    let observable = SparsePauliOp::single(2, 1usize, Pauli::X, 1.0).unwrap();
    let input = Tensor::new(TensorData::FlatC64 {
        data: angles
            .into_iter()
            .flat_map(|angle| {
                [
                    Complex64::new(angle.cos(), 0.0),
                    Complex64::new(angle.sin(), 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ]
            })
            .collect(),
        shape: vec![2, 4],
    })
    .unwrap();
    let output = BatchStateVectorSimulator::from_state_tensor(2, input)
        .unwrap()
        .run(&circuit, &observable)
        .unwrap();
    let storage = output.storage();
    let values = storage.as_f64_slice().unwrap();
    0.43 * values[0] - 0.29 * values[1]
}

/// 验证批量 run 路径对通用门、初始 C64 态和电路参数的梯度。
#[test]
fn batch_run_should_backpropagate_initial_state_and_generic_gate_parameters() {
    let angles = [0.19_f64, -0.34_f64];
    let theta_value = 0.23_f64;
    let phi_value = -0.41_f64;
    let input = Tensor::new(TensorData::FlatC64 {
        data: angles
            .into_iter()
            .flat_map(|angle| {
                [
                    Complex64::new(angle.cos(), 0.0),
                    Complex64::new(angle.sin(), 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ]
            })
            .collect(),
        shape: vec![2, 4],
    })
    .unwrap();
    input.set_requires_grad(true);
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
    let observable = SparsePauliOp::single(2, 1usize, Pauli::X, 1.0).unwrap();
    let weights = [0.43_f64, -0.29_f64];
    let output = BatchStateVectorSimulator::from_state_tensor(2, input.clone())
        .unwrap()
        .run(&circuit, &observable)
        .unwrap();
    output
        .backward_with_grad(Tensor::new(weights.to_vec()).unwrap())
        .unwrap();

    let epsilon = 1e-6;
    let theta_numerical = (generic_batch_loss(angles, theta_value + epsilon, phi_value)
        - generic_batch_loss(angles, theta_value - epsilon, phi_value))
        / (2.0 * epsilon);
    let phi_numerical = (generic_batch_loss(angles, theta_value, phi_value + epsilon)
        - generic_batch_loss(angles, theta_value, phi_value - epsilon))
        / (2.0 * epsilon);
    assert_close(scalar_value(&theta.grad().unwrap()), theta_numerical);
    assert_close(scalar_value(&phi.grad().unwrap()), phi_numerical);

    let values = input
        .grad()
        .unwrap()
        .storage()
        .as_c64_slice()
        .unwrap()
        .to_vec();
    let analytic_direction = angles
        .into_iter()
        .enumerate()
        .map(|(batch, angle)| {
            let offset = batch * 4;
            let direction = [
                Complex64::new(-angle.sin(), 0.0),
                Complex64::new(angle.cos(), 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ];
            values[offset..offset + 4]
                .iter()
                .zip(direction)
                .map(|(gradient, direction)| (gradient.conj() * direction).re)
                .sum::<f64>()
        })
        .sum::<f64>();
    let angle_numerical = (generic_batch_loss(
        [angles[0] + epsilon, angles[1] + epsilon],
        theta_value,
        phi_value,
    ) - generic_batch_loss(
        [angles[0] - epsilon, angles[1] - epsilon],
        theta_value,
        phi_value,
    )) / (2.0 * epsilon);
    assert_close(analytic_direction, angle_numerical);
}

/// 验证关闭自动微分记录时 run 接口返回不带反向节点的前向结果。
#[test]
fn run_should_skip_context_when_grad_is_disabled() {
    let theta = Tensor::new(0.27_f64).unwrap();
    theta.set_requires_grad(true);
    let mut circuit = Circuit::new(1).unwrap();
    let parameter = circuit.add_parameter_tensor(theta).unwrap();
    circuit.ry_param(parameter, 0usize).unwrap();
    let observable = SparsePauliOp::single(1, 0usize, Pauli::Z, 1.0).unwrap();
    let simulator = StateVectorSimulator::new(1).unwrap();
    let output = {
        let _guard = no_grad();
        simulator.run(&circuit, &observable).unwrap()
    };
    assert!(!output.requires_grad());
    assert_close(scalar_value(&output), 0.27_f64.cos());
}

/// 验证 run 路径为 F32 电路参数保留 F32 梯度类型。
#[test]
fn run_should_preserve_f32_parameter_gradient_dtype() {
    let theta = Tensor::new(0.27_f32).unwrap();
    let mut circuit = Circuit::new(1).unwrap();
    let parameter = circuit.add_parameter_tensor(theta.clone()).unwrap();
    circuit.rx_param(parameter, 0usize).unwrap();
    let observable = SparsePauliOp::single(1, 0usize, Pauli::Z, 1.0).unwrap();
    StateVectorSimulator::new(1)
        .unwrap()
        .run(&circuit, &observable)
        .unwrap()
        .backward()
        .unwrap();
    let gradient = theta.grad().unwrap();
    assert_eq!(gradient.dtype(), DType::F32);
    assert_close(
        gradient.storage().as_f32_slice().unwrap()[0] as f64,
        -0.27_f64.sin(),
    );
}

/// 验证状态 API 仍可执行演化、读取振幅并从最终态采样。
#[test]
fn state_api_should_preserve_evolution_amplitudes_and_sampling() {
    let mut circuit = Circuit::new(2).unwrap();
    circuit.h(0usize).unwrap().cnot(0usize, 1usize).unwrap();
    let mut simulator = StateVectorSimulator::new(2).unwrap();
    simulator.apply_circuit(&circuit).unwrap();
    let amplitudes = simulator.amplitudes().unwrap();
    let values = amplitudes.storage().as_c64_slice().unwrap().to_vec();
    let scale = 2.0_f64.sqrt().recip();
    assert_complex_close(values[0], Complex64::new(scale, 0.0));
    assert_complex_close(values[3], Complex64::new(scale, 0.0));
    let counts = simulator.sample_counts(1_000, Some(42)).unwrap();
    assert_eq!(counts.values().sum::<usize>(), 1_000);
    assert!(counts.keys().all(|key| key == "00" || key == "11"));
}
