use arcqml_analysis::{
    AnalysisError, bloch_vector, fidelity, marginal_probabilities, probabilities,
};
use arcqml_circuit::Circuit;
use arcqml_core::{Tensor, TensorData};
use arcqml_sim::StateVectorSimulator;
use num_complex::Complex64;

/// 构造 Bell 态模拟器以覆盖固定态分析接口。
fn bell_simulator() -> StateVectorSimulator {
    let mut circuit = Circuit::new(2).unwrap();
    circuit.h(0usize).unwrap().cnot(0usize, 1usize).unwrap();

    let mut simulator = StateVectorSimulator::new(2).unwrap();
    simulator.apply_circuit(&circuit).unwrap();
    simulator
}

/// 构造带一个可训练 RY 参数的单量子位模拟器并返回该参数。
fn ry_simulator(theta_value: f64) -> (StateVectorSimulator, Tensor) {
    let theta = Tensor::new(theta_value).unwrap();
    let mut circuit = Circuit::new(1).unwrap();
    let parameter = circuit.add_parameter_tensor(theta.clone()).unwrap();
    circuit.ry_param(parameter, 0usize).unwrap();

    let mut simulator = StateVectorSimulator::new(1).unwrap();
    simulator.apply_circuit(&circuit).unwrap();
    (simulator, theta)
}

/// 构造经 RY 和 CNOT 演化的两量子位模拟器，以验证非平凡边缘概率归约。
fn entangled_ry_simulator(theta_value: f64) -> (StateVectorSimulator, Tensor) {
    let theta = Tensor::new(theta_value).unwrap();
    let mut circuit = Circuit::new(2).unwrap();
    let parameter = circuit.add_parameter_tensor(theta.clone()).unwrap();
    circuit
        .ry_param(parameter, 0usize)
        .unwrap()
        .cnot(0usize, 1usize)
        .unwrap();

    let mut simulator = StateVectorSimulator::new(2).unwrap();
    simulator.apply_circuit(&circuit).unwrap();
    (simulator, theta)
}

/// 返回 F64 Tensor 的数据副本，便于在测试中进行数值断言。
fn f64_values(tensor: &Tensor) -> Vec<f64> {
    tensor.storage().as_f64_slice().unwrap().to_vec()
}

/// 返回 F64 标量 Tensor 的唯一数值。
fn scalar_f64(tensor: &Tensor) -> f64 {
    assert!(tensor.shape().is_empty());
    f64_values(tensor)[0]
}

/// 断言两个 F64 值在测试容差内相等。
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

/// 断言两个 F64 切片逐项在测试容差内相等。
fn assert_values_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_close(actual, expected);
    }
}

/// 验证概率和边缘概率返回 Tensor，且保留既有的小端输出顺序。
#[test]
fn analysis_should_calculate_tensor_probabilities_and_marginals() {
    let simulator = bell_simulator();
    let full_probabilities = probabilities(&simulator).unwrap();
    assert_eq!(full_probabilities.shape(), &[4]);
    assert_values_close(&f64_values(&full_probabilities), &[0.5, 0.0, 0.0, 0.5]);

    let marginal = marginal_probabilities(&simulator, &[0]).unwrap();
    assert_eq!(marginal.shape(), &[2]);
    assert_values_close(&f64_values(&marginal), &[0.5, 0.5]);

    let two_qubit_marginal = marginal_probabilities(&simulator, &[0, 1]).unwrap();
    assert_eq!(two_qubit_marginal.shape(), &[4]);
    assert_values_close(&f64_values(&two_qubit_marginal), &[0.5, 0.0, 0.0, 0.5]);
}

/// 验证保真度返回可微 F64 标量 Tensor，Bloch 向量仍为数值分析接口。
#[test]
fn analysis_should_calculate_tensor_fidelity_and_bloch_vectors() {
    let bell = bell_simulator();
    let zero = StateVectorSimulator::new(2).unwrap();
    assert_close(scalar_f64(&fidelity(&bell, &bell).unwrap()), 1.0);
    assert_close(scalar_f64(&fidelity(&bell, &zero).unwrap()), 0.5);

    let mut plus_circuit = Circuit::new(1).unwrap();
    plus_circuit.h(0usize).unwrap();
    let mut plus = StateVectorSimulator::new(1).unwrap();
    plus.apply_circuit(&plus_circuit).unwrap();
    let bloch = bloch_vector(&plus, 0).unwrap();
    assert_close(bloch.x, 1.0);
    assert_close(bloch.y, 0.0);
    assert_close(bloch.z, 0.0);
}

/// 验证振幅 Tensor 可接收复数上游梯度并反传到量子门参数。
#[test]
fn amplitudes_should_backpropagate_to_gate_parameters() {
    let theta_value = 0.4;
    let (simulator, theta) = ry_simulator(theta_value);
    let amplitudes = simulator.amplitudes().unwrap();
    assert!(amplitudes.requires_grad());
    amplitudes
        .backward_with_grad(
            Tensor::new(TensorData::FlatC64 {
                data: vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                shape: vec![2],
            })
            .unwrap(),
        )
        .unwrap();

    assert_close(
        scalar_f64(&theta.grad().unwrap()),
        0.5 * (theta_value / 2.0).cos(),
    );
}

/// 验证概率和分段边缘归约均可自动反传到量子门参数。
#[test]
fn probabilities_and_marginals_should_backpropagate_to_gate_parameters() {
    let theta_value = 0.4;
    let (simulator, theta) = ry_simulator(theta_value);
    let distribution = probabilities(&simulator).unwrap();
    distribution
        .backward_with_grad(Tensor::new(vec![0.0_f64, 1.0]).unwrap())
        .unwrap();
    assert_close(scalar_f64(&theta.grad().unwrap()), 0.5 * theta_value.sin());

    let (simulator, theta) = entangled_ry_simulator(theta_value);
    let marginal = marginal_probabilities(&simulator, &[1]).unwrap();
    marginal
        .backward_with_grad(Tensor::new(vec![0.0_f64, 1.0]).unwrap())
        .unwrap();
    assert_close(scalar_f64(&theta.grad().unwrap()), 0.5 * theta_value.sin());
}

/// 验证保真度可自动反传到参与比较的量子态参数。
#[test]
fn fidelity_should_backpropagate_to_gate_parameters() {
    let theta_value = 0.2;
    let target_angle: f64 = 0.8;
    let (simulator, theta) = ry_simulator(theta_value);
    let target = StateVectorSimulator::from_state_tensor(
        1,
        Tensor::new(TensorData::FlatC64 {
            data: vec![
                Complex64::new((target_angle / 2.0).cos(), 0.0),
                Complex64::new((target_angle / 2.0).sin(), 0.0),
            ],
            shape: vec![2],
        })
        .unwrap(),
    )
    .unwrap();

    let value = fidelity(&simulator, &target).unwrap();
    assert_close(
        scalar_f64(&value),
        ((theta_value - target_angle) / 2.0).cos().powi(2),
    );
    value.backward().unwrap();
    assert_close(
        scalar_f64(&theta.grad().unwrap()),
        -0.5 * (theta_value - target_angle).sin(),
    );
}

/// 验证边缘概率接口会拒绝空、重复和越界的量子位选择。
#[test]
fn analysis_should_validate_marginal_qubit_selection() {
    let simulator = bell_simulator();
    assert!(matches!(
        marginal_probabilities(&simulator, &[]),
        Err(AnalysisError::EmptyQubitSelectionError)
    ));
    assert!(matches!(
        marginal_probabilities(&simulator, &[0, 0]),
        Err(AnalysisError::DuplicateQubitError { index: 0 })
    ));
    assert!(matches!(
        marginal_probabilities(&simulator, &[2]),
        Err(AnalysisError::QubitOutOfRangeError {
            index: 2,
            num_qubits: 2,
        })
    ));
}
