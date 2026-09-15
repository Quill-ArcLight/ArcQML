use arcqml_circuit::prelude::*;
use arcqml_core::Tensor;
use arcqml_unitary::prelude::*;
use num_complex::Complex64;

/// 断言两个浮点数在数值容差内相等。
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-8,
        "actual={actual:?}, expected={expected:?}"
    );
}

/// 构造由固定 RY 门定义的单量子比特目标酉矩阵。
fn ry_target(theta: f64) -> DenseUnitary {
    let mut circuit = Circuit::new(1).unwrap();
    circuit.ry_fixed(theta, 0usize).unwrap();
    unitary_from_circuit(&circuit).unwrap()
}

/// 构造由固定 RY 和 RZ 门定义的单量子比特目标酉矩阵。
fn two_gate_target(ry: f64, rz: f64) -> DenseUnitary {
    let mut circuit = Circuit::new(1).unwrap();
    circuit.ry_fixed(ry, 0usize).unwrap();
    circuit.rz_fixed(rz, 0usize).unwrap();
    unitary_from_circuit(&circuit).unwrap()
}

/// 验证只相差全局相位的酉矩阵具有单位保真度。
#[test]
fn unitary_loss_ignores_global_phase() {
    let phase = Complex64::from_polar(1.0, 0.37);
    let target = DenseUnitary::from_row_major(
        vec![
            phase,
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            phase,
        ],
        2,
    )
    .unwrap();
    let circuit = Circuit::new(1).unwrap();

    assert_close(unitary_fidelity(&target, &circuit).unwrap(), 1.0);
    assert_close(unitary_loss_value(&target, &circuit).unwrap(), 0.0);
}

/// 验证解析梯度与中心差分梯度一致，并写入线路参数。
#[test]
fn unitary_fit_writes_an_analytic_gradient() {
    let target = ry_target(0.8);
    let mut circuit = Circuit::new(1).unwrap();
    circuit.ry(0.2, 0usize).unwrap();
    let id = ParameterId::new(0);

    let loss = unitary_loss(&target, &circuit).unwrap();
    loss.backward().unwrap();
    let stored = circuit
        .parameter(id)
        .unwrap()
        .grad()
        .unwrap()
        .storage()
        .as_f64_slice()
        .unwrap()[0];

    let epsilon = 1e-6;
    circuit
        .parameter(id)
        .unwrap()
        .set_tensor(Tensor::new(0.2 + epsilon).unwrap());
    let plus = unitary_loss_value(&target, &circuit).unwrap();
    circuit
        .parameter(id)
        .unwrap()
        .set_tensor(Tensor::new(0.2 - epsilon).unwrap());
    let minus = unitary_loss_value(&target, &circuit).unwrap();
    let numerical = (plus - minus) / (2.0 * epsilon);

    assert_close(stored, numerical);
}

/// 验证逆门恢复的前向状态可为多个参数门计算正确梯度。
#[test]
fn unitary_fit_restores_forward_state_for_multiple_gates() {
    let target = two_gate_target(0.8, -0.3);
    let mut circuit = Circuit::new(1).unwrap();
    circuit.ry(0.2, 0usize).unwrap();
    circuit.rz(0.1, 0usize).unwrap();

    let loss = unitary_loss(&target, &circuit).unwrap();
    loss.backward().unwrap();
    let gradients = circuit
        .parameters()
        .iter()
        .map(|parameter| parameter.grad().unwrap().storage().as_f64_slice().unwrap()[0])
        .collect::<Vec<_>>();
    let epsilon = 1e-6;
    for (index, value) in [0.2, 0.1].into_iter().enumerate() {
        let id = ParameterId::new(index);
        circuit
            .parameter(id)
            .unwrap()
            .set_tensor(Tensor::new(value + epsilon).unwrap());
        let plus = unitary_loss_value(&target, &circuit).unwrap();
        circuit
            .parameter(id)
            .unwrap()
            .set_tensor(Tensor::new(value - epsilon).unwrap());
        let minus = unitary_loss_value(&target, &circuit).unwrap();
        let numerical = (plus - minus) / (2.0 * epsilon);
        assert_close(gradients[index], numerical);
        circuit
            .parameter(id)
            .unwrap()
            .set_tensor(Tensor::new(value).unwrap());
    }
}

/// 验证拟合入口拒绝量子比特数不一致的目标与线路。
#[test]
fn unitary_fit_rejects_mismatched_qubit_counts() {
    let target = ry_target(0.0);
    let circuit = Circuit::new(2).unwrap();

    assert!(matches!(
        unitary_loss(&target, &circuit),
        Err(UnitaryError::QubitCountMismatchError {
            target: 1,
            circuit: 2,
        })
    ));
}

/// 验证可微酉矩阵损失可通过统一的 Tensor.backward 接口写入参数梯度。
#[test]
fn unitary_loss_should_integrate_with_tensor_autograd() {
    let target = ry_target(0.8);
    let mut circuit = Circuit::new(1).unwrap();
    circuit.ry(0.2, 0usize).unwrap();
    let loss = unitary_loss(&target, &circuit).unwrap();
    let loss_value = loss.storage().as_f64_slice().unwrap()[0];
    loss.backward().unwrap();

    let gradient = circuit.parameters()[0]
        .grad()
        .unwrap()
        .storage()
        .as_f64_slice()
        .unwrap()[0];
    assert_close(loss_value, unitary_loss_value(&target, &circuit).unwrap());

    let epsilon = 1e-6;
    circuit.parameters()[0].set_tensor(Tensor::new(0.2 + epsilon).unwrap());
    let plus = unitary_loss_value(&target, &circuit).unwrap();
    circuit.parameters()[0].set_tensor(Tensor::new(0.2 - epsilon).unwrap());
    let minus = unitary_loss_value(&target, &circuit).unwrap();
    assert_close(gradient, (plus - minus) / (2.0 * epsilon));
}

/// 验证统一酉矩阵损失会把标量上游梯度正确乘入解析梯度。
#[test]
fn unitary_loss_should_scale_gradient_by_upstream_gradient() {
    let target = ry_target(0.8);
    let mut circuit = Circuit::new(1).unwrap();
    circuit.ry(0.2, 0usize).unwrap();
    let loss = unitary_loss(&target, &circuit).unwrap();
    loss.backward().unwrap();
    let expected = circuit.parameters()[0]
        .grad()
        .unwrap()
        .storage()
        .as_f64_slice()
        .unwrap()[0];
    circuit.parameters()[0].zero_grad();

    unitary_loss(&target, &circuit)
        .unwrap()
        .backward_with_grad(Tensor::new(2.5_f64).unwrap())
        .unwrap();

    let gradient = circuit.parameters()[0]
        .grad()
        .unwrap()
        .storage()
        .as_f64_slice()
        .unwrap()[0];
    assert_close(gradient, 2.5 * expected);
}
