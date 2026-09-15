use arcqml_circuit::prelude::*;
use arcqml_kernel::prelude::*;
use num_complex::Complex64;

/// 断言两个复数在数值容差内相等。
fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1e-12,
        "actual={actual:?}, expected={expected:?}"
    );
}

/// 验证共享 kernel 会原地执行单量子比特门。
#[test]
fn kernel_applies_x_gate_in_place() {
    let mut amplitudes = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

    apply_gate(&mut amplitudes, 1, &Gate::X, &[Qubit::new(0)]).unwrap();

    assert_complex_close(amplitudes[0], Complex64::new(0.0, 0.0));
    assert_complex_close(amplitudes[1], Complex64::new(1.0, 0.0));
}

#[test]
fn single_state_kernel_rejects_invalid_state_lengths_without_panicking() {
    let mut amplitudes = vec![Complex64::new(1.0, 0.0)];
    let qubits = [Qubit::new(0)];

    assert!(matches!(
        apply_gate(&mut amplitudes, 1, &Gate::X, &qubits),
        Err(KernelError::StateLengthError {
            num_qubits: 1,
            expected: 2,
            actual: 1,
        })
    ));
    assert!(matches!(
        apply_adjoint_gate(&mut amplitudes, 1, &Gate::X, &qubits),
        Err(KernelError::StateLengthError {
            num_qubits: 1,
            expected: 2,
            actual: 1,
        })
    ));
    assert!(matches!(
        parameter_derivative(&amplitudes, 1, &Gate::X, &qubits, 0),
        Err(KernelError::StateLengthError {
            num_qubits: 1,
            expected: 2,
            actual: 1,
        })
    ));
}

#[test]
fn single_state_kernel_rejects_unrepresentable_state_dimensions() {
    let mut amplitudes = vec![Complex64::new(1.0, 0.0)];

    assert!(matches!(
        apply_gate(
            &mut amplitudes,
            usize::BITS as usize,
            &Gate::X,
            &[Qubit::new(0)],
        ),
        Err(KernelError::StateDimensionOverflowError { .. })
    ));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn batch_kernel_rejects_qubit_counts_that_would_truncate_to_u32() {
    let mut amplitudes = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let num_qubits = u32::MAX as usize + 1;

    assert!(matches!(
        apply_gate_batch(
            &mut amplitudes,
            1,
            num_qubits,
            &Gate::X,
            &[Qubit::new(0)],
        ),
        Err(KernelError::StateDimensionOverflowError {
            num_qubits: actual
        }) if actual == num_qubits
    ));
}

/// 验证共轭转置 kernel 会撤销前向门作用。
#[test]
fn kernel_adjoint_reverses_phase_gate() {
    let mut amplitudes = vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];

    apply_gate(&mut amplitudes, 1, &Gate::S, &[Qubit::new(0)]).unwrap();
    apply_adjoint_gate(&mut amplitudes, 1, &Gate::S, &[Qubit::new(0)]).unwrap();

    assert_complex_close(amplitudes[0], Complex64::new(0.0, 0.0));
    assert_complex_close(amplitudes[1], Complex64::new(1.0, 0.0));
}

/// 验证未绑定参数在执行前会返回明确错误。
#[test]
fn kernel_rejects_unbound_parameters() {
    let mut amplitudes = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let gate = Gate::rx(ParameterId::new(0));

    assert!(matches!(
        apply_gate(&mut amplitudes, 1, &gate, &[Qubit::new(0)]),
        Err(KernelError::UnboundParameterError { .. })
    ));
}

#[test]
fn rotation_derivative_into_matches_central_difference_on_every_qubit() {
    let input = vec![
        Complex64::new(0.25, -0.10),
        Complex64::new(-0.15, 0.30),
        Complex64::new(0.40, 0.05),
        Complex64::new(-0.35, -0.20),
        Complex64::new(0.10, 0.45),
        Complex64::new(-0.25, 0.15),
        Complex64::new(0.05, -0.40),
        Complex64::new(0.30, 0.20),
    ];
    let theta = 0.37;
    let epsilon = 1e-6;

    for gate_name in ["rx", "ry", "rz"] {
        for wire in 0..3 {
            let qubits = [Qubit::new(wire)];
            let gate = match gate_name {
                "rx" => Gate::rx(theta),
                "ry" => Gate::ry(theta),
                "rz" => Gate::rz(theta),
                _ => unreachable!(),
            };
            let mut actual = vec![Complex64::new(123.0, -456.0); input.len()];
            parameter_derivative_into(&input, &mut actual, 3, &gate, &qubits, 0).unwrap();

            let mut plus = input.clone();
            let plus_gate = match gate_name {
                "rx" => Gate::rx(theta + epsilon),
                "ry" => Gate::ry(theta + epsilon),
                "rz" => Gate::rz(theta + epsilon),
                _ => unreachable!(),
            };
            apply_gate(&mut plus, 3, &plus_gate, &qubits).unwrap();

            let mut minus = input.clone();
            let minus_gate = match gate_name {
                "rx" => Gate::rx(theta - epsilon),
                "ry" => Gate::ry(theta - epsilon),
                "rz" => Gate::rz(theta - epsilon),
                _ => unreachable!(),
            };
            apply_gate(&mut minus, 3, &minus_gate, &qubits).unwrap();

            for ((actual, plus), minus) in actual.iter().zip(plus).zip(minus) {
                let expected = (plus - minus) / (2.0 * epsilon);
                assert!(
                    (*actual - expected).norm() < 1e-8,
                    "gate={gate_name}, wire={wire}, actual={actual:?}, expected={expected:?}"
                );
            }
        }
    }
}

#[test]
fn batch_kernel_applies_and_reverses_ry_for_every_row() {
    let original = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let mut amplitudes = original.clone();
    let gate = Gate::ry(std::f64::consts::PI);

    apply_gate_batch(&mut amplitudes, 2, 1, &gate, &[Qubit::new(0)]).unwrap();
    assert_complex_close(amplitudes[0], Complex64::new(0.0, 0.0));
    assert_complex_close(amplitudes[1], Complex64::new(1.0, 0.0));
    assert_complex_close(amplitudes[2], Complex64::new(-1.0, 0.0));
    assert_complex_close(amplitudes[3], Complex64::new(0.0, 0.0));

    apply_adjoint_gate_batch(&mut amplitudes, 2, 1, &gate, &[Qubit::new(0)]).unwrap();
    for (actual, expected) in amplitudes.into_iter().zip(original) {
        assert_complex_close(actual, expected);
    }
}

#[test]
fn batch_kernel_applies_cnot_to_each_row_with_little_endian_qubits() {
    let mut amplitudes = vec![
        // 第 0 个 batch 行：|01>，因此控制量子位 0 为一。
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        // 第 1 个 batch 行：|10>，因此控制量子位 0 为零。
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    ];

    apply_gate_batch(
        &mut amplitudes,
        2,
        2,
        &Gate::cnot(),
        &[Qubit::new(0), Qubit::new(1)],
    )
    .unwrap();

    assert_complex_close(amplitudes[3], Complex64::new(1.0, 0.0));
    assert_complex_close(amplitudes[6], Complex64::new(1.0, 0.0));
}
