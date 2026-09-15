use arcqml_circuit::prelude::*;
use arcqml_unitary::prelude::*;
use num_complex::Complex64;

/// 断言两个复数在数值容差内相等。
fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1e-12,
        "actual={actual:?}, expected={expected:?}"
    );
}

/// 验证 Hadamard 门输出标准行主序矩阵。
#[test]
fn circuit_unitary_returns_a_row_major_hadamard_matrix() {
    let mut circuit = Circuit::new(1).unwrap();
    circuit.h(0usize).unwrap();

    let unitary = unitary_from_circuit(&circuit).unwrap();
    let scale = 1.0 / 2.0_f64.sqrt();
    let expected = [
        Complex64::new(scale, 0.0),
        Complex64::new(scale, 0.0),
        Complex64::new(scale, 0.0),
        Complex64::new(-scale, 0.0),
    ];

    assert_eq!(unitary.dimension(), 2);
    for (actual, expected) in unitary.as_row_major().iter().copied().zip(expected) {
        assert_complex_close(actual, expected);
    }
}

/// 验证 CNOT 矩阵遵循小端量子位约定。
#[test]
fn circuit_unitary_respects_little_endian_cnot_ordering() {
    let mut circuit = Circuit::new(2).unwrap();
    circuit.cnot(0usize, 1usize).unwrap();

    let unitary = unitary_from_circuit(&circuit).unwrap();
    let expected = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];

    for (actual, expected) in unitary.as_row_major().iter().copied().zip(expected) {
        assert_complex_close(actual, Complex64::new(expected, 0.0));
    }
}

/// 验证稠密酉矩阵可从 Tensor 构建并通过完整酉性检查。
#[test]
fn dense_unitary_validates_a_hadamard_matrix() {
    let scale = 1.0 / 2.0_f64.sqrt();
    let unitary = DenseUnitary::from_row_major(
        vec![
            Complex64::new(scale, 0.0),
            Complex64::new(scale, 0.0),
            Complex64::new(scale, 0.0),
            Complex64::new(-scale, 0.0),
        ],
        2,
    )
    .unwrap();

    unitary.validate(1e-12).unwrap();
    assert_eq!(unitary.num_qubits(), 1);
}

/// 验证非酉输入会在显式验证时返回错误。
#[test]
fn dense_unitary_rejects_non_unitary_data_during_validation() {
    let unitary = DenseUnitary::from_row_major(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        2,
    )
    .unwrap();

    assert!(matches!(
        unitary.validate(1e-12),
        Err(UnitaryError::NonUnitaryMatrixError { .. })
    ));
}

/// 验证可以从框架 Tensor 导入独立的稠密酉矩阵。
#[test]
fn dense_unitary_imports_a_tensor() {
    let mut circuit = Circuit::new(1).unwrap();
    circuit.h(0usize).unwrap();
    let source = unitary_from_circuit(&circuit).unwrap();
    let tensor = arcqml_core::Tensor::new(arcqml_core::TensorData::FlatC64 {
        data: source.as_row_major().to_vec(),
        shape: vec![2, 2],
    })
    .unwrap();
    let imported = DenseUnitary::from_tensor(&tensor).unwrap();

    assert_eq!(imported, source);
}

/// 验证外部 Tensor 入口会拒绝非酉矩阵。
#[test]
fn unitary_from_tensor_rejects_a_non_unitary_tensor() {
    let tensor = arcqml_core::Tensor::new(arcqml_core::TensorData::FlatC64 {
        data: vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
        shape: vec![2, 2],
    })
    .unwrap();

    assert!(matches!(
        unitary_from_tensor(&tensor),
        Err(UnitaryError::NonUnitaryMatrixError { .. })
    ));
}
