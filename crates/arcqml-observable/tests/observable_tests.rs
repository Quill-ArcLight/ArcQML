use arcqml_observable::prelude::*;

use num_complex::Complex64;

// ============================================================
// Pauli 算符
// ============================================================

#[test]
fn pauli_name_matrix_and_multiply_should_work() {
    // 验证 Pauli 的名称、单位算符判断、矩阵形式和基本乘法相位。
    assert_eq!(Pauli::X.name(), "X");
    assert!(!Pauli::X.is_identity());
    assert!(Pauli::I.is_identity());

    let x = Pauli::X.matrix();

    assert_eq!(x[1], Complex64::new(1.0, 0.0));
    assert_eq!(x[2], Complex64::new(1.0, 0.0));

    let (phase, pauli) = Pauli::X.multiply(Pauli::Y);

    assert_eq!(phase, Complex64::new(0.0, 1.0));
    assert_eq!(pauli, Pauli::Z);

    let (phase, pauli) = Pauli::Y.multiply(Pauli::X);

    assert_eq!(phase, Complex64::new(0.0, -1.0));
    assert_eq!(pauli, Pauli::Z);
}

// ============================================================
// Pauli 串
// ============================================================

#[test]
fn pauli_string_should_canonicalize_ops() {
    // 验证 PauliString 会过滤 I 项，并按 qubit 下标排序。
    let string = PauliString::new(
        3,
        vec![
            PauliOp::new(2usize, Pauli::Z),
            PauliOp::new(0usize, Pauli::X),
            PauliOp::new(1usize, Pauli::I),
        ],
    )
    .unwrap();

    assert_eq!(string.num_qubits(), 3);
    assert_eq!(string.len(), 2);
    assert_eq!(string.ops()[0], PauliOp::new(0usize, Pauli::X));
    assert_eq!(string.ops()[1], PauliOp::new(2usize, Pauli::Z));
    assert_eq!(string.pauli_on(1usize).unwrap(), Pauli::I);
    assert!(!string.is_identity());
}

#[test]
fn pauli_string_should_validate_inputs() {
    // 验证 PauliString 会拒绝空系统、越界 qubit 和重复 qubit。
    assert!(matches!(
        PauliString::identity(0),
        Err(ObservableError::EmptyQubitError)
    ));

    let out_of_range = PauliString::x(2, 2usize);

    assert!(matches!(
        out_of_range,
        Err(ObservableError::QubitOutOfRangeError {
            index: 2,
            num_qubits: 2
        })
    ));

    let duplicate = PauliString::new(
        2,
        vec![
            PauliOp::new(0usize, Pauli::X),
            PauliOp::new(0usize, Pauli::Z),
        ],
    );

    assert!(matches!(
        duplicate,
        Err(ObservableError::DuplicateQubitError { index: 0 })
    ));
}

#[test]
fn pauli_string_pauli_on_should_validate_qubit_range() {
    // 验证查询指定 qubit 的 Pauli 时会检查 qubit 是否越界。
    let string = PauliString::identity(2).unwrap();

    assert!(matches!(
        string.pauli_on(2usize),
        Err(ObservableError::QubitOutOfRangeError {
            index: 2,
            num_qubits: 2
        })
    ));
}

#[test]
fn pauli_string_commutation_should_work() {
    // 验证 PauliString 的对易判断符合 Pauli 代数规则。
    let x0 = PauliString::x(2, 0usize).unwrap();
    let z0 = PauliString::z(2, 0usize).unwrap();
    let z1 = PauliString::z(2, 1usize).unwrap();
    let x0z1 = PauliString::new(
        2,
        vec![
            PauliOp::new(0usize, Pauli::X),
            PauliOp::new(1usize, Pauli::Z),
        ],
    )
    .unwrap();

    assert!(!x0.commutes_with(&z0).unwrap());
    assert!(x0.commutes_with(&z1).unwrap());
    assert!(!x0z1.commutes_with(&z0).unwrap());
}

#[test]
fn pauli_string_multiply_should_work() {
    // 验证 PauliString 相乘会正确累积局部相位并合并每个 qubit 上的 Pauli。
    let x0 = PauliString::x(2, 0usize).unwrap();
    let y0 = PauliString::y(2, 0usize).unwrap();
    let z1 = PauliString::z(2, 1usize).unwrap();

    let (phase, product) = x0.multiply(&y0).unwrap();

    assert_eq!(phase, Complex64::new(0.0, 1.0));
    assert_eq!(product.pauli_on(0usize).unwrap(), Pauli::Z);
    assert_eq!(product.pauli_on(1usize).unwrap(), Pauli::I);

    let (_, product) = product.multiply(&z1).unwrap();

    assert_eq!(product.len(), 2);
    assert_eq!(product.pauli_on(0usize).unwrap(), Pauli::Z);
    assert_eq!(product.pauli_on(1usize).unwrap(), Pauli::Z);
}

#[test]
fn pauli_string_binary_ops_should_validate_num_qubits() {
    // 验证对易判断和乘法都要求两个 PauliString 属于同样大小的系统。
    let x_two_qubits = PauliString::x(2, 0usize).unwrap();
    let x_one_qubit = PauliString::x(1, 0usize).unwrap();

    assert!(matches!(
        x_two_qubits.commutes_with(&x_one_qubit),
        Err(ObservableError::QubitCountMismatchError {
            expected: 2,
            actual: 1
        })
    ));

    assert!(matches!(
        x_two_qubits.multiply(&x_one_qubit),
        Err(ObservableError::QubitCountMismatchError {
            expected: 2,
            actual: 1
        })
    ));
}

// ============================================================
// 稀疏 Pauli 算符
// ============================================================

#[test]
fn pauli_term_and_sparse_pauli_op_should_work() {
    // 验证 PauliTerm 和 SparsePauliOp 能保存系数、PauliString 和系统 qubit 数量。
    let z0 = PauliString::z(2, 0usize).unwrap();
    let x1 = PauliString::x(2, 1usize).unwrap();
    let term0 = PauliTerm::new(0.5, z0).unwrap();
    let term1 = PauliTerm::new(-1.2, x1).unwrap();
    let operator = SparsePauliOp::new(2, vec![term0.clone(), term1.clone()]).unwrap();

    assert_eq!(term0.coefficient(), 0.5);
    assert_eq!(operator.num_qubits(), 2);
    assert_eq!(operator.len(), 2);
    assert!(!operator.is_empty());
    assert_eq!(operator.terms()[1], term1);
}

#[test]
fn sparse_pauli_op_from_pauli_string_and_single_should_work() {
    // 验证 SparsePauliOp 可以从单个 PauliString 或单 qubit Pauli 项构造。
    let z0 = PauliString::z(2, 0usize).unwrap();
    let from_string = SparsePauliOp::from_pauli_string(0.5, z0).unwrap();
    let single = SparsePauliOp::single(2, 1usize, Pauli::X, -1.0).unwrap();

    assert_eq!(from_string.num_qubits(), 2);
    assert_eq!(from_string.len(), 1);
    assert_eq!(from_string.terms()[0].coefficient(), 0.5);
    assert_eq!(
        single.terms()[0].pauli_string().pauli_on(1usize).unwrap(),
        Pauli::X
    );
}

#[test]
fn sparse_pauli_op_constant_should_create_identity_term() {
    // 验证常数 SparsePauliOp 会被表示为 identity PauliString 上的一项。
    let operator = SparsePauliOp::constant(2, -0.75).unwrap();

    assert_eq!(operator.num_qubits(), 2);
    assert_eq!(operator.len(), 1);
    assert_eq!(operator.terms()[0].coefficient(), -0.75);
    assert!(operator.terms()[0].pauli_string().is_identity());
}

#[test]
fn sparse_pauli_op_add_scale_and_simplify_should_work() {
    // 验证添加项、合并相同 PauliString，以及按常数缩放 SparsePauliOp。
    let z0 = PauliString::z(2, 0usize).unwrap();
    let term0 = PauliTerm::new(1.0, z0.clone()).unwrap();
    let mut operator = SparsePauliOp::zero(2).unwrap();

    operator.add_term(term0).unwrap();
    operator.add_pauli_string(2.0, z0).unwrap();

    assert_eq!(operator.len(), 2);

    let simplified = operator.simplify(1e-12).unwrap();

    assert_eq!(simplified.len(), 1);
    assert_eq!(simplified.terms()[0].coefficient(), 3.0);

    let scaled = simplified.scale(0.5).unwrap();

    assert_eq!(scaled.terms()[0].coefficient(), 1.5);
}

#[test]
fn sparse_pauli_op_simplify_should_drop_small_terms() {
    // 验证 simplify 会删除绝对值不超过 tolerance 的项。
    let z0 = PauliString::z(2, 0usize).unwrap();
    let mut operator = SparsePauliOp::zero(2).unwrap();

    operator
        .add_term(PauliTerm::new(1.0e-13, z0).unwrap())
        .unwrap();

    let simplified = operator.simplify(1.0e-12).unwrap();

    assert!(simplified.is_empty());
}

#[test]
fn sparse_pauli_op_should_validate_inputs() {
    // 验证 SparsePauliOp 会拒绝空系统、非有限系数和 qubit 数量不匹配的项。
    assert!(matches!(
        SparsePauliOp::zero(0),
        Err(ObservableError::EmptyQubitError)
    ));

    let bad_term = PauliTerm::new(f64::NAN, PauliString::identity(1).unwrap());

    assert!(matches!(
        bad_term,
        Err(ObservableError::NonFiniteCoefficientError { .. })
    ));

    let term = PauliTerm::new(1.0, PauliString::identity(2).unwrap()).unwrap();
    let operator = SparsePauliOp::new(1, vec![term]);

    assert!(matches!(
        operator,
        Err(ObservableError::QubitCountMismatchError {
            expected: 1,
            actual: 2
        })
    ));
}

#[test]
fn sparse_pauli_op_should_validate_simplify_tolerance() {
    // 验证 simplify 的 tolerance 必须是有限且非负的实数。
    let operator = SparsePauliOp::zero(1).unwrap();

    assert!(matches!(
        operator.simplify(f64::NAN),
        Err(ObservableError::InvalidToleranceError { .. })
    ));

    assert!(matches!(
        operator.simplify(-1.0),
        Err(ObservableError::InvalidToleranceError { .. })
    ));
}

#[test]
fn hamiltonian_alias_should_keep_vqe_style_code_readable() {
    // 验证 Hamiltonian 仍然是 SparsePauliOp 的别名，方便 VQE 语境继续使用物理名称。
    let z0 = PauliString::z(1, 0usize).unwrap();
    let hamiltonian = Hamiltonian::from_pauli_string(1.0, z0).unwrap();

    assert_eq!(hamiltonian.len(), 1);
    assert_eq!(hamiltonian.terms()[0].coefficient(), 1.0);
}

/// 验证 SparsePauliOp 可以直接从 JSON 读取多项和常数项。
#[test]
fn sparse_pauli_op_from_json_should_construct_operator() {
    let operator = SparsePauliOp::from_json(
        r#"
        {
          "num_qubits": 3,
          "terms": [
            {
              "coefficient": -0.5,
              "paulis": [
                { "qubit": 2, "pauli": "X" },
                { "qubit": 0, "pauli": "Z" }
              ]
            },
            {
              "coefficient": 0.25,
              "paulis": []
            }
          ]
        }
        "#,
    )
    .unwrap();

    assert_eq!(operator.num_qubits(), 3);
    assert_eq!(operator.len(), 2);
    assert_eq!(operator.terms()[0].coefficient(), -0.5);
    assert_eq!(
        operator.terms()[0].pauli_string().pauli_on(0usize).unwrap(),
        Pauli::Z
    );
    assert_eq!(
        operator.terms()[0].pauli_string().pauli_on(2usize).unwrap(),
        Pauli::X
    );
    assert!(operator.terms()[1].pauli_string().is_identity());
}

/// 验证 JSON 中未知的 Pauli 名称会返回解析错误。
#[test]
fn sparse_pauli_op_from_json_should_reject_unknown_pauli() {
    let error = SparsePauliOp::from_json(
        r#"{
          "num_qubits": 1,
          "terms": [{"coefficient": 1.0, "paulis": [{"qubit": 0, "pauli": "I"}]}]
        }"#,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        ObservableError::JsonDeserializationError { .. }
    ));
}
