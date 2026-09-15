use arcqml_circuit::prelude::*;
use arcqml_core::Tensor;

use num_complex::Complex64;

// ============================================================
// 量子比特
// ============================================================

/// 测试 Qubit 可以通过 new 和 usize.into 创建，并正确返回索引。
#[test]
fn qubit_new_and_from_usize_should_work() {
    let qubit = Qubit::new(2);

    assert_eq!(qubit.index(), 2);

    let qubit: Qubit = 3usize.into();

    assert_eq!(qubit.index(), 3);
}

// ============================================================
// 参数
// ============================================================

/// 测试 ParameterId 与 Gateparam 的固定参数、非固定参数行为是否正确。
#[test]
fn parameter_fixed_and_param_should_work() {
    let fixed = Gateparam::fixed(0.5);

    assert_eq!(fixed.as_fixed(), Some(0.5));
    assert_eq!(fixed.as_parameter(), None);
    assert!(fixed.is_fixed());

    let id = ParameterId::new(2);
    let parameter = Gateparam::param(id);

    assert_eq!(parameter.as_fixed(), None);
    assert_eq!(parameter.as_parameter(), Some(id));
    assert!(parameter.is_parameter());
}

// ============================================================
// 量子门
// ============================================================

/// 测试常用 Gate 的名称、arity 和参数化状态是否正确。
#[test]
fn gate_metadata_should_work() {
    let h = Gate::h();

    assert_eq!(h.name(), "H");
    assert_eq!(h.arity(), 1);
    assert!(!h.is_parameterized());

    let rx = Gate::rx(Gateparam::param(ParameterId::new(0)));

    assert_eq!(rx.name(), "Rx");
    assert_eq!(rx.arity(), 1);
    assert!(rx.is_parameterized());

    let cnot = Gate::cnot();

    assert_eq!(cnot.name(), "CNot");
    assert_eq!(cnot.arity(), 2);
}

/// 测试所有内置 Gate 的名称和 arity 是否正确。
#[test]
fn gate_builtin_metadata_should_work() {
    let cases = [
        (Gate::i(), "I", 1),
        (Gate::x(), "X", 1),
        (Gate::y(), "Y", 1),
        (Gate::z(), "Z", 1),
        (Gate::h(), "H", 1),
        (Gate::s(), "S", 1),
        (Gate::t(), "T", 1),
        (Gate::rx(0.1), "Rx", 1),
        (Gate::ry(0.1), "Ry", 1),
        (Gate::rz(0.1), "Rz", 1),
        (Gate::phase(0.1), "Phase", 1),
        (Gate::cnot(), "CNot", 2),
        (Gate::cz(), "CZ", 2),
        (Gate::swap(), "Swap", 2),
        (Gate::toffoli(), "Toffoli", 3),
    ];

    for (gate, name, arity) in cases {
        assert_eq!(gate.name(), name);
        assert_eq!(gate.arity(), arity);
    }
}

#[test]
fn common_gate_metadata_and_multi_parameter_ids_should_work() {
    let fixed_cases = [
        (Gate::sdg(), "Sdg", 1),
        (Gate::tdg(), "Tdg", 1),
        (Gate::sx(), "SX", 1),
        (Gate::sxdg(), "SXdg", 1),
        (Gate::cy(), "CY", 2),
        (Gate::ch(), "CH", 2),
        (Gate::cs(), "CS", 2),
        (Gate::ct(), "CT", 2),
        (Gate::iswap(), "iSwap", 2),
        (Gate::dcx(), "DCX", 2),
        (Gate::ecr(), "ECR", 2),
        (Gate::cswap(), "CSwap", 3),
        (Gate::mcx(3), "MCX", 4),
    ];
    for (gate, name, arity) in fixed_cases {
        assert_eq!(gate.name(), name);
        assert_eq!(gate.arity(), arity);
        assert!(!gate.is_parameterized());
    }

    let ids = [
        ParameterId::new(3),
        ParameterId::new(7),
        ParameterId::new(11),
    ];
    let u3 = Gate::u3(ids[0], ids[1], ids[2]);
    assert_eq!(u3.name(), "U3");
    assert_eq!(u3.parameter_ids(), ids.to_vec());
    assert!(u3.is_parameterized());

    let fsim = Gate::fsim(ids[0], ids[1]);
    assert_eq!(fsim.parameter_ids(), ids[..2].to_vec());
    assert_eq!(Gate::u1(0.2).name(), "Phase");
    assert_eq!(Gate::u2(0.2, 0.3).name(), "U3");
}

#[test]
fn circuit_common_gate_builders_should_work() {
    let mut circuit = Circuit::new(4).unwrap();
    circuit
        .sx(0usize)
        .unwrap()
        .cy(0usize, 1usize)
        .unwrap()
        .cswap(0usize, 1usize, 2usize)
        .unwrap()
        .mcx([0usize, 1usize, 2usize], 3usize)
        .unwrap()
        .u3(0.1, 0.2, 0.3, 0usize)
        .unwrap()
        .rzz(0.4, 1usize, 2usize)
        .unwrap()
        .fsim(0.5, 0.6, 2usize, 3usize)
        .unwrap();

    assert_eq!(circuit.len(), 7);
    assert_eq!(circuit.num_parameters(), 6);
    assert!(circuit.is_parameterized());
    circuit.validate().unwrap();
}

/// 测试自定义酉门会校验矩阵元素数量。
#[test]
fn custom_unitary_should_validate_shape() {
    let matrix = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];

    let gate = Gate::custom_unitary("U", 1, matrix).unwrap();

    assert_eq!(gate.name(), "U");
    assert_eq!(gate.arity(), 1);

    let bad = Gate::custom_unitary("Bad", 2, vec![Complex64::new(1.0, 0.0)]);

    assert!(matches!(
        bad,
        Err(CircuitError::InvalidUnitaryShapeError { .. })
    ));

    let non_unitary = Gate::custom_unitary(
        "NonUnitary",
        1,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    );

    assert!(matches!(
        non_unitary,
        Err(CircuitError::NonUnitaryMatrixError { .. })
    ));
}

/// 测试 Operation 会校验门的 arity 和重复 qubit。
#[test]
fn operation_should_validate_arity_and_duplicates() {
    let operation = Operation::new(Gate::cnot(), vec![Qubit::new(0), Qubit::new(1)]).unwrap();

    assert_eq!(operation.gate().name(), "CNot");
    assert_eq!(operation.qubits(), &[Qubit::new(0), Qubit::new(1)]);

    let arity_error = Operation::new(Gate::cnot(), vec![Qubit::new(0)]);

    assert!(matches!(
        arity_error,
        Err(CircuitError::GateArityError { .. })
    ));

    let duplicate_error = Operation::new(Gate::cnot(), vec![Qubit::new(0), Qubit::new(0)]);

    assert!(matches!(
        duplicate_error,
        Err(CircuitError::DuplicateQubitError { index: 0 })
    ));
}

/// 测试直接构造 CustomUnitary 变体时，Operation::new 仍然会校验矩阵形状。
#[test]
fn operation_should_validate_direct_custom_unitary_shape() {
    let gate = Gate::CustomUnitary {
        name: "Bad".to_string(),
        arity: 1,
        matrix: vec![Complex64::new(1.0, 0.0)],
    };

    let result = Operation::new(gate, vec![Qubit::new(0)]);

    assert!(matches!(
        result,
        Err(CircuitError::InvalidUnitaryShapeError { .. })
    ));
}

// ============================================================
// 量子电路
// ============================================================

/// 测试 Circuit 可以构建常见门，并正确记录门数量、qubit 和参数化状态。
#[test]
fn circuit_should_build_common_gates() {
    let mut circuit = Circuit::new(3).unwrap();

    circuit
        .h(0usize)
        .unwrap()
        .rx_fixed(0.3, 1usize)
        .unwrap()
        .cnot(0usize, 1usize)
        .unwrap()
        .swap(1usize, 2usize)
        .unwrap()
        .toffoli(0usize, 1usize, 2usize)
        .unwrap();

    assert_eq!(circuit.num_qubits(), 3);
    assert_eq!(circuit.len(), 5);
    assert!(!circuit.is_empty());
    assert!(!circuit.is_parameterized());
    assert_eq!(circuit.operations()[0].gate().name(), "H");
    assert_eq!(
        circuit.operations()[2].qubits(),
        &[Qubit::new(0), Qubit::new(1)]
    );

    circuit.validate().unwrap();
}

/// 测试 Circuit 添加自动参数化旋转门时，会创建并追踪参数。
#[test]
fn circuit_should_create_and_track_parameters() {
    let mut circuit = Circuit::new(1).unwrap();

    circuit.rz(0.2, 0usize).unwrap();

    assert!(circuit.is_parameterized());
    assert_eq!(circuit.num_parameters(), 1);
    assert_eq!(
        circuit.parameter_name(ParameterId::new(0)).unwrap(),
        "rz_q0_theta_0"
    );
    assert_eq!(
        circuit.parameter_scalar_value(ParameterId::new(0)).unwrap(),
        0.2
    );
}

/// 测试 Circuit 可以根据参数名查找 ParameterId。
#[test]
fn circuit_should_find_parameter_id_by_name() {
    let mut circuit = Circuit::new(1).unwrap();

    circuit.rz(0.2, 0usize).unwrap();

    let id = circuit.parameter_id("rz_q0_theta_0").unwrap();

    assert_eq!(id.index(), 0);
    assert_eq!(circuit.parameter_id("missing"), None);
}

/// 测试 Circuit 可以添加 f32 标量 Tensor 参数，并以 f64 标量读取。
#[test]
fn circuit_should_add_f32_tensor_parameter() {
    let mut circuit = Circuit::new(1).unwrap();
    let id = circuit
        .add_parameter_tensor(Tensor::new(0.25_f32).unwrap())
        .unwrap();

    assert_eq!(id.index(), 0);
    assert!((circuit.parameter_scalar_value(id).unwrap() - 0.25).abs() < 1e-6);
}

/// 测试 Circuit 会拒绝非标量参数和不支持 dtype 的参数。
#[test]
fn circuit_should_reject_invalid_parameter_tensor() {
    let mut circuit = Circuit::new(1).unwrap();

    let non_scalar = circuit.add_parameter_tensor(Tensor::new(vec![1.0_f64, 2.0]).unwrap());

    assert!(matches!(
        non_scalar,
        Err(CircuitError::InvalidParameterShapeError { .. })
    ));

    let unsupported_dtype = circuit.add_parameter_tensor(Tensor::new(1_i64).unwrap());

    assert!(matches!(
        unsupported_dtype,
        Err(CircuitError::UnsupportedParameterDTypeError { .. })
    ));
}

/// 测试 Circuit 可以让多个门共享同一个 ParameterId。
#[test]
fn circuit_should_share_parameters_by_id() {
    let mut circuit = Circuit::new(2).unwrap();
    let theta = circuit.add_parameter(0.4).unwrap();

    circuit
        .ry_param(theta, 0usize)
        .unwrap()
        .ry_param(theta, 1usize)
        .unwrap();

    assert_eq!(circuit.num_parameters(), 1);
    assert_eq!(circuit.len(), 2);
    assert!(circuit.is_parameterized());
}

/// 测试 Circuit 会拒绝无效输入，包括空线路规模、越界 qubit 和重复 qubit。
#[test]
fn circuit_should_reject_invalid_inputs() {
    assert!(matches!(
        Circuit::new(0),
        Err(CircuitError::EmptyCircuitError)
    ));

    let mut circuit = Circuit::new(2).unwrap();

    let out_of_range = circuit.x(2usize);

    assert!(matches!(
        out_of_range,
        Err(CircuitError::QubitOutOfRangeError {
            index: 2,
            num_qubits: 2
        })
    ));

    let duplicate = circuit.cnot(0usize, 0usize);

    assert!(matches!(
        duplicate,
        Err(CircuitError::DuplicateQubitError { index: 0 })
    ));
}

/// 测试 Circuit 会拒绝无效 ParameterId。
#[test]
fn circuit_should_reject_invalid_parameter_id() {
    let mut circuit = Circuit::new(1).unwrap();
    let id = ParameterId::new(0);

    assert!(matches!(
        circuit.parameter(id),
        Err(CircuitError::InvalidParameterIdError { .. })
    ));

    assert!(matches!(
        circuit.parameter_name(id),
        Err(CircuitError::InvalidParameterIdError { .. })
    ));

    assert!(matches!(
        circuit.parameter_scalar_value(id),
        Err(CircuitError::InvalidParameterIdError { .. })
    ));

    assert!(matches!(
        circuit.rx_param(id, 0usize),
        Err(CircuitError::InvalidParameterIdError { .. })
    ));

    assert!(matches!(
        circuit.add_gate(Gate::rx(id), vec![Qubit::new(0)]),
        Err(CircuitError::InvalidParameterIdError { .. })
    ));
}

/// 测试 Circuit::clear 会清空所有操作和参数。
#[test]
fn circuit_clear_should_remove_operations_and_parameters() {
    let mut circuit = Circuit::new(1).unwrap();

    circuit.rx(0.5, 0usize).unwrap().h(0usize).unwrap();

    assert_eq!(circuit.len(), 2);
    assert_eq!(circuit.num_parameters(), 1);

    circuit.clear();

    assert!(circuit.is_empty());
    assert_eq!(circuit.num_parameters(), 0);
    assert!(!circuit.is_parameterized());
}

/// 测试 Circuit::depth 会把不相交的门放进同一层，并保持依赖顺序。
#[test]
fn circuit_depth_should_count_parallel_layers() {
    let mut circuit = Circuit::new(3).unwrap();

    circuit
        .h(0usize)
        .unwrap()
        .x(2usize)
        .unwrap()
        .cnot(0usize, 1usize)
        .unwrap()
        .z(2usize)
        .unwrap()
        .swap(1usize, 2usize)
        .unwrap();

    // H(q0) 和 X(q2) 是第一层，CNOT(q0, q1) 与 Z(q2) 是第二层，
    // 最后的 SWAP 同时依赖 q1 与 q2，因此单独构成第三层。
    assert_eq!(circuit.depth(), 3);
}

#[test]
fn circuit_clone_should_not_share_parameter_state() {
    let mut circuit = Circuit::new(1).unwrap();
    circuit.ry(0.2, 0usize).unwrap();
    circuit.parameters()[0].set_grad(Tensor::new(0.3_f64).unwrap());

    let cloned = circuit.clone();
    cloned.parameters()[0].set_tensor(Tensor::new(0.8_f64).unwrap());
    cloned.parameters()[0].set_grad(Tensor::new(0.7_f64).unwrap());

    assert_eq!(
        circuit.parameter_scalar_value(ParameterId::new(0)).unwrap(),
        0.2
    );
    assert_eq!(
        cloned.parameter_scalar_value(ParameterId::new(0)).unwrap(),
        0.8
    );
    assert_eq!(
        circuit.parameters()[0]
            .grad()
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[0.3]
    );
    assert_eq!(
        cloned.parameters()[0]
            .grad()
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[0.7]
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn custom_unitary_should_reject_arities_that_would_truncate_to_u32() {
    let arity = u32::MAX as usize + 1;
    let gate = Gate::CustomUnitary {
        name: "overflow".to_string(),
        arity,
        matrix: vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    };

    assert!(matches!(
        gate.validate(),
        Err(CircuitError::UnitaryDimensionOverflowError { arity: actual }) if actual == arity
    ));
}
