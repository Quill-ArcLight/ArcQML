use arcqml_circuit::{Circuit, CircuitError, Gate, Gateparam, ParameterId, Qubit};

/// 验证 JSON 往返保留电路结构和参数引用，但不保留参数运行时数值。
#[test]
fn json_round_trip_preserves_structure_without_parameter_values() {
    let mut circuit = Circuit::new(2).unwrap();
    let theta = circuit.add_parameter(0.75).unwrap();
    circuit.h(0).unwrap();
    circuit
        .add_gate(Gate::ry(theta), vec![Qubit::new(1)])
        .unwrap();
    circuit
        .add_gate(Gate::rz(0.25), vec![Qubit::new(1)])
        .unwrap();
    circuit.cnot(0, 1).unwrap();
    let json = circuit.to_json().unwrap();
    let restored = Circuit::from_json(&json).unwrap();

    assert_eq!(restored.num_qubits(), 2);
    assert_eq!(restored.len(), 4);
    assert_eq!(
        restored.parameter_name(ParameterId::new(0)).unwrap(),
        "parameter_0"
    );
    assert_eq!(
        restored
            .parameter_scalar_value(ParameterId::new(0))
            .unwrap(),
        0.0
    );
    assert_eq!(
        restored.operations()[1].gate().parameters(),
        vec![Gateparam::param(ParameterId::new(0))]
    );
    assert_eq!(
        restored.operations()[2].gate().parameters(),
        vec![Gateparam::fixed(0.25)]
    );
    assert!(!json.contains("0.75"));
}

/// 验证 JSON 中不存在的参数引用会在恢复时被拒绝。
#[test]
fn from_json_rejects_unknown_parameter_reference() {
    let error = Circuit::from_json(
        r#"{"qubit_count":1,"parameters":[],"operations":[{"gate":{"kind":"rx","parameter":{"kind":"parameter","id":0}},"qubits":[0]}]}"#,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        CircuitError::InvalidParameterIdError {
            index: 0,
            parameter_count: 0
        }
    ));
}
