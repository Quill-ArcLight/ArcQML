use arcqml_circuit::{AppendReport, Circuit, CircuitError, ParameterBinding, ParameterId};
use arcqml_core::Tensor;

/// 默认追加应按顺序移动操作、续接参数标识与名称，并清空梯度。
#[test]
fn append_should_move_operations_continue_parameters_and_clear_gradients() {
    let mut left = Circuit::new(2).unwrap();
    left.ry(0.2, 0usize).unwrap();
    left.parameters()[0].set_grad(Tensor::new(0.7_f64).unwrap());

    let mut right = Circuit::new(2).unwrap();
    right.h(1usize).unwrap();
    right.rz(-0.3, 1usize).unwrap();
    right.parameters()[0].set_grad(Tensor::new(-0.4_f64).unwrap());

    let report = left.append(right).unwrap();

    assert_append_report(&report, &[ParameterId::new(1)], 1, 2);
    assert_eq!(left.len(), 3);
    assert_eq!(left.num_parameters(), 2);
    assert_eq!(
        left.parameter_name(ParameterId::new(1)).unwrap(),
        "rz_q1_theta_1"
    );
    assert!(
        left.parameters()
            .iter()
            .all(|parameter| parameter.grad().is_none())
    );
}

/// 显式绑定应复用左侧参数，而不将右侧同名槽位转移进来。
#[test]
fn append_with_bindings_should_reuse_left_parameter() {
    let mut left = Circuit::new(2).unwrap();
    let shared = left.add_parameter(0.5).unwrap();
    left.ry_param(shared, 0usize).unwrap();

    let mut right = Circuit::new(2).unwrap();
    right.rz(0.8, 1usize).unwrap();

    let report = left
        .append_with_bindings(right, &[ParameterBinding::new(ParameterId::new(0), shared)])
        .unwrap();

    assert_append_report(&report, &[shared], 0, 1);
    assert_eq!(left.num_parameters(), 1);
    assert_eq!(left.operations()[1].gate().parameter_ids(), vec![shared]);
    assert_eq!(left.parameter_scalar_value(shared).unwrap(), 0.5);
}

/// 右侧内部共享参数在重映射后仍必须保持共享关系。
#[test]
fn append_should_preserve_right_internal_parameter_sharing() {
    let mut left = Circuit::new(2).unwrap();
    left.rx(0.1, 0usize).unwrap();
    let mut right = Circuit::new(2).unwrap();
    let theta = right.add_parameter(0.9).unwrap();
    right.ry_param(theta, 0usize).unwrap();
    right.rz_param(theta, 1usize).unwrap();

    let report = left.append(right).unwrap();
    let remapped = report.parameter_mapping()[0];

    assert_eq!(remapped, ParameterId::new(1));
    assert_eq!(left.operations()[1].gate().parameter_ids(), vec![remapped]);
    assert_eq!(left.operations()[2].gate().parameter_ids(), vec![remapped]);
}

/// 追加前发生的兼容性错误不得修改左侧电路。
#[test]
fn append_failure_should_leave_left_circuit_unchanged() {
    let mut left = Circuit::new(1).unwrap();
    left.ry(0.2, 0usize).unwrap();
    let mut right = Circuit::new(2).unwrap();
    right.rz(0.3, 1usize).unwrap();

    let error = left.append(right).unwrap_err();

    assert!(matches!(
        error,
        CircuitError::AppendQubitCountMismatchError { left: 1, right: 2 }
    ));
    assert_eq!(left.len(), 1);
    assert_eq!(left.num_parameters(), 1);
    assert_eq!(
        left.parameter_name(ParameterId::new(0)).unwrap(),
        "ry_q0_theta_0"
    );
}

/// 非法绑定也必须在移动右侧内容前被拒绝。
#[test]
fn append_with_invalid_binding_should_leave_left_circuit_unchanged() {
    let mut left = Circuit::new(1).unwrap();
    left.ry(0.2, 0usize).unwrap();
    let mut right = Circuit::new(1).unwrap();
    right.rz(0.3, 0usize).unwrap();

    let error = left
        .append_with_bindings(
            right,
            &[ParameterBinding::new(
                ParameterId::new(0),
                ParameterId::new(1),
            )],
        )
        .unwrap_err();

    assert!(matches!(
        error,
        CircuitError::InvalidBindingTargetParameterIdError {
            index: 1,
            parameter_count: 1
        }
    ));
    assert_eq!(left.len(), 1);
    assert_eq!(left.num_parameters(), 1);
}

/// 非标准名称也应稳定地追加最终参数编号。
#[test]
fn append_should_continue_custom_parameter_name_with_target_id() {
    let mut left = Circuit::new(1).unwrap();
    left.add_parameter(0.1).unwrap();
    let mut right = Circuit::new(1).unwrap();
    right.add_parameter(0.2).unwrap();
    right.parameters()[0].set_name("shared_theta");

    left.append(right).unwrap();

    assert_eq!(
        left.parameter_name(ParameterId::new(1)).unwrap(),
        "shared_theta_1"
    );
}

/// 追加报告应准确描述参数映射和移动数量。
fn assert_append_report(
    report: &AppendReport,
    mapping: &[ParameterId],
    parameters: usize,
    operations: usize,
) {
    assert_eq!(report.parameter_mapping(), mapping);
    assert_eq!(report.appended_parameters(), parameters);
    assert_eq!(report.appended_operations(), operations);
}
