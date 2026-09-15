use arcqml_circuit::prelude::*;
use arcqml_core::Tensor;

/// 演示默认拼接与显式参数绑定两种电路追加方式。
fn main() -> Result<(), Box<dyn std::error::Error>> {
    append_with_independent_parameters()?;
    append_with_shared_parameter()?;
    Ok(())
}

/// 默认拼接：右侧参数全部作为独立参数续接到左侧。
fn append_with_independent_parameters() -> Result<(), Box<dyn std::error::Error>> {
    let mut left = Circuit::new(2)?;
    left.h(0usize)?.ry(0.25, 0usize)?.cnot(0usize, 1usize)?;

    let mut right = Circuit::new(2)?;
    right.rz(-0.4, 1usize)?.x(0usize)?;

    // 右侧参数 ID 0 被重映射为左侧参数 ID 1。
    let report = left.append(right)?;

    assert_eq!(left.len(), 5);
    assert_eq!(left.num_parameters(), 2);
    assert_eq!(report.parameter_mapping(), &[ParameterId::new(1)]);
    assert_eq!(left.parameter_name(ParameterId::new(1))?, "rz_q1_theta_1");

    println!("默认拼接完成：");
    print_append_report(&left, &report);
    Ok(())
}

/// 显式绑定：B 的部分参数共享 A 的参数，其它参数续接到 A 之后。
fn append_with_shared_parameter() -> Result<(), Box<dyn std::error::Error>> {
    let mut left = Circuit::new(2)?;

    // A 的参数 0 将作为两条电路共同使用的参数。
    let shared = left.add_parameter(0.6)?;
    left.ry_param(shared, 0usize)?;
    left.rz(0.1, 1usize)?;
    left.parameters()[shared.index()].set_grad(Tensor::new(0.9_f64)?);

    let mut right = Circuit::new(2)?;
    right.ry(-0.2, 0usize)?;
    let right_shared = right
        .parameter_id("ry_q0_theta_0")
        .expect("自动参数必须存在");

    // B 内部复用它的参数 0；拼接绑定后两个门都会引用 A 的 shared 参数。
    right.ry_param(right_shared, 1usize)?;
    right.rz(0.35, 1usize)?;
    right.parameters()[right_shared.index()].set_grad(Tensor::new(-0.3_f64)?);

    let report =
        left.append_with_bindings(right, &[ParameterBinding::new(right_shared, shared)])?;

    // B 参数 0 绑定到 A 参数 0；B 参数 1 未绑定，因此续接为 A 参数 2。
    assert_eq!(
        report.parameter_mapping(),
        &[ParameterId::new(0), ParameterId::new(2)]
    );
    assert_eq!(report.appended_parameters(), 1);
    assert_eq!(left.num_parameters(), 3);
    assert_eq!(left.parameter_name(ParameterId::new(2))?, "rz_q1_theta_2");

    // 结构改变后，旧梯度没有意义，append 会清空所有参数梯度。
    assert!(
        left.parameters()
            .iter()
            .all(|parameter| parameter.grad().is_none())
    );

    // B 中前两个门应当都已重写为 A 的 shared 参数。
    assert_eq!(left.operations()[2].gate().parameter_ids(), vec![shared]);
    assert_eq!(left.operations()[3].gate().parameter_ids(), vec![shared]);

    println!("\n参数绑定拼接完成：");
    print_append_report(&left, &report);
    Ok(())
}

/// 输出拼接结果，便于核对最终参数表和参数映射。
fn print_append_report(circuit: &Circuit, report: &AppendReport) {
    println!("  操作数：{}", circuit.len());
    println!("  参数数：{}", circuit.num_parameters());
    println!("  B 参数映射：{:?}", report.parameter_mapping());
    println!("  新增参数：{}", report.appended_parameters());
    println!("  新增操作：{}", report.appended_operations());
    for index in 0..circuit.num_parameters() {
        let id = ParameterId::new(index);
        println!(
            "  参数 {index}: {} = {}",
            circuit.parameter_name(id).unwrap(),
            circuit.parameter_scalar_value(id).unwrap()
        );
    }
}
