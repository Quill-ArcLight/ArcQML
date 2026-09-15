use arcqml_circuit::prelude::*;

fn main() -> CircuitResult<()> {
    let mut circuit = Circuit::new(2)?;

    // 自动创建一个可训练的标量 Parameter
    circuit.rz(0.2, 0usize)?;
    let auto_theta = circuit.parameter_id("rz_q0_theta_0").unwrap();

    println!("auto_theta: {:?}", auto_theta);

    // 手动创建 ParameterId 便于多个门可以共享同一个参数
    // 自动创建的 Parameter 也可以用于多个门，但是获取 ParameterId 不是很方便
    let shared_theta = circuit.add_parameter(0.4)?;

    println!("shared_theta: {:?}", shared_theta);

    circuit
        .h(0usize)?
        .cnot(0usize, 1usize)?
        .ry_param(shared_theta, 0usize)?
        .ry_param(shared_theta, 1usize)?
        // *_fixed 版本不会创建 Parameter。
        .rx_fixed(-0.1, 1usize)?;

    circuit.validate()?;

    println!("qubits: {}", circuit.num_qubits());
    println!("operations: {}", circuit.len());
    println!("parameters: {}", circuit.num_parameters());
    println!(
        "automatic parameter: name={}, value={}",
        circuit.parameter_name(auto_theta)?,
        circuit.parameter_scalar_value(auto_theta)?
    );
    println!(
        "shared parameter: name={}, value={}",
        circuit.parameter_name(shared_theta)?,
        circuit.parameter_scalar_value(shared_theta)?
    );

    for (index, operation) in circuit.operations().iter().enumerate() {
        println!(
            "operation {index}: gate={}, qubits={:?}",
            operation.gate().name(),
            operation.qubits()
        );
    }

    Ok(())
}
