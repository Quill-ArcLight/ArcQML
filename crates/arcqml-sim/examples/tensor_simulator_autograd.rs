use arcqml_circuit::prelude::*;
use arcqml_core::prelude::*;
use arcqml_linalg::prelude::*;
use arcqml_observable::prelude::*;
use arcqml_sim::prelude::*;

/// 演示使用参数化电路计算期望值并通过自动微分获得梯度。
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // 用 Tensor 建立可训练的旋转角，并把它注册到 Circuit 参数表。
    let theta = Tensor::new(0.4_f64)?;
    let mut circuit = Circuit::new(1)?;
    let theta_id = circuit.add_parameter_tensor(theta.clone())?;
    circuit.ry_param(theta_id, 0usize)?;

    // 期望值公式为 <0| Ry(theta)† Z Ry(theta) |0> = cos(theta)。
    // SparsePauliOp 首次参与状态矢量计算时会自动编译并缓存 Hamiltonian，无需额外 API。
    let observable = SparsePauliOp::single(1, 0usize, Pauli::Z, 1.0)?;
    let simulator = StateVectorSimulator::new(circuit.num_qubits())?;
    let expectation = simulator.run(&circuit, &observable)?;
    println!("expectation: {}", expectation.value()?);

    // 期望值是普通 F64 Tensor，因此可以继续交给 arcqml-linalg 组成损失。
    let loss = square(&expectation)?;
    loss.backward()?;
    println!("loss: {}", loss.value()?);
    println!(
        "d(loss)/d(theta): {:?}",
        theta.grad().unwrap().storage().as_f64_slice()
    );

    Ok(())
}
