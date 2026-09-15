use arcqml_circuit::prelude::*;
use arcqml_core::prelude::*;
use arcqml_observable::prelude::*;
use arcqml_sim::prelude::*;

/// 演示 U3、受控旋转、Pauli 演化和 fSim 的多参数解析梯度。
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let theta = Tensor::new(0.31_f64)?;
    let phi = Tensor::new(-0.17_f64)?;
    let lambda = Tensor::new(0.42_f64)?;

    let mut circuit = Circuit::new(2)?;
    let theta_id = circuit.add_parameter_tensor(theta.clone())?;
    let phi_id = circuit.add_parameter_tensor(phi.clone())?;
    let lambda_id = circuit.add_parameter_tensor(lambda.clone())?;

    circuit
        .h(0usize)?
        .u3_params(theta_id, phi_id, lambda_id, 0usize)?
        .crx(0.23, 0usize, 1usize)?
        .rzz(0.19, 0usize, 1usize)?
        .fsim(0.27, -0.11, 0usize, 1usize)?;

    let observable = SparsePauliOp::single(2, 0usize, Pauli::Z, 1.0)?;
    let simulator = StateVectorSimulator::new(circuit.num_qubits())?;
    let energy = simulator.run(&circuit, &observable)?;
    energy.backward()?;

    println!("<Z0> = {}", energy.value()?);
    println!("dE/dtheta = {}", theta.grad().unwrap().value()?);
    println!("dE/dphi = {}", phi.grad().unwrap().value()?);
    println!("dE/dlambda = {}", lambda.grad().unwrap().value()?);
    Ok(())
}
