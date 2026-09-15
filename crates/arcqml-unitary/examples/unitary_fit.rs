use arcqml_circuit::Circuit;
use arcqml_optim::Adam;
use arcqml_unitary::{DenseUnitary, unitary_fidelity, unitary_from_circuit, unitary_loss};
use std::error::Error;

/// 构造一个固定参数的目标酉矩阵。
fn build_target(theta: f64) -> Result<DenseUnitary, Box<dyn Error>> {
    let mut circuit = Circuit::new(1)?;
    circuit.ry_fixed(theta, 0usize)?;
    Ok(unitary_from_circuit(&circuit)?)
}

/// 使用 Adam 将一条含参 RY 线路拟合到目标酉矩阵。
fn main() -> Result<(), Box<dyn Error>> {
    // 以 RY(1.2) 电路对应的酉矩阵作为拟合目标。
    let target_angle = 1.2;
    let target = build_target(target_angle)?;

    // 待优化电路从 RY(0) 开始。
    let mut ansatz = Circuit::new(1)?;
    ansatz.ry(0.0, 0usize)?;

    // 使用 Adam 更新电路中的可训练参数。
    let mut optimizer = Adam::new(0.05, 0.9, 0.999, 1e-8, 0.0)?;

    for step in 0..200 {
        // 损失函数返回标量 Tensor；backward 将梯度累积到电路参数。
        let loss = unitary_loss(&target, &ansatz)?;
        let loss_value = loss.storage().as_f64_slice().unwrap()[0];
        loss.backward()?;
        optimizer.step(ansatz.parameters())?;
        optimizer.zero_grad(ansatz.parameters());

        if step % 20 == 0 || step == 199 {
            println!(
                "step={step:03} loss={:.8} fidelity={:.8}",
                loss_value,
                unitary_fidelity(&target, &ansatz)?
            );
        }
    }

    // 读取拟合后的第一个电路参数并报告最终保真度。
    let fitted_angle = ansatz.parameter_scalar_value(arcqml_circuit::ParameterId::new(0))?;
    let fidelity = unitary_fidelity(&target, &ansatz)?;
    println!("target angle = {target_angle:.6}");
    println!("fitted angle = {fitted_angle:.6}");
    println!("final fidelity = {fidelity:.10}");
    Ok(())
}
