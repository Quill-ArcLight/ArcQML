use arcqml::checkpoint::save_weights;
use arcqml::prelude::*;
use std::path::PathBuf;

/// 训练示例：拟合单个 Pauli-Z 期望值并保存电路参数。
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let output_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/arcqml/examples/trained_circuit.json"));

    let mut circuit = Circuit::new(2)?;
    circuit.h(0usize)?;
    circuit.ry(0.20, 0usize)?;
    circuit.rz(-0.35, 0usize)?;
    circuit.cnot(0usize, 1usize)?;
    circuit.ry(0.45, 1usize)?;
    circuit.crx(-0.25, 1usize, 0usize)?;
    circuit.rzz(0.30, 0usize, 1usize)?;
    circuit.u3(0.15, -0.10, 0.40, 1usize)?;

    let observable = SparsePauliOp::single(2, 0usize, Pauli::Z, 1.0)?;
    let target = Tensor::new(-0.35_f64)?;
    let mut optimizer = Adam::new(0.06, 0.9, 0.999, 1e-8, 0.0)?;

    for epoch in 1..=150 {
        let simulator = StateVectorSimulator::new(circuit.num_qubits())?;
        let prediction = simulator.run(&circuit, &observable)?;
        let loss = mse_loss(&prediction, &target)?;
        loss.backward()?;
        optimizer.step(circuit.parameters())?;
        optimizer.zero_grad(circuit.parameters());

        if epoch == 1 || epoch % 25 == 0 {
            println!(
                "epoch {epoch:>3}: expectation = {:.6}, loss = {:.8}",
                scalar(&prediction)?,
                scalar(&loss)?,
            );
        }
    }

    save_weights(&circuit, &output_path)?;
    println!(
        "saved {} parameters to {}",
        circuit.num_parameters(),
        output_path.display()
    );
    Ok(())
}

/// 将单元素实数 Tensor 转换为便于日志输出的 f64。
fn scalar(tensor: &Tensor) -> std::result::Result<f64, Box<dyn std::error::Error>> {
    Ok(tensor.value()?)
}
