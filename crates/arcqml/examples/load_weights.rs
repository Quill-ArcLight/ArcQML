use arcqml::checkpoint::load_weights;
use arcqml::prelude::*;
use std::path::PathBuf;

/// 加载保存的参数，并输出对应电路的 Pauli-Z 期望值。
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let checkpoint_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("crates/arcqml/examples/trained_circuit.json"));

    // 必须重建与 train_and_save.rs 完全相同的电路结构及参数创建顺序。
    let mut circuit = Circuit::new(2)?;
    circuit.h(0usize)?;
    circuit.ry(0.20, 0usize)?;
    circuit.rz(-0.35, 0usize)?;
    circuit.cnot(0usize, 1usize)?;
    circuit.ry(0.45, 1usize)?;
    circuit.crx(-0.25, 1usize, 0usize)?;
    circuit.rzz(0.30, 0usize, 1usize)?;
    circuit.u3(0.15, -0.10, 0.40, 1usize)?;

    load_weights(&circuit, &checkpoint_path)?;

    let observable = SparsePauliOp::single(2, 0usize, Pauli::Z, 1.0)?;
    let simulator = StateVectorSimulator::new(circuit.num_qubits())?;
    let expectation = simulator.run(&circuit, &observable)?;
    println!(
        "loaded {} parameters from {}",
        circuit.num_parameters(),
        checkpoint_path.display()
    );
    println!("<Z0> = {:.6}", scalar(&expectation)?);
    Ok(())
}

/// 将单元素实数 Tensor 转换为便于日志输出的 f64。
fn scalar(tensor: &Tensor) -> std::result::Result<f64, Box<dyn std::error::Error>> {
    Ok(tensor.value()?)
}
