use arcqml_circuit::{Circuit, CircuitError, CircuitResult};
use std::{fs, path::PathBuf};

/// 演示将参数化电路保存为 JSON，并从同一文件恢复电路结构。
fn main() -> CircuitResult<()> {
    let mut circuit = Circuit::new(2)?;
    let theta = circuit.add_parameter(0.75)?;
    circuit.h(0)?;
    circuit.ry_param(theta, 1)?;
    circuit.cnot(0, 1)?;
    circuit.rz_fixed(0.25, 1)?;

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("json_circuit.json");
    let json = circuit.to_json()?;
    fs::write(&path, json).map_err(|error| CircuitError::JsonSerializationError {
        message: error.to_string(),
    })?;

    let json =
        fs::read_to_string(&path).map_err(|error| CircuitError::JsonDeserializationError {
            message: error.to_string(),
        })?;
    let restored = Circuit::from_json(&json)?;

    println!("JSON 文件: {}", path.display());
    println!("恢复后的量子比特数: {}", restored.num_qubits());
    println!("恢复后的操作数: {}", restored.len());
    println!(
        "恢复后的参数占位值: {}",
        restored.parameter_scalar_value(theta)?
    );
    Ok(())
}
