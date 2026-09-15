use arcqml_observable::{ObservableError, ObservableResult, SparsePauliOp};
use std::{fs, path::PathBuf};

/// 演示从 examples 目录中的 JSON 文件读取 SparsePauliOp 哈密顿量。
fn main() -> ObservableResult<()> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("sparse_pauli_op.json");
    let json =
        fs::read_to_string(&path).map_err(|error| ObservableError::JsonDeserializationError {
            message: error.to_string(),
        })?;
    let hamiltonian = SparsePauliOp::from_json(&json)?;

    println!("JSON 文件: {}", path.display());
    println!("量子比特数: {}", hamiltonian.num_qubits());
    println!("Pauli 项数: {}", hamiltonian.len());
    for (index, term) in hamiltonian.terms().iter().enumerate() {
        println!("第 {index} 项系数: {}", term.coefficient());
    }
    Ok(())
}
