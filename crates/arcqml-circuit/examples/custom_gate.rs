use arcqml_circuit::prelude::*;
use num_complex::Complex64;

fn main() -> CircuitResult<()> {
    // 一个作用于 1 个 qubit 的自定义 X 门，矩阵为 2 x 2，所以需要 4 个元素
    let custom_x = Gate::custom_unitary(
        "custom_x",
        1,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )?;

    let mut circuit = Circuit::new(1)?;
    circuit.add_gate(custom_x, vec![Qubit::new(0)])?;
    circuit.validate()?;

    let operation = &circuit.operations()[0];
    println!("gate: {}", operation.gate().name());
    println!("arity: {}", operation.gate().arity());
    println!("qubits: {:?}", operation.qubits());

    Ok(())
}
