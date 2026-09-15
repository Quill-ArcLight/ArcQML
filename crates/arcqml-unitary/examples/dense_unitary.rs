use arcqml_circuit::prelude::*;
use arcqml_unitary::prelude::*;
use std::{env, error::Error};

/// 读取命令行指定的量子位数量，默认使用四量子位电路。
fn parse_num_qubits() -> Result<usize, Box<dyn Error>> {
    let Some(value) = env::args().nth(1) else {
        return Ok(4);
    };

    let num_qubits = value.parse::<usize>()?;
    if num_qubits == 0 {
        return Err("the number of qubits must be positive".into());
    }

    Ok(num_qubits)
}

/// 构造包含单量子位旋转和相邻纠缠门的演示电路。
fn build_circuit(num_qubits: usize) -> Result<Circuit, Box<dyn Error>> {
    let mut circuit = Circuit::new(num_qubits)?;

    for qubit in 0..num_qubits {
        circuit.h(qubit)?;
        circuit.ry_fixed((qubit + 1) as f64 * 0.17, qubit)?;
    }

    for control in 0..(num_qubits - 1) {
        circuit.cnot(control, control + 1)?;
    }

    circuit.phase_fixed(std::f64::consts::PI / 7.0, num_qubits - 1)?;
    Ok(circuit)
}

/// 将行主序 C64 酉矩阵逐行输出到标准输出。
fn print_unitary(unitary: &DenseUnitary) -> Result<(), Box<dyn Error>> {
    let dimension = unitary.dimension();
    let values = unitary.as_row_major();

    println!("U has shape [{dimension}, {dimension}]:");
    for row in 0..dimension {
        for column in 0..dimension {
            let value = values[row * dimension + column];
            print!("{:+.6}{:+.6}i ", value.re, value.im);
        }
        println!();
    }

    Ok(())
}

/// 构造电路、计算整体酉矩阵并输出矩阵元素。
fn main() -> Result<(), Box<dyn Error>> {
    let num_qubits = parse_num_qubits()?;
    let circuit = build_circuit(num_qubits)?;
    let unitary = unitary_from_circuit(&circuit)?;

    println!(
        "Built a {num_qubits}-qubit circuit with {} operations.",
        circuit.len()
    );
    print_unitary(&unitary)
}
