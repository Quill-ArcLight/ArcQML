use arcqml_observable::prelude::*;

fn main() -> ObservableResult<()> {
    // 创建一个 3 qubit PauliString，输入顺序可以任意
    // PauliString::new 会自动按 qubit 编号排序，并且不会存储 I 项
    let string = PauliString::new(
        3,
        vec![
            PauliOp::new(2usize, Pauli::Z),
            PauliOp::new(0usize, Pauli::X),
            PauliOp::new(1usize, Pauli::I),
        ],
    )?;

    println!("num_qubits = {}", string.num_qubits());
    println!("stored_ops = {}", string.len());
    println!("pauli_on_q0 = {}", string.pauli_on(0usize)?.name());
    println!("pauli_on_q1 = {}", string.pauli_on(1usize)?.name());
    println!("pauli_on_q2 = {}", string.pauli_on(2usize)?.name());

    // x、y、z 是创建单项 PauliString 的快捷方法
    let x0 = PauliString::x(2, 0usize)?; // IX
    let y0 = PauliString::y(2, 0usize)?;
    let z1 = PauliString::z(2, 1usize)?;

    // PauliString 相乘会返回整体相位和相乘后的新 PauliString
    // 这里 X(0) * Y(0) = i * Z(0)
    let (phase, product) = x0.multiply(&y0)?;

    println!("X0 * Y0 phase = {}", phase);
    println!(
        "X0 * Y0 result on q0 = {}",
        product.pauli_on(0usize)?.name()
    );
    // 作用在不同 qubit 上的 X(0) 与 Z(1) 可以交换
    println!("X0 commutes with Z1 = {}", x0.commutes_with(&z1)?);

    Ok(())
}
