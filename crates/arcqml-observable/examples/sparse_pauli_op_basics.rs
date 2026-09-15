use arcqml_observable::prelude::*;

fn main() -> ObservableResult<()> {
    // 构造 Pauli 项
    let z0 = PauliString::z(2, 0usize)?;
    let x1 = PauliString::x(2, 1usize)?;
    // 这一项同时作用于两个 qubit
    let z0_x1 = PauliString::new(
        2,
        vec![
            PauliOp::new(0usize, Pauli::Z),
            PauliOp::new(1usize, Pauli::X),
        ],
    )?;

    // SparsePauliOp 表示多项 PauliString 的线性组合
    let mut observable = SparsePauliOp::zero(2)?;
    // 这里重复加入 Z(0)，后面通过 simplify 合并同类项
    observable
        .add_pauli_string(0.5, z0.clone())?
        .add_pauli_string(1.0, z0)?
        .add_pauli_string(-1.2, x1)?
        .add_pauli_string(0.25, z0_x1)?;

    println!("terms before simplify = {}", observable.len());

    // 合并相同 PauliString 项
    let simplified = observable.simplify(1.0e-12)?;

    println!("terms after simplify = {}", simplified.len());

    // 遍历化简后的每一项，PauliString 内只统计非 I 的算符
    for (index, term) in simplified.terms().iter().enumerate() {
        println!(
            "term {}: coefficient = {}, non_identity_ops = {}",
            index,
            term.coefficient(),
            term.pauli_string().len()
        );
    }

    Ok(())
}
