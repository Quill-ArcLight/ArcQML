use arcqml_observable::{Pauli, PauliOp, PauliString, PauliTerm, SparsePauliOp};
use num_complex::Complex64;

/// 断言两个复振幅在浮点误差范围内相等。
fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1.0e-12,
        "actual={actual:?}, expected={expected:?}"
    );
}

/// 验证混合 `X`/`Y`/`Z` 项的紧凑执行计划与手算状态作用一致。
#[test]
fn compiled_hamiltonian_should_apply_mixed_pauli_terms() {
    let scale = 1.0 / 2.0_f64.sqrt();
    let state = vec![Complex64::new(scale, 0.0), Complex64::new(0.0, scale)];
    let mut observable = SparsePauliOp::constant(1, 0.5).unwrap();
    observable
        .add_pauli_string(2.0, PauliString::single(1, 0usize, Pauli::X).unwrap())
        .unwrap();
    observable
        .add_pauli_string(-3.0, PauliString::single(1, 0usize, Pauli::Y).unwrap())
        .unwrap();
    observable
        .add_pauli_string(4.0, PauliString::single(1, 0usize, Pauli::Z).unwrap())
        .unwrap();

    let (value, hamiltonian_state) = observable
        .statevector_expectation_and_apply(&state)
        .unwrap();

    assert!((value + 2.5).abs() < 1.0e-12);
    assert_complex_close(
        hamiltonian_state[0],
        Complex64::new(1.5 * scale, 2.0 * scale),
    );
    assert_complex_close(
        hamiltonian_state[1],
        Complex64::new(2.0 * scale, -6.5 * scale),
    );
}

/// 验证 `add_term` 会失效缓存，下一次计算使用更新后的 Hamiltonian。
#[test]
fn sparse_pauli_op_should_recompile_after_add_term() {
    let scale = 1.0 / 2.0_f64.sqrt();
    let state = vec![Complex64::new(scale, 0.0), Complex64::new(0.0, scale)];
    let mut observable = SparsePauliOp::single(1, 0usize, Pauli::Z, 1.0).unwrap();

    assert!(observable.statevector_expectation(&state).unwrap().abs() < 1.0e-12);
    observable
        .add_term(
            PauliTerm::new(
                1.0,
                PauliString::new(1, vec![PauliOp::new(0usize, Pauli::Y)]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

    assert!((observable.statevector_expectation(&state).unwrap() - 1.0).abs() < 1.0e-12);
}
