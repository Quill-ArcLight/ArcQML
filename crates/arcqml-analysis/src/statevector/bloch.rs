use crate::{AnalysisError, AnalysisResult};
use arcqml_sim::StateVectorSimulator;

/// 单量子位的 Bloch 球坐标。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlochVector {
    /// Pauli-X 期望值。
    pub x: f64,
    /// Pauli-Y 期望值。
    pub y: f64,
    /// Pauli-Z 期望值。
    pub z: f64,
}

/// 计算指定量子比特约化态的 Bloch 向量。
///
/// 返回分量 `(⟨X⟩, ⟨Y⟩, ⟨Z⟩)`。计算直接读取模拟器当前状态，不修改状态，也不进行
/// 抽样。
///
/// # Errors
///
/// 当 `qubit` 不在 `0..simulator.num_qubits()` 范围内，或无法读取模拟器振幅时返回错误。
pub fn bloch_vector(simulator: &StateVectorSimulator, qubit: usize) -> AnalysisResult<BlochVector> {
    if qubit >= simulator.num_qubits() {
        return Err(AnalysisError::QubitOutOfRangeError {
            index: qubit,
            num_qubits: simulator.num_qubits(),
        });
    }

    let state = simulator.amplitudes()?;
    let storage = state.storage();
    let arcqml_core::Storage::C64(amplitudes) = &*storage else {
        unreachable!("StateVectorSimulator::amplitudes always returns C64 tensors")
    };
    let mask = 1usize << qubit;
    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;

    for index in 0..amplitudes.len() {
        if index & mask != 0 {
            continue;
        }
        let zero = amplitudes[index];
        let one = amplitudes[index | mask];
        let coherence = zero.conj() * one;
        x += 2.0 * coherence.re;
        y += 2.0 * coherence.im;
        z += zero.norm_sqr() - one.norm_sqr();
    }

    Ok(BlochVector { x, y, z })
}
