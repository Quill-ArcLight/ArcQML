use super::single::StateVectorSimulator;
use crate::{SimError, SimResult, statevector::state::single as state};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::collections::BTreeMap;

/// 末端全量子位测量的结果计数。
pub type MeasurementCounts = BTreeMap<String, usize>;

impl StateVectorSimulator {
    /// 从当前状态的全量子比特计算基分布中抽样，返回各位串的出现次数。
    ///
    /// 此方法不坍缩或修改模拟器状态。位串按 `q{n-1}…q0` 显示；相同的 `seed`
    /// 对相同状态给出可复现结果，`None` 则使用随机种子。
    ///
    /// # Errors
    ///
    /// 当 `shots` 为零，或内部状态的 dtype、布局、形状、连续性或归一化无效时返回错误。
    pub fn sample_counts(&self, shots: usize, seed: Option<u64>) -> SimResult<MeasurementCounts> {
        if shots == 0 {
            return Err(SimError::InvalidShotsError);
        }

        let probabilities = state::tensor_amplitudes(self.num_qubits(), &self.state)?
            .into_iter()
            .map(|amplitude| amplitude.norm_sqr())
            .collect::<Vec<_>>();
        let total_probability: f64 = probabilities.iter().sum();
        let seed = seed.unwrap_or_else(rand::random);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut counts = MeasurementCounts::new();

        for _ in 0..shots {
            let basis_index = sample_basis_index(&probabilities, total_probability, &mut rng);
            *counts
                .entry(format_basis_state(basis_index, self.num_qubits()))
                .or_default() += 1;
        }

        Ok(counts)
    }
}

fn sample_basis_index(probabilities: &[f64], total_probability: f64, rng: &mut StdRng) -> usize {
    let target = rng.random::<f64>() * total_probability;
    let mut cumulative = 0.0;

    for (index, probability) in probabilities.iter().copied().enumerate() {
        cumulative += probability;
        if target < cumulative {
            return index;
        }
    }

    // 归一化校验允许微小浮点误差；累计误差尾部统一归入最后一个基态。
    probabilities.len() - 1
}

fn format_basis_state(basis_index: usize, num_qubits: usize) -> String {
    format!("{basis_index:0width$b}", width = num_qubits)
}
