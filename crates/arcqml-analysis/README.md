# arcqml-analysis

`arcqml-analysis` 从 `StateVectorSimulator` 的当前纯态计算精确概率、边缘概率、纯态保真度和单量子比特 Bloch 向量。所有函数都为只读。

## 添加依赖

```toml
[dependencies]
arcqml-analysis = { path = "PATH_TO_ARCQML/crates/arcqml-analysis" }
arcqml-circuit = { path = "PATH_TO_ARCQML/crates/arcqml-circuit" }
arcqml-sim = { path = "PATH_TO_ARCQML/crates/arcqml-sim" }
```

## 最小示例

```rust
use arcqml_analysis::{bloch_vector, marginal_probabilities, probabilities};
use arcqml_circuit::Circuit;
use arcqml_sim::StateVectorSimulator;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(2)?;
    circuit.h(0usize)?.cnot(0usize, 1usize)?;

    let mut simulator = StateVectorSimulator::new(2)?;
    simulator.apply_circuit(&circuit)?;

    println!("P = {:?}", probabilities(&simulator)?);
    println!("P(q0) = {:?}", marginal_probabilities(&simulator, &[0])?);
    println!("Bloch(q0) = {:?}", bloch_vector(&simulator, 0usize)?);
    Ok(())
}
```

## 概率和边缘概率

对状态振幅，计算基概率为：

$$
p_j=|\psi_j|^2
$$

`probabilities` 返回形状为 `[2^n]` 的可微 `F64` Tensor，数组下标与模拟器内部量子比特排序方式一致。Bell 态结果为 `[0.5, 0.0, 0.0, 0.5]`。

`marginal_probabilities(simulator, qubits)` 对未选择的量子比特求和，返回形状为 `[2^k]` 的可微 Tensor。输出基位始终按量子比特编号从高到低排列；例如选择 `[0, 2]` 时，四个位置对应 `q2q0 = 00, 01, 10, 11`。

## 纯态保真度

`fidelity(lhs, rhs)` 返回可微 `F64` 标量：

$$
F(|\psi\rangle,|\phi\rangle)
=
|\langle\psi|\phi\rangle|^2
$$

两个模拟器的量子比特数必须相同。数学上保真度位于 `[0, 1]`；实现为保持前向和反向表达式一致，不对浮点结果强制截断，因此极端舍入情况下可能略微越界。

## Bloch 向量

`bloch_vector(simulator, qubit)` 返回普通 Rust 结构体 `BlochVector { x, y, z }`：

$$
\vec{r}=
(\langle X\rangle,\langle Y\rangle,\langle Z\rangle)
$$

单量子比特纯态的向量长度为一；纠缠多量子比特状态的单比特约化态可能位于 Bloch 球内部。Bloch 结果是 `f64` 结构体，不属于自动微分图；概率、边缘概率与保真度返回的 Tensor 则保留自动微分关系。

## 错误与边界

- 量子比特索引越界、重复选择或状态维度不匹配会返回 `AnalysisError`。
- 本 crate 只支持单态 `StateVectorSimulator`，不提供 batch 分析函数。
- 这些接口计算精确结果；需要有限 shots 的统计结果时使用单态模拟器的 `sample_counts`。

模拟器语义见 [`arcqml-sim`](../arcqml-sim/README.md)。许可证见 [混合许可说明](../../README.md#发行状态与许可证)。
