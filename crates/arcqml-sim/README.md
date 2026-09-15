# arcqml-sim

`arcqml-sim` 提供理想纯态量子电路的状态向量模拟器。`StateVectorSimulator` 处理一个状态，`BatchStateVectorSimulator` 让一批独立初态共享同一条电路与参数。


## 添加依赖

```toml
[dependencies]
arcqml-circuit = { path = "PATH_TO_ARCQML/crates/arcqml-circuit" }
arcqml-observable = { path = "PATH_TO_ARCQML/crates/arcqml-observable" }
arcqml-sim = { path = "PATH_TO_ARCQML/crates/arcqml-sim" }
```

构建前设置 `ARCQML_RUNTIME_LIB_DIR`，或直接依赖门面 crate `arcqml`。

## 单态模拟器

```rust
use arcqml_circuit::Circuit;
use arcqml_sim::StateVectorSimulator;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(2)?;
    circuit.h(0usize)?.cnot(0usize, 1usize)?;

    let mut simulator = StateVectorSimulator::new(2)?;
    simulator.apply_circuit(&circuit)?;

    assert_eq!(simulator.amplitudes()?.shape(), &[4]);
    println!("{:?}", simulator.sample_counts(1_000, Some(42))?);
    Ok(())
}
```

`new(n)` 创建零态：

$$
|0\rangle^{\otimes n}
$$

`from_state_tensor` 可导入自定义初态。输入必须是长度为 `2^n`、连续、归一化的 `C64` Tensor。

## 两类执行方式

### 修改当前状态

`apply_gate`、`apply_operation` 与 `apply_circuit` 会原地更新模拟器状态。随后可读取振幅、期望值或抽样。

### 非修改式可微运行

```rust
let output = simulator.run(&circuit, &observable)?;
output.backward()?;
```

`run` 从模拟器当前状态开始执行给定电路，返回可微期望值，但不修改 `simulator`。单态输出是 `F64` 标量 Tensor。

如果当前状态或电路参数需要梯度，`run` 会建立一个量子自定义节点，并在反向传播时调用伴随算法。期望值为：

$$
f(\theta)=
\langle\psi|
U(\theta)^\dagger H U(\theta)
|\psi\rangle
$$

## API 速查

| 接口 | 作用 | 是否修改状态 |
| --- | --- | :---: |
| `new` / `from_state_tensor` | 创建模拟器 | — |
| `reset` | 恢复每条状态为零态 | 是 |
| `apply_operation` / `apply_gate` | 执行一个操作或门 | 是 |
| `apply_circuit` | 执行整条电路 | 是 |
| `amplitudes` | 返回保留自动微分关系的 `C64` Tensor | 否 |
| `expectation_pauli_string` | 当前态的单项期望值 | 否 |
| `expectation_observable` / `expectation_hamiltonian` | 当前态的 Pauli 和期望值 | 否 |
| `run` | 当前态出发的非修改式可微电路运行 | 否 |
| `sample_counts` | 当前单态的末端全量子比特抽样 | 否 |

## BatchStateVectorSimulator

batch 初态 Tensor 形状为：

$$
[B,2^n]
$$

每一行必须独立归一化并采用连续 row-major 存储。

```rust
use arcqml_sim::BatchStateVectorSimulator;

let simulator = BatchStateVectorSimulator::from_state_tensor(
    num_qubits,
    initial_states,
)?;
let values = simulator.run(&circuit, &observable)?;
assert_eq!(values.shape(), &[simulator.batch_size()]);
```

batch `run` 输出 `[B]` Tensor。整条量子电路只占自动微分图中的一个节点；反向传播可同时累积初始 batch 状态和电路参数的梯度。没有任何父节点需要梯度时，不保留伴随上下文。

## 抽样与端序

`sample_counts(shots, seed)` 只属于单态模拟器。`shots` 必须大于零；固定 seed 可复现实验。

- 内部基态由大到小排序，`q0` 是最低有效位。
- 输出 bitstring 按 `q[n-1]...q[0]` 显示。
- 两量子比特系统只执行 `x(0)` 时，抽样键为 `"01"`。
- 抽样不执行电路、不修改状态，也不模拟坍缩。

## TODO

- 添加支持 GPU、噪声、密度矩阵、量子信道、中途测量、经典条件控制。
- 添加 batch 模拟器抽样接口。

概率、边缘概率、保真度和 Bloch 向量位于 [`arcqml-analysis`](../arcqml-analysis/README.md)。
