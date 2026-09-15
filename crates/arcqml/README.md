# arcqml

`arcqml` 是 ArcQML 的 Rust 门面 crate。普通应用优先依赖它：一个入口即可使用电路、状态向量模拟、可观测量、自动微分、损失函数、优化器、状态分析、检查点、酉矩阵和可视化能力。


## 添加依赖

当前发行方式使用本地路径依赖：

```toml
[dependencies]
arcqml = { path = "PATH_TO_ARCQML/crates/arcqml" }
```

编译前必须设置 `ARCQML_RUNTIME_LIB_DIR`。例如在发行包根目录使用 Windows PowerShell：

```powershell
$env:ARCQML_RUNTIME_LIB_DIR = (Resolve-Path ".\libs\x86_64-pc-windows-msvc").Path
cargo check -p arcqml
```

默认启用 `parallel` feature。若需要严格控制依赖，可关闭默认 feature：

```toml
arcqml = { path = "PATH_TO_ARCQML/crates/arcqml", default-features = false }
```

## 最小示例

```rust
use arcqml::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(2)?;
    circuit.h(0usize)?.cnot(0usize, 1usize)?;

    let observable = SparsePauliOp::single(2, 0usize, Pauli::Z, 1.0)?;
    let simulator = StateVectorSimulator::new(2)?;
    let expectation = simulator.run(&circuit, &observable)?;

    println!("<Z0> = {}", expectation.value()?);
    Ok(())
}
```

`prelude` 适合示例、应用和交互式探索。编写库时，如果希望依赖关系和名称来源更清晰，可以从具体模块按需导入：

```rust
use arcqml::circuit::Circuit;
use arcqml::observable::{Pauli, SparsePauliOp};
use arcqml::sim::StateVectorSimulator;
```

## 模块导航

| 模块 | 主要内容 |
| --- | --- |
| `arcqml::core` | `Tensor`、`Parameter`、自动微分与梯度模式 |
| `arcqml::linalg` | 可微逐元素运算、线性代数、归约与广播 |
| `arcqml::circuit` | `Circuit`、`Gate`、`Qubit`、参数引用与拼接 |
| `arcqml::observable` | `PauliString`、`SparsePauliOp`、`Hamiltonian` |
| `arcqml::sim` | 单态与 batch 状态向量模拟器 |
| `arcqml::analysis` | 概率、边缘概率、保真度与 Bloch 向量 |
| `arcqml::loss` | MSE、L1、二元 NLL 与交叉熵 |
| `arcqml::optim` | SGD、Adam 与更新统计 |
| `arcqml::checkpoint` | 电路参数权重的 JSON 保存与加载 |
| `arcqml::unitary` | 整体稠密酉矩阵、保真度和拟合 |
| `arcqml::visualization` | Unicode 文本电路图与 SVG |

`init_rayon` 与 `rayon_num_threads` 还会直接从 crate 根导出。

## 一次完整训练步骤

```rust
use arcqml::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(1)?;
    circuit.ry(0.3, 0usize)?;

    let observable = SparsePauliOp::single(1, 0usize, Pauli::Z, 1.0)?;
    let simulator = StateVectorSimulator::new(1)?;
    let target = Tensor::new(0.2_f64)?;
    let mut optimizer = Adam::new(0.05, 0.9, 0.999, 1e-8, 0.0)?;

    optimizer.zero_grad(circuit.parameters());
    let prediction = simulator.run(&circuit, &observable)?;
    let loss = mse_loss(&prediction, &target)?;
    loss.backward()?;
    optimizer.step(circuit.parameters())?;

    println!("loss = {}", loss.value()?);
    Ok(())
}
```

量子期望值为：

$$
f(\theta)=\langle\psi|U(\theta)^\dagger H U(\theta)|\psi\rangle
$$

`run` 把整条量子电路作为一个可微节点接入 Tensor 计算图。`backward()` 将上游梯度继续传到电路参数。

## 权重与电路结构

参数权重和电路结构是两个不同概念：

- `Circuit::to_json` / `Circuit::from_json` 保存电路拓扑与参数名称，但不保存参数当前值；加载后参数值初始化为零。
- `save_weights` / `load_weights` 保存参数名称、dtype 与值，但不保存门拓扑。

```rust
use arcqml::checkpoint::{load_weights, save_weights};

save_weights(&circuit, "weights.json")?;
load_weights(&restored_circuit, "weights.json")?;
```

当前代码没有保存 Adam 动量或完整训练状态的 `save_checkpoint` / `load_checkpoint` API。需要恢复训练时，应由应用另外保存优化器状态、训练步数和随机状态。

