# ArcQML

**中文** | [English](README_en.md)

ArcQML 是一个用 Rust 语言原生实现的量子机器学习框架，提供了量子电路构建、状态向量模拟、可观测量期望值计算、自动微分和参数优化等核心能力。框架既支持单个量子态的计算，也支持批量状态模拟，并可使用 Pauli 算符及其线性组合表示可观测量。量子电路计算得到的期望值以可微分张量形式输出，可以继续参与损失函数及其他经典可微运算；执行反向传播时，梯度能够沿完整计算图传递至量子电路参数。对于含参量子电路，ArcQML 可以利用伴随法计算电路参数梯度，再与 Adam、SGD 等经典优化器配合完成混合量子—经典算法的训练。

这些能力可以用于变分量子本征求解、量子神经网络、量子态分析和小规模酉矩阵综合等任务。框架使用 Rust 原生编译与 CPU 并行。当前版本仅支持在 CPU 上进行模拟，后续计划增加对 GPU 的支持。本预览版的低层接口存在已知内存安全限制，见 [SECURITY.md](SECURITY.md)。

> 当前版本：`0.1.0`（Windows 实验性预览版）。ArcQML 采用混合许可，包含公开源码与闭源 Runtime。请先阅读[发行状态与许可证](#发行状态与许可证)，再决定是否用于公开分发或生产环境。

## 为什么使用 ArcQML

- **一条可微计算链**：量子期望值以 `Tensor` 返回，可继续接损失函数并调用 `backward()`。
- **Rust 与 Python 双接口**：Rust 提供完整能力，Python 原生扩展覆盖常用的电路、模拟、分析和训练流程。
- **单态与批量执行**：同一条参数化电路可作用于一个初态或一批行主序初态。
- **适合教学与实验**：内置常用量子门、Pauli Hamiltonian、SGD、Adam、状态分析、电路图和权重保存。

可以先把 ArcQML 理解为四层：

1. `Circuit` 描述“要执行哪些量子门”；
2. `(Batch)StateVectorSimulator` 描述“从哪个量子态开始计算”；
3. `SparsePauliOp` 描述“最后观察什么物理量”；
4. `Tensor`、`loss` 与 `optimizer` 负责“如何根据结果更新参数”。

## 30 秒上手：Python

当前发行包提供的 wheel 适用于 **CPython 3.11、Windows x86_64**：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ./wheels/x86_64-pc-windows-msvc/arcqml-0.1.0-cp311-cp311-win_amd64.whl
```

安装后可用以下命令验证：

```bash
python -c "import arcqml; print(arcqml.__version__)"
```

下面的程序构造 Bell 态，读取精确概率并做末端抽样：

```python
import arcqml

circuit = arcqml.Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)

simulator = arcqml.StateVectorSimulator(2)
simulator.apply_circuit(circuit)

probabilities = arcqml.analysis.probabilities(simulator).numpy()
counts = simulator.sample_counts(1_000, seed=42)

print(probabilities)  # 约为 [0.5, 0.0, 0.0, 0.5]
print(counts)         # 只会出现 "00" 和 "11"
```

Linux、其他 Python 版本、macOS 与 ARM 平台不在本次预览版验证范围内，未随发行快照提供预编译 wheel。需要从源码构建时，请参阅 [Python README](python/README.md)。

## 30 秒上手：Rust

Rust 接口使用 Rust 2024 edition，本次预编译 Runtime 要求 Rust 1.98.0 工具链。应用项目通过路径依赖使用本发行包：

```toml
[dependencies]
arcqml = { path = "PATH_TO_ARCQML/crates/arcqml" }
```

构建前必须把 `ARCQML_RUNTIME_LIB_DIR` 指向与目标三元组匹配的 Runtime 库目录。

Windows PowerShell：

```powershell
$env:RUSTUP_TOOLCHAIN = "1.98.0"
$env:ARCQML_RUNTIME_LIB_DIR = "PATH_TO_ARCQML\libs\x86_64-pc-windows-msvc"
cargo run
```

Linux（后续平台说明，本次未附 Linux 二进制）：

```bash
export ARCQML_RUNTIME_LIB_DIR="PATH_TO_ARCQML/libs/x86_64-unknown-linux-gnu"
cargo run
```

最小程序：

```rust
use arcqml::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(2)?;
    circuit.h(0usize)?.cnot(0usize, 1usize)?;

    let mut simulator = StateVectorSimulator::new(2)?;
    simulator.apply_circuit(&circuit)?;

    println!("amplitudes = {:?}", simulator.amplitudes()?);
    println!("probabilities = {:?}", probabilities(&simulator)?);
    println!("counts = {:?}", simulator.sample_counts(1_000, Some(42))?);
    Ok(())
}
```

## 第一个可训练电路

下面让一个单量子比特旋转门的 Pauli-Z 期望值靠近目标值。`ry` 创建可训练参数；如需创建固定角度应改用 Rust API 中的 `ry_fixed`。

```python
import arcqml

circuit = arcqml.Circuit(1)
circuit.ry(0.3, 0)

observable = arcqml.PauliSum.z(1, 0)
simulator = arcqml.StateVectorSimulator(1)
optimizer = arcqml.Adam(learning_rate=0.05)
target = arcqml.tensor(0.2)

for step in range(20):
    prediction = simulator.run(circuit, observable)
    loss = arcqml.mse_loss(prediction, target)
    loss.backward()
    optimizer.step(circuit)
    optimizer.zero_grad(circuit)

print("loss =", loss.item())
print("parameters =", circuit.parameter_values())
```

ArcQML 的均方误差采用半均方定义：

$$
L = \frac{1}{2N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$

`backward()` 会累积梯度，所以每轮训练都需要清空旧梯度。`no_grad()` 可用于验证和推理，避免创建自动微分图。

## 功能总览

| 能力 | Rust | Python | 主要入口 |
| --- | :---: | :---: | --- |
| 参数化量子电路 | 完整门集 | 常用门子集 | `Circuit` |
| 单态状态向量模拟 | ✓ | ✓ | `StateVectorSimulator` |
| 行主序 batch 模拟 | ✓ | ✓ | `BatchStateVectorSimulator` |
| Pauli 和期望值 | ✓ | ✓ | `SparsePauliOp` / `PauliSum` |
| 量子伴随梯度 | ✓ | ✓ | `run` + `Tensor.backward` |
| Tensor 与线性代数 | ✓ | 基础 Tensor | `arcqml-core` / `arcqml-linalg` |
| 损失函数 | ✓ | 部分 | `arcqml-loss` |
| 优化器 | ✓ | 部分 | `arcqml-optim` |
| 概率、边缘概率、保真度、Bloch 向量 | ✓ | ✓ | `arcqml-analysis` |
| 末端抽样 | 仅单态 | 仅单态 | `sample_counts` |
| 电路文本图与 SVG | ✓ | — | `arcqml-visualization` |
| 电路结构 JSON | ✓ | — | `Circuit::to_json` / `from_json` |
| 参数权重 JSON | ✓ | — | `save_weights` / `load_weights` |
| 整体酉矩阵与拟合 | ✓ | — | `arcqml-unitary` |

## 重要约定

- 量子比特从 `0` 开始编号。
- 内部状态索引采用从大到小排序，`q0` 是最低有效位。
- 抽样字符串按 `q[n-1]...q[0]` 展示；两量子比特系统只执行 `x(0)` 时得到的是 `"01"`。
- `apply_circuit` 会修改模拟器当前状态；`run` 从当前状态计算但不修改模拟器。
- `sample_counts` 只对当前单态状态抽样，不执行电路，也不造成状态坍缩。
- batch 状态 Tensor 形状为 `[batch_size, 2^num_qubits]`，每行必须归一化并采用连续行主序存储。
- `Tensor::clone()` 复制共享句柄；需要独立数据时使用 `deep_clone()`。

## TODO

- CUDA、GPU 后端和分布式执行；
- 稀疏态、张量网络或矩阵乘积态模拟；
- 噪声模型、密度矩阵、量子信道；
- 中途测量、态坍缩、经典条件控制；
- 量子真机或云后端接入；
- batch 抽样；
- Python 侧的完整 Rust 门集、电路 JSON、权重检查点和酉矩阵 API。

## 示例与文档

- [技术手册](docs/ArcQML技术手册.md)：数学约定、数据布局、自动微分与错误排查。
- [H₂ VQE 教程](docs/tutorial/h2_vqe.md)：量子化学中的变分基态求解。
- [QNN 信用分类教程](docs/tutorial/qnn_german_credit.md)：批量量子神经网络训练。
- [自定义损失教程](docs/tutorial/custom_loss.md)：组合 Rust 算子、验证梯度、接入训练与编写专用反向。
- [Rust 示例](examples/rust) 与 [Python 示例](examples/python)。

## 发行状态与许可证

ArcQML 采用混合许可，这是源码可用发行，不是 OSI 认可的开源许可：

- 自有公开源码采用 [ArcQML 非商业源码许可](LICENSE)：只允许非商业用途，修改和集成作品需依条款公开完整源码；商业使用另行授权。
- 官方闭源 Runtime 采用 [非商业二进制许可](LICENSE-RUNTIME)。公开源码许可为未经修改的官方 Runtime 设置例外，不要求公开其私有实现；用户集成代码仍需公开。
- wheel 中的公开框架、闭源 Runtime 和第三方组件分别适用各自许可。第三方材料保留原许可，见 [第三方说明](THIRD_PARTY_NOTICES.md) 及包内版本化许可正文与 SBOM。
- 以前合法取得的 MIT/Apache 许可副本，其已授予权利不因本次发行政策变更而撤回。

维护人：liuxl；商业授权及许可联系：quill@arclightquantum.com。

本次为 Windows 实验性预览版，使用 Rust 1.98.0 和 CPython 3.11。内存安全问题 S01、Linux 验收和格式整理暂缓，详见 [发行说明](RELEASE_NOTES.md) 与 [安全限制](SECURITY.md)。仅使用本次随附的库和 wheel，勿混用历史产物。
