# ArcQML Python

ArcQML Python 是基于 PyO3 的原生扩展，为 Python 用户提供可微量子电路、CPU 纯态模拟、状态分析和 Adam 训练接口。量子电路的输出是 ArcQML `Tensor`：先构造 loss，再调用 `loss.backward()`，梯度就会回传到电路参数。

> 当前版本：`0.1.0`。Python API 是 Rust 完整 API 的常用功能子集。

## 安装

### 使用发行包 wheel

当前随仓库提供的 wheel 仅支持 **CPython 3.11、Windows x86_64**。以下命令在仓库根目录执行：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ./wheels/x86_64-pc-windows-msvc/arcqml-0.1.0-cp311-cp311-win_amd64.whl
```

官方 wheel 已包含匹配的 ArcQML Runtime，更多版本将在后续放出。

### 从源码构建

源码声明支持 Python `3.9` 或更高版本。还需要 Rust 1.98.0、Maturin，以及与当前平台和 ArcQML 版本匹配的 Runtime 静态库。以下命令同样在仓库根目录执行：

```powershell
$env:RUSTUP_TOOLCHAIN = "1.98.0"
$env:ARCQML_RUNTIME_LIB_DIR = (Resolve-Path ".\libs\x86_64-pc-windows-msvc").Path
python -m pip install "maturin>=1.7,<2.0"
maturin develop --release
```

验证安装：

```bash
python -c "import arcqml; print(arcqml.__version__)"
```

## 第一个量子程序

```python
import arcqml

circuit = arcqml.Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)

simulator = arcqml.StateVectorSimulator(2)
simulator.apply_circuit(circuit)

print(simulator.amplitudes().numpy())
print(arcqml.analysis.probabilities(simulator).numpy())
print(simulator.sample_counts(1_000, seed=42))
```

这个程序得到 Bell 态：精确概率集中在 `00` 与 `11`。`apply_circuit` 修改模拟器当前状态；`sample_counts` 只读取该状态，不会重新执行电路或造成状态坍缩。

## 最小训练循环

```python
import arcqml

circuit = arcqml.Circuit(1)
circuit.ry(0.3, 0)
observable = arcqml.PauliSum.z(1, 0)
simulator = arcqml.StateVectorSimulator(1)
optimizer = arcqml.Adam(learning_rate=0.05)
target = arcqml.tensor(0.2)

for epoch in range(20):
    optimizer.zero_grad(circuit)
    prediction = simulator.run(circuit, observable)
    loss = arcqml.mse_loss(prediction, target)
    loss.backward()
    optimizer.step(circuit)

print(loss.item())
print(circuit.parameter_values())
```

推荐顺序是：

```text
zero_grad → run → loss → backward → step
```

梯度会累积，所以不要省略 `zero_grad`。验证或推理时使用：

```python
with arcqml.no_grad():
    prediction = simulator.run(circuit, observable)
```

## API 总览

| 对象或函数 | 用途 |
| --- | --- |
| `Tensor` / `tensor` | 数值、动态自动微分图、梯度与 NumPy 转换 |
| `no_grad` | 临时关闭自动微分图记录 |
| `Circuit` | 构造固定门与参数化量子电路 |
| `PauliSum` | 构造实系数 Pauli Hamiltonian |
| `StateVectorSimulator` | 单态演化、期望值、振幅和抽样 |
| `BatchStateVectorSimulator` | 一批初态共享同一电路的演化与期望值 |
| `mse_loss` | 半均方误差 |
| `binary_cross_entropy_with_logits` | 数值稳定的二元 logits 交叉熵 |
| `Adam` | 更新电路参数 |
| `analysis` | 概率、边缘概率、保真度与 Bloch 向量 |
| `init_rayon` / `rayon_num_threads` | 初始化或读取 CPU 并行线程数 |

### `Circuit`

当前 Python 门集为：

- 固定门：`h`、`x`、`y`、`z`；
- 可训练单比特门：`rx`、`ry`、`rz`、`phase`、`u3`；
- 双比特门：`cnot`。

`num_qubits` 与 `num_parameters` 是只读属性。`parameter_values()` 返回参数名字典；`gradients()` 只有在反向传播已产生梯度后才能调用。

### `Tensor`

`arcqml.tensor(data, requires_grad=False)` 接受 Python 数值、数值列表、已有 `Tensor`，以及 C 连续的 NumPy `float64` 或 `complex128` 数组。

常用接口：

- `shape`、`dtype`、`requires_grad`、`is_leaf`；
- `backward(gradient=None)`、`grad()`、`detach()`；
- `numpy()` 与标量专用的 `item()`。

`backward()` 直接用于标量；非标量 Tensor 必须传入形状相同的上游梯度。`grad()` 在没有梯度时返回 `None`。

### `PauliSum`

```python
observable = arcqml.PauliSum.z(2, 0, coefficient=0.6)
observable.add_x(1, coefficient=0.25)
observable.add_term("YZ", [0, 1], coefficient=-0.2)
observable.add_identity(coefficient=0.1)
```

`add_term(paulis, qubits)` 中两个序列长度必须相同，Pauli 字符只允许 `I`、`X`、`Y`、`Z`，量子比特不能重复或越界。

### 单态与 batch

单态初始振幅是长度为 `2 ** num_qubits` 的 C 连续 `numpy.complex128` 一维数组。batch 初态是形状为 `[batch_size, 2 ** num_qubits]` 的二维数组，每行是一条归一化状态：

```python
import numpy as np
import arcqml

states = np.ascontiguousarray(
    [
        [1.0 + 0.0j, 0.0j],
        [0.0j, 1.0 + 0.0j],
    ],
    dtype=np.complex128,
)

simulator = arcqml.BatchStateVectorSimulator.from_amplitudes(1, states)
values = simulator.run(circuit, observable)  # shape 为 [2]
```

`BatchStateVectorSimulator(num_qubits, batch_size)` 默认把每行初始化为零态。batch 模拟器没有 `sample_counts`。

## 索引与数据布局

- 量子比特从 `0` 开始编号，`q0` 是内部基态索引的最低有效位。
- 抽样字符串按 `q[n-1]...q[0]` 展示。
- NumPy 输入必须为 C 连续；需要时使用 `numpy.ascontiguousarray`。
- 振幅使用 `complex128`，常用训练目标使用 `float64`。

## TODO

- GPU、噪声、密度矩阵、量子信道、中途测量、经典条件控制或真机后端。
- 其它 Rust 完整功能以及对应 TODO。

端到端示例参见 [`python/examples`](examples) 与仓库级 [`examples/python`](../examples/python)。许可证为 [混合许可说明](../README.md#发行状态与许可证)，发行状态参见[根 README](../README.md#发行状态与许可证)。
