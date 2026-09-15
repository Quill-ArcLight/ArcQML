# ArcQML 技术手册


## 目录

- [1. 阅读说明与能力边界](#1-阅读说明与能力边界)
  - [1.1 框架定位](#11-框架定位)
  - [1.2 当前支持的能力](#12-当前支持的能力)
  - [1.3 当前限制](#13-当前限制)
- [2. 发行包与环境配置](#2-发行包与环境配置)
  - [2.1 发行目录](#21-发行目录)
  - [2.2 Rust 环境与 Runtime 配置](#22-rust-环境与-runtime-配置)
  - [2.3 Python 环境与 wheel 安装](#23-python-环境与-wheel-安装)
  - [2.4 一次可训练前向与反向的数据流](#24-一次可训练前向与反向的数据流)
- [3. 数学、索引与存储约定](#3-数学索引与存储约定)
  - [3.1 纯态与归一化](#31-纯态与归一化)
  - [3.2 量子比特索引与二进制字符串显示](#32-量子比特索引与二进制字符串显示)
  - [3.3 局部矩阵与多量子比特门顺序](#33-局部矩阵与多量子比特门顺序)
- [4. Tensor、Parameter 与自动微分](#4-tensorparameter-与自动微分)
  - [4.1 Tensor 的数据与元数据](#41-tensor-的数据与元数据)
  - [4.2 自动微分图与梯度生命周期](#42-自动微分图与梯度生命周期)
  - [4.3 Parameter 的语义](#43-parameter-的语义)
- [5. Circuit、Operation 与参数管理](#5-circuitoperation-与参数管理)
  - [5.1 Circuit 的结构](#51-circuit-的结构)
  - [5.2 内置门范围](#52-内置门范围)
  - [5.3 参数（含参门）的三种添加方式](#53-参数含参门的三种添加方式)
  - [5.4 共享参数与梯度累积](#54-共享参数与梯度累积)
  - [5.5 校验、复制、拼接](#55-校验复制拼接)
  - [5.6 参数表达式的当前限制](#56-参数表达式的当前限制)
- [6. Pauli 可观测量与 Hamiltonian](#6-pauli-可观测量与-hamiltonian)
  - [6.1 PauliString](#61-paulistring)
  - [6.2 SparsePauliOp 与编译缓存](#62-sparsepauliop-与编译缓存)
  - [6.3 期望值语义](#63-期望值语义)
- [7. 单态与批量状态向量模拟](#7-单态与批量状态向量模拟)
  - [7.1 StateVectorSimulator：有状态与无状态接口](#71-statevectorsimulator有状态与无状态接口)
  - [7.2 初态导入与验证](#72-初态导入与验证)
  - [7.3 BatchStateVectorSimulator](#73-batchstatevectorsimulator)
- [8. 量子伴随自动微分](#8-量子伴随自动微分)
  - [8.1 前向定义](#81-前向定义)
  - [8.2 单态伴随反向公式](#82-单态伴随反向公式)
  - [8.3 batch 的上游梯度与 VJP](#83-batch-的上游梯度与-vjp)
  - [8.4 no_grad 与训练图内存](#84-no_grad-与训练图内存)
- [9. 状态分析与末端抽样](#9-状态分析与末端抽样)
  - [9.1 精确概率与边缘概率](#91-精确概率与边缘概率)
  - [9.2 纯态保真度与 Bloch 向量](#92-纯态保真度与-bloch-向量)
  - [9.3 sample_counts 的精确行为](#93-sample_counts-的精确行为)
- [10. 损失函数、优化器与训练循环](#10-损失函数优化器与训练循环)
  - [10.1 损失函数：输入约束与前向公式](#101-损失函数输入约束与前向公式)
  - [10.2 优化器](#102-优化器)
  - [10.3 推荐训练循环](#103-推荐训练循环)
  - [10.4 自定义损失](#104-自定义损失)
- [11. 权重检查点与整体酉矩阵拟合](#11-权重检查点与整体酉矩阵拟合)
  - [11.1 权重检查点的文件契约](#111-权重检查点的文件契约)
  - [11.2 DenseUnitary 的数据契约](#112-denseunitary-的数据契约)
  - [11.3 酉保真度、损失与解析梯度](#113-酉保真度损失与解析梯度)
- [12. 可视化、Python 绑定与并行执行](#12-可视化python-绑定与并行执行)
  - [12.1 文本与 SVG 电路图](#121-文本与-svg-电路图)
  - [12.2 Python 当前导出接口](#122-python-当前导出接口)
  - [12.3 Rayon 线程池](#123-rayon-线程池)

## 1. 阅读说明与能力边界

### 1.1 框架定位

ArcQML 是一个以 Rust 原生实现、同时提供 Rust 与 Python 接口的量子机器学习框架。用户可以通过统一接口构建固定或含参量子电路，使用状态向量模拟器执行单个或批量量子态的演化，并以 Pauli 算符及其线性组合定义待测量的可观测量。框架将电路输出组织为可微分 `Tensor`；期望值不仅可以直接读取，还可以继续参与损失函数、经典张量运算和其他可微计算，从而把量子计算与经典计算纳入同一条自动微分链路。

对于含参量子电路，ArcQML Runtime 提供门作用、门导数、电路计划执行和伴随反向的原生数值能力；公开 Rust 层负责电路与参数管理、可观测量编译及期望值计算、batch 外层调度、自动微分图、损失函数和优化器。调用 `backward` 后，梯度可由经典损失沿计算图回传至电路参数，再由 `Adam` 或 `SGD` 等优化器进行更新。该能力组合适用于变分量子本征求解、量子神经网络、量子态分析和小规模酉矩阵综合等任务。

### 1.2 当前支持的能力

| 能力域 | 当前实现 |
| --- | --- |
| 量子态 | 归一化 `C64` 稠密纯态状态向量 |
| 电路 | 固定门、参数化门、自定义酉门、参数共享与拼接 |
| 训练 | `Tensor` 自动微分与量子伴随法 |
| 可观测量 | `PauliString` 与实系数 `SparsePauliOp` |
| 分析 | 概率、边缘概率、保真度、Bloch 向量、末端抽样 |
| 损失与优化 | Rust 支持 `MSE`/`L1`/交叉熵、算子组合自定义损失与 `SGD`/`Adam` |
| 用户接口 | Rust 完整接口；Python 原生扩展提供常用电路、模拟、分析与训练接口 |

### 1.3 当前限制

当前后端仅支持 CPU 上的 `C64` 稠密纯态状态向量模拟，暂未包含 CUDA、稀疏态、噪声模型、密度矩阵、量子信道、中途测量或经典条件控制等。`BatchStateVectorSimulator` 表示多条独立初态共享同一电路和参数，当前暂未提供批量抽样接口。权重检查点只保存电路参数，暂未保存电路拓扑、优化器状态、训练数据位置或随机数状态。Python API 暴露的是常用功能子集；需要完整能力时，应使用 Rust 原生接口。后续能力以实际发行说明为准。

## 2. 发行包与环境配置

### 2.1 发行目录

以下路径均以用户取得的 ArcQML 发行目录为基准，将示例中的 `PATH_TO_YOUR_FILES` 替换为实际存放发行包的目录，并保持发行包内部的相对目录结构不变。

```text
PATH_TO_YOUR_FILES/ArcQML/
├── crates/                 # Rust 接口及配套功能
├── libs/
│   ├── x86_64-pc-windows-msvc/
│   │   └── arcqml_runtime_private.lib
│   └── ...
├── wheels/
│   ├── x86_64-pc-windows-msvc/
│   │   └── arcqml-0.1.0-cp311-cp311-win_amd64.whl
│   └── ...
├── examples/               # 完整示例
├── docs/                   # 技术手册、API 文档与教程
├── Cargo.toml
├── LICENSE-RUNTIME
└── ...
```


其他操作系统、CPU 架构和 Python 版本的支持，以后续实际提供的构建产物为准。

### 2.2 Rust 环境与 Runtime 配置


```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

安装后执行 `rustc --version` 和 `cargo --version` 确认命令可用。然后进入发行包根目录，将环境变量 `ARCQML_RUNTIME_LIB_DIR` 指向与当前平台匹配的 `libs` 子目录。该变量用于链接预编译 Runtime。

Windows PowerShell：

```powershell
Set-Location "PATH_TO_YOUR_FILES\ArcQML"
$env:ARCQML_RUNTIME_LIB_DIR = (Resolve-Path ".\libs\x86_64-pc-windows-msvc").Path
cargo check -p arcqml
```


```bash
cd PATH_TO_YOUR_FILES/ArcQML
export ARCQML_RUNTIME_LIB_DIR="$PWD/libs/x86_64-unknown-linux-gnu"
cargo check -p arcqml
```

环境变量只对当前终端会话生效；打开新终端后需要重新设置。`cargo check` 只检查源码和类型，不验证最终应用链接。

用户自己的 Rust 项目可以通过路径依赖引用发行包中的接口：

```toml
[dependencies]
arcqml = { path = "PATH_TO_YOUR_FILES/ArcQML/crates/arcqml" }
```

运行该项目时同样需要先设置 `ARCQML_RUNTIME_LIB_DIR`。如果复制或移动发行包，应同步更新依赖路径与 Runtime 路径。

### 2.3 Python 环境与 wheel 安装


```bash
cd PATH_TO_YOUR_FILES/ArcQML
conda create -n arcqml-example python=3.11 -y
conda activate arcqml-example
python -m pip install --upgrade pip
```

wheel 已静态嵌入与其版本匹配的 ArcQML Runtime，因此 Python 用户不需要设置 `ARCQML_RUNTIME_LIB_DIR`，也不需要安装 Rust、Maturin 或从源码编译扩展。可用 `python -c "import arcqml; print(arcqml)"` 验证导入是否成功。若 pip 报告 wheel 与平台不兼容，应先核对 `python --version`、`python -c "import platform; print(platform.machine())"` 与操作系统。

### 2.4 一次可训练前向与反向的数据流

```text
Circuit.parameters()
    │  参数值在执行时绑定到 Gate
    ▼
(Batch)StateVectorSimulator.run(circuit, observable)
    │  接口层校验并绑定参数，Runtime 执行状态演化，公开 Rust 层计算期望值
    ▼
F64 标量或 [B] F64 Tensor
    │  与普通 Tensor loss 组成统一自动微分图
    ▼
loss.backward()
    │  经典 VJP → Runtime 量子伴随计算 → Parameter.grad
    ▼
optimizer.step(circuit.parameters())
```

整条量子电路作为自定义 `Tensor` 操作接入自动微分图。公开 Rust 层负责参数与形状校验、可观测量计算、计算图连接和优化器更新；Runtime 负责门作用、电路计划执行及伴随反向数值计算。`run` 的父节点连接初态与电路参数，因此外部损失函数无需了解量子门内部执行过程，量子期望值也能继续参与经典可微计算。

## 3. 数学、索引与存储约定

### 3.1 纯态与归一化

对 `n` 个量子比特，ArcQML 使用长度为 `d` 的复数向量表示纯态，其中状态空间维数为：

$$
d=2^n.
$$

纯态及其归一化条件为：

$$
\lvert\psi\rangle
=\sum_{x=0}^{2^n-1}\alpha_x\lvert x\rangle,
\qquad
\sum_{x=0}^{2^n-1}\lvert\alpha_x\rvert^2=1.
$$

从 `Tensor` 导入单态时，框架检查整个向量的归一化平方范数；导入 batch 时则逐行检查。考虑到浮点误差，两种接口都要求：

$$
\left\lvert
\sum_x\lvert\alpha_x\rvert^2-1
\right\rvert
\le 10^{-10}.
$$

默认构造器创建全零计算基态：

$$
\lvert 0\cdots 0\rangle.
$$

该状态的第 0 个振幅为 1，其余振幅均为 0。

### 3.2 量子比特索引与二进制字符串显示

ArcQML 规定量子比特 `q0` 对应计算基索引的最低有效位，`q1` 对应下一位，依次类推。计算基与数值索引的关系为：

$$
\lvert x\rangle
=\lvert q_{n-1}\cdots q_1q_0\rangle,
\qquad
x=\sum_{k=0}^{n-1}q_k2^k.
$$

输出二进制字符串时按量子比特编号从大到小排列，即最左侧为 `q[n-1]`、最右侧为 `q0`。例如，在两个量子比特的系统中，内部索引 1 显示为 `01`。

| 二量子比特内部索引 | 二进制显示 | `q1` 的值 | `q0` 的值 |
| --- | --- | --- | --- |
| 0 | 00 | 0 | 0 |
| 1 | 01 | 0 | 1 |
| 2 | 10 | 1 | 0 |
| 3 | 11 | 1 | 1 |

### 3.3 局部矩阵与多量子比特门顺序

多量子比特门按调用参数确定局部基顺序：传入的 `qubits[0]` 对应局部矩阵索引的最低位，`qubits[1]` 对应下一位。对于 `cnot`、`cp`、`crx`、`cry` 和 `crz` 等受控门，第一个量子比特是控制位，第二个量子比特是目标位；例如 `cnot(0, 1)` 表示 `q0` 控制、`q1` 为目标。交换类门虽然在数学上具有对称性，调用时仍需传入两个合法且不同的量子比特索引。

稠密整体酉矩阵的公开行主序数据先使用输出计算基索引，再使用输入计算基索引。另一方面，整体酉的内部批量计算先将每个计算基输入作为一列式 batch 状态，最终再转置到公开矩阵布局。

## 4. Tensor、Parameter 与自动微分

### 4.1 Tensor 的数据与元数据

`Tensor` 是用户直接操作的数值对象。它把 `Storage`、版本号、`TensorMeta` 和 `AutogradMeta` 组合在共享句柄中。`TensorMeta` 包含 `shape`、`dtype`、`device`、`layout`、`strides` 和 `offset`；公开查询包括 `shape`、`ndim`、`numel`、`dtype`、`device`、`layout`、`is_contiguous`、`strides` 和 `offset`。

| `dtype` | 典型用途 | 自动微分可作为叶子变量 |
| --- | --- | --- |
| `F32` | 单精度实数参数或数据 | 可以。 |
| `F64` | 量子参数、期望值、损失和默认 `zeros` | 可以。 |
| `C64` | 双精度复数（两个 f64，共 16 字节），对应 `Complex64` / NumPy `complex128` | 可以。 |
| `I64` | 整数标签或索引数据 | 不可以。 |
| `Bool` | 布尔数据 | 不可以。 |

`Tensor::clone()` 只复制句柄，仍共享底层 `Storage`、版本号和自动微分元数据。`deep_clone()` 复制数值、Tensor 元数据和已有梯度，保留 `requires_grad` 标志，但创建独立存储、独立版本计数的新叶子 Tensor，不继承原计算图；它不是可微复制算子。`storage_mut()` 取得写锁，并在 `guard` 释放时递增共享版本号；这个版本号用于在反向前检测前向数据被原地修改的情形。

### 4.2 自动微分图与梯度生命周期

只有叶子 `Tensor` 可以开启 `requires_grad`；可求导叶子 `dtype` 仅限 `F32`、`F64`、`C64`。由可微操作产生的输出会在当前线程的梯度记录开关开启且至少一个父节点需要梯度时记录图。默认只持久保存叶子梯度；中间 `Tensor` 必须显式调用 `retain_grad()` 才会保留。

`backward()` 只为 `shape=[]` 的标量输出自动构造上游梯度 1。非标量输出必须使用 `backward_with_grad(gradient)`，且上游梯度与输出在形状、`dtype` 等方面兼容。默认 `backward` 会释放本次遍历的图节点；若要复用图，应从第一次反向起使用 `backward_with_grad_retain_graph`，不能在图释放后补救。`retain_grad()` 只保留中间梯度，不保留计算图。梯度以累积方式写入，因此训练迭代必须显式调用 `zero_grad`。共享前向的多个损失优先合成一个标量后反向。

### 4.3 Parameter 的语义

`Parameter` 是 `Tensor` 的语义包装，是框架中可训练参数的主要数值对象。`Parameter::new` 要求底层 `Tensor` 是叶子且 `dtype` 为 `F32`、`F64` 或 `C64`，并会将其 `requires_grad` 置为 `true`；默认 `trainable` 为 `true`。非法输入会 panic，需要可恢复错误时使用 `Parameter::try_new`。`Circuit` 参数登记检查 `numel=1` 且为 `F32`/`F64`，训练时应使用 `shape=[]` 的标量参数；量子门角度训练不接受 `C64` 参数。

`freeze()` 将 `trainable` 设为 `false`，优化器随后会跳过该参数；它不会同时关闭底层 `Tensor` 的 `requires_grad`，因此该参数仍可能接收梯度。`unfreeze()` 会恢复 `trainable=true`，并确保底层 `Tensor` 开启梯度记录。换言之，`requires_grad` 控制是否参与反向传播，`trainable` 控制优化器是否更新参数。

## 5. Circuit、Operation 与参数管理

### 5.1 Circuit 的结构

`Circuit` 保存 `num_qubits`、按执行顺序排列的 `Operation` 列表，以及 `Parameter` 列表。`add_gate` 将 `Gate` 与 `qubit` 列表包装为 `Operation`；调用模拟器后，接口层才会校验和绑定参数，并由 ArcQML Runtime 将 `Gate` 作用到当前量子态。

`Circuit::depth()` 以量子比特资源为依据计算并行层数：每个 `Operation` 的层是其涉及各量子比特当前层的最大值加一，并把该层写回所有参与量子比特。

### 5.2 内置门范围

| 分类 | Rust `Circuit` 方法 | 参数行为 |
| --- | --- | --- |
| 单比特固定 | `i`, `x`, `y`, `z`, `h`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg` | 无训练参数。 |
| 单比特参数 | `rx`, `ry`, `rz`, `phase`, `u1`, `u2`, `u3` | 默认创建新参数；部分门提供 `_fixed` 或 `_param` 变体。 |
| 受控门 | `cnot`/`cx`, `cy`, `cz`, `ch`, `cs`, `ct`, `cp`, `crx`, `cry`, `crz` | 输入的前两个 `qubit` 参数依次是控制位与目标位。 |
| 双/多比特 | `swap`, `iswap`, `dcx`, `ecr`, `rxx`, `ryy`, `rzz`, `rzx`, `fsim`, `toffoli`, `cswap`, `mcx` | 旋转类和 `fSim` 可训练；其余固定。 |
| 扩展 | `Gate::custom_unitary(name, arity, matrix)` | 传入 `row-major` `C64` 矩阵，创建时验证维度和酉性。 |

Python `Circuit` 当前提供 `h`、`x`、`y`、`z`、`rx`、`ry`、`rz`、`phase`、`u3` 和 `cnot`。其门集小于 Rust `Circuit`；需要其它受控门、多量子比特门或自定义酉门时，应使用 Rust API。

### 5.3 参数（含参门）的三种添加方式

| 方式 | 例子 | 参数表行为 | 适合场景 |
| --- | --- | --- | --- |
| 自动注册 | `circuit.ry(0.3, 0)` | 生成 `F64` 标量 `Parameter` 与稳定自动名称 | 一般可训练门。 |
| 固定参数 | `circuit.ry_fixed(0.3, 0)` | `Gateparam::Fixed`；不登记 `Parameter` | 编码常数、不可训练层。 |
| 显式共享 | `id = add_parameter(...)`；`ry_param(id, 0)` | 多个门指向同一个 `ParameterId` | 对称 ansatz 与共享权重。 |

自动门参数名称由 `gate`、角色、量子比特与参数表下标组成，例如 `ry_q0_theta_0`。手动调用 `add_parameter_tensor` 时，默认名为 `parameter_<index>`。名称是 `checkpoint` 匹配依据的一部分，因此一旦模型需要长期持久化，改变建模顺序或门名称规则会影响加载兼容性。

### 5.4 共享参数与梯度累积

```rust
use arcqml::prelude::*;

let mut circuit = Circuit::new(1)?;
let theta = circuit.add_parameter(0.3)?;
circuit.ry_param(theta, 0usize)?;
circuit.rz_param(theta, 0usize)?;
// 反向时，两处操作对同一 theta 的贡献相加
```

每个参数化 `Operation` 只保留 `ParameterId`；执行前由接口层读取 `Circuit` 当前 `Parameter` `Tensor` 的数值并交给 Runtime。因此优化器更新参数后，不必重建电路，下一次运行会自动使用新角度。伴随反向也按 `ParameterId` 累加多处门操作的梯度贡献，从而实现共享参数的梯度累积。

### 5.5 校验、复制、拼接

`validate()` 会检查电路量子比特数非零、`Operation` 的 `qubit` 合法、`Gate` 参数引用属于当前参数表，以及参数名称非空且不重复。`Circuit::clone()` 调用 `deep_clone()`，所以复制出的 `Circuit` 不与原电路共享参数值或梯度。

`append(right)` 消费右电路并追加其操作，对未绑定参数做重映射；`append_with_bindings(right, bindings)` 可将右电路指定 `ParameterId` 显式绑定到左电路已有参数。两条电路的量子比特数必须相同，且成功追加后结果电路会清空参数梯度。

### 5.6 参数表达式的当前限制

当前版本的参数化门只能在参数槽位中保存固定标量或单个 `ParameterId`，尚不能直接保存由多个参数组成的表达式。因此，`-x`、`x + y` 等表达式不能作为门参数传入；同一个 `ParameterId` 可以被多个门共享，但不能在 `Circuit` 内声明参数之间的算术关系。

如果需要使用参数表达式，现阶段可在电路外部计算表达式的数值，再将结果作为固定参数或独立参数写入电路。需要注意的是，这种方式不会自动保留表达式与原始参数之间的依赖关系：例如把 `x + y` 的计算结果登记为新的 `Parameter` 后，量子伴随反向只会得到该新参数的梯度，不会自动按链式法则把梯度分配给 `x` 和 `y`。训练过程中还需要在每次前向计算前重新计算并更新该参数值。

当前版本不支持把非叶子 Tensor 表达式直接登记为门参数；这种调用还可能触发 panic。若需要训练表达式的原始变量，调用方必须自行实现参数映射的 VJP/链式梯度；每轮重新写入数值不会自动建立依赖关系。一般训练应优先使用单个 `ParameterId` 和参数共享。量子输出继续接经典可微损失则受支持。

## 6. Pauli 可观测量与 Hamiltonian

### 6.1 PauliString

`PauliString` 表示若干 `PauliOp` 的张量积，只存储非 `I` 项，并按 `qubit` 下标排序。构造时检查 `qubit` 范围和重复量子比特。`identity`、`single`、`x`、`y`、`z` 提供快捷构造。两个 `PauliString` 在同一 `qubit` 上同时非 `I` 且 Pauli 不同的位置数量为偶数时对易。

$$
P=P_{n-1}\otimes\cdots\otimes P_1\otimes P_0,
\qquad
P_q\in\{I,X,Y,Z\}.
$$

例如二量子比特的 `X(0)` 对应 `I ⊗ X`，将 `|00⟩` 映射到 `|01⟩`。`multiply(rhs)` 返回 (复相位, 结果 `PauliString`)。这是 Pauli 代数辅助功能；`SparsePauliOp` 的公开系数则被限制为有限实数，因此可观测量期望值是实数。

### 6.2 SparsePauliOp 与编译缓存

`SparsePauliOp` 是 `PauliTerm` 的集合，每个 `term` 为有限实数 `coefficient` 和 `PauliString`。`Hamiltonian`、`PauliSum`、`PauliObservable` 是它的类型别名。构造时要求 `num_qubits` 非零且所有项的量子比特数一致；`scale` 和 `simplify` 也拒绝非有限因子或非法 `tolerance`。

$$
H=\sum_j c_jP_j,
\qquad
c_j\in\mathbb{R}.
$$

首次进行状态向量计算时，公开 Rust 层会为 `SparsePauliOp` 准备可复用的执行表示：对角项可以合并，非对角 Pauli 串会转换为位掩码，用于公开可观测量实现中的期望值及 `H|ψ⟩` 计算。增加 `term` 会使已有缓存失效；`clone` 复制可观测量定义，但不复制执行缓存。因此，在训练循环中复用同一个 `Hamiltonian`，可以避免每轮重复准备相同的执行信息。

```rust
let mut h = SparsePauliOp::single(2, 0usize, Pauli::Z, 0.5)?;
h.add_pauli_string(-1.0, PauliString::x(2, 1usize)?)?;
// H = 0.5 Z(0) − 1.0 X(1)
```

### 6.3 期望值语义

对于当前纯态，单态模拟器的 `expectation_observable` 返回 `F64` 标量 `Tensor`；batch 版本返回 `shape` 为 `[B]` 的 `F64` `Tensor`，每个元素对应一行初态。期望值定义为：

$$
\langle H\rangle
=\mathrm{Re}\left(
\langle\psi\rvert H\lvert\psi\rangle
\right).
$$

`run` 创建的可微路径会用于伴随反向，以避免再次构造同一可观测量作用结果。

## 7. 单态与批量状态向量模拟

### 7.1 StateVectorSimulator：有状态与无状态接口

| 接口 | 状态影响 | 返回值 | 主要用途 |
| --- | --- | --- | --- |
| `new` / `from_state_tensor` | 创建初态 | 模拟器 | 默认零态或导入已归一化自定义态。 |
| `apply_gate` / `apply_operation` | 修改当前 `state` | `&mut Self` | 逐步执行固定参数 `Gate`/`Operation`。 |
| `apply_circuit` | 修改当前 `state` | `&mut Self` | 读取 `Circuit` 当前参数并逐门执行。 |
| `run` | 不修改当前 `state` | 可微 `F64` 标量 `Tensor` | 训练与函数式前向。 |
| `amplitudes` | 方法本身不改态 | 当前 `C64` Tensor 的共享句柄 | 读取振幅，保留已有图连接。 |
| `sample_counts` | 只读 | `BTreeMap<String, usize>` | 当前态的末端全量子比特抽样。 |

值得注意的是，`apply_circuit` 与 `run` 不能混为一谈。前者改变模拟器 `state`，所以多次调用会在已演化状态上继续作用；后者从调用方当前 `state` 复制出前向缓冲区并返回期望值，不改变 `self`。若需要反复从零态执行同一电路，可使用新模拟器或在 `apply_circuit` 前调用 `reset`；训练循环通常使用 `run`。

Rust `amplitudes()` 返回的共享 Tensor 不是独立数值副本；通过其存储执行原地写入会影响原状态，也可能破坏归一化或已记录图的版本约束。Python `numpy()` 则复制为独立数组。

### 7.2 初态导入与验证

单态 `from_state_tensor` 接受连续、CPU、`Dense` 的 `C64` 一维 `Tensor`，`shape` 必须严格为 `[d]`；batch 版本要求二维 `[B, d]`、batch 大小非零，并逐行检查归一化。其中，`d` 是状态空间维数，两种接口均采用该节给出的归一化容差。

从 Python NumPy 导入时，单态要求 C 连续 `complex128` 一维数组，batch 要求 C 连续 `complex128` 二维数组。绑定首先检查 Python 维度，再复制到 `FlatC64` `Tensor`，随后由 Rust 模拟器执行同样的归一化与布局验证。

### 7.3 BatchStateVectorSimulator

`BatchStateVectorSimulator` 的 `state` 形状是 `[batch_size, d]`，每行是一条彼此独立的纯态。`apply_circuit` 对每一行施加相同电路；`run` 返回 `shape` 为 `[batch_size]` 的 `F64` `Tensor`。Runtime 按行主序解释批量状态；启用 `parallel` feature 时，公开 Rust 层使用 Rayon 进行 batch 外层任务调度。因此线程数、批量大小、状态维度和内存带宽都会影响实际性能。

**batch 的含义**  batch 不是“一个量子态中的并行分支”，而是一批独立初态共享同一条 `Circuit` 与同一个 `SparsePauliOp`。它不能表达每条样本使用不同门拓扑或不同量子比特数的情形。Batch 路径适合在移植传统机器学习任务处理较大数据集时使用。

## 8. 量子伴随自动微分

### 8.1 前向定义

对于按执行顺序排列的多个量子门，ArcQML 的 `run` 从初态开始逐门演化得到最终态，并计算 `Hamiltonian` 的期望值。第 `k` 个门对应的状态演化和最终目标函数定义为：

$$
\lvert\psi_k\rangle
=U_k\lvert\psi_{k-1}\rangle,
\qquad
f=\mathrm{Re}\left(
\langle\psi_m\rvert H\lvert\psi_m\rangle
\right).
$$

若初态或任一 `Circuit` `Parameter` 需要梯度，且当前线程启用梯度记录，`run` 会保存最终态、`Hamiltonian` 作用于最终态的结果、已绑定操作和参数槽位映射，并创建一个自定义 `F64` `Tensor` autograd 节点。否则它只执行前向，不保存伴随上下文。

### 8.2 单态伴随反向公式

设外层实标量损失为 `L`，单态期望值为 `f`，标量上游梯度为 `g = ∂L/∂f`。为避免重复计入上游权重，以下公式使用未缩放的伴随态：

$$
\lvert\lambda_m\rangle=H\lvert\psi_m\rangle,
\qquad
\lvert\lambda_{k-1}\rangle=U_k^\dagger\lvert\lambda_k\rangle.
$$

按逆序施加门的共轭转置可以恢复每个门的输入态。参数 `p` 在多个门中出现时，梯度为各处贡献之和：

$$
\frac{\partial L}{\partial p}
=2g\sum_{k:\,p\text{ 出现在 }U_k}
\mathrm{Re}\left[
\langle\lambda_k\rvert
\frac{\partial U_k}{\partial p}
\lvert\psi_{k-1}\rangle
\right].
$$

实现可在反向开始时把 `g` 乘入伴随缓冲区，此后不再额外乘 `g`。若初态需要梯度，ArcQML 返回的复梯度采用 `∂L/∂Re(ψ₀) + i ∂L/∂Im(ψ₀)` 约定：

$$
\nabla_{\psi_0}L=2g\lvert\lambda_0\rangle.
$$

它等于两倍的共轭 Wirtinger 偏导，不应与未乘 2 的偏导符号混用。

参数梯度则按照 `Parameter` 的 `F32` 或 `F64` `dtype` 构造成标量 `Tensor`。该过程只需保存最终态和已绑定门信息，不需要保存每一层完整状态。

### 8.3 batch 的上游梯度与 VJP

batch `run` 的输出不是标量而是 `[B]` 向量，因此它本身不能直接调用无参 `backward()`。通常应先用 `mse_loss` 或二元 `logits` 交叉熵等经典损失函数将其归约为标量。若上层图传回 `grad_output`，batch 伴随节点要求它是 `shape` 为 `[B]` 的 `F64` `Tensor`，并把每个上游梯度乘到对应 batch 行的 `adjoint_state` 中。

因此 batch 模式实现的是向量-雅可比积（VJP）：外部经典网络或损失决定每个样本输出的上游权重，量子伴随器再将其回传到共有 `Circuit` 参数和可微初态。

### 8.4 no_grad 与训练图内存

用 `no_grad()` 或 Python 的 `with arcqml.no_grad():` 包围 `run` 时，`records_gradients` 为 `false`，运行不保留伴随上下文，返回普通前向 `Tensor`。验证、推理、基准测试和只读状态分析应采用此模式；需要反向时，不应先把预测转为 `f64` 或 NumPy 数组，因为那会离开 `Tensor` 图。

Rust 必须绑定作用域守卫，单独执行 `no_grad();` 会在语句结束时立即恢复记录：

```rust
{
    let _guard = no_grad();
    let prediction = simulator.run(&circuit, &observable)?;
    // 在此作用域内读取 prediction。
}
```

梯度记录开关仅作用于当前线程。Python 当前应每次新建上下文，即使用 `with arcqml.no_grad():`；不要缓存或重复进入同一个上下文对象。

## 9. 状态分析与末端抽样

### 9.1 精确概率与边缘概率

`analysis::probabilities` 读取当前 `amplitudes`，并计算 `square(abs(amplitudes))`。返回 `shape` 为 `[d]` 的可微 `F64` `Tensor`；第 `index` 项对应同一数值的计算基索引，其中 `q0` 是该索引的最低有效位。

`marginal_probabilities` 先验证选择集合非空、无重复且索引范围合法，然后将选择的 `qubit` 按编号降序排序。对每个完整基态索引，它依次移入选中位形成输出 `segment id`，再用 `segment_sum` 汇总概率。选择 `[0, 2]` 时，输出按照 `q2q0` 排列，依次对应 `00`、`01`、`10` 和 `11`，与输入数组中的选择顺序无关。

### 9.2 纯态保真度与 Bloch 向量

`fidelity(lhs, rhs)` 先检查两个 `simulator` 的量子比特数相等，再计算两个量子态内积的模平方：

$$
F\!\left(\lvert\psi\rangle,\lvert\phi\rangle\right)
=\left\lvert\langle\psi\vert\phi\rangle\right\rvert^2.
$$

返回值不执行区间截断，以保持前向表达式与反向规则一致；理论上相同归一化纯态的结果为 1、正交态为 0，但浮点舍入可能产生极小的区间外误差。

`bloch_vector` 对指定 `qubit` 的每对零分支振幅与一分支振幅计算相干项，并按照下式累加 Bloch 向量三个分量：

$$
\begin{aligned}
r_x&=2\mathrm{Re}\left(\sum a_0^*a_1\right),\\
r_y&=2\mathrm{Im}\left(\sum a_0^*a_1\right),\\
r_z&=\sum\left(\lvert a_0\rvert^2-\lvert a_1\rvert^2\right).
\end{aligned}
$$

该函数返回普通 `BlochVector {x, y, z}`，当前实现没有把它作为可微 `Tensor` 返回。

### 9.3 sample_counts 的精确行为

`sample_counts` 仅属于单态模拟器。它要求 `shots` 为正整数，读取当前已归一化 `C64` `state`，取每个振幅的 `norm_sqr` 作为离散分布，使用给定 `seed` 或新随机 `seed` 初始化 `StdRng`，然后独立抽取指定次数。它不执行 `Circuit`，不写回 `state`，不产生测量坍缩，也不建立可微节点。

```rust
let mut sim = StateVectorSimulator::new(2)?;
sim.apply_circuit(&bell)?;
let counts = sim.sample_counts(1_000, Some(42))?;
// keys 使用 q1q0 显示，如 "00" 与 "11"。
```

## 10. 损失函数、优化器与训练循环

### 10.1 损失函数：输入约束与前向公式

| 函数 | 关键约束 |
| --- | --- |
| `mse_loss` | `prediction` 与 `target` 非空、同 dtype（`F32`/`F64`），形状可广播。 |
| `l1_loss` | `prediction` 与 `target` 非空、同 dtype（`F32`/`F64`），形状可广播。 |
| `binary_nll_loss` | `prediction` 与 `labels` 均为一维且长度相同；标签只能取 0 或 1。 |
| `BCE with logits` | `logits` 与 `targets` 的 `shape` 和 `dtype` 相同；`target` 取值位于 0 到 1 之间。 |
| `cross_entropy_loss` | `logits` 为二维 `Tensor`；`labels` 长度等于 batch 大小，每个标签都是合法类别索引。 |

这些损失依赖当前 CPU Dense 数值算子。设 `N` 为广播后误差 Tensor 的元素总数，`prediction` 和 `target` 的第 `i` 个广播后元素分别记为预测值与目标值。例如 `[B, C]` 误差的 `N = B × C`。均方误差损失定义为：

$$
L_{\mathrm{MSE}}
=\frac{1}{2N}\sum_{i=1}^{N}
\left(\hat{y}_i-y_i\right)^2.
$$

`L1` 损失定义为：

$$
L_{\mathrm{L1}}
=\frac{1}{N}\sum_{i=1}^{N}
\left\lvert\hat{y}_i-y_i\right\rvert.
$$

`binary_nll_loss` 的 `prediction` 表示 Pauli-Z 期望值，而不是普通概率或 `logit`。框架先对期望值执行截断：

$$
z_i^{\mathrm{clip}}
=\mathrm{clamp}\left(
z_i,-1+2\varepsilon,1-2\varepsilon
\right),
\qquad
\varepsilon=10^{-12}.
$$

随后，将截断后的期望值转换为两个类别的概率：

$$
p_i(0)=\frac{1+z_i^{\mathrm{clip}}}{2},
\qquad
p_i(1)=\frac{1-z_i^{\mathrm{clip}}}{2}.
$$

对应的二分类负对数似然损失为：

$$
L_{\mathrm{binary\ NLL}}
=-\frac{1}{N}\sum_{i=1}^{N}
\left[
(1-y_i)\log p_i(0)+y_i\log p_i(1)
\right],
\qquad
y_i\in\{0,1\}.
$$

直接接收 `logits` 的二元交叉熵采用数值稳定形式，其中 `N` 为 logits 的元素总数：

$$
L_{\mathrm{BCE}}
=\frac{1}{N}\sum_{i=1}^{N}
\left[
\max(x_i,0)+\log\!\left(1+e^{-\lvert x_i\rvert}\right)-y_ix_i
\right].
$$

对于包含 B 个样本和 C 个类别的多分类 `logits`，交叉熵损失定义为：

$$
L_{\mathrm{CE}}
=-\frac{1}{B}\sum_{b=1}^{B}
\log\!\left[
\mathrm{softmax}(X_b)_{y_b}
\right].
$$

### 10.2 优化器

#### SGD：实现是带耦合 L2 项的更新

`Sgd::step` 对 `trainable` 且 `requires_grad`、并且已有 `grad` 的 `Parameter` 更新。对于 `F32` 或 `F64`，每个元素执行以下更新：

$$
\theta\leftarrow\theta-\eta\left(g+\lambda\theta\right).
$$

$$
\eta=\text{学习率},
\qquad
g=\text{当前梯度},
\qquad
\lambda=\text{权重衰减系数}.
$$

无梯度参数计入 `skipped_no_grad`，冻结或 `requires_grad=false` 的参数计入 `skipped_frozen`。

#### Adam：偏置校正与权重衰减位置

`Adam` 在首次 `step` 时按参数切片中的位置建立状态槽位，后续调用必须保持参数数量和顺序不变。优化器保存迭代步数、一阶矩和二阶矩。每次更新时，首先把耦合式 `L2` 项加入当前梯度：

$$
\widetilde{g}_t
=g_t+\lambda\theta_{t-1}.
$$

随后更新一阶矩和二阶矩：

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)\widetilde{g}_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)\widetilde{g}_t^2.
\end{aligned}
$$

完成偏置校正后，再更新参数：

$$
\begin{aligned}
\widehat{m}_t&=\frac{m_t}{1-\beta_1^t},\\
\widehat{v}_t&=\frac{v_t}{1-\beta_2^t},\\
\theta_t&=\theta_{t-1}
-\eta\frac{\widehat{m}_t}{\sqrt{\widehat{v}_t}+\varepsilon}.
\end{aligned}
$$

因此，这里的 `weight_decay` 会进入矩估计，不等同于 `AdamW` 的解耦权重衰减。

构造器要求 `learning_rate`、`weight_decay` 为有限非负数，`beta1` 和 `beta2` 位于 `[0, 1)`，`epsilon` 为有限正数。训练中必须复用同一个 `Adam` 实例；重新创建会丢失动量与步数。

### 10.3 推荐训练循环

```rust
let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8, 0.0)?;
for _ in 0..steps {
    let prediction = simulator.run(&circuit, &observable)?;
    let loss = mse_loss(&prediction, &target)?;
    loss.backward()?;
    optimizer.step(circuit.parameters())?;
    optimizer.zero_grad(circuit.parameters());
}
```

建议在每轮 `optimizer.step` 之后调用 `zero_grad`，清空当前梯度，避免下一轮意外累积。不要在 `backward` 之前替换或原地修改参与前向计算的 `Parameter` `Tensor`，否则版本检查可能报告前向数据已被修改。参数更新应交给 `optimizer.step` 完成。

### 10.4 自定义损失

Rust 用户可以将量子输出继续接到 `sub`、`square`、`mul`、`mean` 等可微算子，构造自己的标量损失。普通算子组合无需手写反向；需要为图外计算或整体表达式指定导数时，可使用 `CustomOp` 与 `apply_custom_op`。自动微分遵循实际计算图的局部反向规则，不会对任意等价数学表达式自动进行符号化简。

完整的输入契约、加权 MSE、batch 电路训练、解析/数值梯度检查和专用反向示例见[自定义损失教程](tutorial/custom_loss.md)与[自定义损失示例](../examples/rust/custom_loss.rs)。当前 Python 尚未导出对应的通用算子组合与 `CustomOp` 接口。

## 11. 权重检查点与整体酉矩阵拟合

### 11.1 权重检查点的文件契约

`save_weights` 写入格式化 JSON，固定 `format` 为 `arcqml/weights`。每个参数记录 `name`、`dtype` 与 `value`；记录先按名称排序，使相同状态得到稳定顺序。写入只接受 `numel=1` 的 `F32`/`F64` `Circuit` `Parameter`。

```json
{
  "format": "arcqml/weights",
  "num_qubits": 2,
  "parameters": [
    {"name": "ry_q0_theta_0", "dtype": "f64", "value": 0.36154268969437336}
  ]
}
```

`load_weights` 先校验 `format`、量子比特数、参数名集合、重复名称、`dtype` 和 JSON 解析出的有限 `value`，并预先构造替换 Tensor；然后写入参数并清空梯度。它不会比对门拓扑，因此调用方必须以预期的同构 `Circuit` 重建模型。

### 11.2 DenseUnitary 的数据契约

`DenseUnitary` 保存 `num_qubits`、`dimension` 和 `row-major` `C64` 数据。`dimension` 必须是 2 的幂；公开数据先使用输出计算基索引，再使用输入计算基索引。`DenseUnitary::from_tensor` 只检查 `C64`、二维方阵、连续复制后的长度和有限性；公共 `unitary_from_tensor` 在此基础上使用以下默认容差完整验证酉性：

$$
\varepsilon_{\mathrm{unitary}}=10^{-10}.
$$

`unitary_from_circuit` 通过对所有计算基初态组成的内部 batch 执行 `Circuit`，再转置为公开 `row-major` 布局。对于 `n` 个量子比特，其空间复杂度为：

$$
O(4^n).
$$

单份矩阵数据约占 `16 × 4^n` 字节，内部计算还需额外缓冲区。因此，该功能适用于小规模电路比较和综合。

### 11.3 酉保真度、损失与解析梯度

目标整体酉矩阵与当前电路整体酉矩阵的迹重叠、整体酉保真度和损失分别定义为：

$$
\begin{aligned}
s&=\mathrm{Tr}\left(U_{\mathrm{target}}^{\dagger}U(\boldsymbol{\theta})\right),\\
F_U&=\mathrm{clamp}\left(\frac{\lvert s\rvert^2}{d^2},0,1\right),\\
L_U&=1-F_U.
\end{aligned}
$$

与第 9 章的纯态 `fidelity` 不同，此处会显式执行从 0 到 1 的区间截断，以限制数值舍入造成的越界。

`unitary_loss` 返回可纳入统一自动微分的 `F64` 标量。其反向先按逆序恢复前向 `StateBatch`，再按照未截断目标的解析式累积每个参数的梯度，并乘上外层标量上游梯度：

$$
\frac{\partial L_U}{\partial p}
=-\frac{2}{d^2}
\mathrm{Re}\left(
s^*\frac{\partial s}{\partial p}
\right).
$$

该梯度实现不在前向截断区间外额外置零。对精确酉矩阵，未截断保真度理论上不超过 1；若把未经酉性验证的 `DenseUnitary` 当作目标，可能得到被截断的损失与上述梯度不一致的结果。`unitary_loss` 本身只检查与电路的量子比特数兼容，目标应通过 `unitary_from_tensor` 或显式 `validate` 验证。数值舍入导致的轻微越界仍需按该实现语义理解。

## 12. 可视化、Python 绑定与并行执行

### 12.1 文本与 SVG 电路图

Rust 顶层接口中的 `draw` 只把 `Circuit` 渲染为 Unicode 字符串：`q0` 位于最上方，单量子比特门显示为边框，受控门显示为控制点和连接线，参数化门以 `p<index>` 或固定数值显示。

```rust
let mut circuit = Circuit::new(3)?;
circuit.h(0usize)?.cnot(0usize, 1usize)?.rz(0.3, 2usize)?;
println!("depth = {}", circuit.depth());
println!("{}", draw(&circuit));
```

`arcqml::visualization::draw_svg(&circuit)` 返回 SVG 字符串；`write_svg(&circuit, path)` 写入文件并返回 `Result`，调用方应处理写入错误：

```rust
arcqml::visualization::write_svg(&circuit, "circuit.svg")?;
```

### 12.2 Python 当前导出接口

| Python 对象/函数 | 对应 Rust 概念 | 注意事项 |
| --- | --- | --- |
| `Tensor` / `tensor` / `no_grad` | `arcqml::core::Tensor` 与梯度模式 | `numpy()` 或 `item()` 是显式离开 `Tensor` 表示的边界。 |
| `Circuit` | `arcqml::circuit::Circuit` | 只绑定常用门集；参数名值与梯度以 `dict` 返回。 |
| `StateVectorSimulator` | 单态 simulator | `from_amplitudes` 要求 C 连续 `complex128` 一维 NumPy 数组。 |
| `BatchStateVectorSimulator` | batch simulator | `from_amplitudes` 要求 C 连续 `complex128` 二维行主序数组。 |
| `PauliSum` | `SparsePauliOp` | 用于 `run` 的 `observable`。 |
| `mse_loss` / `BCE logits` | Rust 损失函数接口的子集 | Python 当前未导出 `L1`、`binary_nll` 和多分类 `CE`。 |
| `Adam` | `arcqml::optim::Adam` | 默认 `beta1=0.9`、`beta2=0.999`、`epsilon=1e-8`、`weight_decay=0`。 |

Python 的 `simulator.run` 不修改当前态，`apply_circuit` 会修改当前态；这与 Rust 语义一致。Python `BatchStateVectorSimulator` 同样暂未支持 `sample_counts`。Python `Circuit.gradients()` 在任一参数尚未产生梯度时会按绑定层实现报告错误，故训练中应先调用 `backward` 再读取。

### 12.3 Rayon 线程池

默认启用 `parallel` feature 时，Rust 顶层接口提供 `init_rayon(num_threads)` 与 `rayon_num_threads()`，Python 同样导出这两个函数。初始化全局 Rayon 工作线程池必须发生在首次并行计算之前；批量路径在样本层进行任务调度，因此线程数、batch 大小、状态维度和内存带宽都会影响实际性能。线程池只能初始化一次，应用应在首次模拟之前完成配置。
