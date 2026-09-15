# arcqml-observable

`arcqml-observable` 定义 Pauli 字符串及其稀疏实系数线性组合，用来表示量子测量量与 Hamiltonian。`Hamiltonian`、`PauliSum` 和 `PauliObservable` 都是 `SparsePauliOp` 的类型别名。

## 添加依赖

```toml
[dependencies]
arcqml-observable = { path = "PATH_TO_ARCQML/crates/arcqml-observable" }
```

## 从单项开始

```rust
use arcqml_observable::{Pauli, PauliOp, PauliString};

let term = PauliString::new(
    2,
    vec![
        PauliOp::new(0usize, Pauli::Z),
        PauliOp::new(1usize, Pauli::X),
    ],
)?;
```

这表示一个 Pauli 张量积项。未显式列出的量子比特视为单位算符。`PauliString::x`、`y`、`z`、`single` 与 `identity` 可用于常见构造。

`new` 会拒绝越界或重复量子比特。`commutes_with` 判断两个字符串是否对易；`multiply` 返回相乘产生的复相位与新字符串。

## 构造 Pauli 和

一般可观测量表示为：

$$
H=\sum_{k}c_kP_k
$$

其中每个系数必须是有限实数。

```rust
use arcqml_observable::{Pauli, PauliString, SparsePauliOp};

let mut hamiltonian = SparsePauliOp::single(
    2,
    0usize,
    Pauli::Z,
    0.5,
)?;
hamiltonian.add_pauli_string(
    -1.0,
    PauliString::x(2, 1usize)?,
)?;

let simplified = hamiltonian.simplify(1e-12)?;
```

常用接口：

| 类别 | 接口 |
| --- | --- |
| 构造 | `new`、`zero`、`single`、`identity`、`constant`、`from_pauli_string` |
| 编辑 | `add_term`、`add_pauli_string` |
| 变换 | `scale`、`simplify` |
| 查询 | `num_qubits`、`len`、`is_empty`、`terms` |
| 导入 | `from_json` |

同一个 `SparsePauliOp` 中所有项必须具有相同量子比特数。`simplify(tolerance)` 合并相同 Pauli 项并移除绝对值不超过容差的系数。

## 在模拟器中测量

```rust
let value = simulator.expectation_observable(&hamiltonian)?;
```

期望值定义为：

$$
\langle H\rangle
=
\langle\psi|H|\psi\rangle
=
\sum_k c_k\langle\psi|P_k|\psi\rangle
$$

单态返回 `F64` 标量 Tensor，batch 返回形状为 `[batch_size]` 的 `F64` Tensor。两者均可参与自动微分。单项可以使用 `expectation_pauli_string`。

`statevector_expectation` 等低层接口直接接收连续复振幅切片；普通应用优先通过 [`arcqml-sim`](../arcqml-sim/README.md) 调用，以获得状态校验与自动微分集成。

许可证见 [混合许可说明](../../README.md#发行状态与许可证)。
