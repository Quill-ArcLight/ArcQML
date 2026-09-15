# arcqml-unitary

`arcqml-unitary` 从量子电路生成整体稠密酉矩阵，并计算与目标酉矩阵之间的全局相位不变保真度、损失和解析梯度。它适合小规模门综合与电路验证，不适合大规模量子比特系统。

## 添加依赖

```toml
[dependencies]
arcqml-circuit = { path = "PATH_TO_ARCQML/crates/arcqml-circuit" }
arcqml-unitary = { path = "PATH_TO_ARCQML/crates/arcqml-unitary" }
```

本 crate 需要匹配的 ArcQML Runtime 静态库。构建配置见[根 README](../../README.md#30-秒上手rust)。

## 生成整体酉矩阵

```rust
use arcqml_circuit::Circuit;
use arcqml_unitary::unitary_from_circuit;

let mut circuit = Circuit::new(1)?;
circuit.h(0usize)?;

let unitary = unitary_from_circuit(&circuit)?;
assert_eq!(unitary.dimension(), 2);
assert_eq!(unitary.as_row_major().len(), 4);
```

`DenseUnitary` 使用行主序 `C64` 数据，矩阵索引约定为 `U[output, input]`。`unitary_from_tensor` 可从二维 `C64` Tensor 导入，并使用默认容差 `1e-10` 验证酉性。

对于 `n` 个量子比特，矩阵维度与元素数分别为：

$$
d=2^n
$$

$$
d^2=4^n
$$

内存会快速增长。构造前应限制量子比特数，并处理维度溢出或分配失败错误。

## 比较和拟合

```rust
use arcqml_circuit::Circuit;
use arcqml_unitary::{unitary_from_circuit, unitary_loss};

let mut target_circuit = Circuit::new(1)?;
target_circuit.ry_fixed(0.8, 0usize)?;
let target = unitary_from_circuit(&target_circuit)?;

let mut ansatz = Circuit::new(1)?;
ansatz.ry(0.2, 0usize)?;

let loss = unitary_loss(&target, &ansatz)?;
loss.backward()?;
println!("loss = {}", loss.value()?);
```

保真度定义为：

$$
F(U_{\mathrm{target}},U(\theta))
=
\frac{
\left|\mathrm{Tr}\left(U_{\mathrm{target}}^\dagger U(\theta)\right)\right|^2
}{d^2}
$$

损失为：

$$
L(\theta)=1-F(U_{\mathrm{target}},U(\theta))
$$

该定义不受整体相位影响。实现会把浮点保真度截断到 `[0, 1]`。

## API 速查

| 接口 | 返回 | 用途 |
| --- | --- | --- |
| `unitary_from_circuit` | `DenseUnitary` | 生成电路整体矩阵 |
| `unitary_from_tensor` | `DenseUnitary` | 导入并验证二维 `C64` Tensor |
| `unitary_fidelity` | `f64` | 只计算全局相位不变保真度 |
| `unitary_loss_value` | `f64` | 只计算损失，适合日志和数值检查 |
| `unitary_loss` | 标量 `Tensor` | 建立可微节点并支持 `backward()` |

`unitary_loss` 使用解析伴随方法把梯度累积到电路参数。目标矩阵与电路的量子比特数必须相同，电路可训练参数必须是 `F32` 或 `F64`。

内部通过批量演化全部计算基生成整体矩阵，不逐门显式构造完整矩阵；但最终仍需要保存 `4^n` 个复数。

许可证见 [混合许可说明](../../README.md#发行状态与许可证)。
