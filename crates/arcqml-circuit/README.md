# arcqml-circuit

`arcqml-circuit` 定义量子电路的数据结构：量子比特数、按执行顺序排列的门操作，以及参数化门引用的参数表。它只描述电路，不负责执行；执行请使用 [`arcqml-sim`](../arcqml-sim/README.md)。

## 添加依赖

```toml
[dependencies]
arcqml-circuit = { path = "PATH_TO_ARCQML/crates/arcqml-circuit" }
```

## 构造第一条电路

```rust
use arcqml_circuit::Circuit;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(2)?;
    circuit.h(0usize)?.cnot(0usize, 1usize)?;

    assert_eq!(circuit.num_qubits(), 2);
    assert_eq!(circuit.len(), 2);
    assert_eq!(circuit.depth(), 2);
    circuit.validate()?;
    Ok(())
}
```

量子比特从 `0` 开始编号。`Circuit::new(0)` 会返回错误。对受控门，第一个参数是控制位，第二个参数是目标位。

## 内置门

| 类别 | 便捷方法 |
| --- | --- |
| 单比特固定门 | `i`、`x`、`y`、`z`、`h`、`s`、`sdg`、`t`、`tdg`、`sx`、`sxdg` |
| 单比特参数门 | `rx`、`ry`、`rz`、`phase`、`u1`、`u2`、`u3` |
| 受控参数门 | `cp`、`crx`、`cry`、`crz` |
| 双比特固定门 | `cnot` / `cx`、`cy`、`cz`、`ch`、`cs`、`ct`、`swap`、`iswap`、`dcx`、`ecr` |
| 双比特参数门 | `rxx`、`ryy`、`rzz`、`rzx`、`fsim` |
| 多比特门 | `toffoli`、`cswap`、`mcx` |

角参数均使用弧度。Pauli 旋转门采用：

$$
R_P(\theta)=
\cos\left(\frac{\theta}{2}\right)I
-i\sin\left(\frac{\theta}{2}\right)P
$$

也可以先用 `Gate` 构造门，再调用 `add_gate`。自定义门使用 `Gate::custom_unitary(name, arity, matrix)`；矩阵是行主序 `Vec<Complex64>`，元素数必须为：

$$
4^{\mathrm{arity}}
$$

构造时会检查维度、有限值与酉性。

## 可训练、固定与共享参数

参数门有三种常用写法。

### 自动创建可训练参数

```rust
let mut circuit = Circuit::new(1)?;
circuit.ry(0.3, 0usize)?;
assert_eq!(circuit.num_parameters(), 1);
```

### 使用固定参数

```rust
circuit.ry_fixed(0.3, 0usize)?;
```

`_fixed` 方法不会把角度加入参数表，适合编码常量或不可训练门。

### 复用同一个参数

```rust
let theta = circuit.add_parameter(0.3)?;
circuit.ry_param(theta, 0usize)?;
circuit.rz_param(theta, 0usize)?;
```

两个门共享同一个 `ParameterId`，反向传播时梯度贡献会累加到同一参数。`add_parameter_tensor` 可接受标量 `F32` 或 `F64` Tensor。

## 查询、复制与拼接

| 接口 | 用途 |
| --- | --- |
| `operations` | 按执行顺序读取门操作 |
| `parameters` | 读取参数切片，供优化器使用 |
| `parameter` / `parameter_id` / `parameter_name` | 按 ID 或名称查找参数 |
| `depth` | 按量子比特资源计算电路深度 |
| `validate` | 验证量子比特、参数引用与名称唯一性 |
| `clear` | 删除全部操作和参数，保留量子比特数 |
| `clone` / `deep_clone` | 创建不共享参数存储与梯度的深副本 |

`append(right)` 消费右电路并追加操作，自动重映射其参数 ID。`append_with_bindings` 可把右侧参数绑定到左侧已有参数：

```rust
use arcqml_circuit::{Circuit, ParameterBinding};

let mut left = Circuit::new(1)?;
let shared = left.add_parameter(0.3)?;
left.ry_param(shared, 0usize)?;

let mut right = Circuit::new(1)?;
right.rz(0.2, 0usize)?;
let source = right.parameter_id("rz_q0_theta_0").unwrap();

let report = left.append_with_bindings(
    right,
    &[ParameterBinding::new(source, shared)],
)?;
assert_eq!(report.appended_operations(), 1);
```

两条电路的量子比特数必须一致。拼接成功后会清空结果电路中的参数梯度。

## 电路 JSON

```rust
let structure = circuit.to_json()?;
let restored = Circuit::from_json(&structure)?;
```

电路 JSON 保存门结构、量子比特和参数名称；`from_json` 会把可训练参数初始化为 `F64` 零标量。

训练后的参数值应另用 [`arcqml-checkpoint`](../arcqml-checkpoint/README.md) 保存。应用若需要完整模型，应同时管理电路结构 JSON 与权重 JSON，并保持版本一致。

