# arcqml-checkpoint

`arcqml-checkpoint` 把 `Circuit` 的可训练参数保存为可读、顺序稳定的 JSON，并以“全部校验通过后再写入”的方式加载权重。

## 添加依赖

```toml
[dependencies]
arcqml-checkpoint = { path = "PATH_TO_ARCQML/crates/arcqml-checkpoint" }
arcqml-circuit = { path = "PATH_TO_ARCQML/crates/arcqml-circuit" }
```

通过门面 crate 使用时：

```rust
use arcqml::checkpoint::{load_weights, save_weights};
```

## 保存与加载

```rust
use arcqml_checkpoint::{load_weights, save_weights};
use arcqml_circuit::Circuit;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut trained = Circuit::new(1)?;
    trained.ry(0.3, 0usize)?;
    save_weights(&trained, "weights.json")?;

    // 应用代码负责重建相同参数命名的电路。
    let mut restored = Circuit::new(1)?;
    restored.ry(0.0, 0usize)?;
    load_weights(&restored, "weights.json")?;
    Ok(())
}
```

`load_weights` 接受 `&Circuit`，因为参数内部使用共享可变状态；它不会改变门拓扑。

## JSON 格式

```json
{
  "format": "arcqml/weights",
  "num_qubits": 2,
  "parameters": [
    {
      "name": "ry_q0_theta_0",
      "dtype": "f64",
      "value": 0.36154268969437336
    }
  ]
}
```

参数按名称排序写入，因此相同参数状态会得到稳定的记录顺序。公开数据结构包括 `WeightCheckpoint`、`ParameterRecord` 与格式常量 `FORMAT`。

## 加载契约

加载前会完整检查：

- `format` 必须是 `arcqml/weights`；
- 量子比特数必须相同；
- 参数名集合必须完全一致，且不能重复；
- dtype 必须匹配，当前只支持标量 `f32` 与 `f64`；
- 参数值必须是有限数值。

只有全部检查通过后才更新目标参数并清空旧梯度。校验失败不会留下部分更新。

## TODO

权重文件不包含：

- 电路门结构与执行顺序；
- `ParameterId` 的内部引用关系；
- 梯度与自动微分图；
- Adam 的一阶矩、二阶矩和步数；
- 训练数据、随机数状态或应用配置。

当前代码只提供 `save_weights` 与 `load_weights`，没有完整训练状态的 `save_checkpoint` / `load_checkpoint`。以上功能会在后续版本中更新。

