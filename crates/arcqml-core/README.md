# arcqml-core

`arcqml-core` 是 ArcQML 的基础数据层，定义共享所有权的 CPU 稠密 `Tensor`、可训练 `Parameter`、底层 `Storage`、元数据与动态自动微分引擎。大多数应用通过门面 crate `arcqml` 使用这些类型；只有需要精确控制 Tensor 和梯度时才直接依赖本 crate。

## 添加依赖

```toml
[dependencies]
arcqml-core = { path = "PATH_TO_ARCQML/crates/arcqml-core" }
arcqml-linalg = { path = "PATH_TO_ARCQML/crates/arcqml-linalg" } # 仅自动微分示例需要
```

默认启用 `parallel` feature，用于 Rayon 线程池支持。

## Tensor 入门

```rust
use arcqml_core::{DType, Tensor};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let scalar = Tensor::new(0.3_f64)?;
    let vector = Tensor::new(vec![1.0_f64, 2.0, 3.0])?;
    let matrix = Tensor::zeros_with_dtype(vec![2, 3], DType::F32)?;

    assert_eq!(scalar.shape(), &[]);
    assert_eq!(vector.shape(), &[3]);
    assert_eq!(matrix.shape(), &[2, 3]);
    Ok(())
}
```

支持的存储类型为 `F32`、`F64`、`C64`、`I64` 与 `Bool`。当前计算设备与布局只有 `Device::Cpu` 和 `Layout::Dense`；`Cuda` 与 `Sparse` 只是类型占位，尚无计算实现。

## 常用接口

| 类别 | 接口 |
| --- | --- |
| 构造 | `new`、`zeros`、`zeros_with_dtype`、`randn`、带 seed 的随机构造 |
| 元数据 | `shape`、`ndim`、`numel`、`dtype`、`device`、`layout`、`strides` |
| 内存 | `storage`、`storage_mut`、`is_contiguous`、`contiguous` |
| 视图 | `reshape`、`transpose`、`transpose_dims` |
| 自动微分 | `set_requires_grad`、`backward`、`backward_with_grad`、`grad`、`zero_grad` |
| 图控制 | `retain_grad`、`detach`、`no_grad`、`is_grad_enabled` |
| 复制 | `clone`、`deep_clone` |

`Tensor::clone()` 只复制共享句柄，底层存储和自动微分元数据仍然共享。需要完全独立的数据和梯度状态时使用 `deep_clone()`。

转置通常产生非连续 view。传给要求连续 row-major buffer 的算子或模拟器前，应先调用 `contiguous()`。

## 自动微分

```rust
use arcqml_core::Tensor;
use arcqml_linalg::{mul, sum};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let x = Tensor::new(vec![1.0_f64, 2.0, 3.0])?;
    x.set_requires_grad(true);

    let y = sum(&mul(&x, &x)?)?;
    y.backward()?;

    println!("gradient = {:?}", x.grad());
    Ok(())
}
```

反向传播遵循链式法则：

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
$$

- `backward()` 只适用于标量输出。
- 非标量输出使用 `backward_with_grad(upstream)`，且上游梯度形状与 dtype 必须匹配。
- 叶子 Tensor 默认保留梯度；中间 Tensor 只有调用 `retain_grad()` 后才保留。
- 梯度会累积；开始新一轮优化前应调用 `zero_grad()`。
- `no_grad()` 返回作用域守卫，适合验证与推理。

## Parameter

`Parameter` 为 Tensor 增加“可训练参数”语义，电路与优化器共享同一参数状态：

```rust
use arcqml_core::{Parameter, Tensor};

let parameter = Parameter::try_new(Tensor::new(0.3_f64)?)?;
parameter.set_name("theta");
parameter.freeze();
assert!(!parameter.trainable());
parameter.unfreeze();
```

`Parameter` 底层只支持 `F32`、`F64` 与 `C64`，但当前 SGD 和 Adam 只更新 `F32` 与 `F64`。`freeze` 会使优化器跳过参数；它不删除已有梯度。

## 并行线程

启用 `parallel` feature 后，可在第一次并行计算前设置 Rayon 全局线程数：

```rust
arcqml_core::init_rayon(4)?;
println!("threads = {}", arcqml_core::rayon_num_threads());
```

全局线程池只能初始化一次。应用应在启动阶段调用，而不是在训练循环中重复调用。

错误统一使用 `ArcQmlError` / `Result`。高级算子位于 [`arcqml-linalg`](../arcqml-linalg/README.md)，完整框架入口见 [`arcqml`](../arcqml/README.md)。许可证见 [混合许可说明](../../README.md#发行状态与许可证)。
