# arcqml-optim

`arcqml-optim` 根据 `Parameter` 或叶子 `Tensor` 中已经累积的梯度原地更新数值。当前提供 SGD 与 Adam；两者只更新 `F32` 和 `F64` 参数。

## 添加依赖

```toml
[dependencies]
arcqml-optim = { path = "PATH_TO_ARCQML/crates/arcqml-optim" }
```

## 推荐训练顺序

```text
forward → loss → backward → step → zero_grad
```

```text
创建并复用 optimizer
循环：
    prediction = forward(...)
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step(parameters)
    optimizer.zero_grad(parameters)
```

梯度由 `backward()` 累积。推荐在 `step` 后清空当前梯度，以避免下个循环中错误的梯度累加。

## SGD

```rust
use arcqml_optim::Sgd;

let optimizer = Sgd::new(0.01, 0.0)?;
let stats = optimizer.step(circuit.parameters())?;
```

更新规则为：

$$
\theta_{t+1}
=
\theta_t-\eta\left(g_t+\lambda\theta_t\right)
$$

这里的 `weight_decay` 是与梯度耦合的 L2 项。`set_learning_rate` 与 `set_weight_decay` 可修改后续步骤使用的超参数。

## Adam

```rust
use arcqml_optim::Adam;

let mut optimizer = Adam::new(
    0.01,  // learning_rate
    0.9,   // beta1
    0.999, // beta2
    1e-8,  // epsilon
    0.0,   // weight_decay
)?;
let stats = optimizer.step(circuit.parameters())?;
```

ArcQML Adam 先把耦合权重衰减加入梯度：

$$
\tilde{g}_t=g_t+\lambda\theta_t
$$

然后更新一阶矩与二阶矩：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)\tilde{g}_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)\tilde{g}_t^2
$$

经过偏差修正后更新参数：

$$
\theta_{t+1}
=
\theta_t-
\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

Adam 状态槽按传入切片的位置对应参数。训练期间必须复用同一个优化器，并保持参数数量、顺序、dtype 和元素数稳定。每次调用 `step` 都会增加 `step_count`，即使所有参数都被跳过。

## 公共行为

| 接口 | 用途 |
| --- | --- |
| `step(&[Parameter])` | 更新电路或模型参数 |
| `step_tensors(&[Tensor])` | 直接更新启用梯度的叶子 Tensor |
| `zero_grad` | 清空 Parameter 梯度 |
| `zero_grad_tensors` | 清空 Tensor 梯度 |

`StepStats` 提供 `updated`、`skipped_frozen` 与 `skipped_no_grad` 计数。被冻结、未启用梯度或尚无梯度的项会被跳过，不会被当作零梯度更新。

超参数必须是有限数值；学习率和权重衰减非负，`beta1` 与 `beta2` 位于 `[0, 1)`，`epsilon` 为正数。输入非法时返回 `OptimError`。

当前检查点模块不保存 Adam 状态，重建优化器会重新从零矩开始。许可证见 [混合许可说明](../../README.md#发行状态与许可证)。
