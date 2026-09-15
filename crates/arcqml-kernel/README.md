# arcqml-kernel

`arcqml-kernel` 是 ArcQML 状态向量模拟器的底层执行层。它负责把电路参数解析为可执行门、校验振幅缓冲区与量子比特，并通过稳定 ABI 调用 ArcQML Runtime 完成门应用、伴随门应用和参数导数计算。

> **普通应用不应直接依赖本 crate。** 请优先使用 [`arcqml-sim`](../arcqml-sim/README.md) 或门面 crate [`arcqml`](../arcqml/README.md)，它们提供状态管理、可观测量和自动微分集成。

## 组件边界

```text
arcqml-circuit
      │ 参数与 Gate
      ▼
arcqml-kernel
      │ 校验、绑定、ABI 调度
      ▼
arcqml-runtime-sys / arcqml-runtime-abi
      │ 稳定 C ABI
      ▼
ArcQML Runtime
```


## 添加依赖

仅限开发模拟器或 Runtime 集成层时：

```toml
[dependencies]
arcqml-kernel = { path = "PATH_TO_ARCQML/crates/arcqml-kernel" }
```

编译前设置 `ARCQML_RUNTIME_LIB_DIR`，并确保 Runtime 与接口源码版本一致。

## 参数绑定

参数化门在 `arcqml-circuit` 中保存 `ParameterId`，不复制参数数值。执行前使用：

- `resolve_gate(circuit, gate)`：解析门与其参数 Tensor；
- `bind_gate(circuit, gate)`：读取当前标量值并生成 `BoundGate`；
- `BoundParameter`：记录参数 ID 与 Tensor 绑定。

因此优化器更新参数后，下一次绑定会读取新值。

## 底层状态向量接口

| 接口 | 作用 |
| --- | --- |
| `apply_gate` | 对单态振幅原地施加固定参数门 |
| `apply_gate_batch` | 对连续行主序 batch 原地施加门 |
| `apply_adjoint_gate` | 对单态施加门的共轭转置 |
| `apply_adjoint_gate_batch` | 对 batch 施加门的共轭转置 |
| `parameter_derivative` | 返回单态参数导数态 |
| `parameter_derivative_into` | 把单态参数导数写入调用方缓冲区 |
| `parameter_derivative_batch` | 返回 batch 参数导数态 |

单态振幅长度必须满足：

$$
|\psi|=2^n
$$

batch 使用连续行主序布局：

$$
[B,2^n]
$$

参数导数接口计算：

$$
\frac{\partial}{\partial p}\left(U(p)|\psi\rangle\right)
$$

它不是完整 loss 梯度；调用方仍需构造伴随状态和向量—Jacobian 乘积。普通用户应让 `StateVectorSimulator::run` 完成这一过程。

## 不建议直接调用此 crate !!!