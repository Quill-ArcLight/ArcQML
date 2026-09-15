# arcqml-loss

`arcqml-loss` 提供面向 ArcQML Tensor 的可微损失函数。返回值都是标量 Tensor，可直接调用 `backward()`，把梯度传回经典 Tensor 或量子电路参数。

## 添加依赖

```toml
[dependencies]
arcqml-core = { path = "PATH_TO_ARCQML/crates/arcqml-core" }
arcqml-loss = { path = "PATH_TO_ARCQML/crates/arcqml-loss" }
```

## 最小示例

```rust
use arcqml_core::Tensor;
use arcqml_loss::mse_loss;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let prediction = Tensor::new(0.8_f64)?;
    prediction.set_requires_grad(true);
    let target = Tensor::new(1.0_f64)?;
    let loss = mse_loss(&prediction, &target)?;
    loss.backward()?;
    println!("loss = {}", loss.value()?);
    Ok(())
}
```

## 自定义损失

Rust 用户可以直接组合 `arcqml-linalg` 算子，返回标量 Tensor 后调用 `backward()`，无需实现新的损失类型。例如，与上述依赖一起添加 `arcqml-linalg` 后：

```rust
use arcqml_core::Tensor;
use arcqml_linalg::{mean, mul, square, sub};

let prediction = Tensor::new(vec![0.8_f64, -0.4])?;
prediction.set_requires_grad(true);
let target = Tensor::new(vec![0.2_f64, -0.2])?;
let weights = Tensor::new(vec![1.0_f64, 3.0])?;
let error = sub(&prediction, &target)?;
let loss = mul(&mean(&mul(&weights, &square(&error)?)?)?, &Tensor::new(0.5_f64)?)?;
loss.backward()?; // prediction 的梯度约为 [0.3, -0.3]
```

这里按元素数归约，而不是按权重和归约。完整的输入校验、量子电路训练、梯度检查和 `CustomOp` 专用反向见[自定义损失教程](../../docs/tutorial/custom_loss.md)与[可运行 Rust 示例](../../examples/rust/custom_loss.rs)。当前 Python 尚未导出通用算子组合接口。

## 损失函数

### 半均方误差

`mse_loss(prediction, target)` 使用：

$$
L_{\mathrm{MSE}}=
\frac{1}{2N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$

### 平均绝对误差

`l1_loss(prediction, target)` 使用：

$$
L_{\mathrm{L1}}=
\frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|
$$

### Pauli-Z 二元负对数似然

`binary_nll_loss(prediction, labels)` 把每个预测值解释为 Pauli-Z 期望值：

$$
p(0)=\frac{1+z}{2},\qquad
p(1)=\frac{1-z}{2}
$$

输入必须是一维 Tensor，标签是等长的 `0` 或 `1` 切片。概率会在 `1e-12` 边界处截断，避免对数奇点。

### 二元 logits 交叉熵

`binary_cross_entropy_with_logits_loss(logits, targets)` 使用数值稳定形式：

$$
L=
\frac{1}{N}\sum_i
\left[
\max(x_i,0)+\log\left(1+e^{-|x_i|}\right)-y_i x_i
\right]
$$

`targets` 必须与 `logits` 形状和 dtype 完全相同，且每个值位于闭区间 `[0, 1]`。

反向使用整体导数 `(sigmoid(x) - y) / N`，包括零 logits；若 targets 需要梯度，其导数为 `-x / N`。两者均乘以上游梯度，不沿前向的 clamp/abs 组合求导。

### 多分类交叉熵

`cross_entropy_loss(logits, labels)` 接受形状为 `[batch_size, class_count]` 的 logits，以及每个样本的类别索引。内部使用稳定的 `log_softmax`：

$$
L=-\frac{1}{B}\sum_{i=1}^{B}\log p_{i,y_i}
$$

标签必须小于类别数。

## 输入约束

- 输入必须非空，且 dtype 为 `F32` 或 `F64`。
- MSE 与 L1 要求两侧 dtype 一致，形状可按底层 NumPy 风格规则广播。
- 二元 logits 交叉熵不做广播，两个输入形状必须完全一致。
- 多分类交叉熵只接受二维 logits。
- 已覆盖的输入校验失败时返回 `LossError`；接口并未统一拒绝所有非有限输入，也不保证计算结果总是有限。

当前 `binary_nll_loss` 使用的 `1e-12` 截断量在 F32 的 ±1 附近不能有效表示，边界输入可能得到 NaN 或无穷值。该路径应优先使用 F64，并在训练中检查结果有限性；它与接收 logits 的 BCE 不是同一种输入语义。

量子训练中，`prediction` 通常来自 `StateVectorSimulator::run` 或 batch `run`。完整训练流程见 [`arcqml`](../arcqml/README.md#一次完整训练步骤)，优化器见 [`arcqml-optim`](../arcqml-optim/README.md)。许可证见 [混合许可说明](../../README.md#发行状态与许可证)。
