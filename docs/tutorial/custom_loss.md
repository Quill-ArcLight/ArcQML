# 使用内置算子定义自定义损失

本教程面向 ArcQML 0.1.0 的 **Rust 接口**。我们将用内置算子实现加权半均方误差，把它接到量子电路输出之后，并验证梯度与训练结果。最后介绍如何通过 `CustomOp` 为损失中的某个操作提供专用反向规则。

完整可运行程序：[examples/rust/custom_loss.rs](../../examples/rust/custom_loss.rs)。它不需要下载数据，包含经典梯度检查、量子电路训练和自定义反向示例。下文代码片段按这个程序的结构讲解，完整运行时以该文件为准。

当前 Python 扩展提供内置 MSE、BCE-with-logits 和 Tensor 的反向接口，但尚未导出通用 Tensor 算术、归约算子及 `CustomOp`。本教程的算子组合写法不能直接翻译为当前 Python API；把输出转换为 NumPy 后计算损失，也不会自动连接回 ArcQML 的计算图。

## 1. 准备与运行

在仓库根目录执行。使用当前稳定版 Rust，并按[技术手册的环境配置](../ArcQML技术手册.md#22-rust-环境与-runtime-配置)准备匹配平台的 Runtime；当前源码使用了 Rust 1.88 起才支持的语法，完整依赖的最低版本尚未单独验证。

Windows PowerShell：

```powershell
$env:ARCQML_RUNTIME_LIB_DIR = (Resolve-Path ".\libs\x86_64-pc-windows-msvc").Path
cargo run -p arcqml --example custom_loss
```

Linux x86_64 GNU：

```bash
export ARCQML_RUNTIME_LIB_DIR="$PWD/libs/x86_64-unknown-linux-gnu"
cargo run -p arcqml --example custom_loss
```

如果是在自己的应用项目中使用，可添加路径依赖：

```toml
[dependencies]
arcqml = { path = "PATH_TO_ARCQML/crates/arcqml" }
num-complex = "0.4" # 本例构造复数初态时使用
```

Runtime 路径仍需指向发行包中的对应库。仅使用经典 Tensor 和损失算子时，可以直接依赖 `arcqml-core` 与 `arcqml-linalg`；本例还会执行量子电路，因此使用门面 crate `arcqml`。

## 2. 自定义损失就是返回 Tensor 的函数

只要前向计算使用带反向规则的 ArcQML 算子，就可以把它们组合成新的损失，无需登记一个“损失函数类型”，也无需手写每个输入的梯度。

以加权半均方误差为例，设预测、目标和固定权重均包含 N 个元素：

$$
L=\frac{1}{2N}\sum_{i=1}^{N}w_i(\hat y_i-y_i)^2,
\qquad w_i\ge 0.
$$

这里按**元素总数 N**平均，不按权重和平均。把全部权重乘以 2，会让损失和梯度都乘以 2。若需要按权重和归一化，应另外定义分母，并拒绝权重和为零的输入。

本教程采用以下输入契约，便于集中学习计算图：

- `prediction`、`target`、`weights` 均为非空、有限的 CPU Dense F64 Tensor。
- 三个输入的形状完全相同，本例不启用广播。
- 使用完整连续存储，存储长度等于逻辑元素数量；不接收转置、偏移或共享大存储的子视图。
- 权重非负。目标和权重默认作为常量，不设置 `requires_grad`。

这些是示例函数主动检查的约束，并不代表 ArcQML 的所有算子都只支持 F64。量子模拟器的 `run` 返回 F64，因此本例统一使用 F64 常量，避免隐式类型转换的误解。

程序首先导入接口并定义应用层错误类型：

```rust
use arcqml::prelude::*;
use num_complex::Complex64;

type AppResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
```

完整示例中的 `validate_input` 检查上述类型、布局和有限性约束。损失函数在校验之后，仅用四类可微运算构造计算图：

```rust
fn weighted_mse(prediction: &Tensor, target: &Tensor, weights: &Tensor) -> AppResult<Tensor> {
    for tensor in [prediction, target, weights] {
        validate_input(tensor)?;
    }
    if prediction.shape() != target.shape() || prediction.shape() != weights.shape() {
        return Err("prediction, target and weights must have the same shape".into());
    }
    if weights.storage().as_f64_slice().unwrap().iter().any(|&w| w < 0.0) {
        return Err("weights must be nonnegative".into());
    }
    let error = sub(prediction, target)?;
    let weighted_squared_error = mul(weights, &square(&error)?)?;
    Ok(mul(&mean(&weighted_squared_error)?, &Tensor::new(0.5_f64)?)?)
}
```

数据流为 `prediction → sub → square → mul → mean → mul → loss`。读取存储在这里仅用于输入校验；真正构造损失的数值运算始终在 Tensor 图中完成。`mean` 把任意受支持形状归约成 `shape=[]` 的标量，使结果可以直接调用无参 `backward()`。

返回 `AppResult<Tensor>` 是示例的错误处理选择，不是框架强制要求。应用也可以定义自己的错误类型。

## 3. 先验证经典梯度

在接入量子电路之前，用两个预测值验证损失：

```rust
let prediction = Tensor::new(vec![0.8_f64, -0.4])?;
prediction.set_requires_grad(true);
let target = Tensor::new(vec![0.2_f64, -0.2])?;
let weights = Tensor::new(vec![1.0_f64, 3.0])?;

let loss = weighted_mse(&prediction, &target, &weights)?;
loss.backward()?;
println!("loss = {}", loss.value()?); // 约 0.12
println!("gradient = {:?}", prediction.grad());
```

解析导数为：

$$
\frac{\partial L}{\partial\hat y_i}
=\frac{w_i(\hat y_i-y_i)}{N}.
$$

因此预测梯度应为 `[0.3, -0.3]`。`set_requires_grad(true)` 必须在前向构图之前设置；常量张量不会因为之后调用 `backward()` 就自动成为求导变量。

完整示例还使用中心差分进行独立检查：

$$
\frac{\partial L}{\partial\hat y_i}
\approx\frac{L(\hat y+h e_i)-L(\hat y-h e_i)}{2h},
\qquad h=10^{-6}.
$$

示例对解析结果使用 `1e-12` 的绝对容差，对差分结果使用 `1e-8`。差分步长与容差需要结合 dtype 和数值尺度选择，不能把这一组参数原样用于所有损失。

## 4. 接入可训练量子电路

准备一批两条初态：第一行为 `|0⟩`，第二行为 `|1⟩`。它们共享一个可训练的 RY 门，并测量 Pauli-Z：

```rust
let zero = Complex64::new(0.0, 0.0);
let one = Complex64::new(1.0, 0.0);
let states = Tensor::new(TensorData::FlatC64 {
    data: vec![one, zero, zero, one],
    shape: vec![2, 2],
})?;
let simulator = BatchStateVectorSimulator::from_state_tensor(1, states)?;
let mut circuit = Circuit::new(1)?;
circuit.ry(0.3, 0usize)?;
let observable = SparsePauliOp::single(1, 0usize, Pauli::Z, 1.0)?;
let target = Tensor::new(vec![0.2_f64, -0.2])?;
let weights = Tensor::new(vec![1.0_f64, 3.0])?;
let optimizer = Sgd::new(0.1, 0.0)?;
```

`run` 返回 `[2]` 形状的预测，分别为 `cos(theta)` 和 `-cos(theta)`。门参数已在 `ry` 中登记为可训练参数，不需要对预测再调用 `set_requires_grad`。

每轮训练都重新构造前向图：

```rust
for _ in 0..80 {
    let prediction = simulator.run(&circuit, &observable)?;
    let loss = weighted_mse(&prediction, &target, &weights)?;
    loss.backward()?;
    optimizer.step(circuit.parameters())?;
    optimizer.zero_grad(circuit.parameters());
}
```

本例使用新建的电路参数，首轮没有遗留梯度；每轮更新后清空梯度。如果在进入循环前已对这些参数执行过反向传播，应额外清空一次。

损失的反向规则先生成预测梯度，再由量子伴随节点计算电路参数梯度。损失函数本身不需要了解 RY 门如何求导。

此例可以进一步化简为：

$$
L(\theta)=(\cos\theta-0.2)^2,
\qquad
\frac{\partial L}{\partial\theta}
=-2(\cos\theta-0.2)\sin\theta.
$$

示例在第一步检查这个解析梯度，从而验证“自定义损失 → batch 输出 → 电路参数”的完整链路。

最后重新运行前向读取更新后的预测。Rust 的 `no_grad` 必须绑定守卫，才能在整个作用域内关闭梯度记录：

```rust
let _guard = no_grad();
let prediction = simulator.run(&circuit, &observable)?;
let final_loss = weighted_mse(&prediction, &target, &weights)?.value()?;
```

在随附 Windows x86_64 Runtime 上验证时，程序输出如下；不同构建的末尾小数可能略有差异：

```text
Gradient checks passed (weighted MSE and custom log-cosh).
Circuit gradient: -0.4464343907 (analytic: -0.4464343907).
step 00: loss = 0.5705332118
step 20: loss = 0.0041978216
step 40: loss = 0.0000008903
step 60: loss = 0.0000000002
final predictions: [0.20000018681924192, -0.20000018681924192]
final loss: 3.490e-14
```

## 5. 扩展损失时保持计算图连接

多项损失可以用 `add` 合成一个标量，再调用一次 `backward()`。例如把自定义误差与另一个标量正则项组合为 `add(&data_loss, &regularization)?`。正则项也必须由原参数 Tensor 的可微运算构成。

以下行为会影响梯度链路：

| 操作 | 对计算图的影响 |
| --- | --- |
| 用 `sub`、`square`、`mul`、`mean` 等可微算子计算 | 在当前线程启用梯度记录且存在可微输入时连接计算图 |
| `loss.value()` 用于日志，然后仍对原 `loss` 反向 | 不影响原 Tensor 的计算图 |
| 用 `value()`、存储或 NumPy 提取数值，在图外计算后重新 `Tensor::new` | 新建叶子张量，原来的依赖关系不会自动恢复 |
| `detach()` | 断开返回值的上游图；底层存储仍共享 |
| `deep_clone()` | 创建独立叶子副本，不继承原计算图 |
| `clone()` | 复制共享句柄，保留同一个图节点 |
| 在 `no_grad` 作用域中计算损失 | 不记录这些前向运算，不能靠事后打开梯度恢复 |

默认反向会释放图。如果多项损失共享前向结果，优先合并后反向；确需分别反向时，应在第一次以及后续仍需保留图的反向中使用 `backward_with_grad_retain_graph`。`retain_grad()` 只保留中间梯度，不会保留计算图。当前共享图释放后的错误检查还不完整，不应依赖重复反向一定报错。

本教程的例子使用独立连续张量和每轮一次反向。当前仓库仍有涉及视图的归约、梯度累加和优化器更新问题；仅检查 `is_contiguous()` 或调用 `contiguous()` 并不能排除共享较大底层存储的子视图问题。不要把本例通过理解为所有 dtype、视图与算子组合均已验证。

## 6. 什么时候需要专用反向

通常不需要手写反向。但以下情况需要单独考虑：前向调用了 ArcQML 图外的计算；现有算子没有所需反向规则；或者前向为了数值稳定使用了不可微中间表达式，而需要明确指定整体函数的导数。

例如 BCE-with-logits 的稳定前向包含 `max(x,0)` 与 `abs(x)`。当前 `clamp` 在边界选择梯度 1，实数 `abs` 在零点选择梯度 0；把这两个局部规则机械组合，不能在零点得到 BCE 的整体导数。内置 `binary_cross_entropy_with_logits_loss` 已使用专用反向：

$$
\frac{\partial L}{\partial x_i}=\frac{\sigma(x_i)-y_i}{N}.
$$

因此应直接复用内置 BCE，不要为了重写相同功能而重新拼接这段稳定前向。这个修复属于当前 Rust 源码；旧 wheel 需要重新构建才能包含它。

自动微分沿实际计算图应用局部反向规则，不会普遍执行符号化简。对整体可微的函数，应检查关键点的整体导数；对 L1 等本身不可微的函数，应说明选择的次梯度。不可微点的中心差分不一定等于框架约定的反向值。

## 7. 进阶：用 CustomOp 定义 log-cosh 的反向

示例中还实现了一个标量残差损失：

$$
\ell(r)=\log\cosh r
=|r|+\log(1+e^{-2|r|})-\log 2,
\qquad \ell'(r)=\tanh r.
$$

它在小残差处近似 `r²/2`，在大残差处近似 `|r| - log(2)`。稳定前向避免直接计算大输入的 `cosh`；专用反向直接计算 `tanh`。这个例子用于展示显式反向机制，并不表示所有含 `abs` 的组合都会产生错误梯度。

`CustomOp` 需要提供三部分：

| 成员 | 职责 |
| --- | --- |
| `name` | 为图节点提供诊断名称 |
| `forward` | 返回一个不需要梯度的输出，以及反向需要的上下文 |
| `backward` | 根据上下文和上游梯度，按输入顺序返回梯度槽 |

完整示例的 `LogCosh` 只接受一个有限 F64 标量，使用 `Context = f64` 保存残差快照。其核心前向为：

```rust
let r = input.value()?;
let a = r.abs();
let value = a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2;
Ok((Tensor::new(value)?, r))
```

反向返回的不是单独的 `tanh(r)`，而是乘上上游梯度后的局部向量—雅可比积（VJP）：

```rust
Ok(vec![Some(Tensor::new(grad_output.value()? * r.tanh())?)])
```

这里允许在前向中使用普通 `f64` 运算，是因为 `apply_custom_op` 会用我们提供的反向规则重新建立输入到输出的连接：

```rust
let prediction = Tensor::new(0.8_f64)?;
prediction.set_requires_grad(true);
let target = Tensor::new(0.2_f64)?;
let residual = sub(&prediction, &target)?;
let loss = apply_custom_op(LogCosh, std::slice::from_ref(&residual))?;
loss.backward()?;
// prediction 的梯度约为 tanh(0.6)。
```

`apply_custom_op` 自动在禁用梯度记录的作用域内执行 `forward`，再把结果作为一个原子节点连接到输入。不应在 `forward` 或 `backward` 内对父节点调用 `backward()`，也不应自行更新参数。

如果操作有多个输入，必须返回同样数量、相同顺序的 `Option<Tensor>`；每个梯度应匹配对应输入的形状和 dtype。`None` 表示不返回该输入的梯度，不能用来掩盖一个实际需要训练却尚未实现的导数。广播操作需要在反向中把梯度归约回原输入形状，不能直接返回广播后的张量。

本例仅提供一阶标量反向，没有实现 batch log-cosh，也没有承诺高阶自动微分。需要多元素版本时，应同时设计归约方式、上下文和对应 VJP。完整程序检查了零点、正负残差、±1000 的前向有限性，以及上游梯度为 3 时的缩放结果。

## 8. 编写自己的损失时如何验证

先确定损失公式和输入契约，再检查以下行为：

1. 小型已知输入的损失值正确，返回标量且 dtype 符合预期。
2. 光滑区域的梯度与解析式或有限差分一致；零点等关键位置单独检查。
3. 将损失乘以一个非单位系数后，梯度按相同比例缩放。
4. batch 归约的分母正确，特别区分元素数、样本数和权重和。
5. 接入电路后，参数梯度存在，并在可解析的小电路上验证链式法则。
6. 非法 shape、dtype、空输入和无效权重得到明确错误。

训练损失下降可以作为端到端检查，但不能代替梯度正确性验证。新增支持的 dtype、广播方式或布局，应有相应的验证用例。

相关文档：[内置损失说明](../../crates/arcqml-loss/README.md)、[Tensor 与自动微分](../../crates/arcqml-core/README.md)、[算子列表](../../crates/arcqml-linalg/README.md)、[技术手册](../ArcQML技术手册.md)。
