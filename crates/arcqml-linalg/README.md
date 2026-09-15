# arcqml-linalg

`arcqml-linalg` 为 `arcqml-core::Tensor` 提供可微数学、线性代数、形状变换、归约与 NumPy 风格广播。每个公开算子都能在需要时向动态自动微分图登记反向规则。

## 添加依赖

```toml
[dependencies]
arcqml-core = { path = "PATH_TO_ARCQML/crates/arcqml-core" }
arcqml-linalg = { path = "PATH_TO_ARCQML/crates/arcqml-linalg" }
```

## 最小示例

```rust
use arcqml_core::Tensor;
use arcqml_linalg::{TensorLinalgExt, mean};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let lhs = Tensor::new(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]])?;
    let rhs = Tensor::new(vec![vec![5.0_f64], vec![6.0]])?;

    let product = lhs.matmul(&rhs)?;
    let average = mean(&product)?;

    assert_eq!(product.shape(), &[2, 1]);
    assert_eq!(average.shape(), &[]);
    Ok(())
}
```

同一运算通常既有自由函数，也有 `TensorLinalgExt` 方法。库代码可以按喜好选择，但应保持项目风格一致。

## 算子总览

| 类别 | 函数 |
| --- | --- |
| 二元逐元素与广播 | `add`、`sub`、`mul`、`div` |
| 一元逐元素 | `neg`、`square`、`sqrt`、`abs`、`exp`、`log`、`sigmoid`、`tanh`、`clamp` |
| 复数 | `conj` |
| 线性代数 | `dot`、`matmul`、`transpose`、`l2_norm` |
| 形状 | `reshape` |
| 全局归约 | `sum`、`mean`、`max`、`min` |
| 维度归约 | `sum_dim`、`mean_dim`、`segment_sum` |
| 归一化 | `softmax`、`log_softmax`、`logsumexp` |

二维矩阵乘法要求：

$$
[m,k]@[k,n]\rightarrow[m,n]
$$

`dot` 只接受一维向量。对于 `C64`，点积不会自动对左操作数取复共轭；需要 Hermitian 内积时应显式调用 `conj`。

## 广播与形状

`add`、`sub`、`mul` 与 `div` 按 NumPy 风格从末尾维度对齐：两个维度相等，或其中一个为 `1` 时才兼容。

```rust
use arcqml_core::Tensor;
use arcqml_linalg::add;

let matrix = Tensor::new(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]])?;
let row = Tensor::new(vec![10.0_f64, 20.0])?;
let result = add(&matrix, &row)?;
assert_eq!(result.shape(), &[2, 2]);
```

`sum`、`mean`、`max` 和 `min` 对全部元素归约并返回标量。`sum_dim`、`mean_dim` 与 `logsumexp` 通过 `axis` 和 `keepdim` 控制输出形状。

## 连续性与 dtype

- 数值内核以连续 row-major buffer 为主要输入。
- `transpose` 产生的通常是非连续 view；必要时调用 `contiguous()`。
- `Bool` 不是数值线性代数输入。
- 各算子支持的 dtype 不完全相同；类型不兼容时返回 `LinalgError`，不会静默转换。
- `l2_norm` 返回 `F64` 标量。
- `ndarray_bridge` 提供 `Tensor` 与 `ndarray::ArrayD` 的转换和借用视图。

复数反向传播采用面向实值损失的共轭 Wirtinger VJP 约定。若最终目标不是实值标量，应先明确所需的复梯度语义。

