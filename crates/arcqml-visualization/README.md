# arcqml-visualization

`arcqml-visualization` 把 `Circuit` 渲染为 Unicode 文本图或 SVG 矢量图。渲染只读取电路，不执行量子门，也不修改电路与参数。

## 添加依赖

```toml
[dependencies]
arcqml-circuit = { path = "PATH_TO_ARCQML/crates/arcqml-circuit" }
arcqml-visualization = { path = "PATH_TO_ARCQML/crates/arcqml-visualization" }
```

## 终端文本图

```rust
use arcqml_circuit::Circuit;
use arcqml_visualization::draw;

let mut circuit = Circuit::new(2)?;
circuit.h(0usize)?.cnot(0usize, 1usize)?;

println!("{}", draw(&circuit));
```

文本图适合终端、日志与测试失败信息：

- `q0` 位于最上方；
- 单比特门显示为带边框方块；
- 受控门显示控制点、目标与连接线；
- 其他多比特门显示跨量子线门框；
- 可训练参数显示为 `p0`、`p1` 等编号；
- 固定参数直接显示数值。

电路的并行深度由 `Circuit::depth()` 计算，不由渲染器推断。

## SVG 输出

`draw_svg` 返回完整 SVG 字符串；`write_svg` 直接写入文件：

```rust
use arcqml_visualization::{draw_svg, write_svg};

let svg = draw_svg(&circuit);
assert!(svg.contains("<svg"));

write_svg(&circuit, "circuit.svg")?;
```

SVG 适合 README、技术报告和网页。文件写入失败会返回 `std::io::Error`；`draw` 与 `draw_svg` 本身返回字符串。

运行内置示例：

```powershell
cargo run -p arcqml-visualization --example complex_circuit
cargo run -p arcqml-visualization --example svg_circuit
```

生成的参考文件位于 [`examples/svg_circuit.svg`](examples/svg_circuit.svg)。普通用户也可从门面 crate 使用 `arcqml::visualization::{draw, draw_svg, write_svg}`；`arcqml::prelude::*` 目前只额外重导出 `draw`。

许可证见 [混合许可说明](../../README.md#发行状态与许可证)。
