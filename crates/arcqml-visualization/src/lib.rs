//! 量子电路的只读文本与 SVG 渲染。
//!
//! 渲染函数不会执行或修改电路。文本图适合终端和日志，SVG 适合文档、网页和报告。
//! 两种格式均把 `q0` 放在最上方，并用参数编号表示可训练门参数。
//!
//! # 示例
//!
//! ```
//! use arcqml_circuit::Circuit;
//! use arcqml_visualization::draw;
//!
//! let mut circuit = Circuit::new(2)?;
//! circuit.h(0usize)?.cnot(0usize, 1usize)?;
//! let diagram = draw(&circuit);
//! assert!(diagram.contains("q0"));
//! # Ok::<(), arcqml_circuit::CircuitError>(())
//! ```

mod svg;
mod text;

pub use svg::{draw_svg, write_svg};
pub use text::draw;
