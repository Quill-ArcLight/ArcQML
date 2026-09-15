use arcqml_circuit::{Circuit, CircuitResult};
use arcqml_visualization::{draw, write_svg};
use std::path::PathBuf;

type ExampleResult<T> = Result<T, Box<dyn std::error::Error>>;

/// 构建用于展示 SVG 绘图能力的五量子比特参考电路。
fn build_example_circuit() -> CircuitResult<Circuit> {
    let mut circuit = Circuit::new(5)?;
    circuit.h(0)?;
    circuit.ry(0.37, 1)?;
    circuit.x(4)?;
    circuit.cnot(0, 2)?;
    circuit.crz(0.63, 2, 3)?;
    circuit.rzz(0.41, 1, 3)?;
    circuit.swap(0, 4)?;
    circuit.toffoli(0, 2, 4)?;
    Ok(circuit)
}

/// 输出文本电路图，并将同一电路导出为 SVG 矢量图文件。
fn main() -> ExampleResult<()> {
    let circuit = build_example_circuit()?;
    let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("svg_circuit.svg");

    write_svg(&circuit, &output_path)?;

    println!("{}", draw(&circuit));
    println!("SVG 矢量图已生成：{}", output_path.display());
    Ok(())
}
