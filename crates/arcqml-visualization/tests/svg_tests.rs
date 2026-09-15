use arcqml_circuit::{Circuit, Gate, Qubit};
use arcqml_visualization::draw_svg;
use num_complex::Complex64;

/// 验证 SVG 包含量子线路、单比特门、受控门和交换门图元。
#[test]
fn renders_common_gate_shapes() {
    let mut circuit = Circuit::new(3).unwrap();
    circuit.h(0).unwrap();
    circuit.ry(0.25, 1).unwrap();
    circuit.cnot(0, 2).unwrap();
    circuit.swap(1, 2).unwrap();

    let svg = draw_svg(&circuit);

    assert!(svg.starts_with("<svg"));
    assert!(svg.contains(">H</text>"));
    assert!(svg.contains(">Ry(p0)</text>"));
    assert!(svg.contains("<circle"));
    assert!(svg.contains("  </g>\n</svg>"));
}

/// 验证公开 `draw_svg` API 会转义自定义门标签中的 XML 特殊字符。
#[test]
fn draw_svg_escapes_custom_gate_xml_characters() {
    let mut circuit = Circuit::new(1).unwrap();
    let gate = Gate::custom_unitary(
        "A<&>\"'",
        1,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap();
    circuit.add_gate(gate, vec![Qubit::new(0)]).unwrap();

    let svg = draw_svg(&circuit);

    assert!(svg.contains("A&lt;&amp;&gt;&quot;&apos;"));
}
