use arcqml_circuit::{Circuit, Gate, Gateparam, Operation};
use std::fmt::Write as _;
use std::io;
use std::path::Path;

const LEFT_MARGIN: f64 = 68.0;
const RIGHT_MARGIN: f64 = 28.0;
const TOP_MARGIN: f64 = 36.0;
const BOTTOM_MARGIN: f64 = 36.0;
const TRACK_GAP: f64 = 56.0;
const MIN_COLUMN_WIDTH: f64 = 58.0;
const COLUMN_GAP: f64 = 16.0;
const GATE_HEIGHT: f64 = 32.0;
const CONTROL_RADIUS: f64 = 4.5;
const TARGET_RADIUS: f64 = 11.0;

/// 将量子电路渲染为 SVG 矢量图字符串。
///
/// 输出把 `q0` 放在最上方；可训练参数显示为 `pN`，固定浮点参数最多显示三位小数。
/// 此函数只读取电路，不执行量子门或修改电路状态。
pub fn draw_svg(circuit: &Circuit) -> String {
    let layers = schedule_layers(circuit);
    let columns = layout_columns(&layers);
    let circuit_width = columns
        .last()
        .map(|column| column.x + column.width)
        .unwrap_or(0.0);
    let width = LEFT_MARGIN + circuit_width + RIGHT_MARGIN;
    let height =
        TOP_MARGIN + BOTTOM_MARGIN + TRACK_GAP * circuit.num_qubits().saturating_sub(1) as f64;
    let mut svg = String::new();

    write_svg_header(&mut svg, width, height);
    render_wires(&mut svg, circuit, width);
    for (layer, column) in layers.iter().zip(&columns) {
        for operation in layer {
            render_operation(&mut svg, operation, column.x, column.width);
        }
    }
    svg.push_str("  </g>\n</svg>\n");
    svg
}

/// 将量子电路 SVG 矢量图写入指定文件。
///
/// 目标文件已存在时会被覆盖；父目录不会自动创建。
///
/// # Errors
///
/// 当无法创建、写入或刷新目标 SVG 文件时返回 I/O 错误。
pub fn write_svg(circuit: &Circuit, path: impl AsRef<Path>) -> io::Result<()> {
    std::fs::write(path, draw_svg(circuit))
}

/// 写入 SVG 文档头和画布背景。
fn write_svg_header(svg: &mut String, width: f64, height: f64) {
    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.0}\" height=\"{height:.0}\" viewBox=\"0 0 {width:.0} {height:.0}\" role=\"img\" aria-label=\"量子电路图\">"
    );
    svg.push_str("  <rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n");
    svg.push_str("  <g stroke=\"#1f2937\" stroke-width=\"1.5\" fill=\"none\" stroke-linecap=\"round\" stroke-linejoin=\"round\">\n");
}

/// 绘制量子比特标签和水平线路。
fn render_wires(svg: &mut String, circuit: &Circuit, width: f64) {
    let wire_end = width - RIGHT_MARGIN + 4.0;
    for qubit in 0..circuit.num_qubits() {
        let y = track_y(qubit);
        let _ = writeln!(
            svg,
            "    <text x=\"{:.1}\" y=\"{:.1}\" fill=\"#111827\" stroke=\"none\" font-family=\"Arial, sans-serif\" font-size=\"15\" text-anchor=\"end\" dominant-baseline=\"middle\">q{qubit}</text>",
            LEFT_MARGIN - 12.0,
            y
        );
        let _ = writeln!(
            svg,
            "    <line x1=\"{LEFT_MARGIN:.1}\" y1=\"{y:.1}\" x2=\"{wire_end:.1}\" y2=\"{y:.1}\" stroke=\"#6b7280\"/>"
        );
    }
}

/// 按量子比特占用关系为操作安排可并行的绘图层。
fn schedule_layers(circuit: &Circuit) -> Vec<Vec<&Operation>> {
    let mut layers = Vec::<Vec<&Operation>>::new();
    for operation in circuit.operations() {
        let earliest_layer = layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                layer
                    .iter()
                    .any(|other| spans_overlap(operation, other))
                    .then_some(index + 1)
            })
            .max()
            .unwrap_or(0);
        if let Some(layer) = layers
            .iter_mut()
            .skip(earliest_layer)
            .find(|layer| layer.iter().all(|other| !spans_overlap(operation, other)))
        {
            layer.push(operation);
        } else {
            layers.push(vec![operation]);
        }
    }
    layers
}

/// 计算每一绘图层的横坐标和宽度。
fn layout_columns(layers: &[Vec<&Operation>]) -> Vec<Column> {
    let mut cursor = 0.0;
    layers
        .iter()
        .map(|layer| {
            let width = layer
                .iter()
                .map(|operation| required_width(operation))
                .fold(MIN_COLUMN_WIDTH, f64::max);
            let column = Column {
                x: LEFT_MARGIN + cursor,
                width,
            };
            cursor += width + COLUMN_GAP;
            column
        })
        .collect()
}

/// 判断两个操作在垂直方向的可视范围是否重叠。
fn spans_overlap(left: &Operation, right: &Operation) -> bool {
    let (left_min, left_max) = operation_span(left);
    let (right_min, right_max) = operation_span(right);
    left_min <= right_max && right_min <= left_max
}

/// 返回一个操作涉及量子比特的最小和最大编号。
fn operation_span(operation: &Operation) -> (usize, usize) {
    let min = operation
        .qubits()
        .iter()
        .map(|qubit| qubit.index())
        .min()
        .expect("合法操作至少作用于一个量子比特");
    let max = operation
        .qubits()
        .iter()
        .map(|qubit| qubit.index())
        .max()
        .expect("合法操作至少作用于一个量子比特");
    (min, max)
}

/// 根据门类型绘制对应的 SVG 图元。
fn render_operation(svg: &mut String, operation: &Operation, x: f64, width: f64) {
    match operation.gate() {
        Gate::CNot => render_controlled_gate(svg, operation, x, width, "X", 1),
        Gate::CY => render_controlled_gate(svg, operation, x, width, "Y", 1),
        Gate::CZ => render_controlled_gate(svg, operation, x, width, "Z", 1),
        Gate::CH => render_controlled_gate(svg, operation, x, width, "H", 1),
        Gate::CS => render_controlled_gate(svg, operation, x, width, "S", 1),
        Gate::CT => render_controlled_gate(svg, operation, x, width, "T", 1),
        Gate::CPhase(parameter) => render_controlled_gate(
            svg,
            operation,
            x,
            width,
            &format!("P({})", parameter_label(parameter)),
            1,
        ),
        Gate::CRx(parameter) => render_controlled_gate(
            svg,
            operation,
            x,
            width,
            &format!("Rx({})", parameter_label(parameter)),
            1,
        ),
        Gate::CRy(parameter) => render_controlled_gate(
            svg,
            operation,
            x,
            width,
            &format!("Ry({})", parameter_label(parameter)),
            1,
        ),
        Gate::CRz(parameter) => render_controlled_gate(
            svg,
            operation,
            x,
            width,
            &format!("Rz({})", parameter_label(parameter)),
            1,
        ),
        Gate::Toffoli => render_controlled_gate(svg, operation, x, width, "X", 2),
        Gate::MCX { controls } => render_controlled_gate(svg, operation, x, width, "X", *controls),
        Gate::Swap | Gate::CSwap => render_swap(svg, operation, x, width),
        gate if gate.arity() == 1 => {
            render_gate_box(
                svg,
                track_y(operation.qubits()[0].index()),
                x,
                width,
                &gate_label(gate),
            );
        }
        gate => render_multi_gate(svg, operation, x, width, &gate_label(gate)),
    }
}

/// 绘制受控门的控制点、连接线和目标门。
fn render_controlled_gate(
    svg: &mut String,
    operation: &Operation,
    x: f64,
    width: f64,
    target_label: &str,
    controls: usize,
) {
    let qubits = operation.qubits();
    let center_x = x + width * 0.5;
    let y_values = qubits
        .iter()
        .map(|qubit| track_y(qubit.index()))
        .collect::<Vec<_>>();
    let min_y = y_values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_y = y_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let _ = writeln!(
        svg,
        "    <line x1=\"{center_x:.1}\" y1=\"{min_y:.1}\" x2=\"{center_x:.1}\" y2=\"{max_y:.1}\"/>"
    );
    for qubit in &qubits[..controls] {
        let y = track_y(qubit.index());
        let _ = writeln!(
            svg,
            "    <circle cx=\"{center_x:.1}\" cy=\"{y:.1}\" r=\"{CONTROL_RADIUS:.1}\" fill=\"#111827\" stroke=\"none\"/>"
        );
    }
    let target_y = track_y(qubits[controls].index());
    if target_label == "X" {
        render_x_target(svg, center_x, target_y);
    } else {
        render_gate_box(svg, target_y, x, width, target_label);
    }
}

/// 绘制 CNOT、Toffoli 或多控制 X 的圆圈加号目标。
fn render_x_target(svg: &mut String, x: f64, y: f64) {
    let _ = writeln!(
        svg,
        "    <circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"{TARGET_RADIUS:.1}\" fill=\"white\"/>"
    );
    let _ = writeln!(
        svg,
        "    <line x1=\"{:.1}\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\"/>",
        x - TARGET_RADIUS,
        x + TARGET_RADIUS
    );
    let _ = writeln!(
        svg,
        "    <line x1=\"{x:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{:.1}\"/>",
        y - TARGET_RADIUS,
        y + TARGET_RADIUS
    );
}

/// 绘制交换门或受控交换门。
fn render_swap(svg: &mut String, operation: &Operation, x: f64, width: f64) {
    let qubits = operation.qubits();
    let center_x = x + width * 0.5;
    let y_values = qubits
        .iter()
        .map(|qubit| track_y(qubit.index()))
        .collect::<Vec<_>>();
    let min_y = y_values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_y = y_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let _ = writeln!(
        svg,
        "    <line x1=\"{center_x:.1}\" y1=\"{min_y:.1}\" x2=\"{center_x:.1}\" y2=\"{max_y:.1}\"/>"
    );
    if matches!(operation.gate(), Gate::CSwap) {
        let control_y = track_y(qubits[0].index());
        let _ = writeln!(
            svg,
            "    <circle cx=\"{center_x:.1}\" cy=\"{control_y:.1}\" r=\"{CONTROL_RADIUS:.1}\" fill=\"#111827\" stroke=\"none\"/>"
        );
        render_cross(svg, center_x, track_y(qubits[1].index()));
        render_cross(svg, center_x, track_y(qubits[2].index()));
    } else {
        render_cross(svg, center_x, track_y(qubits[0].index()));
        render_cross(svg, center_x, track_y(qubits[1].index()));
    }
}

/// 绘制交换门端点的叉号。
fn render_cross(svg: &mut String, x: f64, y: f64) {
    const HALF_SIZE: f64 = 7.0;
    let _ = writeln!(
        svg,
        "    <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"/>",
        x - HALF_SIZE,
        y - HALF_SIZE,
        x + HALF_SIZE,
        y + HALF_SIZE
    );
    let _ = writeln!(
        svg,
        "    <line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\"/>",
        x - HALF_SIZE,
        y + HALF_SIZE,
        x + HALF_SIZE,
        y - HALF_SIZE
    );
}

/// 绘制跨越多条量子线路的多比特门矩形。
fn render_multi_gate(svg: &mut String, operation: &Operation, x: f64, width: f64, label: &str) {
    let (min_qubit, max_qubit) = operation_span(operation);
    let top = track_y(min_qubit) - GATE_HEIGHT * 0.5;
    let height = track_y(max_qubit) - track_y(min_qubit) + GATE_HEIGHT;
    let center_y = top + height * 0.5;
    let _ = writeln!(
        svg,
        "    <rect x=\"{x:.1}\" y=\"{top:.1}\" width=\"{width:.1}\" height=\"{height:.1}\" rx=\"3\" fill=\"white\"/>"
    );
    write_gate_label(svg, x + width * 0.5, center_y, label);
}

/// 绘制单比特门或受控门目标的矩形和标签。
fn render_gate_box(svg: &mut String, y: f64, x: f64, width: f64, label: &str) {
    let top = y - GATE_HEIGHT * 0.5;
    let _ = writeln!(
        svg,
        "    <rect x=\"{x:.1}\" y=\"{top:.1}\" width=\"{width:.1}\" height=\"{GATE_HEIGHT:.1}\" rx=\"3\" fill=\"white\"/>"
    );
    write_gate_label(svg, x + width * 0.5, y, label);
}

/// 写入居中的门标签，并转义 XML 特殊字符。
fn write_gate_label(svg: &mut String, x: f64, y: f64, label: &str) {
    let escaped = escape_xml(label);
    let _ = writeln!(
        svg,
        "    <text x=\"{x:.1}\" y=\"{y:.1}\" fill=\"#111827\" stroke=\"none\" font-family=\"Arial, sans-serif\" font-size=\"14\" text-anchor=\"middle\" dominant-baseline=\"middle\">{escaped}</text>"
    );
}

/// 根据门标签长度计算满足可读性的列宽。
fn required_width(operation: &Operation) -> f64 {
    let label = match operation.gate() {
        Gate::CNot | Gate::Toffoli | Gate::MCX { .. } | Gate::Swap | Gate::CSwap => "X".to_string(),
        Gate::CY => "Y".to_string(),
        Gate::CZ => "Z".to_string(),
        Gate::CH => "H".to_string(),
        Gate::CS => "S".to_string(),
        Gate::CT => "T".to_string(),
        Gate::CPhase(parameter) => format!("P({})", parameter_label(parameter)),
        Gate::CRx(parameter) => format!("Rx({})", parameter_label(parameter)),
        Gate::CRy(parameter) => format!("Ry({})", parameter_label(parameter)),
        Gate::CRz(parameter) => format!("Rz({})", parameter_label(parameter)),
        gate => gate_label(gate),
    };
    MIN_COLUMN_WIDTH.max(label.chars().count() as f64 * 8.0 + 24.0)
}

/// 返回用于图中展示的门名称和参数标签。
fn gate_label(gate: &Gate) -> String {
    match gate {
        Gate::Rx(parameter) => format!("Rx({})", parameter_label(parameter)),
        Gate::Ry(parameter) => format!("Ry({})", parameter_label(parameter)),
        Gate::Rz(parameter) => format!("Rz({})", parameter_label(parameter)),
        Gate::Phase(parameter) => format!("P({})", parameter_label(parameter)),
        Gate::U3(theta, phi, lambda) => format!(
            "U3({},{},{})",
            parameter_label(theta),
            parameter_label(phi),
            parameter_label(lambda)
        ),
        Gate::Rxx(parameter) => format!("Rxx({})", parameter_label(parameter)),
        Gate::Ryy(parameter) => format!("Ryy({})", parameter_label(parameter)),
        Gate::Rzz(parameter) => format!("Rzz({})", parameter_label(parameter)),
        Gate::Rzx(parameter) => format!("Rzx({})", parameter_label(parameter)),
        Gate::FSim(theta, phi) => {
            format!("fSim({},{})", parameter_label(theta), parameter_label(phi))
        }
        Gate::CustomUnitary { name, .. } => name.clone(),
        _ => gate.name().to_string(),
    }
}

/// 返回固定值或可训练参数在图中的简短标签。
fn parameter_label(parameter: &Gateparam) -> String {
    match parameter {
        Gateparam::Fixed(value) => format_value(*value),
        Gateparam::Param(id) => format!("p{}", id.index()),
    }
}

/// 将浮点参数格式化为紧凑的三位小数文本。
fn format_value(value: f64) -> String {
    let mut formatted = format!("{value:.3}");
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.pop();
    }
    formatted
}

/// 返回指定量子比特对应的垂直坐标。
fn track_y(qubit: usize) -> f64 {
    TOP_MARGIN + TRACK_GAP * qubit as f64
}

/// 对门名中的 XML 特殊字符进行转义。
fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// 保存一个绘图层的横坐标和宽度。
#[derive(Debug, Clone, Copy)]
struct Column {
    x: f64,
    width: f64,
}
