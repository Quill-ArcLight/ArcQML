use arcqml_circuit::{Circuit, Gate, Gateparam, Operation, Qubit};

const TRACK_HEIGHT: usize = 4;
const MIN_COLUMN_WIDTH: usize = 5;

/// 将量子电路渲染为适合终端和日志输出的 Unicode 文本图。
///
/// 输出把 `q0` 放在最上方；可训练参数显示为 `pN`，固定浮点参数最多显示三位小数。
/// 此函数只读取电路，不执行量子门或修改电路状态。
pub fn draw(circuit: &Circuit) -> String {
    let layers = schedule_layers(circuit);
    let prefix_width = format!("q{}: ", circuit.num_qubits() - 1).chars().count();
    let columns = layout_columns(&layers);
    let height = circuit.num_qubits() * TRACK_HEIGHT - 1;
    let width = prefix_width + 1 + columns.iter().map(|column| column.width + 2).sum::<usize>();
    let mut canvas = Canvas::new(width, height);

    for qubit in 0..circuit.num_qubits() {
        let y = track_y(qubit);
        canvas.write(y, 0, &format!("q{qubit}: "));
        canvas.horizontal(y, prefix_width, width);
    }

    for (layer, column) in layers.iter().zip(columns.iter()) {
        for operation in layer {
            render_operation(
                &mut canvas,
                operation,
                prefix_width + 1 + column.start,
                column.width,
            );
        }
    }

    canvas.finish()
}

#[derive(Debug, Clone, Copy)]
struct Column {
    start: usize,
    width: usize,
}

fn layout_columns(layers: &[Vec<&Operation>]) -> Vec<Column> {
    let mut cursor = 0usize;
    layers
        .iter()
        .map(|layer| {
            let width = layer
                .iter()
                .map(|operation| required_width(operation))
                .max()
                .unwrap_or(MIN_COLUMN_WIDTH);
            let column = Column {
                start: cursor,
                width,
            };
            cursor += width + 2;
            column
        })
        .collect()
}

fn schedule_layers(circuit: &Circuit) -> Vec<Vec<&Operation>> {
    let mut layers = Vec::<Vec<&Operation>>::new();

    for operation in circuit.operations() {
        let earliest_layer = layers
            .iter()
            .enumerate()
            .filter_map(|(index, layer)| {
                layer
                    .iter()
                    .any(|other| visual_spans_overlap(operation, other))
                    .then_some(index + 1)
            })
            .max()
            .unwrap_or(0);

        if let Some(layer) = layers.iter_mut().skip(earliest_layer).find(|layer| {
            layer
                .iter()
                .all(|other| !visual_spans_overlap(operation, other))
        }) {
            layer.push(operation);
        } else {
            layers.push(vec![operation]);
        }
    }

    layers
}

fn visual_spans_overlap(left: &Operation, right: &Operation) -> bool {
    let (left_min, left_max) = operation_span(left);
    let (right_min, right_max) = operation_span(right);
    left_min <= right_max && right_min <= left_max
}

fn operation_span(operation: &Operation) -> (usize, usize) {
    let min = operation
        .qubits()
        .iter()
        .map(|qubit| qubit.index())
        .min()
        .unwrap();
    let max = operation
        .qubits()
        .iter()
        .map(|qubit| qubit.index())
        .max()
        .unwrap();
    (min, max)
}

fn render_operation(canvas: &mut Canvas, operation: &Operation, x: usize, width: usize) {
    match operation.gate() {
        Gate::CNot => render_controlled_gate(canvas, operation, x, width, "X", 1),
        Gate::CY => render_controlled_gate(canvas, operation, x, width, "Y", 1),
        Gate::CZ => render_controlled_gate(canvas, operation, x, width, "Z", 1),
        Gate::CH => render_controlled_gate(canvas, operation, x, width, "H", 1),
        Gate::CS => render_controlled_gate(canvas, operation, x, width, "S", 1),
        Gate::CT => render_controlled_gate(canvas, operation, x, width, "T", 1),
        Gate::CPhase(parameter) => render_controlled_gate(
            canvas,
            operation,
            x,
            width,
            &format!("P({})", parameter_label(parameter)),
            1,
        ),
        Gate::CRx(parameter) => render_controlled_gate(
            canvas,
            operation,
            x,
            width,
            &format!("Rx({})", parameter_label(parameter)),
            1,
        ),
        Gate::CRy(parameter) => render_controlled_gate(
            canvas,
            operation,
            x,
            width,
            &format!("Ry({})", parameter_label(parameter)),
            1,
        ),
        Gate::CRz(parameter) => render_controlled_gate(
            canvas,
            operation,
            x,
            width,
            &format!("Rz({})", parameter_label(parameter)),
            1,
        ),
        Gate::Toffoli => render_controlled_gate(canvas, operation, x, width, "X", 2),
        Gate::MCX { controls } => {
            render_controlled_gate(canvas, operation, x, width, "X", *controls)
        }
        Gate::Swap | Gate::CSwap => render_swap(canvas, operation, x, width),
        gate if gate.arity() == 1 => {
            render_single_gate(
                canvas,
                operation.qubits()[0].index(),
                x,
                width,
                &gate_label(gate),
            );
        }
        gate => render_multi_gate(canvas, operation, x, width, &gate_label(gate)),
    }
}

fn render_single_gate(canvas: &mut Canvas, qubit: usize, x: usize, width: usize, label: &str) {
    canvas.gate_box(track_y(qubit), x, width, label);
}

fn render_controlled_gate(
    canvas: &mut Canvas,
    operation: &Operation,
    x: usize,
    width: usize,
    target_label: &str,
    controls: usize,
) {
    let qubits = operation.qubits();
    let target = qubits[controls].index();
    let center = x + width / 2;
    let min_y = qubits
        .iter()
        .map(|qubit| track_y(qubit.index()))
        .min()
        .unwrap();
    let max_y = qubits
        .iter()
        .map(|qubit| track_y(qubit.index()))
        .max()
        .unwrap();
    canvas.vertical(center, min_y, max_y);

    for qubit in &qubits[..controls] {
        canvas.set(track_y(qubit.index()), center, '●');
    }
    canvas.gate_box(track_y(target), x, width, target_label);
}

fn render_swap(canvas: &mut Canvas, operation: &Operation, x: usize, width: usize) {
    let qubits = operation.qubits();
    let center = x + width / 2;
    let min_y = qubits
        .iter()
        .map(|qubit| track_y(qubit.index()))
        .min()
        .unwrap();
    let max_y = qubits
        .iter()
        .map(|qubit| track_y(qubit.index()))
        .max()
        .unwrap();
    canvas.vertical(center, min_y, max_y);

    match operation.gate() {
        Gate::CSwap => {
            canvas.set(track_y(qubits[0].index()), center, '●');
            canvas.set(track_y(qubits[1].index()), center, '×');
            canvas.set(track_y(qubits[2].index()), center, '×');
        }
        Gate::Swap => {
            canvas.set(track_y(qubits[0].index()), center, '×');
            canvas.set(track_y(qubits[1].index()), center, '×');
        }
        _ => unreachable!("render_swap only accepts swap gates"),
    }
}

fn render_multi_gate(
    canvas: &mut Canvas,
    operation: &Operation,
    x: usize,
    width: usize,
    label: &str,
) {
    let min_y = operation
        .qubits()
        .iter()
        .map(|qubit| track_y(qubit.index()))
        .min()
        .unwrap();
    let max_y = operation
        .qubits()
        .iter()
        .map(|qubit| track_y(qubit.index()))
        .max()
        .unwrap();
    canvas.multi_gate_box(min_y, max_y, operation.qubits(), x, width, label);
}

fn required_width(operation: &Operation) -> usize {
    let label = match operation.gate() {
        Gate::CNot
        | Gate::CY
        | Gate::CZ
        | Gate::CH
        | Gate::CS
        | Gate::CT
        | Gate::Toffoli
        | Gate::MCX { .. } => "X".to_string(),
        Gate::CPhase(parameter) => format!("P({})", parameter_label(parameter)),
        Gate::CRx(parameter) => format!("Rx({})", parameter_label(parameter)),
        Gate::CRy(parameter) => format!("Ry({})", parameter_label(parameter)),
        Gate::CRz(parameter) => format!("Rz({})", parameter_label(parameter)),
        Gate::Swap | Gate::CSwap => "X".to_string(),
        gate => gate_label(gate),
    };
    MIN_COLUMN_WIDTH.max(label.chars().count() + 4)
}

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

fn parameter_label(parameter: &Gateparam) -> String {
    match parameter {
        Gateparam::Fixed(value) => format_value(*value),
        Gateparam::Param(id) => format!("p{}", id.index()),
    }
}

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

fn track_y(qubit: usize) -> usize {
    qubit * TRACK_HEIGHT + 1
}

#[derive(Debug)]
struct Canvas {
    cells: Vec<Vec<char>>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            cells: vec![vec![' '; width]; height],
        }
    }

    fn set(&mut self, y: usize, x: usize, value: char) {
        self.cells[y][x] = value;
    }

    fn write(&mut self, y: usize, x: usize, value: &str) {
        for (offset, character) in value.chars().enumerate() {
            self.set(y, x + offset, character);
        }
    }

    fn horizontal(&mut self, y: usize, start: usize, end: usize) {
        for x in start..end {
            self.set(y, x, '─');
        }
    }

    fn vertical(&mut self, x: usize, start: usize, end: usize) {
        for y in start..=end {
            self.set(y, x, '│');
        }
    }

    fn gate_box(&mut self, y: usize, x: usize, width: usize, label: &str) {
        self.set(y - 1, x, '┌');
        self.horizontal(y - 1, x + 1, x + width - 1);
        self.set(y - 1, x + width - 1, '┐');

        self.set(y, x, '┤');
        for interior_x in x + 1..x + width - 1 {
            self.set(y, interior_x, ' ');
        }
        self.write_centered(y, x + 1, width - 2, label);
        self.set(y, x + width - 1, '├');

        self.set(y + 1, x, '└');
        self.horizontal(y + 1, x + 1, x + width - 1);
        self.set(y + 1, x + width - 1, '┘');
    }

    fn multi_gate_box(
        &mut self,
        min_y: usize,
        max_y: usize,
        qubits: &[Qubit],
        x: usize,
        width: usize,
        label: &str,
    ) {
        let top = min_y - 1;
        let bottom = max_y + 1;
        self.set(top, x, '┌');
        self.horizontal(top, x + 1, x + width - 1);
        self.set(top, x + width - 1, '┐');

        for y in top + 1..bottom {
            self.set(y, x, '│');
            self.set(y, x + width - 1, '│');
        }

        for qubit in qubits {
            let y = track_y(qubit.index());
            self.set(y, x, '┤');
            self.set(y, x + width - 1, '├');
        }

        for y in (min_y..=max_y).step_by(TRACK_HEIGHT) {
            if !qubits.iter().any(|qubit| track_y(qubit.index()) == y) {
                self.set(y, x, '┼');
                self.set(y, x + width - 1, '┼');
            }
        }

        let label_y = (min_y + max_y) / 2;
        self.write_centered(label_y, x + 1, width - 2, label);

        self.set(bottom, x, '└');
        self.horizontal(bottom, x + 1, x + width - 1);
        self.set(bottom, x + width - 1, '┘');
    }

    fn write_centered(&mut self, y: usize, x: usize, width: usize, value: &str) {
        let value_width = value.chars().count();
        let start = x + (width.saturating_sub(value_width)) / 2;
        self.write(y, start, value);
    }

    fn finish(self) -> String {
        self.cells
            .into_iter()
            .map(|row| row.into_iter().collect::<String>().trim_end().to_string())
            .collect::<Vec<_>>()
            .join("\n")
    }
}
