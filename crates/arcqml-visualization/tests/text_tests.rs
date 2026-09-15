use arcqml_circuit::Circuit;
use arcqml_visualization::draw;

#[test]
fn draw_should_render_boxes_controls_and_parameter_labels() {
    let mut circuit = Circuit::new(2).unwrap();
    circuit.h(0usize).unwrap().ry(0.25, 1usize).unwrap();
    circuit.cnot(0usize, 1usize).unwrap();

    let diagram = draw(&circuit);

    assert!(diagram.contains('┌'));
    assert!(diagram.contains('┐'));
    assert!(diagram.contains('┤'));
    assert!(diagram.contains('├'));
    assert!(diagram.contains('└'));
    assert!(diagram.contains('┘'));
    assert!(diagram.contains('H'));
    assert!(diagram.contains("Ry(p0)"));
    assert!(diagram.contains('●'));
    assert!(diagram.contains('│'));
    assert!(diagram.contains('X'));
    assert!(diagram.lines().any(|line| line.contains("q0:")));
    assert!(diagram.lines().any(|line| line.contains("q1:")));
}

#[test]
fn draw_should_render_multi_qubit_gates_and_swaps() {
    let mut circuit = Circuit::new(3).unwrap();
    circuit.rzz(0.5, 0usize, 2usize).unwrap();
    circuit.swap(0usize, 1usize).unwrap();
    circuit.toffoli(0usize, 1usize, 2usize).unwrap();

    let diagram = draw(&circuit);

    assert!(diagram.contains("Rzz(p0)"));
    assert!(diagram.contains('×'));
    assert_eq!(diagram.matches('●').count(), 2);
}

#[test]
fn draw_should_keep_later_overlapping_operations_to_the_right() {
    let mut circuit = Circuit::new(3).unwrap();
    circuit
        .x(1usize)
        .unwrap()
        .cnot(0usize, 1usize)
        .unwrap()
        .rzz(0.5, 0usize, 2usize)
        .unwrap()
        .u3(0.1, 0.2, 0.3, 2usize)
        .unwrap();

    let diagram = draw(&circuit);
    let rzz_column = diagram
        .lines()
        .find_map(|line| line.find("Rzz(p0)"))
        .unwrap();
    let u3_column = diagram
        .lines()
        .find_map(|line| line.find("U3(p1,p2,p3)"))
        .unwrap();

    assert!(rzz_column < u3_column);
}
