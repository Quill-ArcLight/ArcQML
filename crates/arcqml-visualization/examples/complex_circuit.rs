use arcqml_circuit::Circuit;
use arcqml_visualization::draw;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(5)?;
    circuit
        .h(0usize)?
        .ry_fixed(0.37, 1usize)?
        .x(4usize)?
        .cnot(0usize, 2usize)?
        .crz(0.63, 2usize, 3usize)?
        .rzz(0.41, 1usize, 3usize)?
        .swap(0usize, 4usize)?
        .toffoli(0usize, 2usize, 4usize)?
        .fsim(0.22, -0.18, 1usize, 2usize)?
        .u3(0.5, 0.25, -0.4, 3usize)?;

    println!("depth = {}", circuit.depth());
    println!();
    println!("{}", draw(&circuit));
    Ok(())
}
