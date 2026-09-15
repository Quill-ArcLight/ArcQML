use arcqml::prelude::*;
use arcqml_visualization::write_svg;

type AppResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const NUM_QUBITS: usize = 4;
const ACTIVE_ELECTRONS: usize = 2;
const LAYERS: usize = 6;
const STEPS: usize = 100;
const EXACT_GROUND_ENERGY: f64 = -1.136_189_454_207_826_6;
const HARTREE_FOCK_ENERGY: f64 = -1.117_349_035_056_280_5;
const CHEMICAL_ACCURACY: f64 = 1.6e-3;

fn h2_ansatz() -> AppResult<Circuit> {
    let mut circuit = Circuit::new(NUM_QUBITS)?;

    // Jordan–Wigner 编码下的 Hartree–Fock 初态：q0、q1 被占据。
    for qubit in 0..ACTIVE_ELECTRONS {
        circuit.x(qubit)?;
    }

    let mut parameter_index = 0usize;
    for _ in 0..LAYERS {
        for qubit in 0..NUM_QUBITS {
            let ry = 0.04 * (0.37 * (parameter_index + 1) as f64).sin();
            parameter_index += 1;
            let rz = 0.04 * (0.37 * (parameter_index + 1) as f64).sin();
            parameter_index += 1;
            circuit.ry(ry, qubit)?;
            circuit.rz(rz, qubit)?;
        }
        for control in 0..NUM_QUBITS {
            circuit.cnot(control, (control + 1) % NUM_QUBITS)?;
        }
    }
    Ok(circuit)
}

fn energy(
    simulator: &StateVectorSimulator,
    circuit: &Circuit,
    hamiltonian: &SparsePauliOp,
) -> AppResult<f64> {
    Ok(simulator.run(circuit, hamiltonian)?.value()?)
}

fn main() -> AppResult<()> {
    let hamiltonian = SparsePauliOp::from_json(include_str!(
        "../data/h2_hamiltonian.json"
    ))?;
    let circuit = h2_ansatz()?;
    let simulator = StateVectorSimulator::new(NUM_QUBITS)?;
    let mut optimizer = Adam::new(0.05, 0.9, 0.999, 1e-8, 0.0)?;

    println!("H2/STO-3G, Jordan-Wigner, {} qubits", NUM_QUBITS);
    println!("Hamiltonian terms : {}", hamiltonian.len());
    println!("Trainable params  : {}", circuit.num_parameters());
    println!("Hartree-Fock      : {HARTREE_FOCK_ENERGY:.12} Ha");
    println!("Exact ground      : {EXACT_GROUND_ENERGY:.12} Ha");

    for step in 1..=STEPS {
        let energy_tensor = simulator.run(&circuit, &hamiltonian)?;
        energy_tensor.backward()?;
        optimizer.step(circuit.parameters())?;
        optimizer.zero_grad(circuit.parameters());

        if step == 1 || step % 10 == 0 {
            let current = energy(&simulator, &circuit, &hamiltonian)?;
            println!(
                "step {step:>3}: energy = {current:.12} Ha, |error| = {:.3e} Ha",
                (current - EXACT_GROUND_ENERGY).abs()
            );
        }
    }

    write_svg(&circuit, "./examples/rust/h2_vqe_circuit.svg");

    let final_energy = energy(&simulator, &circuit, &hamiltonian)?;
    let absolute_error = (final_energy - EXACT_GROUND_ENERGY).abs();
    println!("\nVQE energy        : {final_energy:.12} Ha");
    println!("Absolute error    : {absolute_error:.6e} Ha");
    println!(
        "Chemical accuracy: {} (threshold {:.1e} Ha)",
        if absolute_error < CHEMICAL_ACCURACY {
            "reached"
        } else {
            "not reached"
        },
        CHEMICAL_ACCURACY
    );
    Ok(())
}
