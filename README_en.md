# ArcQML

ArcQML is a quantum machine learning framework implemented natively in Rust. It provides core capabilities including quantum circuit construction, state-vector simulation, observable expectation value calculation, automatic differentiation, and parameter optimization. The framework supports both single-state computation and batched-state simulation, and observables can be represented using Pauli operators and their linear combinations. Expectation values computed by quantum circuits are returned as differentiable tensors, allowing them to participate in loss functions and other classical differentiable operations. During backpropagation, gradients can propagate through the complete computation graph to the quantum circuit parameters. For parameterized quantum circuits, ArcQML can compute circuit parameter gradients using the adjoint method and combine them with classical optimizers such as Adam and SGD to train hybrid quantum-classical algorithms.

These capabilities can be used for tasks such as variational quantum eigensolving (VQE), quantum neural networks (QNN), quantum state analysis, and small-scale unitary matrix synthesis. The framework uses native Rust compilation and CPU parallelism. The current version supports CPU simulation only, with GPU support planned for a future release. The low-level interfaces in this preview release have known memory-safety limitations; see [SECURITY.md](SECURITY.md).

> Current version: `0.1.0` (experimental preview for Windows). ArcQML uses a mixed licensing model that includes publicly available source code and a closed-source Runtime. Please read [Release Status and Licensing](#release-status-and-licensing) before deciding whether to use it for public distribution or in production environments.

## Why Use ArcQML

- **A single differentiable computation chain**: Quantum expectation values are returned as `Tensor` objects, which can be passed to loss functions followed by a call to `backward()`.
- **Both Rust and Python APIs**: Rust provides the full set of capabilities, while the native Python extension covers common circuit, simulation, analysis, and training workflows.
- **Single- and batched- state execution**: The same parameterized circuit can operate on either a single initial state or a batch of row-major initial states.
- **Suitable for teaching and experimentation**: Common quantum gates, Pauli Hamiltonians, SGD, Adam, state analysis, circuit diagrams, and weight saving are built in.

ArcQML can initially be understood as four layers:

1. `Circuit` describes which quantum gates to execute;
2. `(Batch)StateVectorSimulator` describes the quantum state from which computation begins;
3. `SparsePauliOp` describes the physical quantity to observe at the end;
4. `Tensor`, `loss`, and `optimizer` determine how parameters are updated based on the results.

## Get Started in 30 Seconds: Python

The wheel included in the current release is built for **CPython 3.11、Windows x86_64**:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ./wheels/x86_64-pc-windows-msvc/arcqml-0.1.0-cp311-cp311-win_amd64.whl
```

After installation, verify it with the following command:

```bash
python -c "import arcqml; print(arcqml.__version__)"
```

The following program constructs a Bell state, reads the exact probabilities, and performs terminal sampling:

```python
import arcqml

circuit = arcqml.Circuit(2)
circuit.h(0)
circuit.cnot(0, 1)

simulator = arcqml.StateVectorSimulator(2)
simulator.apply_circuit(circuit)

probabilities = arcqml.analysis.probabilities(simulator).numpy()
counts = simulator.sample_counts(1_000, seed=42)

print(probabilities)  # Approximately [0.5, 0.0, 0.0, 0.5]
print(counts)         # Only "00" and "11" will appear
```

Linux, other Python versions, macOS, and ARM platforms are outside the validation scope of this preview release, and no prebuilt wheels for them are included in the release snapshot. If you need to build from source, see the [Python README](python/README.md).

## Get Started in 30 Seconds: Rust

The Rust API uses the Rust 2024 edition, and the prebuilt Runtime in this release requires the Rust 1.98.0 toolchain. Application projects can use this release package through a path dependency:

```toml
[dependencies]
arcqml = { path = "PATH_TO_ARCQML/crates/arcqml" }
```

Before building, `ARCQML_RUNTIME_LIB_DIR` must point to the Runtime library directory matching the target triple.

Windows PowerShell:

```powershell
$env:RUSTUP_TOOLCHAIN = "1.98.0"
$env:ARCQML_RUNTIME_LIB_DIR = "PATH_TO_ARCQML\libs\x86_64-pc-windows-msvc"
cargo run
```

Linux (for reference regarding future platform support; Linux binaries are not included in this release):

```bash
export ARCQML_RUNTIME_LIB_DIR="PATH_TO_ARCQML/libs/x86_64-unknown-linux-gnu"
cargo run
```

Minimal program:

```rust
use arcqml::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let mut circuit = Circuit::new(2)?;
    circuit.h(0usize)?.cnot(0usize, 1usize)?;

    let mut simulator = StateVectorSimulator::new(2)?;
    simulator.apply_circuit(&circuit)?;

    println!("amplitudes = {:?}", simulator.amplitudes()?);
    println!("probabilities = {:?}", probabilities(&simulator)?);
    println!("counts = {:?}", simulator.sample_counts(1_000, Some(42))?);
    Ok(())
}
```

## Your First Trainable Circuit

The following example trains the Pauli-Z expectation value of a single-qubit rotation gate toward a target value. `ry` creates a trainable parameter; to create a fixed angle instead, use `ry_fixed` in the Rust API.

```python
import arcqml

circuit = arcqml.Circuit(1)
circuit.ry(0.3, 0)

observable = arcqml.PauliSum.z(1, 0)
simulator = arcqml.StateVectorSimulator(1)
optimizer = arcqml.Adam(learning_rate=0.05)
target = arcqml.tensor(0.2)

for step in range(20):
    prediction = simulator.run(circuit, observable)
    loss = arcqml.mse_loss(prediction, target)
    loss.backward()
    optimizer.step(circuit)
    optimizer.zero_grad(circuit)

print("loss =", loss.item())
print("parameters =", circuit.parameter_values())
```

ArcQML defines mean squared error as half the mean squared error:

$$
L = \frac{1}{2N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2
$$

`backward()` accumulates gradients, so old gradients must be cleared in every training iteration. `no_grad()` can be used for validation and inference to avoid creating an automatic differentiation graph.

## Feature Overview

| Capability | Rust | Python | Primary Entry Point |
| --- | :---: | :---: | --- |
| Parameterized quantum circuits | Full gate set | Common gate subset | `Circuit` |
| Single-state vector simulation | ✓ | ✓ | `StateVectorSimulator` |
| Batch-state vector simulation | ✓ | ✓ | `BatchStateVectorSimulator` |
| Pauli sums and expectation values | ✓ | ✓ | `SparsePauliOp` / `PauliSum` |
| Quantum adjoint gradients | ✓ | ✓ | `run` + `Tensor.backward` |
| Tensor and linear algebra | ✓ | Basic Tensor | `arcqml-core` / `arcqml-linalg` |
| Loss functions | ✓ | Partial | `arcqml-loss` |
| Optimizers | ✓ | Partial | `arcqml-optim` |
| Probabilities, marginal probabilities, fidelity, and Bloch vectors | ✓ | ✓ | `arcqml-analysis` |
| Terminal sampling | Single state only | Single state only | `sample_counts` |
| Text circuit diagrams and SVG | ✓ | — | `arcqml-visualization` |
| Circuit structure JSON | ✓ | — | `Circuit::to_json` / `from_json` |
| Parameter weights JSON | ✓ | — | `save_weights` / `load_weights` |
| Unitary matrices and fitting | ✓ | — | `arcqml-unitary` |

## Important Conventions

- Qubits are numbered starting from `0`.
- Internal state indices are ordered from high to low, with `q0` as the least significant bit.
- Sampling strings are displayed as `q[n-1]...q[0]`; in a two-qubit system, applying only `x(0)` produces `"01"`.
- `apply_circuit` modifies the simulator's current state; `run` computes from the current state without modifying the simulator.
- `sample_counts` samples only from the current single state. It does not execute a circuit or cause state collapse.
- A batch state Tensor has the shape `[batch_size, 2^num_qubits]`; each row must be normalized and stored in contiguous row-major order.
- `Tensor::clone()` copies a shared handle; use `deep_clone()` when independent data is required.

## TODO

- CUDA, GPU backends, and distributed execution;
- sparse-state, tensor-network, or matrix-product-state simulation;
- noise models, density matrices, and quantum channels;
- mid-circuit measurement, state collapse, and classical conditional control;
- access to real quantum hardware or cloud backends;
- batch sampling;
- the complete Rust gate set, circuit JSON, weight checkpoints, and unitary matrix APIs in Python.

## Examples and Documentation

- [Technical Manual](docs/ArcQML技术手册.md): Mathematical conventions, data layout, automatic differentiation, and troubleshooting.
- [H₂ VQE Tutorial](docs/tutorial/h2_vqe.md): Variational ground-state solving in quantum chemistry.
- [QNN Credit Classification Tutorial](docs/tutorial/qnn_german_credit.md): Batched quantum neural network training.
- [Custom Loss Tutorial](docs/tutorial/custom_loss.md): Composing Rust operators, validating gradients, integrating training, and writing custom backward implementations.
- [Rust Examples](examples/rust) and [Python Examples](examples/python).

## Release Status and Licensing

ArcQML uses a mixed licensing model. This is a source-available release, not one distributed under an OSI-approved open-source license:

- Proprietary publicly available source code is licensed under the [ArcQML Non-Commercial Source License](LICENSE): only non-commercial use is permitted, and the complete source code of modifications and integrated works must be made public according to its terms; commercial use requires separate authorization.
- The official closed-source Runtime is licensed under the [Non-Commercial Binary License](LICENSE-RUNTIME). The public source license grants an exception for the unmodified official Runtime and does not require disclosure of its private implementation; user integration code must still be made public.
- The public framework code, closed-source Runtime, and third-party components included in the wheel are each subject to their respective licenses. Third-party materials retain their original licenses; see [Third-Party Notices](THIRD_PARTY_NOTICES.md), the versioned license texts included in the package, and the SBOM.
- Rights previously granted under lawfully obtained MIT/Apache-licensed copies are not revoked by this change in release policy.

Maintainer: liuxl; commercial authorization and licensing contact: quill@arclightquantum.com.

This is an experimental preview release for Windows, using Rust 1.98.0 and CPython 3.11. Memory-safety issue S01, Linux acceptance testing, and formatting cleanup have been deferred; see the [Release Notes](RELEASE_NOTES.md) and [Security Limitations](SECURITY.md) for details. Use only the libraries and wheel included with this release, and do not mix them with historical artifacts.
