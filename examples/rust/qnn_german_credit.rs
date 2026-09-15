use arcqml::prelude::*;
use num_complex::Complex64;
use std::{cmp::Ordering, f64::consts::PI, fs};
use arcqml_visualization::write_svg;

type AppResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const QUBITS: usize = 10;
const STATE_DIMENSION: usize = 1 << QUBITS;
const PARAMETERS_PER_LAYER: usize = 37;
const LAYERS: usize = 1;
const EPOCHS: usize = 5;
const BATCH_SIZE: usize = 100;
const LEARNING_RATE: f64 = 0.01;
const SEED: u64 = 4;
const DATA_PATH: &str = "examples/data/german_credit.csv";
const FEATURE_COLUMNS: [&str; QUBITS] = [
    "Account Balance",
    "Payment Status of Previous Credit",
    "Purpose",
    "Value Savings/Stocks",
    "Length of current employment",
    "Guarantors",
    "Most valuable available asset",
    "Concurrent Credits",
    "Type of apartment",
    "No of Credits at this Bank",
];

#[derive(Clone)]
struct Sample {
    features: [f64; QUBITS],
    label: f64,
}

struct Split {
    train: Vec<Sample>,
    validation: Vec<Sample>,
    test: Vec<Sample>,
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        radius * (2.0 * PI * self.uniform()).cos()
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let selected = (self.next_u64() as usize) % (index + 1);
            values.swap(index, selected);
        }
    }
}

fn load_data(path: &str) -> AppResult<Split> {
    let text = fs::read_to_string(path)?;
    let mut lines = text.lines();
    let headers: Vec<&str> = lines.next().ok_or("CSV is empty")?.split(',').collect();
    let label_column = headers
        .iter()
        .position(|name| *name == "Creditability")
        .ok_or("missing Creditability column")?;
    let feature_indices: Vec<usize> = FEATURE_COLUMNS
        .iter()
        .map(|wanted| {
            headers
                .iter()
                .position(|name| name == wanted)
                .ok_or_else(|| format!("missing feature column: {wanted}"))
        })
        .collect::<std::result::Result<Vec<_>, String>>()?;

    let mut samples = Vec::new();
    for (row_index, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != headers.len() {
            return Err(format!("invalid CSV row {}", row_index + 2).into());
        }
        let mut features = [0.0; QUBITS];
        for (output, &column) in features.iter_mut().zip(&feature_indices) {
            *output = fields[column].parse()?;
        }
        // 原始数据：1=信用良好、0=信用不良；基准将“信用不良”作为正类。
        let creditability: u8 = fields[label_column].parse()?;
        samples.push(Sample {
            features,
            label: f64::from(1 - creditability),
        });
    }

    // 与 benchmark 预处理相同，将每列 min-max 缩放到 [π/2, 3π/2]。
    for column in 0..QUBITS {
        let minimum = samples
            .iter()
            .map(|sample| sample.features[column])
            .fold(f64::INFINITY, f64::min);
        let maximum = samples
            .iter()
            .map(|sample| sample.features[column])
            .fold(f64::NEG_INFINITY, f64::max);
        for sample in &mut samples {
            sample.features[column] =
                PI / 2.0 + (sample.features[column] - minimum) / (maximum - minimum) * PI;
        }
    }

    Rng::new(SEED).shuffle(&mut samples);
    let test = samples.split_off(900);
    let validation = samples.split_off(800);
    Ok(Split {
        train: samples,
        validation,
        test,
    })
}

fn encode_product_states(samples: &[Sample]) -> AppResult<Tensor> {
    let mut amplitudes = Vec::with_capacity(samples.len() * STATE_DIMENSION);
    for sample in samples {
        let mut state = vec![Complex64::new(1.0, 0.0)];
        for qubit in (0..QUBITS).rev() {
            let angle = sample.features[qubit];
            let half = angle / 2.0;
            let phase_zero = Complex64::from_polar(1.0, -half) * half.cos();
            let phase_one =
                Complex64::new(0.0, -1.0) * Complex64::from_polar(1.0, half) * half.sin();
            let mut expanded = Vec::with_capacity(state.len() * 2);
            for amplitude in state {
                expanded.push(amplitude * phase_zero);
                expanded.push(amplitude * phase_one);
            }
            state = expanded;
        }
        amplitudes.extend(state);
    }
    Ok(Tensor::new(TensorData::FlatC64 {
        data: amplitudes,
        shape: vec![samples.len(), STATE_DIMENSION],
    })?)
}

fn append_ansatz_layer(circuit: &mut Circuit, values: &[f64]) -> AppResult<()> {
    if values.len() != PARAMETERS_PER_LAYER {
        return Err("each ansatz layer requires 37 parameters".into());
    }
    for qubit in 0..QUBITS {
        circuit.rx(values[qubit], qubit)?;
    }
    for (control, target) in [(0, 1), (2, 3), (4, 5), (9, 8), (7, 6)] {
        circuit.cnot(control, target)?;
    }
    for qubit in 1..9 {
        circuit.rx(values[9 + qubit], qubit)?;
    }
    for (control, target) in [(1, 2), (3, 4), (8, 7), (6, 5)] {
        circuit.cnot(control, target)?;
    }
    for qubit in 2..8 {
        circuit.rx(values[16 + qubit], qubit)?;
    }
    for (control, target) in [(2, 3), (4, 5), (7, 6)] {
        circuit.cnot(control, target)?;
    }
    for (parameter, qubit) in [(24, 3), (25, 4), (26, 5), (27, 6)] {
        circuit.ry(values[parameter], qubit)?;
    }
    circuit.cnot(3, 4)?.cnot(6, 5)?;
    circuit
        .rz(values[28], 4)?
        .ry(values[29], 4)?
        .rz(values[30], 4)?;
    circuit
        .rz(values[31], 5)?
        .ry(values[32], 5)?
        .rz(values[33], 5)?;
    circuit.cnot(4, 5)?;
    circuit
        .rz(values[34], 5)?
        .ry(values[35], 5)?
        .rz(values[36], 5)?;
    Ok(())
}

fn build_circuit() -> AppResult<Circuit> {
    let mut rng = Rng::new(SEED);
    let mut circuit = Circuit::new(QUBITS)?;
    for _layer in 0..LAYERS {
        let values: Vec<f64> = (0..PARAMETERS_PER_LAYER).map(|_| rng.normal()).collect();
        append_ansatz_layer(&mut circuit, &values)?;
    }
    Ok(circuit)
}

fn labels(samples: &[Sample]) -> AppResult<Tensor> {
    Ok(Tensor::new(
        samples
            .iter()
            .map(|sample| sample.label)
            .collect::<Vec<_>>(),
    )?)
}

fn loss_and_backward(
    circuit: &Circuit,
    observable: &SparsePauliOp,
    samples: &[Sample],
) -> AppResult<f64> {
    let states = encode_product_states(samples)?;
    let simulator = BatchStateVectorSimulator::from_state_tensor(QUBITS, states)?;
    let logits = simulator.run(circuit, observable)?;
    let loss = binary_cross_entropy_with_logits_loss(&logits, &labels(samples)?)?;
    let value = loss.value()?;
    loss.backward()?;
    Ok(value)
}

fn predict(
    circuit: &Circuit,
    observable: &SparsePauliOp,
    samples: &[Sample],
) -> AppResult<Vec<f64>> {
    let _guard = no_grad();
    let states = encode_product_states(samples)?;
    let simulator = BatchStateVectorSimulator::from_state_tensor(QUBITS, states)?;
    let logits = simulator.run(circuit, observable)?;
    let storage = logits.storage();
    match &*storage {
        Storage::F64(values) => Ok(values.clone()),
        _ => Err("QNN logits must be F64".into()),
    }
}

fn validation(
    circuit: &Circuit,
    observable: &SparsePauliOp,
    samples: &[Sample],
) -> AppResult<(f64, f64)> {
    let scores = predict(circuit, observable, samples)?;
    let targets = labels(samples)?;
    let logits = Tensor::new(scores.clone())?;
    let loss = binary_cross_entropy_with_logits_loss(&logits, &targets)?.value()?;
    let correct = scores
        .iter()
        .zip(samples)
        .filter(|(score, sample)| (**score >= 0.0) == (sample.label == 1.0))
        .count();
    Ok((loss, correct as f64 / samples.len() as f64))
}

fn auc_metrics(scores: &[f64], samples: &[Sample]) -> (f64, f64) {
    let mut pairs: Vec<(f64, bool)> = scores
        .iter()
        .zip(samples)
        .map(|(&score, sample)| (score, sample.label == 1.0))
        .collect();
    pairs.sort_by(|left, right| right.0.partial_cmp(&left.0).unwrap_or(Ordering::Equal));
    let positives = pairs.iter().filter(|pair| pair.1).count() as f64;
    let negatives = pairs.len() as f64 - positives;
    let (mut tp, mut fp) = (0.0, 0.0);
    let (mut previous_tpr, mut previous_fpr, mut previous_recall, mut previous_precision) =
        (0.0, 0.0, 0.0, 1.0);
    let (mut roc_auc, mut pr_auc) = (0.0, 0.0);
    for (_, positive) in pairs {
        if positive {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let tpr = tp / positives;
        let fpr = fp / negatives;
        let precision = tp / (tp + fp);
        roc_auc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0;
        pr_auc += (tpr - previous_recall) * (precision + previous_precision) / 2.0;
        (previous_tpr, previous_fpr) = (tpr, fpr);
        (previous_recall, previous_precision) = (tpr, precision);
    }
    (roc_auc, pr_auc)
}

fn main() -> AppResult<()> {
    let data = load_data(DATA_PATH)?;
    let circuit = build_circuit()?;
    let observable = SparsePauliOp::single(QUBITS, 5usize, Pauli::Z, 1.0)?;
    let mut optimizer = Adam::new(LEARNING_RATE, 0.9, 0.999, 1e-8, 0.0)?;
    let mut rng = Rng::new(SEED);

    println!(
        "samples: train={}, validation={}, test={}",
        data.train.len(),
        data.validation.len(),
        data.test.len()
    );
    println!(
        "qubits={QUBITS}, layers={LAYERS}, parameters={}",
        circuit.num_parameters()
    );
    for epoch in 1..=EPOCHS {
        let mut indices: Vec<usize> = (0..data.train.len()).collect();
        rng.shuffle(&mut indices);
        let mut weighted_loss = 0.0;
        for batch_indices in indices.chunks(BATCH_SIZE) {
            let batch: Vec<Sample> = batch_indices
                .iter()
                .map(|&index| data.train[index].clone())
                .collect();
            let loss = loss_and_backward(&circuit, &observable, &batch)?;
            optimizer.step(circuit.parameters())?;
            optimizer.zero_grad(circuit.parameters());
            weighted_loss += loss * batch.len() as f64;
        }
        let (validation_loss, validation_accuracy) =
            validation(&circuit, &observable, &data.validation)?;
        println!(
            "epoch {epoch:02}/{EPOCHS}: train_loss={:.6}, validation_loss={validation_loss:.6}, validation_accuracy={:.1}%",
            weighted_loss / data.train.len() as f64,
            validation_accuracy * 100.0
        );
    }

    write_svg(&circuit, "./examples/rust/qnn_circuit.svg");

    let test_scores = predict(&circuit, &observable, &data.test)?;
    let (roc_auc, pr_auc) = auc_metrics(&test_scores, &data.test);
    println!("test ROC-AUC={roc_auc:.4}, PR-AUC={pr_auc:.4}");
    Ok(())
}
