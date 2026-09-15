"""使用 ArcQML Python API 训练德国信用数据集 QNN。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import arcqml


QUBITS = 10
STATE_DIMENSION = 1 << QUBITS
PARAMETERS_PER_LAYER = 37
LAYERS = 1
EPOCHS = 5
BATCH_SIZE = 100
LEARNING_RATE = 0.01
SEED = 4
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "german_credit.csv"
FEATURE_COLUMNS = (
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
)


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 CSV，生成角度特征和二分类标签。"""
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    features = np.asarray(
        [[float(row[column]) for column in FEATURE_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    labels = np.asarray(
        [1.0 - float(row["Creditability"]) for row in rows],
        dtype=np.float64,
    )
    minimum = features.min(axis=0)
    maximum = features.max(axis=0)
    features = np.pi / 2.0 + (features - minimum) / (maximum - minimum) * np.pi
    return features, labels


def encode_product_states(features: np.ndarray) -> np.ndarray:
    """将 RX(x) 后接 RZ(x) 的特征编码直接转换为乘积态。"""
    states = np.ones((len(features), 1), dtype=np.complex128)
    for qubit in range(QUBITS - 1, -1, -1):
        angles = features[:, qubit]
        local_states = np.column_stack(
            (
                np.exp(-0.5j * angles) * np.cos(0.5 * angles),
                -1j * np.exp(0.5j * angles) * np.sin(0.5 * angles),
            )
        )
        states = np.einsum("bi,bj->bij", states, local_states).reshape(len(features), -1)
    assert states.shape == (len(features), STATE_DIMENSION)
    return np.ascontiguousarray(states, dtype=np.complex128)


def append_ansatz_layer(circuit: arcqml.Circuit, values: np.ndarray) -> None:
    """添加与 benchmark 相同的 37 参数、52 门 ansatz 层。"""
    if values.shape != (PARAMETERS_PER_LAYER,):
        raise ValueError("each ansatz layer requires 37 parameters")
    for qubit in range(QUBITS):
        circuit.rx(angle=float(values[qubit]), qubit=qubit)
    for control, target in ((0, 1), (2, 3), (4, 5), (9, 8), (7, 6)):
        circuit.cnot(control=control, target=target)
    for qubit in range(1, 9):
        circuit.rx(angle=float(values[9 + qubit]), qubit=qubit)
    for control, target in ((1, 2), (3, 4), (8, 7), (6, 5)):
        circuit.cnot(control=control, target=target)
    for qubit in range(2, 8):
        circuit.rx(angle=float(values[16 + qubit]), qubit=qubit)
    for control, target in ((2, 3), (4, 5), (7, 6)):
        circuit.cnot(control=control, target=target)
    for parameter, qubit in ((24, 3), (25, 4), (26, 5), (27, 6)):
        circuit.ry(angle=float(values[parameter]), qubit=qubit)
    circuit.cnot(control=3, target=4)
    circuit.cnot(control=6, target=5)
    circuit.rz(angle=float(values[28]), qubit=4)
    circuit.ry(angle=float(values[29]), qubit=4)
    circuit.rz(angle=float(values[30]), qubit=4)
    circuit.rz(angle=float(values[31]), qubit=5)
    circuit.ry(angle=float(values[32]), qubit=5)
    circuit.rz(angle=float(values[33]), qubit=5)
    circuit.cnot(control=4, target=5)
    circuit.rz(angle=float(values[34]), qubit=5)
    circuit.ry(angle=float(values[35]), qubit=5)
    circuit.rz(angle=float(values[36]), qubit=5)


def build_circuit(rng: np.random.Generator) -> arcqml.Circuit:
    circuit = arcqml.Circuit(num_qubits=QUBITS)
    for _layer in range(LAYERS):
        append_ansatz_layer(
            circuit=circuit,
            values=rng.standard_normal(PARAMETERS_PER_LAYER),
        )
    return circuit


def predict(circuit: arcqml.Circuit, observable: arcqml.PauliSum, states: np.ndarray) -> np.ndarray:
    simulator = arcqml.BatchStateVectorSimulator.from_amplitudes(
        num_qubits=QUBITS,
        amplitudes=states,
    )
    with arcqml.no_grad():
        return simulator.run(circuit=circuit, observable=observable).numpy()


def validate(
    circuit: arcqml.Circuit,
    observable: arcqml.PauliSum,
    states: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float]:
    scores = predict(circuit=circuit, observable=observable, states=states)
    loss = arcqml.binary_cross_entropy_with_logits(
        logits=arcqml.tensor(scores),
        targets=arcqml.tensor(labels),
    ).item()
    accuracy = np.mean((scores >= 0.0) == (labels == 1.0))
    return float(loss), float(accuracy)


def auc_metrics(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """不依赖 scikit-learn 计算 ROC-AUC 与梯形 PR-AUC。"""
    order = np.argsort(-scores, kind="stable")
    positives = float(labels.sum())
    negatives = float(len(labels) - labels.sum())
    tp = fp = roc_auc = pr_auc = 0.0
    previous_tpr = previous_fpr = previous_recall = 0.0
    previous_precision = 1.0
    for index in order:
        if labels[index] == 1.0:
            tp += 1.0
        else:
            fp += 1.0
        tpr, fpr = tp / positives, fp / negatives
        precision = tp / (tp + fp)
        roc_auc += (fpr - previous_fpr) * (tpr + previous_tpr) / 2.0
        pr_auc += (tpr - previous_recall) * (precision + previous_precision) / 2.0
        previous_tpr, previous_fpr = tpr, fpr
        previous_recall, previous_precision = tpr, precision
    return roc_auc, pr_auc


def main() -> None:
    features, labels = load_data(path=DATA_PATH)
    rng = np.random.default_rng(seed=SEED)
    indices = rng.permutation(len(features))
    train, validation, test = indices[:800], indices[800:900], indices[900:]
    states = encode_product_states(features=features)
    circuit = build_circuit(rng=rng)
    observable = arcqml.PauliSum.z(num_qubits=QUBITS, qubit=5)
    optimizer = arcqml.Adam(learning_rate=LEARNING_RATE)

    print(f"samples: train={len(train)}, validation={len(validation)}, test={len(test)}")
    print(f"qubits={QUBITS}, layers={LAYERS}, parameters={circuit.num_parameters}")
    for epoch in range(1, EPOCHS + 1):
        rng.shuffle(train)
        weighted_loss = 0.0
        for start in range(0, len(train), BATCH_SIZE):
            batch_indices = train[start : start + BATCH_SIZE]
            simulator = arcqml.BatchStateVectorSimulator.from_amplitudes(
                num_qubits=QUBITS,
                amplitudes=np.ascontiguousarray(states[batch_indices]),
            )
            logits = simulator.run(circuit=circuit, observable=observable)
            targets = arcqml.tensor(np.ascontiguousarray(labels[batch_indices]))
            loss = arcqml.binary_cross_entropy_with_logits(logits=logits, targets=targets)
            loss.backward()
            optimizer.step(circuit=circuit)
            optimizer.zero_grad(circuit=circuit)
            weighted_loss += float(loss.item()) * len(batch_indices)

        validation_loss, validation_accuracy = validate(
            circuit=circuit,
            observable=observable,
            states=np.ascontiguousarray(states[validation]),
            labels=np.ascontiguousarray(labels[validation]),
        )
        print(
            f"epoch {epoch:02}/{EPOCHS}: train_loss={weighted_loss / len(train):.6f}, "
            f"validation_loss={validation_loss:.6f}, "
            f"validation_accuracy={validation_accuracy * 100:.1f}%"
        )

    test_scores = predict(
        circuit=circuit,
        observable=observable,
        states=np.ascontiguousarray(states[test]),
    )
    roc_auc, pr_auc = auc_metrics(scores=test_scores, labels=labels[test])
    print(f"test ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")


if __name__ == "__main__":
    main()
