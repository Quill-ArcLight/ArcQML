use arcqml_checkpoint::{
    CheckpointError, ParameterRecord, WeightCheckpoint, load_weights, save_weights,
};
use arcqml_circuit::{Circuit, ParameterId};
use arcqml_core::Tensor;
use std::fs::{self, File};
use std::time::{SystemTime, UNIX_EPOCH};

/// 生成唯一的临时 JSON 文件路径，避免测试之间发生冲突。
fn temporary_path(stem: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{stem}-{}-{nonce}.json", std::process::id()))
}

/// 创建用于检查点测试的双量子比特参数化电路。
fn circuit() -> Circuit {
    let mut circuit = Circuit::new(2).unwrap();
    circuit.ry(0.1, 0).unwrap();
    circuit.rz(-0.2, 1).unwrap();
    circuit
}

/// 验证保存并加载后参数值恢复，且历史梯度会被清空。
#[test]
fn round_trip_restores_weights_and_clears_gradients() {
    let path = temporary_path("arcqml-checkpoint-round-trip");
    let source = circuit();
    source.parameters()[0].set_tensor(Tensor::new(0.75_f64).unwrap());
    source.parameters()[1].set_tensor(Tensor::new(-0.5_f64).unwrap());
    save_weights(&source, &path).unwrap();

    let restored = circuit();
    restored.parameters()[0].set_grad(Tensor::new(1.0_f64).unwrap());
    load_weights(&restored, &path).unwrap();

    assert_eq!(
        restored
            .parameter_scalar_value(ParameterId::new(0))
            .unwrap(),
        0.75
    );
    assert_eq!(
        restored
            .parameter_scalar_value(ParameterId::new(1))
            .unwrap(),
        -0.5
    );
    assert!(restored.parameters()[0].grad().is_none());
    fs::remove_file(path).unwrap();
}

/// 验证严格加载会拒绝目标电路不存在的额外参数。
#[test]
fn strict_load_rejects_an_unexpected_parameter() {
    let path = temporary_path("arcqml-checkpoint-unexpected");
    let source = circuit();
    let checkpoint = WeightCheckpoint {
        format: "arcqml/weights".to_string(),
        num_qubits: 2,
        parameters: vec![
            ParameterRecord {
                name: "ry_q0_theta_0".to_string(),
                dtype: "f64".to_string(),
                value: 0.1,
            },
            ParameterRecord {
                name: "rz_q1_theta_1".to_string(),
                dtype: "f64".to_string(),
                value: -0.2,
            },
            ParameterRecord {
                name: "extra".to_string(),
                dtype: "f64".to_string(),
                value: 1.0,
            },
        ],
    };
    let file = File::create(&path).unwrap();
    serde_json::to_writer(file, &checkpoint).unwrap();

    let error = load_weights(&source, &path).unwrap_err();
    assert!(matches!(
        error,
        CheckpointError::UnexpectedParameterError { .. }
    ));
    fs::remove_file(path).unwrap();
}
