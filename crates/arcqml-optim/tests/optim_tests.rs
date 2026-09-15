use arcqml_core::prelude::*;
use arcqml_optim::prelude::*;

/// 从给定数据和形状创建 F64 Tensor。
fn make_f64_tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::new(TensorData::FlatF64 { data, shape }).unwrap()
}

/// 从给定数据和形状创建 F32 Tensor。
fn make_f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::new(TensorData::FlatF32 { data, shape }).unwrap()
}

/// 从给定数据和形状创建 I64 Tensor。
fn make_i64_tensor(data: Vec<i64>, shape: Vec<usize>) -> Tensor {
    Tensor::new(TensorData::FlatI64 { data, shape }).unwrap()
}

/// 为 Tensor 创建并命名可训练 Parameter。
fn make_parameter(name: &str, tensor: Tensor) -> Parameter {
    let parameter = Parameter::new(tensor);

    parameter.set_name(name);

    parameter
}

/// 断言两个 F64 切片在测试容差内逐元素相等。
fn assert_f64_slice_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
/// 验证 SGD 更新 F64 Parameter 并保持已有 Tensor 句柄同步。
fn sgd_should_update_f64_parameter() {
    let parameter = make_parameter("theta", make_f64_tensor(vec![1.0, 2.0], vec![2]));
    let tensor_handle = parameter.tensor();

    parameter.set_grad(make_f64_tensor(vec![0.1, -0.2], vec![2]));

    let optimizer = Sgd::new(0.5, 0.0).unwrap();
    let stats = optimizer.step(std::slice::from_ref(&parameter)).unwrap();

    assert_eq!(stats.updated(), 1);
    assert_eq!(stats.skipped_frozen(), 0);
    assert_eq!(stats.skipped_no_grad(), 0);
    assert_f64_slice_close(
        parameter.tensor().storage().as_f64_slice().unwrap(),
        &[0.95, 2.1],
    );
    assert_f64_slice_close(
        tensor_handle.storage().as_f64_slice().unwrap(),
        &[0.95, 2.1],
    );
    assert!(parameter.grad().is_some());
}

#[test]
/// 验证 SGD 更新 F32 Parameter。
fn sgd_should_update_f32_parameter() {
    let parameter = make_parameter("theta", make_f32_tensor(vec![1.0, 2.0], vec![2]));

    parameter.set_grad(make_f32_tensor(vec![0.25, -0.5], vec![2]));

    let optimizer = Sgd::new(0.2, 0.0).unwrap();

    optimizer.step(std::slice::from_ref(&parameter)).unwrap();

    let tensor = parameter.tensor();
    let storage = tensor.storage();
    let values = storage.as_f32_slice().unwrap();

    assert!((values[0] - 0.95).abs() < 1e-6);
    assert!((values[1] - 2.1).abs() < 1e-6);
}

#[test]
/// 验证 SGD 在更新中加入耦合权重衰减。
fn sgd_should_apply_weight_decay() {
    let parameter = make_parameter("theta", make_f64_tensor(vec![2.0], vec![]));

    parameter.set_grad(make_f64_tensor(vec![0.5], vec![]));

    let optimizer = Sgd::new(0.1, 0.25).unwrap();

    optimizer.step(std::slice::from_ref(&parameter)).unwrap();

    assert_f64_slice_close(parameter.tensor().storage().as_f64_slice().unwrap(), &[1.9]);
}

#[test]
/// 验证 SGD 跳过冻结 Parameter 和不存在梯度的 Parameter。
fn sgd_should_skip_frozen_and_no_grad_parameters() {
    let frozen = make_parameter("frozen", make_f64_tensor(vec![1.0], vec![]));
    let no_grad = make_parameter("no_grad", make_f64_tensor(vec![2.0], vec![]));

    frozen.freeze();

    let parameters = vec![frozen, no_grad];
    let optimizer = Sgd::new(0.1, 0.0).unwrap();
    let stats = optimizer.step(&parameters).unwrap();

    assert_eq!(stats.updated(), 0);
    assert_eq!(stats.skipped_frozen(), 1);
    assert_eq!(stats.skipped_no_grad(), 1);
    assert_f64_slice_close(
        parameters[0].tensor().storage().as_f64_slice().unwrap(),
        &[1.0],
    );
    assert_f64_slice_close(
        parameters[1].tensor().storage().as_f64_slice().unwrap(),
        &[2.0],
    );
}

#[test]
/// 验证 SGD 清空 Parameter 梯度。
fn sgd_zero_grad_should_clear_all_gradients() {
    let parameter = make_parameter("theta", make_f64_tensor(vec![1.0], vec![]));

    parameter.set_grad(make_f64_tensor(vec![0.1], vec![]));

    let optimizer = Sgd::new(0.1, 0.0).unwrap();

    optimizer.zero_grad(std::slice::from_ref(&parameter));

    assert!(parameter.grad().is_none());
}

#[test]
/// 验证 SGD 可直接更新叶子 Tensor。
fn sgd_should_update_leaf_tensors_directly() {
    let tensor = make_f64_tensor(vec![1.0], vec![]);
    tensor.set_requires_grad(true);
    tensor.set_grad(make_f64_tensor(vec![0.5], vec![]));

    let stats = Sgd::new(0.2, 0.0)
        .unwrap()
        .step_tensors(std::slice::from_ref(&tensor))
        .unwrap();

    assert_eq!(stats.updated(), 1);
    assert_f64_slice_close(tensor.storage().as_f64_slice().unwrap(), &[0.9]);
}

#[test]
/// 验证 Adam 连续更新 Parameter 并保留一阶矩和二阶矩状态。
fn adam_should_update_parameter_and_preserve_moments() {
    let parameter = make_parameter("theta", make_f64_tensor(vec![1.0], vec![]));
    parameter.set_grad(make_f64_tensor(vec![0.5], vec![]));
    let mut optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0).unwrap();

    optimizer.step(std::slice::from_ref(&parameter)).unwrap();
    optimizer.step(std::slice::from_ref(&parameter)).unwrap();

    assert_eq!(optimizer.step_count(), 2);
    assert!((parameter.tensor().storage().as_f64_slice().unwrap()[0] - 0.8).abs() < 1e-8);
}

#[test]
/// 验证 Adam 可直接更新 F32 叶子 Tensor。
fn adam_should_update_leaf_f32_tensor_directly() {
    let tensor = make_f32_tensor(vec![1.0], vec![]);
    tensor.set_requires_grad(true);
    tensor.set_grad(make_f32_tensor(vec![0.25], vec![]));
    let mut optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0).unwrap();

    optimizer
        .step_tensors(std::slice::from_ref(&tensor))
        .unwrap();

    assert!((tensor.storage().as_f32_slice().unwrap()[0] - 0.9).abs() < 1e-5);
}

#[test]
/// 验证 Adam 拒绝非法超参数和变化的参数切片长度。
fn adam_should_validate_hyperparameters_and_state_shape() {
    assert!(matches!(
        Adam::new(0.1, 1.0, 0.999, 1e-8, 0.0),
        Err(OptimError::InvalidHyperParameterError { .. })
    ));

    let first = make_parameter("first", make_f64_tensor(vec![1.0], vec![]));
    first.set_grad(make_f64_tensor(vec![0.1], vec![]));
    let second = make_parameter("second", make_f64_tensor(vec![1.0], vec![]));
    second.set_grad(make_f64_tensor(vec![0.1], vec![]));
    let mut optimizer = Adam::new(0.1, 0.9, 0.999, 1e-8, 0.0).unwrap();

    optimizer.step(std::slice::from_ref(&first)).unwrap();
    assert!(matches!(
        optimizer.step(&[first, second]),
        Err(OptimError::AdamStateMismatchError { .. })
    ));
}

#[test]
/// 验证 SGD 拒绝非法学习率和权重衰减。
fn sgd_should_validate_hyper_parameters() {
    assert!(matches!(
        Sgd::new(-0.1, 0.0),
        Err(OptimError::InvalidHyperParameterError { .. })
    ));

    assert!(matches!(
        Sgd::new(0.1, f64::NAN),
        Err(OptimError::InvalidHyperParameterError { .. })
    ));
}

#[test]
/// 验证 SGD 拒绝元素数量不一致的梯度。
fn sgd_should_reject_grad_shape_mismatch() {
    let parameter = make_parameter("theta", make_f64_tensor(vec![1.0, 2.0], vec![2]));

    parameter.set_grad(make_f64_tensor(vec![0.1], vec![]));

    let optimizer = Sgd::new(0.1, 0.0).unwrap();
    let result = optimizer.step(std::slice::from_ref(&parameter));

    assert!(matches!(
        result,
        Err(OptimError::GradShapeMismatchError { .. })
    ));
}

#[test]
/// 验证 SGD 拒绝数据类型不一致的梯度。
fn sgd_should_reject_grad_dtype_mismatch() {
    let parameter = make_parameter("theta", make_f64_tensor(vec![1.0], vec![]));

    parameter.set_grad(make_f32_tensor(vec![0.1], vec![]));

    let optimizer = Sgd::new(0.1, 0.0).unwrap();
    let result = optimizer.step(std::slice::from_ref(&parameter));

    assert!(matches!(
        result,
        Err(OptimError::GradDTypeMismatchError { .. })
    ));
}

#[test]
/// 验证不支持的 Parameter 数据类型会在创建阶段被拒绝。
fn parameter_should_reject_unsupported_dtype_at_creation() {
    assert!(Parameter::try_new(make_i64_tensor(vec![1], vec![])).is_err());
}
