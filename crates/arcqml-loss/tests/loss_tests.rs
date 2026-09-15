use arcqml_core::Tensor;
use arcqml_loss::prelude::*;

/// 断言两个 F64 值在测试容差内相等。
fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12);
}

#[test]
/// 验证均方误差损失的归约值及其反向梯度。
fn tensor_mse_loss_should_reduce_and_backpropagate() {
    let prediction = Tensor::new(vec![1.0_f64, 3.0]).unwrap();
    let target = Tensor::new(vec![0.0_f64, 1.0]).unwrap();
    prediction.set_requires_grad(true);
    target.set_requires_grad(true);

    let loss = mse_loss(&prediction, &target).unwrap();

    assert_eq!(loss.shape(), &[]);
    assert_close(loss.storage().as_f64_slice().unwrap()[0], 1.25);

    loss.backward().unwrap();

    assert_eq!(
        prediction.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[0.5, 1.0]
    );
    assert_eq!(
        target.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[-0.5, -1.0]
    );
}

#[test]
/// 验证 L1 损失支持广播且保留 F32 梯度类型。
fn tensor_l1_loss_should_support_broadcast_and_f32_gradients() {
    let prediction = Tensor::new(vec![1.0_f32, -2.0, 3.0]).unwrap();
    let target = Tensor::new(1.0_f32).unwrap();
    prediction.set_requires_grad(true);
    target.set_requires_grad(true);

    let loss = l1_loss(&prediction, &target).unwrap();

    assert!((loss.storage().as_f32_slice().unwrap()[0] - 5.0 / 3.0).abs() < 1e-6);

    loss.backward().unwrap();

    assert_eq!(
        prediction.grad().unwrap().storage().as_f32_slice().unwrap(),
        &[0.0, -1.0 / 3.0, 1.0 / 3.0]
    );
    assert!((target.grad().unwrap().storage().as_f32_slice().unwrap()[0]).abs() < 1e-6);
}

#[test]
fn binary_nll_loss_should_reduce_z_expectations_and_backpropagate() {
    let expectations = Tensor::new(vec![0.6_f64, -0.2]).unwrap();
    expectations.set_requires_grad(true);

    let loss = binary_nll_loss(&expectations, &[0, 1]).unwrap();

    assert_eq!(loss.shape(), &[]);
    assert_close(
        loss.storage().as_f64_slice().unwrap()[0],
        -0.5 * (0.8_f64.ln() + 0.6_f64.ln()),
    );

    loss.backward().unwrap();

    let gradient = expectations.grad().unwrap();
    assert_close(gradient.storage().as_f64_slice().unwrap()[0], -0.3125);
    assert_close(gradient.storage().as_f64_slice().unwrap()[1], 5.0 / 12.0);
}

#[test]
/// 验证损失函数拒绝非法数据类型和空输入。
fn tensor_losses_should_validate_dtype_and_empty_input() {
    let integers = Tensor::new(vec![1_i64]).unwrap();
    let empty = Tensor::new(Vec::<f64>::new()).unwrap();

    assert!(matches!(
        mse_loss(&integers, &integers),
        Err(LossError::UnsupportedTensorDType { .. })
    ));
    assert!(matches!(
        l1_loss(&empty, &empty),
        Err(LossError::EmptyTensorError)
    ));
}
