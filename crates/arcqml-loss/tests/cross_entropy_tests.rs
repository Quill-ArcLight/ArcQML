use arcqml_core::{DType, Tensor, TensorData, no_grad};
use arcqml_linalg::mul;
use arcqml_loss::{LossError, binary_cross_entropy_with_logits_loss, cross_entropy_loss};

const TOLERANCE: f64 = 1e-12;

/// 断言两个浮点数在指定容差内相等。
fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < TOLERANCE,
        "实际值为 {actual}，期望值为 {expected}"
    );
}

#[test]
/// 验证多分类交叉熵的归约值与 logits 梯度。
fn cross_entropy_should_reduce_and_backpropagate() {
    let logits = Tensor::new(vec![vec![1.0_f64, 0.0], vec![0.0, 1.0]]).unwrap();
    logits.set_requires_grad(true);

    let loss = cross_entropy_loss(&logits, &[0, 1]).unwrap();

    assert_close(
        loss.storage().as_f64_slice().unwrap()[0],
        (1.0_f64 + (-1.0_f64).exp()).ln(),
    );

    loss.backward().unwrap();

    let probability_error = 1.0 / (2.0 * (1.0_f64.exp() + 1.0));
    let expected_gradients = [
        -probability_error,
        probability_error,
        probability_error,
        -probability_error,
    ];
    for (actual, expected) in logits
        .grad()
        .unwrap()
        .storage()
        .as_f64_slice()
        .unwrap()
        .iter()
        .zip(expected_gradients)
    {
        assert_close(*actual, expected);
    }
}

#[test]
/// 验证二元 logits 交叉熵在大幅度输入上保持数值稳定并得到正确梯度。
fn binary_cross_entropy_with_logits_should_be_stable_and_backpropagate() {
    let logits = Tensor::new(vec![-3.0_f64, 4.0]).unwrap();
    let targets = Tensor::new(vec![0.0_f64, 1.0]).unwrap();
    logits.set_requires_grad(true);

    let loss = binary_cross_entropy_with_logits_loss(&logits, &targets).unwrap();

    let expected_loss =
        0.5 * ((1.0_f64 + (-3.0_f64).exp()).ln() + (1.0_f64 + (-4.0_f64).exp()).ln());
    assert_close(loss.storage().as_f64_slice().unwrap()[0], expected_loss);

    loss.backward().unwrap();

    let expected_gradients = [
        1.0 / (2.0 * (3.0_f64.exp() + 1.0)),
        -1.0 / (2.0 * (4.0_f64.exp() + 1.0)),
    ];
    let gradients = logits.grad().unwrap();
    for (actual, expected) in gradients
        .storage()
        .as_f64_slice()
        .unwrap()
        .iter()
        .zip(expected_gradients)
    {
        assert_close(*actual, expected);
    }
}

#[test]
/// 零 logits 对硬标签和软标签均使用 BCE 整体导数，而非 clamp/abs 的边界导数。
fn binary_cross_entropy_with_logits_should_differentiate_zero() {
    for dtype in [DType::F32, DType::F64] {
        for target in [0.0, 0.25, 0.5, 1.0] {
            let logits = real_tensor(dtype, &[0.0], &[]);
            let targets = real_tensor(dtype, &[target], &[]);
            logits.set_requires_grad(true);

            let loss = binary_cross_entropy_with_logits_loss(&logits, &targets).unwrap();
            assert_values(&loss, &[std::f64::consts::LN_2]);
            loss.backward().unwrap();

            assert_values(&logits.grad().unwrap(), &[0.5 - target]);
            assert!(targets.grad().is_none());
        }
    }
}

#[test]
/// 验证两种精度下的均值归约、上游缩放、非叶子输入及 logits/targets 梯度。
fn binary_cross_entropy_with_logits_should_propagate_both_input_gradients() {
    let values = [-1000.0, -2.0, -1e-6, 0.0, 0.0, 1e-6, 2.0, 1000.0];
    let labels = [1.0, 0.25, 0.0, 0.0, 1.0, 1.0, 0.75, 0.0];
    for dtype in [DType::F32, DType::F64] {
        let inputs = real_tensor(dtype, &values.map(|x| x / 2.0), &[2, 4]);
        let targets = real_tensor(dtype, &labels, &[2, 4]);
        inputs.set_requires_grad(true);
        targets.set_requires_grad(true);
        let logits = mul(&inputs, &real_tensor(dtype, &[2.0], &[])).unwrap();
        let loss = binary_cross_entropy_with_logits_loss(&logits, &targets).unwrap();
        assert!(loss.storage().as_f64_slice().map_or_else(
            || loss.storage().as_f32_slice().unwrap()[0].is_finite(),
            |values| values[0].is_finite(),
        ));
        let scaled_loss = mul(&loss, &real_tensor(dtype, &[-3.0], &[])).unwrap();
        scaled_loss.backward().unwrap();

        // sigmoid(x) = (1 + tanh(x / 2)) / 2，作为独立的解析参照。
        let expected_inputs: Vec<_> = values
            .iter()
            .zip(labels)
            .map(|(&x, y)| -6.0 * (0.5 * (1.0 + (x / 2.0).tanh()) - y) / 8.0)
            .collect();
        let expected_targets = values.map(|x| 3.0 * x / 8.0);
        let input_gradient = inputs.grad().unwrap();
        let target_gradient = targets.grad().unwrap();
        assert_eq!(input_gradient.shape(), &[2, 4]);
        assert_eq!(target_gradient.shape(), &[2, 4]);
        assert_eq!(input_gradient.dtype(), dtype);
        assert_eq!(target_gradient.dtype(), dtype);
        assert_values(&input_gradient, &expected_inputs);
        assert_values(&target_gradient, &expected_targets);
    }
}

#[test]
/// 专用节点仅在需要时记录计算图，也支持仅 targets 需要梯度。
fn binary_cross_entropy_with_logits_should_respect_gradient_mode() {
    let logits = Tensor::new(vec![-2.0_f64, 4.0]).unwrap();
    let targets = Tensor::new(vec![0.25_f64, 0.75]).unwrap();
    assert!(
        !binary_cross_entropy_with_logits_loss(&logits, &targets)
            .unwrap()
            .requires_grad()
    );

    targets.set_requires_grad(true);
    let loss = binary_cross_entropy_with_logits_loss(&logits, &targets).unwrap();
    loss.backward().unwrap();
    assert_values(&targets.grad().unwrap(), &[1.0, -2.0]);
    assert!(logits.grad().is_none());

    let _guard = no_grad();
    assert!(
        !binary_cross_entropy_with_logits_loss(&logits, &targets)
            .unwrap()
            .requires_grad()
    );
}

/// 用相同数据构造两种实数精度的测试张量。
fn real_tensor(dtype: DType, values: &[f64], shape: &[usize]) -> Tensor {
    Tensor::new(match dtype {
        DType::F32 => TensorData::FlatF32 {
            data: values.iter().map(|&value| value as f32).collect(),
            shape: shape.to_vec(),
        },
        DType::F64 => TensorData::FlatF64 {
            data: values.to_vec(),
            shape: shape.to_vec(),
        },
        _ => unreachable!(),
    })
    .unwrap()
}

/// 按张量精度检查完整的输出或梯度值。
fn assert_values(tensor: &Tensor, expected: &[f64]) {
    let (actual, tolerance): (Vec<f64>, f64) = match tensor.dtype() {
        DType::F32 => (
            tensor
                .storage()
                .as_f32_slice()
                .unwrap()
                .iter()
                .map(|&x| f64::from(x))
                .collect(),
            1e-6,
        ),
        DType::F64 => (tensor.storage().as_f64_slice().unwrap().to_vec(), TOLERANCE),
        _ => unreachable!(),
    };
    assert_eq!(actual.len(), expected.len());
    for (actual, &expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < tolerance,
            "实际值为 {actual}，期望值为 {expected}"
        );
    }
}

#[test]
/// 验证交叉熵拒绝与批大小不符或超出类别范围的标签。
fn cross_entropy_should_reject_invalid_labels() {
    let logits = Tensor::new(vec![vec![0.0_f64, 1.0], vec![1.0, 0.0]]).unwrap();

    assert!(matches!(
        cross_entropy_loss(&logits, &[0]),
        Err(LossError::TensorOperationError { .. })
    ));
    assert!(matches!(
        cross_entropy_loss(&logits, &[0, 2]),
        Err(LossError::TensorOperationError { .. })
    ));
}
