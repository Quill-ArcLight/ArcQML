use arcqml_core::prelude::*;
use arcqml_linalg::autograd::{
    matmul_backward_shapes, reduction_backward_shape, reshape_backward_shape,
    transpose_backward_shape,
};
use arcqml_linalg::kernels::*;
use arcqml_linalg::ndarray_bridge::*;
use arcqml_linalg::prelude::*;
use arcqml_linalg::shape::*;

use ndarray::{ArrayD, IxDyn, array};
use num_complex::Complex64;

// ============================================================
// 辅助函数
// ============================================================

fn make_f64_tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::new(TensorData::FlatF64 { data, shape }).unwrap()
}

fn make_f32_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
    Tensor::new(TensorData::FlatF32 { data, shape }).unwrap()
}

fn make_i64_tensor(data: Vec<i64>, shape: Vec<usize>) -> Tensor {
    Tensor::new(TensorData::FlatI64 { data, shape }).unwrap()
}

fn make_c64_tensor(data: Vec<Complex64>, shape: Vec<usize>) -> Tensor {
    Tensor::new(TensorData::FlatC64 { data, shape }).unwrap()
}

fn assert_f64_slice_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

fn assert_c64_slice_close(actual: &[Complex64], expected: &[Complex64]) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert!((*actual - *expected).norm() < 1e-12);
    }
}

// ============================================================
// 形状
// ============================================================

#[test]
fn shape_infer_and_validate_should_work() {
    assert_eq!(infer_matmul_shape(&[2, 3], &[3, 4]).unwrap(), vec![2, 4]);
    assert_eq!(infer_transpose_shape(&[2, 3]).unwrap(), vec![3, 2]);
    assert_eq!(infer_broadcast_shape(&[2, 3], &[3]).unwrap(), vec![2, 3]);
    assert_eq!(
        infer_reduction_shape(&[2, 3], Some(1), false).unwrap(),
        vec![2]
    );
    assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);

    assert!(validate_matmul_shapes(&[2, 3], &[4, 2]).is_err());
    assert!(validate_axis("test", &[2, 3], 2).is_err());
    assert!(validate_reshape_shape(&[2, 3], &[4]).is_err());
}

// ============================================================
// 内核
// ============================================================

#[test]
fn elementwise_kernels_should_work() {
    let lhs = vec![1.0_f64, 2.0, 3.0];
    let rhs = vec![4.0_f64, 5.0, 6.0];
    let mut out = vec![0.0_f64; 3];

    add_f64(&lhs, &rhs, &mut out).unwrap();
    assert_f64_slice_close(&out, &[5.0, 7.0, 9.0]);

    mul_f64(&lhs, &rhs, &mut out).unwrap();
    assert_f64_slice_close(&out, &[4.0, 10.0, 18.0]);

    assert!(add_f64(&lhs, &rhs, &mut [0.0_f64; 2]).is_err());
}

#[test]
fn gemm_and_reduction_kernels_should_work() {
    let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0_f64, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut out = vec![0.0_f64; 4];

    gemm_f64(&a, &b, &mut out, 2, 3, 2).unwrap();
    assert_f64_slice_close(&out, &[58.0, 64.0, 139.0, 154.0]);

    assert_eq!(sum_i64(&[1, 2, 3, 4]).unwrap(), 10);
    assert_eq!(mean_i64(&[2, 4, 6, 8]).unwrap(), 5);
    assert_eq!(max_f64(&[1.0, 9.0, 3.0]).unwrap(), 9.0);
    assert_eq!(min_f64(&[1.0, 9.0, 3.0]).unwrap(), 1.0);
    assert!(sum_f64(&[]).is_err());
}

#[test]
fn gemv_and_integer_overflow_kernels_should_work() {
    let matrix = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vector = vec![10.0_f64, 20.0, 30.0];
    let mut out = vec![0.0_f64; 2];

    gemv_f64(&matrix, &vector, &mut out, 2, 3).unwrap();
    assert_f64_slice_close(&out, &[140.0, 320.0]);

    assert!(sum_i64(&[i64::MAX, 1]).is_err());
    assert!(add_i64(&[i64::MAX], &[1], &mut [0]).is_err());
}

// ============================================================
// 张量操作
// ============================================================

#[test]
fn tensor_add_sub_mul_should_support_broadcast() {
    let lhs = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let rhs = make_f64_tensor(vec![10.0, 20.0, 30.0], vec![3]);

    let added = add(&lhs, &rhs).unwrap();
    assert_eq!(added.shape(), &[2, 3]);
    assert_f64_slice_close(
        added.storage().as_f64_slice().unwrap(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
    );

    let subbed = sub(&lhs, &rhs).unwrap();
    assert_f64_slice_close(
        subbed.storage().as_f64_slice().unwrap(),
        &[-9.0, -18.0, -27.0, -6.0, -15.0, -24.0],
    );

    let multiplied = mul(&lhs, &rhs).unwrap();
    assert_f64_slice_close(
        multiplied.storage().as_f64_slice().unwrap(),
        &[10.0, 40.0, 90.0, 40.0, 100.0, 180.0],
    );
}

#[test]
fn tensor_unary_ops_should_work() {
    let input = make_f64_tensor(vec![1.0, 4.0, 9.0], vec![3]);

    let squared = square(&input).unwrap();
    assert_f64_slice_close(
        squared.storage().as_f64_slice().unwrap(),
        &[1.0, 16.0, 81.0],
    );

    let sqrted = sqrt(&input).unwrap();
    assert_f64_slice_close(sqrted.storage().as_f64_slice().unwrap(), &[1.0, 2.0, 3.0]);

    let complex = make_c64_tensor(
        vec![Complex64::new(3.0, 4.0), Complex64::new(5.0, 12.0)],
        vec![2],
    );
    let absed = abs(&complex).unwrap();
    assert_f64_slice_close(absed.storage().as_f64_slice().unwrap(), &[5.0, 13.0]);
}

/// 验证新增逐元素实数算子的前向计算。
#[test]
fn extra_elementwise_ops_should_work() {
    let input = make_f64_tensor(vec![-1.0, 0.0, 2.0], vec![3]);
    assert_f64_slice_close(
        neg(&input).unwrap().storage().as_f64_slice().unwrap(),
        &[1.0, 0.0, -2.0],
    );
    assert_f64_slice_close(
        exp(&input).unwrap().storage().as_f64_slice().unwrap(),
        &[(-1.0_f64).exp(), 1.0, 2.0_f64.exp()],
    );
    assert_f64_slice_close(
        log(&make_f64_tensor(vec![1.0, std::f64::consts::E], vec![2]))
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[0.0, 1.0],
    );
    assert_f64_slice_close(
        sigmoid(&input).unwrap().storage().as_f64_slice().unwrap(),
        &[0.2689414213699951, 0.5, 0.8807970779778823],
    );
    assert_f64_slice_close(
        tanh(&input).unwrap().storage().as_f64_slice().unwrap(),
        &[(-1.0_f64).tanh(), 0.0, 2.0_f64.tanh()],
    );
    assert_f64_slice_close(
        clamp(&input, -0.5, 1.0)
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[-0.5, 0.0, 1.0],
    );
    let divisor = make_f64_tensor(vec![2.0], vec![]);
    assert_f64_slice_close(
        div(&input, &divisor)
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[-0.5, 0.0, 1.0],
    );
}

/// 验证新增逐元素实数算子的反向规则。
#[test]
fn extra_elementwise_backward_should_work() {
    let input = make_f64_tensor(vec![2.0, 4.0], vec![2]);
    input.set_requires_grad(true);
    sum(&log(&input).unwrap()).unwrap().backward().unwrap();
    assert_f64_slice_close(
        input.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[0.5, 0.25],
    );

    let lhs = make_f64_tensor(vec![2.0, 4.0], vec![2]);
    let rhs = make_f64_tensor(vec![2.0], vec![]);
    lhs.set_requires_grad(true);
    rhs.set_requires_grad(true);
    sum(&div(&lhs, &rhs).unwrap()).unwrap().backward().unwrap();
    assert_f64_slice_close(
        lhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[0.5, 0.5],
    );
    assert_f64_slice_close(
        rhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[-1.5],
    );

    let clamped = make_f64_tensor(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
    clamped.set_requires_grad(true);
    sum(&clamp(&clamped, -1.0, 1.0).unwrap())
        .unwrap()
        .backward()
        .unwrap();
    assert_f64_slice_close(
        clamped.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[0.0, 1.0, 1.0, 1.0, 0.0],
    );
}

/// 验证按维度归约及归一化算子的前向和反向传播。
#[test]
fn dimension_reductions_and_normalized_ops_should_work() {
    let input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(sum_dim(&input, 1, false).unwrap().shape(), &[2]);
    assert_f64_slice_close(
        sum_dim(&input, 1, false)
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[6.0, 15.0],
    );
    assert_eq!(mean_dim(&input, 0, true).unwrap().shape(), &[1, 3]);
    assert_f64_slice_close(
        mean_dim(&input, 0, true)
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[2.5, 3.5, 4.5],
    );

    let softmax_value = softmax(&input, 1).unwrap();
    assert_f64_slice_close(
        softmax_value.storage().as_f64_slice().unwrap(),
        &[
            0.09003057317038046,
            0.24472847105479764,
            0.6652409557748218,
            0.09003057317038046,
            0.24472847105479764,
            0.6652409557748218,
        ],
    );
    let log_softmax_value = log_softmax(&input, 1).unwrap();
    assert_f64_slice_close(
        log_softmax_value.storage().as_f64_slice().unwrap(),
        &[
            -2.4076059644443806,
            -1.4076059644443806,
            -0.4076059644443806,
            -2.4076059644443806,
            -1.4076059644443806,
            -0.4076059644443806,
        ],
    );
    assert_f64_slice_close(
        logsumexp(&input, 1, false)
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[3.4076059644443806, 6.407605964444381],
    );

    let reduced = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    reduced.set_requires_grad(true);
    sum(&mean_dim(&reduced, 1, false).unwrap())
        .unwrap()
        .backward()
        .unwrap();
    assert_f64_slice_close(
        reduced.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[0.5, 0.5, 0.5, 0.5],
    );

    let normalized = make_f64_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    normalized.set_requires_grad(true);
    softmax(&normalized, 0)
        .unwrap()
        .backward_with_grad(make_f64_tensor(vec![1.0, 2.0, 3.0], vec![3]))
        .unwrap();
    assert_f64_slice_close(
        normalized.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[
            -0.1418170936096631,
            -0.14077035746963026,
            0.2825874510792934,
        ],
    );
}

/// 验证非末轴 log-softmax 和保留维度的 logsumexp 能正确传播非均匀上游梯度。
#[test]
fn normalized_backward_should_support_non_last_axis_and_keepdim() {
    let log_softmax_input =
        make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
    log_softmax_input.set_requires_grad(true);
    log_softmax(&log_softmax_input, 1)
        .unwrap()
        .backward_with_grad(make_f64_tensor(
            vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 2.0, 5.0],
            vec![2, 2, 2],
        ))
        .unwrap();

    let first_probability = 1.0 / (1.0 + 2.0_f64.exp());
    let second_probability = 1.0 - first_probability;
    assert_f64_slice_close(
        log_softmax_input
            .grad()
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[
            1.0 - 4.0 * first_probability,
            2.0 - 6.0 * first_probability,
            3.0 - 4.0 * second_probability,
            4.0 - 6.0 * second_probability,
            -1.0 - first_probability,
            -5.0 * first_probability,
            2.0 - second_probability,
            5.0 - 5.0 * second_probability,
        ],
    );

    let logsumexp_input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    logsumexp_input.set_requires_grad(true);
    logsumexp(&logsumexp_input, 1, true)
        .unwrap()
        .backward_with_grad(make_f64_tensor(vec![2.0, -3.0], vec![2, 1]))
        .unwrap();

    let probabilities = [0.09003057317038046, 0.24472847105479764, 0.6652409557748218];
    assert_f64_slice_close(
        logsumexp_input
            .grad()
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[
            2.0 * probabilities[0],
            2.0 * probabilities[1],
            2.0 * probabilities[2],
            -3.0 * probabilities[0],
            -3.0 * probabilities[1],
            -3.0 * probabilities[2],
        ],
    );
}

/// 验证空归约维度会在前向阶段被拒绝，避免无定义的均值和归一化结果。
#[test]
fn dimension_reductions_should_reject_empty_reduction_axes() {
    let empty_axis = make_f64_tensor(Vec::new(), vec![2, 0]);
    for result in [
        sum_dim(&empty_axis, 1, false),
        mean_dim(&empty_axis, 1, false),
        softmax(&empty_axis, 1),
        log_softmax(&empty_axis, 1),
        logsumexp(&empty_axis, 1, false),
    ] {
        assert!(matches!(result, Err(LinalgError::EmptyInputError { .. })));
    }
}

/// 验证 F32 按维度归约不会因维度长度超过 255 而截断除数。
#[test]
fn mean_dim_should_use_the_full_axis_length_for_f32() {
    let input = make_f32_tensor(vec![3.0; 300], vec![300]);
    assert_f64_slice_close(
        &mean_dim(&input, 0, false)
            .unwrap()
            .storage()
            .as_f32_slice()
            .unwrap()
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>(),
        &[3.0],
    );
}

/// 验证归一化和 logsumexp 对极端输入保持有限且正确的数值结果。
#[test]
fn normalized_ops_should_be_numerically_stable_for_extreme_inputs() {
    let input = make_f64_tensor(vec![1000.0, 1001.0, 999.0], vec![3]);
    let probabilities = softmax(&input, 0).unwrap();
    let probability_storage = probabilities.storage();
    let probability_values = probability_storage.as_f64_slice().unwrap();
    assert!(probability_values.iter().all(|value| value.is_finite()));
    assert!((probability_values.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let logs = log_softmax(&input, 0).unwrap();
    assert!(
        logs.storage()
            .as_f64_slice()
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
    let lse = logsumexp(&input, 0, false).unwrap();
    assert!((lse.storage().as_f64_slice().unwrap()[0] - 1001.4076059644444).abs() < 1e-10);
}

#[test]
fn tensor_matmul_dot_reduction_and_norm_should_work() {
    let lhs = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let rhs = make_f64_tensor(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

    let product = matmul(&lhs, &rhs).unwrap();
    assert_eq!(product.shape(), &[2, 2]);
    assert_f64_slice_close(
        product.storage().as_f64_slice().unwrap(),
        &[58.0, 64.0, 139.0, 154.0],
    );

    let a = make_i64_tensor(vec![1, 2, 3], vec![3]);
    let b = make_i64_tensor(vec![4, 5, 6], vec![3]);
    let got_dot = dot(&a, &b).unwrap();
    assert_eq!(got_dot.shape(), &[]);
    assert_eq!(got_dot.storage().as_i64_slice().unwrap(), &[32]);

    assert_eq!(
        sum(&lhs).unwrap().storage().as_f64_slice().unwrap(),
        &[21.0]
    );
    assert_eq!(
        mean(&lhs).unwrap().storage().as_f64_slice().unwrap(),
        &[3.5]
    );
    assert_eq!(max(&lhs).unwrap().storage().as_f64_slice().unwrap(), &[6.0]);
    assert_eq!(min(&lhs).unwrap().storage().as_f64_slice().unwrap(), &[1.0]);

    let norm = l2_norm(&make_f32_tensor(vec![3.0, 4.0], vec![2])).unwrap();
    assert_f64_slice_close(norm.storage().as_f64_slice().unwrap(), &[5.0]);
}

#[test]
fn c64_forward_ops_should_work() {
    let lhs = make_c64_tensor(
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
        vec![1, 2],
    );
    let rhs = make_c64_tensor(
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(1.0, -1.0),
            Complex64::new(3.0, 0.0),
        ],
        vec![2, 2],
    );

    let product = matmul(&lhs, &rhs).unwrap();
    assert_c64_slice_close(
        product.storage().as_c64_slice().unwrap(),
        &[Complex64::new(3.0, -1.0), Complex64::new(5.0, -2.0)],
    );

    let scalar = make_c64_tensor(vec![Complex64::new(2.0, -1.0)], vec![1]);
    let shifted = add(&product, &scalar).unwrap();
    assert_c64_slice_close(
        shifted.storage().as_c64_slice().unwrap(),
        &[Complex64::new(5.0, -2.0), Complex64::new(7.0, -3.0)],
    );
}

#[test]
fn tensor_reshape_and_transpose_views_should_work() {
    let input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);

    let reshaped = reshape(&input, vec![3, 2]).unwrap();
    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.strides(), &[2, 1]);
    assert!(reshaped.is_contiguous());
    assert_f64_slice_close(
        reshaped.storage().as_f64_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );

    let transposed = transpose(&input).unwrap();
    assert_eq!(transposed.shape(), &[3, 2]);
    assert_eq!(transposed.strides(), &[1, 3]);
    assert!(!transposed.is_contiguous());

    assert!(matmul(&transposed, &reshaped).is_err());
}

#[test]
fn tensor_ext_methods_should_work() {
    let lhs = make_f64_tensor(vec![1.0, 2.0], vec![2]);
    let rhs = make_f64_tensor(vec![3.0, 4.0], vec![2]);

    let added = lhs.add(&rhs).unwrap();

    assert_f64_slice_close(added.storage().as_f64_slice().unwrap(), &[4.0, 6.0]);
}

// ============================================================
// ndarray 桥接
// ============================================================

#[test]
fn ndarray_bridge_should_convert_and_view() {
    let array = array![[1.0_f64, 2.0], [3.0, 4.0]].into_dyn();
    let tensor = from_arrayd_f64(array).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_f64_slice_close(
        tensor.storage().as_f64_slice().unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
    );

    with_array_view_f64(&tensor, |view| {
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view[[1, 1]], 4.0);
    })
    .unwrap();

    let array = to_arrayd_f64(&tensor).unwrap();
    assert_eq!(array[[0, 1]], 2.0);
}

#[test]
fn ndarray_bridge_should_support_other_dtypes_and_reject_dtype_mismatches() {
    let complex_array = ArrayD::from_shape_vec(
        IxDyn(&[2]),
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    )
    .unwrap();
    let complex_tensor = from_arrayd_c64(complex_array).unwrap();
    assert_c64_slice_close(
        to_arrayd_c64(&complex_tensor).unwrap().as_slice().unwrap(),
        &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
    );

    let bool_array = ArrayD::from_shape_vec(IxDyn(&[2]), vec![true, false]).unwrap();
    let bool_tensor = from_arrayd_bool(bool_array).unwrap();
    assert_eq!(
        to_arrayd_bool(&bool_tensor).unwrap().as_slice().unwrap(),
        &[true, false]
    );

    let f64_tensor = make_f64_tensor(vec![1.0], vec![1]);
    assert!(to_arrayd_f32(&f64_tensor).is_err());
}

#[test]
fn linalg_errors_should_use_english_messages() {
    let lhs = make_f64_tensor(vec![1.0, 2.0], vec![2]);
    let rhs = make_f64_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let broadcast_error = add(&lhs, &rhs).unwrap_err().to_string();
    assert!(broadcast_error.starts_with("shape mismatch in broadcast"));

    let dtype_error = sum(&Tensor::new(vec![true, false]).unwrap())
        .unwrap_err()
        .to_string();
    let dimension_error = transpose(&make_f64_tensor(vec![1.0, 2.0], vec![2]))
        .unwrap_err()
        .to_string();

    for message in [broadcast_error, dtype_error, dimension_error] {
        assert!(
            !message
                .chars()
                .any(|character| ('\u{4e00}'..='\u{9fff}').contains(&character)),
            "error message contains Chinese text: {message}"
        );
    }
}

// ============================================================
// 自动微分形状辅助函数
// ============================================================

#[test]
fn autograd_shape_helpers_should_work() {
    assert_eq!(
        matmul_backward_shapes(&[2, 3], &[3, 4]).unwrap(),
        (vec![2, 3], vec![3, 4])
    );
    assert_eq!(
        reduction_backward_shape(&[2, 3], Some(1), false).unwrap(),
        vec![2, 3]
    );
    assert_eq!(
        reshape_backward_shape(&[2, 3], &[3, 2]).unwrap(),
        vec![2, 3]
    );
    assert_eq!(transpose_backward_shape(&[2, 3]).unwrap(), vec![2, 3]);
}

// ============================================================
// 自动微分集成
// ============================================================

/// 测试带广播的逐元素乘法会对被广播维度正确累加梯度。
#[test]
fn binary_broadcast_backward_should_accumulate_gradients() {
    let lhs = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let rhs = make_f64_tensor(vec![10.0, 20.0, 30.0], vec![3]);
    lhs.set_requires_grad(true);
    rhs.set_requires_grad(true);

    let loss = sum(&mul(&lhs, &rhs).unwrap()).unwrap();
    loss.backward().unwrap();

    assert_f64_slice_close(
        lhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
    );
    assert_f64_slice_close(
        rhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[5.0, 7.0, 9.0],
    );

    // 加法与减法的广播梯度分别对右侧累加正号和负号。
    let add_lhs = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let add_rhs = make_f64_tensor(vec![10.0, 20.0], vec![2]);
    add_lhs.set_requires_grad(true);
    add_rhs.set_requires_grad(true);
    sum(&add(&add_lhs, &add_rhs).unwrap())
        .unwrap()
        .backward()
        .unwrap();
    assert_f64_slice_close(
        add_lhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[1.0, 1.0, 1.0, 1.0],
    );
    assert_f64_slice_close(
        add_rhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[2.0, 2.0],
    );

    let sub_lhs = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let sub_rhs = make_f64_tensor(vec![10.0, 20.0], vec![2]);
    sub_lhs.set_requires_grad(true);
    sub_rhs.set_requires_grad(true);
    sum(&sub(&sub_lhs, &sub_rhs).unwrap())
        .unwrap()
        .backward()
        .unwrap();
    assert_f64_slice_close(
        sub_lhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[1.0, 1.0, 1.0, 1.0],
    );
    assert_f64_slice_close(
        sub_rhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[-2.0, -2.0],
    );
}

/// 测试 square、sqrt、abs 与 mean 的复合图可以端到端反向传播。
#[test]
fn unary_and_mean_backward_should_propagate_gradients() {
    let input = make_f64_tensor(vec![-4.0, 9.0], vec![2]);
    input.set_requires_grad(true);

    let loss = mean(&abs(&sqrt(&square(&input).unwrap()).unwrap()).unwrap()).unwrap();
    loss.backward().unwrap();

    assert_f64_slice_close(
        input.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[-0.5, 0.5],
    );
}

/// 测试矩阵乘法反向规则分别计算 G @ B^T 和 A^T @ G。
#[test]
fn matmul_backward_should_propagate_gradients_to_both_operands() {
    let lhs = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let rhs = make_f64_tensor(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    lhs.set_requires_grad(true);
    rhs.set_requires_grad(true);

    let loss = sum(&matmul(&lhs, &rhs).unwrap()).unwrap();
    loss.backward().unwrap();

    assert_f64_slice_close(
        lhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[11.0, 15.0, 11.0, 15.0],
    );
    assert_f64_slice_close(
        rhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[4.0, 4.0, 6.0, 6.0],
    );
}

/// 测试点积、L2 范数与极值归约的梯度规则。
#[test]
fn scalar_linalg_backward_should_propagate_gradients() {
    let lhs = make_f64_tensor(vec![1.0, 2.0, 3.0], vec![3]);
    let rhs = make_f64_tensor(vec![4.0, 5.0, 6.0], vec![3]);
    lhs.set_requires_grad(true);
    rhs.set_requires_grad(true);
    let dot_loss = dot(&lhs, &rhs).unwrap();
    dot_loss.backward().unwrap();
    assert_f64_slice_close(
        lhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[4.0, 5.0, 6.0],
    );
    assert_f64_slice_close(
        rhs.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[1.0, 2.0, 3.0],
    );

    let norm_input = make_f64_tensor(vec![3.0, 4.0], vec![2]);
    norm_input.set_requires_grad(true);
    l2_norm(&norm_input).unwrap().backward().unwrap();
    assert_f64_slice_close(
        norm_input.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[0.6, 0.8],
    );

    let extrema_input = make_f64_tensor(vec![1.0, 3.0, 3.0, -2.0], vec![4]);
    extrema_input.set_requires_grad(true);
    max(&extrema_input).unwrap().backward().unwrap();
    assert_f64_slice_close(
        extrema_input
            .grad()
            .unwrap()
            .storage()
            .as_f64_slice()
            .unwrap(),
        &[0.0, 0.5, 0.5, 0.0],
    );

    let min_input = make_f64_tensor(vec![1.0, -2.0, -2.0, 3.0], vec![4]);
    min_input.set_requires_grad(true);
    min(&min_input).unwrap().backward().unwrap();
    assert_f64_slice_close(
        min_input.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[0.0, 0.5, 0.5, 0.0],
    );
}

/// 验证 C64 反传遵循 PyTorch 的共轭 Wirtinger VJP 语义。
#[test]
fn c64_backward_should_use_conjugate_wirtinger_vjp() {
    // y=a*b，显式给定复数上游梯度 g 后，应有 ga=g*conj(b)，gb=g*conj(a)。
    let lhs = make_c64_tensor(
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 1.0)],
        vec![2],
    );
    let rhs = make_c64_tensor(
        vec![Complex64::new(2.0, -1.0), Complex64::new(4.0, 5.0)],
        vec![2],
    );
    lhs.set_requires_grad(true);
    rhs.set_requires_grad(true);
    let mul_grad = vec![Complex64::new(3.0, 4.0), Complex64::new(-2.0, 1.0)];
    mul(&lhs, &rhs)
        .unwrap()
        .backward_with_grad(make_c64_tensor(mul_grad.clone(), vec![2]))
        .unwrap();
    assert_c64_slice_close(
        lhs.grad().unwrap().storage().as_c64_slice().unwrap(),
        &[
            mul_grad[0] * Complex64::new(2.0, -1.0).conj(),
            mul_grad[1] * Complex64::new(4.0, 5.0).conj(),
        ],
    );
    assert_c64_slice_close(
        rhs.grad().unwrap().storage().as_c64_slice().unwrap(),
        &[
            mul_grad[0] * Complex64::new(1.0, 2.0).conj(),
            mul_grad[1] * Complex64::new(-3.0, 1.0).conj(),
        ],
    );

    // y=x^2 与 y=sqrt(x) 的 VJP 分别使用 2*conj(x) 与 1/(2*conj(sqrt(x)))。
    let square_input = make_c64_tensor(vec![Complex64::new(1.0, 2.0)], vec![1]);
    square_input.set_requires_grad(true);
    square(&square_input)
        .unwrap()
        .backward_with_grad(make_c64_tensor(vec![Complex64::new(3.0, -1.0)], vec![1]))
        .unwrap();
    assert_c64_slice_close(
        square_input
            .grad()
            .unwrap()
            .storage()
            .as_c64_slice()
            .unwrap(),
        &[Complex64::new(3.0, -1.0) * 2.0 * Complex64::new(1.0, 2.0).conj()],
    );

    let sqrt_input = make_c64_tensor(vec![Complex64::new(3.0, 4.0)], vec![1]);
    sqrt_input.set_requires_grad(true);
    let sqrt_grad = Complex64::new(2.0, -3.0);
    sqrt(&sqrt_input)
        .unwrap()
        .backward_with_grad(make_c64_tensor(vec![sqrt_grad], vec![1]))
        .unwrap();
    assert_c64_slice_close(
        sqrt_input.grad().unwrap().storage().as_c64_slice().unwrap(),
        &[sqrt_grad / (2.0 * Complex64::new(3.0, 4.0).sqrt().conj())],
    );

    // dot 是非共轭的 sum(a_i*b_i)，因此它的反传同样对另一侧取共轭。
    let dot_lhs = make_c64_tensor(vec![Complex64::new(1.0, 2.0)], vec![1]);
    let dot_rhs = make_c64_tensor(vec![Complex64::new(3.0, -4.0)], vec![1]);
    dot_lhs.set_requires_grad(true);
    dot_rhs.set_requires_grad(true);
    let dot_grad = Complex64::new(2.0, 5.0);
    dot(&dot_lhs, &dot_rhs)
        .unwrap()
        .backward_with_grad(make_c64_tensor(vec![dot_grad], vec![]))
        .unwrap();
    assert_c64_slice_close(
        dot_lhs.grad().unwrap().storage().as_c64_slice().unwrap(),
        &[dot_grad * Complex64::new(3.0, -4.0).conj()],
    );
    assert_c64_slice_close(
        dot_rhs.grad().unwrap().storage().as_c64_slice().unwrap(),
        &[dot_grad * Complex64::new(1.0, 2.0).conj()],
    );

    // Y=A@B 的 C64 VJP 为 GA=G@B^H，GB=A^H@G。
    let matrix_lhs = make_c64_tensor(
        vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)],
        vec![1, 2],
    );
    let matrix_rhs = make_c64_tensor(
        vec![
            Complex64::new(2.0, 1.0),
            Complex64::new(-1.0, 4.0),
            Complex64::new(5.0, -2.0),
            Complex64::new(3.0, 0.5),
        ],
        vec![2, 2],
    );
    matrix_lhs.set_requires_grad(true);
    matrix_rhs.set_requires_grad(true);
    let matrix_grad = vec![Complex64::new(2.0, -1.0), Complex64::new(-3.0, 4.0)];
    matmul(&matrix_lhs, &matrix_rhs)
        .unwrap()
        .backward_with_grad(make_c64_tensor(matrix_grad.clone(), vec![1, 2]))
        .unwrap();
    assert_c64_slice_close(
        matrix_lhs.grad().unwrap().storage().as_c64_slice().unwrap(),
        &[
            matrix_grad[0] * Complex64::new(2.0, 1.0).conj()
                + matrix_grad[1] * Complex64::new(-1.0, 4.0).conj(),
            matrix_grad[0] * Complex64::new(5.0, -2.0).conj()
                + matrix_grad[1] * Complex64::new(3.0, 0.5).conj(),
        ],
    );
    assert_c64_slice_close(
        matrix_rhs.grad().unwrap().storage().as_c64_slice().unwrap(),
        &[
            Complex64::new(1.0, 2.0).conj() * matrix_grad[0],
            Complex64::new(1.0, 2.0).conj() * matrix_grad[1],
            Complex64::new(3.0, -1.0).conj() * matrix_grad[0],
            Complex64::new(3.0, -1.0).conj() * matrix_grad[1],
        ],
    );

    // abs 与 L2 范数输出实值，梯度应与实部、虚部联合最速下降方向一致。
    let abs_input = make_c64_tensor(vec![Complex64::new(3.0, 4.0)], vec![1]);
    abs_input.set_requires_grad(true);
    sum(&abs(&abs_input).unwrap()).unwrap().backward().unwrap();
    assert_c64_slice_close(
        abs_input.grad().unwrap().storage().as_c64_slice().unwrap(),
        &[Complex64::new(0.6, 0.8)],
    );

    let real_loss_input = make_c64_tensor(vec![Complex64::new(3.0, 4.0)], vec![1]);
    real_loss_input.set_requires_grad(true);
    l2_norm(&real_loss_input).unwrap().backward().unwrap();
    assert_c64_slice_close(
        real_loss_input
            .grad()
            .unwrap()
            .storage()
            .as_c64_slice()
            .unwrap(),
        &[Complex64::new(0.6, 0.8)],
    );
}

/// 测试 arcqml-linalg 的 reshape 与 transpose 会复用 arcqml-core 的可微 view 实现。
#[test]
fn reshape_and_transpose_backward_should_propagate_gradient() {
    let input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    input.set_requires_grad(true);

    let reshaped = reshape(&input, vec![4]).unwrap();
    let transposed = transpose(&reshaped.reshape(vec![2, 2]).unwrap()).unwrap();
    transposed
        .backward_with_grad(make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]))
        .unwrap();

    assert_f64_slice_close(
        input.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[1.0, 3.0, 2.0, 4.0],
    );
}

/// 验证复共轭算子遵循 PyTorch 风格的复数反向规则。
#[test]
fn conj_should_preserve_complex_autograd_semantics() {
    let input = make_c64_tensor(
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)],
        vec![2],
    );
    input.set_requires_grad(true);

    let output = conj(&input).unwrap();
    assert_c64_slice_close(
        output.storage().as_c64_slice().unwrap(),
        &[Complex64::new(1.0, -2.0), Complex64::new(-3.0, -4.0)],
    );
    output
        .backward_with_grad(make_c64_tensor(
            vec![Complex64::new(3.0, 4.0), Complex64::new(-1.0, 2.0)],
            vec![2],
        ))
        .unwrap();
    assert_c64_slice_close(
        input.grad().unwrap().storage().as_c64_slice().unwrap(),
        &[Complex64::new(3.0, -4.0), Complex64::new(-1.0, -2.0)],
    );
}

/// 验证固定分段求和的前向汇总、反向广播和输入校验。
#[test]
fn segment_sum_should_aggregate_and_backpropagate() {
    let input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    input.set_requires_grad(true);

    let output = segment_sum(&input, &[1, 0, 1, 2], 3).unwrap();
    assert_f64_slice_close(output.storage().as_f64_slice().unwrap(), &[2.0, 4.0, 4.0]);
    output
        .backward_with_grad(make_f64_tensor(vec![10.0, 20.0, 30.0], vec![3]))
        .unwrap();
    assert_f64_slice_close(
        input.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[20.0, 10.0, 20.0, 30.0],
    );

    assert!(segment_sum(&input, &[0, 1], 2).is_err());
    assert!(segment_sum(&input, &[0, 1, 2, 3], 3).is_err());
}
