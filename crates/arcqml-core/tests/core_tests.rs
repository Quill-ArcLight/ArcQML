use arcqml_core::prelude::*;

use num_complex::Complex64;
use std::sync::Arc;

// ============================================================
// 辅助函数
// ============================================================

fn make_f64_tensor(data: Vec<f64>, shape: Vec<usize>) -> Tensor {
    Tensor::new(TensorData::FlatF64 { data, shape }).unwrap()
}

fn make_c64_tensor(data: Vec<Complex64>, shape: Vec<usize>) -> Tensor {
    Tensor::new(TensorData::FlatC64 { data, shape }).unwrap()
}

#[derive(Debug)]
struct SquareBackward;

impl BackwardFn for SquareBackward {
    fn backward(
        &self,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> arcqml_core::Result<Vec<Option<Tensor>>> {
        let input = &parents[0];
        let input_values = input.storage();
        let grad_values = grad_output.storage();

        let (Storage::F64(input), Storage::F64(grad_output)) = (&*input_values, &*grad_values)
        else {
            panic!("square test backward expects f64 tensors");
        };

        let gradient = input
            .iter()
            .zip(grad_output)
            .map(|(input, grad_output)| 2.0 * input * grad_output)
            .collect();

        let gradient =
            Tensor::from_storage_meta(Storage::F64(gradient), input_meta(input, parents))?;

        Ok(vec![Some(gradient)])
    }
}

#[derive(Debug)]
struct AddBackward;

impl BackwardFn for AddBackward {
    fn backward(
        &self,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> arcqml_core::Result<Vec<Option<Tensor>>> {
        assert_eq!(parents.len(), 2);

        Ok(vec![Some(grad_output.clone()), Some(grad_output.clone())])
    }
}

#[derive(Debug)]
struct IdentityBackward;

impl BackwardFn for IdentityBackward {
    fn backward(
        &self,
        parents: &[Tensor],
        grad_output: &Tensor,
    ) -> arcqml_core::Result<Vec<Option<Tensor>>> {
        assert_eq!(parents.len(), 1);

        Ok(vec![Some(grad_output.clone())])
    }
}

fn input_meta(_input: &[f64], parents: &[Tensor]) -> TensorMeta {
    parents[0].meta().clone()
}

fn square_for_test(input: &Tensor) -> Tensor {
    let values = input.storage();
    let Storage::F64(values) = &*values else {
        panic!("square test forward expects an f64 tensor");
    };

    let storage = Storage::F64(values.iter().map(|value| value * value).collect());

    Tensor::from_operation_named(
        "square_for_test",
        storage,
        input.meta().clone(),
        vec![input.clone()],
        Arc::new(SquareBackward),
    )
    .unwrap()
}

fn add_for_test(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    let lhs_values = lhs.storage();
    let rhs_values = rhs.storage();

    let (Storage::F64(lhs_values), Storage::F64(rhs_values)) = (&*lhs_values, &*rhs_values) else {
        panic!("add test forward expects f64 tensors");
    };

    let storage = Storage::F64(
        lhs_values
            .iter()
            .zip(rhs_values)
            .map(|(lhs, rhs)| lhs + rhs)
            .collect(),
    );

    Tensor::from_operation_named(
        "add_for_test",
        storage,
        lhs.meta().clone(),
        vec![lhs.clone(), rhs.clone()],
        Arc::new(AddBackward),
    )
    .unwrap()
}

fn identity_for_test(input: &Tensor) -> Tensor {
    Tensor::from_operation_named(
        "identity_for_test",
        input.storage().clone(),
        input.meta().clone(),
        vec![input.clone()],
        Arc::new(IdentityBackward),
    )
    .unwrap()
}

/// 验证不可导操作会保留断图上下文并在反向传播时报告。
#[test]
fn non_differentiable_operation_should_report_its_name_and_reason() {
    let input = Tensor::new(0.5_f64).unwrap();
    input.set_requires_grad(true);

    let output = Tensor::from_non_differentiable_operation(
        "external_solver",
        "the external solver has no VJP",
        Storage::F64(vec![0.25]),
        input.meta().clone(),
        vec![input],
    )
    .unwrap();

    let error = output.backward().unwrap_err().to_string();
    assert!(error.contains("external_solver"));
    assert!(error.contains("the external solver has no VJP"));
    assert!(error.contains("detach()"));
}

/// 计算标量三次方的自定义原子操作。
#[derive(Debug, Clone)]
struct CubeCustomOp;

impl CustomOp for CubeCustomOp {
    type Context = f64;

    /// 返回自定义操作的稳定名称。
    fn name(&self) -> &'static str {
        "cube_custom_op"
    }

    /// 计算输入的三次方并保存输入值供反向传播使用。
    fn forward(&self, inputs: &[Tensor]) -> arcqml_core::Result<(Tensor, Self::Context)> {
        if inputs.len() != 1 {
            return Err(ArcQmlError::AutogradError(
                "cube_custom_op expects one input".to_string(),
            ));
        }
        let Storage::F64(values) = &*inputs[0].storage() else {
            return Err(ArcQmlError::AutogradError(
                "cube_custom_op expects an f64 input".to_string(),
            ));
        };
        let value = values[0];
        Ok((Tensor::new(value.powi(3))?, value))
    }

    /// 按局部导数 `3 * x^2` 计算输入梯度。
    fn backward(
        &self,
        input: &Self::Context,
        grad_output: &Tensor,
    ) -> arcqml_core::Result<Vec<Option<Tensor>>> {
        let Storage::F64(values) = &*grad_output.storage() else {
            return Err(ArcQmlError::AutogradError(
                "cube_custom_op expects an f64 upstream gradient".to_string(),
            ));
        };
        Ok(vec![Some(Tensor::new(3.0 * input.powi(2) * values[0])?)])
    }
}

/// 验证自定义操作只提供局部 VJP 时仍可使用统一的反向传播入口。
#[test]
fn custom_op_should_connect_to_the_autograd_graph() {
    let input = Tensor::new(2.0_f64).unwrap();
    input.set_requires_grad(true);

    let output = apply_custom_op(CubeCustomOp, std::slice::from_ref(&input)).unwrap();
    output.backward().unwrap();

    let gradient_tensor = input.grad().unwrap();
    let Storage::F64(gradient) = &*gradient_tensor.storage() else {
        panic!("cube_custom_op should produce an f64 gradient");
    };
    assert_eq!(gradient, &[12.0]);
}

// ============================================================
// 数据类型
// ============================================================

/// 测试数据类型名称与 Display 输出是否正确。
#[test]
fn dtype_name_and_display_should_work() {
    assert_eq!(DType::F64.name(), "f64");
    assert_eq!(DType::C64.name(), "complex64");
    assert_eq!(DType::I64.name(), "i64");
    assert_eq!(DType::Bool.name(), "bool");

    assert_eq!(DType::F64.to_string(), "f64");
    assert_eq!(DType::C64.to_string(), "complex64");
}

// ============================================================
// 设备
// ============================================================

/// 测试设备名称、Display 输出与默认设备是否正确。
#[test]
fn device_name_display_and_default_should_work() {
    let cpu = Device::default();

    assert_eq!(cpu, Device::Cpu);
    assert_eq!(cpu.name(), "cpu");
    assert_eq!(cpu.to_string(), "cpu");

    let cuda = Device::Cuda(0);

    assert_eq!(cuda.name(), "cuda:0");
    assert_eq!(cuda.to_string(), "cuda:0");
}

// ============================================================
// 布局
// ============================================================

/// 测试张量布局名称、Display 输出与默认布局是否正确。
#[test]
fn layout_name_display_and_default_should_work() {
    let layout = Layout::default();

    assert_eq!(layout, Layout::Dense);
    assert_eq!(layout.name(), "dense");
    assert_eq!(layout.to_string(), "dense");

    assert_eq!(Layout::Sparse.name(), "sparse");
    assert_eq!(Layout::Sparse.to_string(), "sparse");
}

// ============================================================
// 存储
// ============================================================

/// 测试不同存储类型返回的 dtype 与元素数量是否正确。
#[test]
fn storage_dtype_and_len_should_work() {
    let f64_storage = Storage::F64(vec![1.0, 2.0, 3.0]);

    assert_eq!(f64_storage.dtype(), DType::F64);
    assert_eq!(f64_storage.len(), 3);

    let c64_storage = Storage::C64(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]);

    assert_eq!(c64_storage.dtype(), DType::C64);
    assert_eq!(c64_storage.len(), 2);

    let i64_storage = Storage::I64(vec![1, 2, 3, 4]);

    assert_eq!(i64_storage.dtype(), DType::I64);
    assert_eq!(i64_storage.len(), 4);

    let bool_storage = Storage::Bool(vec![true, false]);

    assert_eq!(bool_storage.dtype(), DType::Bool);
    assert_eq!(bool_storage.len(), 2);
}

/// 测试按 dtype 获取存储切片时的类型匹配行为。
#[test]
fn storage_typed_slice_should_work() {
    let f64_storage = Storage::F64(vec![1.0, 2.0, 3.0]);

    assert!(!f64_storage.is_empty());
    assert_eq!(f64_storage.as_f64_slice(), Some(&[1.0, 2.0, 3.0][..]));
    assert!(f64_storage.as_i64_slice().is_none());

    let c64_storage = Storage::C64(vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)]);

    assert_eq!(c64_storage.as_c64_slice().unwrap().len(), 2);
    assert!(c64_storage.as_bool_slice().is_none());
}

// ============================================================
// 张量元信息
// ============================================================

/// 测试由连续 shape 创建的 TensorMeta 是否包含正确元信息。
#[test]
fn tensor_meta_new_should_work() {
    let meta = TensorMeta::new(vec![2, 3], DType::F64, Device::Cpu, Layout::Dense).unwrap();

    assert_eq!(meta.shape(), &[2, 3]);
    assert_eq!(meta.dtype(), DType::F64);
    assert_eq!(meta.device(), &Device::Cpu);
    assert_eq!(meta.layout(), Layout::Dense);
    assert_eq!(meta.numel(), 6);
    assert_eq!(meta.strides(), &[3, 1]);
    assert_eq!(meta.offset(), 0);
    assert!(meta.is_contiguous());
    assert_eq!(meta.storage_len_required(), 6);
}

/// 测试标量 TensorMeta 的元素数量和存储需求是否为一。
#[test]
fn tensor_meta_scalar_numel_should_be_one() {
    let meta = TensorMeta::new(vec![], DType::F64, Device::Cpu, Layout::Dense).unwrap();

    assert_eq!(meta.shape(), &[]);
    assert_eq!(meta.numel(), 1);
    assert_eq!(meta.strides(), &[]);
    assert_eq!(meta.storage_len_required(), 1);
}

#[test]
fn tensor_meta_new_should_reject_overflowing_dense_layouts() {
    assert!(matches!(
        TensorMeta::new(vec![usize::MAX, 2], DType::F64, Device::Cpu, Layout::Dense,),
        Err(ArcQmlError::ShapeError(_))
    ));
    assert!(matches!(
        TensorMeta::new(vec![2, usize::MAX], DType::F64, Device::Cpu, Layout::Dense,),
        Err(ArcQmlError::ShapeError(_))
    ));
}

#[test]
fn tensor_meta_should_reject_unsupported_device_and_layout() {
    assert!(matches!(
        TensorMeta::new(
            vec![1],
            DType::F64,
            Device::Cuda(0),
            Layout::Dense,
        ),
        Err(ArcQmlError::NotImplementedError(message)) if message.contains("cuda:0")
    ));
    assert!(matches!(
        TensorMeta::from_parts(
            vec![1],
            vec![1],
            0,
            DType::F64,
            Device::Cpu,
            Layout::Sparse,
        ),
        Err(ArcQmlError::NotImplementedError(message)) if message.contains("sparse")
    ));
}

/// 测试按完整字段构造 TensorMeta 是否保留给定元信息。
#[test]
fn tensor_meta_from_parts_should_work() {
    let meta = TensorMeta::from_parts(
        vec![2, 3],
        vec![3, 1],
        0,
        DType::F64,
        Device::Cpu,
        Layout::Dense,
    )
    .unwrap();

    assert_eq!(meta.shape(), &[2, 3]);
    assert_eq!(meta.dtype(), DType::F64);
    assert_eq!(meta.device(), &Device::Cpu);
    assert_eq!(meta.layout(), Layout::Dense);
    assert_eq!(meta.numel(), 6);
    assert_eq!(meta.strides(), &[3, 1]);
    assert_eq!(meta.offset(), 0);
    assert!(meta.is_contiguous());
}

#[test]
fn tensor_meta_from_parts_should_reject_invalid_layouts() {
    assert!(matches!(
        TensorMeta::from_parts(vec![2], vec![], 0, DType::F64, Device::Cpu, Layout::Dense,),
        Err(ArcQmlError::ShapeError(_))
    ));
    assert!(matches!(
        TensorMeta::from_parts(vec![2], vec![-1], 0, DType::F64, Device::Cpu, Layout::Dense,),
        Err(ArcQmlError::ShapeError(_))
    ));
    assert!(matches!(
        TensorMeta::from_parts(
            vec![usize::MAX, 2],
            vec![2, 1],
            0,
            DType::F64,
            Device::Cpu,
            Layout::Dense,
        ),
        Err(ArcQmlError::ShapeError(_))
    ));
    assert!(matches!(
        TensorMeta::from_parts(
            vec![usize::MAX],
            vec![2],
            0,
            DType::F64,
            Device::Cpu,
            Layout::Dense,
        ),
        Err(ArcQmlError::ShapeError(_))
    ));
}

/// 测试非连续 view 的 TensorMeta 所需底层存储长度是否正确。
#[test]
fn tensor_meta_view_storage_len_should_work() {
    let meta = TensorMeta::from_parts(
        vec![2, 2],
        vec![3, 1],
        1,
        DType::F64,
        Device::Cpu,
        Layout::Dense,
    )
    .unwrap();

    assert_eq!(meta.shape(), &[2, 2]);
    assert_eq!(meta.strides(), &[3, 1]);
    assert_eq!(meta.offset(), 1);
    assert!(!meta.is_contiguous());
    assert_eq!(meta.storage_len_required(), 6);
}

// ============================================================
// 张量创建
// ============================================================

/// 测试 Tensor::new 能创建一个有效的连续张量。
#[test]
fn tensor_new_should_create_valid_tensor() {
    let tensor = Tensor::new(vec![vec![1.0, 2.0], vec![3.0, 4.0]]).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.dtype(), DType::F64);
    assert_eq!(tensor.device(), &Device::Cpu);
    assert_eq!(tensor.layout(), Layout::Dense);
    assert_eq!(tensor.ndim(), 2);
    assert_eq!(tensor.numel(), 4);
    assert_eq!(tensor.strides(), &[2, 1]);
    assert_eq!(tensor.offset(), 0);
    assert!(tensor.is_contiguous());
    assert!(!tensor.requires_grad());
    assert!(tensor.grad().is_none());
}

/// 测试 Tensor::new 能从标量、向量和矩阵推断 shape 与 dtype。
#[test]
fn tensor_new_should_infer_shape_and_dtype_from_input() {
    let scalar = Tensor::new(1.5_f64).unwrap();
    assert_eq!(scalar.shape(), &[]);
    assert_eq!(scalar.dtype(), DType::F64);
    assert_eq!(scalar.storage().as_f64_slice().unwrap(), &[1.5]);

    let vector = Tensor::new(vec![1.0_f32, 2.0, 3.0]).unwrap();
    assert_eq!(vector.shape(), &[3]);
    assert_eq!(vector.dtype(), DType::F32);

    let matrix = Tensor::new(vec![vec![1_i64, 2], vec![3, 4]]).unwrap();
    assert_eq!(matrix.shape(), &[2, 2]);
    assert_eq!(matrix.dtype(), DType::I64);
    assert_eq!(matrix.storage().as_i64_slice().unwrap(), &[1, 2, 3, 4]);

    let mask = Tensor::new(vec![true, false, true]).unwrap();
    assert_eq!(mask.shape(), &[3]);
    assert_eq!(mask.dtype(), DType::Bool);
    assert_eq!(
        mask.storage().as_bool_slice().unwrap(),
        &[true, false, true]
    );
}

/// 测试 Tensor::new 会拒绝每行长度不同的不规则矩阵。
#[test]
fn tensor_new_should_reject_ragged_matrix() {
    let result = Tensor::new(vec![vec![1.0, 2.0], vec![3.0]]);

    assert!(matches!(result, Err(ArcQmlError::ShapeError(_))));
}

/// 测试从扁平 bool 数据创建 Tensor 是否正确保留数据与 shape。
#[test]
fn tensor_from_flat_bool_should_work() {
    let tensor = Tensor::new(TensorData::FlatBool {
        data: vec![true, false, true, false],
        shape: vec![2, 2],
    })
    .unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.dtype(), DType::Bool);
    assert_eq!(
        tensor.storage().as_bool_slice().unwrap(),
        &[true, false, true, false]
    );
}

/// 测试存储 dtype 与 TensorMeta dtype 不一致时会返回对应错误。
#[test]
fn tensor_new_should_reject_dtype_mismatch() {
    let storage = Storage::F64(vec![1.0, 2.0]);

    let meta = TensorMeta::new(vec![2], DType::I64, Device::Cpu, Layout::Dense).unwrap();

    let result = Tensor::from_storage_meta(storage, meta);

    assert!(result.is_err());

    match result.unwrap_err() {
        ArcQmlError::DTypeMismatchError { expected, actual } => {
            assert_eq!(expected, DType::I64);
            assert_eq!(actual, DType::F64);
        }
        other => panic!("expected DTypeMismatchError, got {other:?}"),
    }
}

/// 测试底层存储长度不足以覆盖 shape 时会返回英文 ShapeError 信息。
#[test]
fn tensor_new_should_reject_shape_len_mismatch() {
    let storage = Storage::F64(vec![1.0, 2.0, 3.0]);

    let meta = TensorMeta::new(vec![2, 2], DType::F64, Device::Cpu, Layout::Dense).unwrap();

    let result = Tensor::from_storage_meta(storage, meta);

    assert!(result.is_err());

    match result.unwrap_err() {
        ArcQmlError::ShapeError(message) => {
            assert!(message.contains("storage length insufficient for tensor elements"));
        }
        other => panic!("expected ShapeError, got {other:?}"),
    }
}

/// 测试从完整部件创建 Tensor 时会保留 autograd 元信息。
#[test]
fn tensor_from_parts_should_keep_autograd_meta() {
    let storage = Storage::F64(vec![1.0, 2.0]);

    let meta = TensorMeta::new(vec![2], DType::F64, Device::Cpu, Layout::Dense).unwrap();

    let autograd = AutogradMeta::new(true, true);

    let tensor = Tensor::from_parts(storage, meta, autograd).unwrap();

    assert!(tensor.requires_grad());
    assert!(tensor.grad().is_none());
}

/// 测试 Tensor 可以使用非连续 view 的 TensorMeta 创建。
#[test]
fn tensor_from_parts_should_accept_view_meta() {
    let storage = Storage::F64(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    let meta = TensorMeta::from_parts(
        vec![2, 2],
        vec![3, 1],
        1,
        DType::F64,
        Device::Cpu,
        Layout::Dense,
    )
    .unwrap();

    let tensor = Tensor::from_storage_meta(storage, meta).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.strides(), &[3, 1]);
    assert_eq!(tensor.offset(), 1);
    assert!(!tensor.is_contiguous());
}

// ============================================================
// 张量自动微分状态
// ============================================================

/// 测试 Tensor 的 requires_grad 标记可以被开启和关闭。
#[test]
fn tensor_requires_grad_should_be_mutable() {
    let tensor = make_f64_tensor(vec![1.0, 2.0], vec![2]);

    assert!(!tensor.requires_grad());

    tensor.set_requires_grad(true);

    assert!(tensor.requires_grad());

    tensor.set_requires_grad(false);

    assert!(!tensor.requires_grad());
}

/// 测试 Tensor 梯度的设置、读取与清空行为。
#[test]
fn tensor_grad_set_get_and_zero_should_work() {
    let tensor = make_f64_tensor(vec![1.0, 2.0], vec![2]);

    let grad = make_f64_tensor(vec![0.1, 0.2], vec![2]);

    assert!(tensor.grad().is_none());

    tensor.set_grad(grad);

    let got_grad = tensor.grad();

    assert!(got_grad.is_some());

    let got_grad = got_grad.unwrap();

    assert_eq!(got_grad.shape(), &[2]);
    assert_eq!(got_grad.dtype(), DType::F64);
    assert_eq!(got_grad.numel(), 2);

    tensor.zero_grad();

    assert!(tensor.grad().is_none());
}

/// 测试 Tensor::clone 会复制 TensorMeta，但共享 Storage 和 autograd 状态。
#[test]
fn tensor_clone_should_copy_meta_and_share_storage_and_autograd_state() {
    let tensor = make_f64_tensor(vec![1.0, 2.0], vec![2]);

    let cloned = tensor.clone();

    assert_ne!(
        tensor.meta() as *const TensorMeta,
        cloned.meta() as *const TensorMeta
    );

    assert!(!tensor.requires_grad());
    assert!(!cloned.requires_grad());

    cloned.set_requires_grad(true);

    assert!(tensor.requires_grad());
    assert!(cloned.requires_grad());

    let grad = make_f64_tensor(vec![0.5, 0.6], vec![2]);

    cloned.set_grad(grad);

    assert!(tensor.grad().is_some());
    assert!(cloned.grad().is_some());

    match &mut *cloned.storage_mut() {
        Storage::F64(values) => values[0] = 10.0,
        other => panic!("expected f64 storage, got {other:?}"),
    }

    assert_eq!(tensor.storage().as_f64_slice().unwrap(), &[10.0, 2.0]);
}

/// 测试 Tensor 的深拷贝会复制存储和 autograd 状态。
#[test]
fn tensor_deep_clone_should_copy_storage_and_autograd_meta() {
    let tensor = make_f64_tensor(vec![1.0, 2.0], vec![2]);

    tensor.set_requires_grad(true);

    let cloned = tensor.deep_clone().unwrap();

    assert_eq!(cloned.shape(), &[2]);
    assert_eq!(cloned.dtype(), DType::F64);
    assert_eq!(cloned.storage().as_f64_slice(), Some(&[1.0, 2.0][..]));
    assert!(cloned.requires_grad());

    cloned.set_requires_grad(false);

    assert!(tensor.requires_grad());
    assert!(!cloned.requires_grad());
}

/// 测试不同逻辑 Tensor 可以共享同一份 Storage，并观察彼此的原地更新。
#[test]
fn tensor_view_should_share_mutable_storage() {
    let tensor = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let view_meta = TensorMeta::from_parts(
        vec![2, 2],
        vec![1, 2],
        0,
        DType::F64,
        Device::Cpu,
        Layout::Dense,
    )
    .unwrap();
    let view = tensor.view_with_meta(view_meta).unwrap();

    match &mut *view.storage_mut() {
        Storage::F64(values) => values[1] = 20.0,
        other => panic!("expected f64 storage, got {other:?}"),
    }

    assert_eq!(
        tensor.storage().as_f64_slice().unwrap(),
        &[1.0, 20.0, 3.0, 4.0]
    );
    assert_eq!(view.strides(), &[1, 2]);
}

// ============================================================
// 自动微分
// ============================================================

/// 测试自定义计算图节点可以将标量梯度传播到叶子 Tensor。
#[test]
fn tensor_backward_should_propagate_gradient_to_leaf() {
    let input = Tensor::new(3.0_f64).unwrap();
    input.set_requires_grad(true);

    let output = square_for_test(&input);

    assert!(!output.is_leaf());
    assert!(output.grad().is_none());

    output.backward().unwrap();

    assert_eq!(
        input.grad().unwrap().storage().as_f64_slice(),
        Some(&[6.0][..])
    );
    assert!(output.grad().is_none());
}

/// 测试同一叶子 Tensor 经由多条路径得到的梯度会正确累加。
#[test]
fn tensor_backward_should_accumulate_multiple_paths() {
    let input = Tensor::new(3.0_f64).unwrap();
    input.set_requires_grad(true);

    let output = add_for_test(&square_for_test(&input), &square_for_test(&input));

    output.backward().unwrap();

    assert_eq!(
        input.grad().unwrap().storage().as_f64_slice(),
        Some(&[12.0][..])
    );
}

/// 深链图应使用堆上的显式工作栈，而不是递归耗尽线程调用栈。
#[test]
fn tensor_backward_should_support_deep_chain_without_stack_overflow() {
    const DEPTH: usize = 20_000;

    let input = Tensor::new(1.0_f64).unwrap();
    input.set_requires_grad(true);

    let mut output = input.clone();
    for _ in 0..DEPTH {
        output = identity_for_test(&output);
    }

    output.backward().unwrap();

    assert_eq!(
        input.grad().unwrap().storage().as_f64_slice(),
        Some(&[1.0][..])
    );
}

/// 测试 C64 Tensor 可以作为叶子节点接收自动微分梯度。
#[test]
fn tensor_backward_should_support_c64_leaf_gradient() {
    let input = Tensor::new(Complex64::new(1.0, 2.0)).unwrap();
    input.set_requires_grad(true);

    let output = identity_for_test(&input);

    output.backward().unwrap();

    assert_eq!(
        input.grad().unwrap().storage().as_c64_slice(),
        Some(&[Complex64::new(1.0, 0.0)][..])
    );
}

/// 测试 retain_grad 会保留非叶子 Tensor 的梯度。
#[test]
fn tensor_retain_grad_should_keep_non_leaf_gradient() {
    let input = Tensor::new(3.0_f64).unwrap();
    input.set_requires_grad(true);

    let middle = square_for_test(&input);
    middle.retain_grad();
    let output = square_for_test(&middle);

    output.backward().unwrap();

    assert_eq!(
        middle.grad().unwrap().storage().as_f64_slice(),
        Some(&[18.0][..])
    );
    assert_eq!(
        input.grad().unwrap().storage().as_f64_slice(),
        Some(&[108.0][..])
    );
}

/// 测试 no_grad 作用域不会为前向操作记录计算图。
#[test]
fn no_grad_should_disable_graph_recording() {
    let input = Tensor::new(3.0_f64).unwrap();
    input.set_requires_grad(true);

    let output = {
        let _guard = no_grad();
        square_for_test(&input)
    };

    assert!(!output.requires_grad());
    assert!(output.is_leaf());
}

/// 测试 view 的 backward 会按照 view strides 将梯度散射回父 Tensor。
#[test]
fn tensor_view_backward_should_scatter_gradient_to_parent_layout() {
    let input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    input.set_requires_grad(true);

    let transposed_meta = TensorMeta::from_parts(
        vec![2, 2],
        vec![1, 2],
        0,
        DType::F64,
        Device::Cpu,
        Layout::Dense,
    )
    .unwrap();
    let view = input.view_with_meta(transposed_meta).unwrap();

    view.backward_with_grad(make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]))
        .unwrap();

    assert_eq!(
        input.grad().unwrap().storage().as_f64_slice(),
        Some(&[1.0, 3.0, 2.0, 4.0][..])
    );
}

/// 测试连续 Tensor 的 reshape 共享 Storage，并将梯度恢复为原始形状。
#[test]
fn tensor_reshape_should_share_storage_and_propagate_gradient() {
    let input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    input.set_requires_grad(true);

    let output = input.reshape(vec![3, 2]).unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    assert!(output.is_contiguous());

    // reshape 是 view，修改输出 Storage 会反映到输入 Storage。
    match &mut *output.storage_mut() {
        Storage::F64(values) => values[0] = 10.0,
        other => panic!("expected f64 storage, got {other:?}"),
    }
    assert_eq!(
        input.storage().as_f64_slice(),
        Some(&[10.0, 2.0, 3.0, 4.0, 5.0, 6.0][..])
    );

    // 原地修改会触发版本检查，因此使用独立输入验证正常的反向传播路径。
    let input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    input.set_requires_grad(true);
    let output = input.reshape(vec![3, 2]).unwrap();

    output
        .backward_with_grad(make_f64_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        ))
        .unwrap();

    assert_eq!(input.grad().unwrap().shape(), &[2, 3]);
    assert_eq!(
        input.grad().unwrap().storage().as_f64_slice(),
        Some(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0][..])
    );
}

/// 测试二维 transpose 交换 shape 和 stride，并在反向时还原梯度位置。
#[test]
fn tensor_transpose_should_create_view_and_propagate_gradient() {
    let input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    input.set_requires_grad(true);

    let output = input.transpose().unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    assert_eq!(output.strides(), &[1, 3]);
    assert!(!output.is_contiguous());

    output
        .backward_with_grad(make_f64_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        ))
        .unwrap();

    assert_eq!(
        input.grad().unwrap().storage().as_f64_slice(),
        Some(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0][..])
    );
}

/// 测试 transpose 后的 contiguous 会按逻辑顺序复制数据，且梯度可穿过两个操作。
#[test]
fn tensor_contiguous_should_materialize_non_contiguous_layout_and_propagate_gradient() {
    let input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    input.set_requires_grad(true);
    let transposed = input.transpose().unwrap();

    let output = transposed.contiguous().unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    assert!(output.is_contiguous());
    assert_eq!(
        output.storage().as_f64_slice(),
        Some(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0][..])
    );

    output
        .backward_with_grad(make_f64_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        ))
        .unwrap();

    assert_eq!(
        input.grad().unwrap().storage().as_f64_slice(),
        Some(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0][..])
    );
}

/// 测试非连续 Tensor 的 reshape 会先连续化，再正确地将梯度传回原 Tensor。
#[test]
fn tensor_reshape_should_materialize_non_contiguous_input_and_propagate_gradient() {
    let input = make_f64_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    input.set_requires_grad(true);
    let transposed = input.transpose().unwrap();

    let output = transposed.reshape(vec![2, 3]).unwrap();

    assert_eq!(output.shape(), &[2, 3]);
    assert!(output.is_contiguous());
    assert_eq!(
        output.storage().as_f64_slice(),
        Some(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0][..])
    );

    output
        .backward_with_grad(make_f64_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ))
        .unwrap();

    assert_eq!(
        input.grad().unwrap().storage().as_f64_slice(),
        Some(&[1.0, 3.0, 5.0, 2.0, 4.0, 6.0][..])
    );
}

/// 测试 forward 后原地修改参与计算的 Tensor 会在 backward 时返回错误。
#[test]
fn tensor_backward_should_reject_inplace_modified_parent() {
    let input = Tensor::new(3.0_f64).unwrap();
    input.set_requires_grad(true);
    let output = square_for_test(&input);

    match &mut *input.storage_mut() {
        Storage::F64(values) => values[0] = 4.0,
        other => panic!("expected f64 storage, got {other:?}"),
    }

    assert!(matches!(
        output.backward(),
        Err(ArcQmlError::AutogradError(_))
    ));
}

/// 测试默认 backward 会释放计算图，retain_graph 允许一次额外复用。
#[test]
fn tensor_backward_should_release_or_retain_graph_as_requested() {
    let input = Tensor::new(3.0_f64).unwrap();
    input.set_requires_grad(true);
    let output = square_for_test(&input);

    output
        .backward_with_grad_retain_graph(Tensor::new(1.0_f64).unwrap())
        .unwrap();
    output.backward().unwrap();

    assert_eq!(
        input.grad().unwrap().storage().as_f64_slice(),
        Some(&[12.0][..])
    );
    assert!(matches!(
        output.backward(),
        Err(ArcQmlError::AutogradError(_))
    ));
}

/// 测试默认 AutogradMeta 的梯度需求、叶子状态和梯度为空的语义。
#[test]
fn autograd_meta_default_should_work() {
    let meta = AutogradMeta::default();

    assert!(!meta.requires_grad());
    assert!(meta.is_leaf());
    assert!(meta.grad().is_none());
}

/// 测试 AutogradMeta 各状态字段和梯度的设置及清空行为。
#[test]
fn autograd_meta_setters_should_work() {
    let mut meta = AutogradMeta::new(false, true);

    assert!(!meta.requires_grad());
    assert!(meta.is_leaf());

    meta.set_requires_grad(true);
    meta.set_leaf(false);

    assert!(meta.requires_grad());
    assert!(!meta.is_leaf());

    let grad = make_f64_tensor(vec![0.1, 0.2], vec![2]);

    meta.set_grad(grad);

    assert!(meta.grad().is_some());

    meta.zero_grad();

    assert!(meta.grad().is_none());
}

// ============================================================
// 参数
// ============================================================

/// 测试创建 Parameter 会启用训练和梯度计算。
#[test]
fn parameter_new_should_enable_requires_grad_and_trainable() {
    let tensor = make_f64_tensor(vec![1.0, 2.0], vec![2]);

    assert!(!tensor.requires_grad());

    let parameter = Parameter::new(tensor);

    assert!(parameter.trainable());
    assert!(parameter.requires_grad());
    assert!(parameter.name().is_none());
}

/// 测试 Parameter 名称可以设置和清除。
#[test]
fn parameter_name_should_be_mutable() {
    let tensor = make_f64_tensor(vec![1.0, 2.0], vec![2]);

    let parameter = Parameter::new(tensor);

    assert_eq!(parameter.name(), None);

    parameter.set_name("theta");

    assert_eq!(parameter.name(), Some("theta".to_string()));

    parameter.clear_name();

    assert_eq!(parameter.name(), None);
}

/// 测试禁用训练不会关闭已有 Tensor 的梯度计算，重新启用训练会确保梯度计算开启。
#[test]
fn parameter_trainable_should_not_disable_requires_grad() {
    let tensor = make_f64_tensor(vec![1.0, 2.0], vec![2]);

    let parameter = Parameter::new(tensor);

    assert!(parameter.trainable());
    assert!(parameter.requires_grad());

    parameter.set_trainable(false);

    assert!(!parameter.trainable());
    assert!(parameter.requires_grad());

    parameter.set_trainable(true);

    assert!(parameter.trainable());
    assert!(parameter.requires_grad());
}

/// 测试冻结只禁止优化器更新，解冻后恢复可训练状态。
#[test]
fn parameter_freeze_and_unfreeze_should_work() {
    let tensor = make_f64_tensor(vec![1.0, 2.0], vec![2]);

    let parameter = Parameter::new(tensor);

    parameter.freeze();

    assert!(!parameter.trainable());
    assert!(parameter.requires_grad());

    parameter.unfreeze();

    assert!(parameter.trainable());
    assert!(parameter.requires_grad());
}

/// 测试 Parameter 梯度的设置、读取与清空行为。
#[test]
fn parameter_grad_set_get_and_zero_should_work() {
    let tensor = make_f64_tensor(vec![1.0, 2.0], vec![2]);

    let parameter = Parameter::new(tensor);

    assert!(parameter.grad().is_none());

    let grad = make_f64_tensor(vec![0.01, 0.02], vec![2]);

    parameter.set_grad(grad);

    let got_grad = parameter.grad();

    assert!(got_grad.is_some());

    let got_grad = got_grad.unwrap();

    assert_eq!(got_grad.shape(), &[2]);
    assert_eq!(got_grad.dtype(), DType::F64);
    assert_eq!(got_grad.numel(), 2);

    parameter.zero_grad();

    assert!(parameter.grad().is_none());
}

/// 测试 Parameter 返回的 Tensor 拷贝具有正确的张量元信息。
#[test]
fn parameter_tensor_should_return_tensor_copy_with_correct_meta() {
    let tensor = make_c64_tensor(
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
        vec![2],
    );

    let parameter = Parameter::new(tensor);

    assert_eq!(parameter.tensor().dtype(), DType::C64);
    assert_eq!(parameter.tensor().shape(), &[2]);
    assert_eq!(parameter.tensor().numel(), 2);
}

/// 测试替换 Parameter 的 Tensor 时会保留名称、训练状态和梯度。
#[test]
fn parameter_set_tensor_should_keep_state_and_grad() {
    let tensor = make_f64_tensor(vec![1.0, 2.0], vec![2]);
    let parameter = Parameter::new(tensor);

    parameter.set_name("weight");
    parameter.freeze();
    parameter.set_grad(make_f64_tensor(vec![0.1, 0.2], vec![2]));

    let new_tensor = make_f64_tensor(vec![3.0, 4.0], vec![2]);

    parameter.set_tensor(new_tensor);

    assert_eq!(parameter.name(), Some("weight".to_string()));
    assert!(!parameter.trainable());
    assert!(!parameter.requires_grad());
    assert_eq!(
        parameter.tensor().storage().as_f64_slice().unwrap(),
        &[3.0, 4.0]
    );
    assert_eq!(
        parameter.grad().unwrap().storage().as_f64_slice().unwrap(),
        &[0.1, 0.2]
    );
}

/// 测试克隆 Parameter 后，两个句柄共享同一份内部状态。
#[test]
fn parameter_clone_should_share_inner_state() {
    let tensor = make_f64_tensor(vec![1.0], vec![]);
    let parameter = Parameter::new(tensor);
    let cloned = parameter.clone();

    cloned.set_tensor(make_f64_tensor(vec![2.0], vec![]));

    assert_eq!(parameter.tensor().storage().as_f64_slice().unwrap(), &[2.0]);
}

/// 测试嵌套 no_grad 在非 LIFO 释放时仍会保持正确状态。
#[test]
fn no_grad_should_support_nested_and_non_lifo_drop() {
    assert!(is_grad_enabled());

    let outer = no_grad();
    assert!(!is_grad_enabled());

    let inner = no_grad();
    assert!(!is_grad_enabled());

    drop(outer);
    assert!(!is_grad_enabled());

    drop(inner);
    assert!(is_grad_enabled());
}

/// 测试一个线程的 no_grad 不会关闭其他线程的梯度记录。
#[test]
fn no_grad_should_be_thread_local() {
    let guard = no_grad();
    assert!(!is_grad_enabled());

    let other_thread_enabled = std::thread::spawn(is_grad_enabled).join().unwrap();
    assert!(other_thread_enabled);

    drop(guard);
    assert!(is_grad_enabled());
}

/// 测试仅可为浮点或复数叶子 Tensor 开启 requires_grad。
#[test]
fn requires_grad_should_reject_non_differentiable_or_non_leaf_tensors() {
    let integer = Tensor::new(1_i64).unwrap();
    assert!(integer.try_set_requires_grad(true).is_err());
    assert!(!integer.requires_grad());

    let input = Tensor::new(2.0_f64).unwrap();
    input.set_requires_grad(true);
    let output = square_for_test(&input);

    assert!(!output.is_leaf());
    assert!(output.try_set_requires_grad(true).is_err());
}

/// 测试 Parameter 的可失败构造和替换接口会拒绝无效 Tensor。
#[test]
fn parameter_should_reject_invalid_tensor_without_mutating_state() {
    let integer = Tensor::new(1_i64).unwrap();
    assert!(Parameter::try_new(integer).is_err());

    let parameter = Parameter::new(Tensor::new(0.5_f64).unwrap());
    let input = Tensor::new(2.0_f64).unwrap();
    input.set_requires_grad(true);
    let non_leaf = square_for_test(&input);

    assert!(parameter.try_set_tensor(non_leaf).is_err());
    assert_eq!(
        parameter.tensor().storage().as_f64_slice(),
        Some(&[0.5][..])
    );
}

/// 测试替换 Parameter 时仅保留形状和数据类型兼容的旧梯度。
#[test]
fn parameter_set_tensor_should_clear_incompatible_gradient() {
    let parameter = Parameter::new(make_f64_tensor(vec![1.0, 2.0], vec![2]));
    parameter.set_grad(make_f64_tensor(vec![0.1, 0.2], vec![2]));

    parameter
        .try_set_tensor(make_f64_tensor(vec![3.0], vec![1]))
        .unwrap();

    assert!(parameter.grad().is_none());
}

/// 测试 value 会读取单元素实数 Tensor，并拒绝不支持的形状和数据类型。
#[test]
fn tensor_value_should_read_real_scalars() {
    assert_eq!(Tensor::new(1.25_f64).unwrap().value().unwrap(), 1.25);
    assert_eq!(Tensor::new(-0.5_f32).unwrap().value().unwrap(), -0.5);
    assert_eq!(make_f64_tensor(vec![3.0], vec![1]).value().unwrap(), 3.0);
    assert!(make_f64_tensor(vec![1.0, 2.0], vec![2]).value().is_err());
    assert!(
        Tensor::new(Complex64::new(1.0, 0.0))
            .unwrap()
            .value()
            .is_err()
    );
}

#[cfg(test)]
mod tests {
    use super::checked_power_of_two;

    #[test]
    fn checked_power_of_two_should_respect_platform_width() {
        assert_eq!(checked_power_of_two(0), Some(1));
        assert_eq!(checked_power_of_two(1), Some(2));
        assert_eq!(
            checked_power_of_two(usize::BITS as usize - 1),
            Some(1usize << (usize::BITS - 1))
        );
        assert_eq!(checked_power_of_two(usize::BITS as usize), None);
    }
}
