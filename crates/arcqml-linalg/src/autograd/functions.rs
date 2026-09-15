use crate::ops::broadcast_index;
use arcqml_core::{ArcQmlError, BackwardFn, Result, Storage, Tensor};
use num_complex::Complex64;
#[cfg(feature = "parallel")]
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// 二元逐元素算子的种类：Add，Sub，Mul。
#[derive(Debug, Clone, Copy)]
pub(crate) enum BinaryKind {
    Add,
    Sub,
    Mul,
}

/// 一元逐元素算子的种类：Square，Sqrt，Abs。
#[derive(Debug, Clone, Copy)]
pub(crate) enum UnaryKind {
    Square,
    Sqrt,
    Abs,
}

/// 标量归约算子的种类：Sum，Mean。
#[derive(Debug, Clone, Copy)]
pub(crate) enum ReductionKind {
    Sum,
    Mean,
}

/// 极值归约算子的种类：Max，Min。
#[derive(Debug, Clone, Copy)]
pub(crate) enum ExtremumKind {
    Max,
    Min,
}

/// 支持广播的二元逐元素反向规则。
#[derive(Debug)]
pub(crate) struct BinaryBackward {
    pub(crate) kind: BinaryKind,
    pub(crate) output_shape: Vec<usize>,
}

impl BackwardFn for BinaryBackward {
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 2 {
            return Err(ArcQmlError::AutogradError(
                "binary backward expects two parents".to_string(),
            ));
        }

        let lhs = &parents[0];
        let rhs = &parents[1];
        let lhs_shape = lhs.shape().to_vec();
        let rhs_shape = rhs.shape().to_vec();
        let lhs_meta = lhs.meta().clone();
        let rhs_meta = rhs.meta().clone();
        let lhs_storage = lhs.storage();
        let rhs_storage = rhs.storage();
        let grad_storage = grad_output.storage();

        match (&*lhs_storage, &*rhs_storage, &*grad_storage) {
            (Storage::F32(lhs), Storage::F32(rhs), Storage::F32(grad)) => {
                let (lhs_grad, rhs_grad) = binary_backward_values(
                    lhs,
                    rhs,
                    grad,
                    &lhs_shape,
                    &rhs_shape,
                    &self.output_shape,
                    self.kind,
                    |value| value,
                );
                Ok(vec![
                    Some(make_f32_gradient(lhs_grad, lhs_meta.clone())?),
                    Some(make_f32_gradient(rhs_grad, rhs_meta.clone())?),
                ])
            }
            (Storage::F64(lhs), Storage::F64(rhs), Storage::F64(grad)) => {
                let (lhs_grad, rhs_grad) = binary_backward_values(
                    lhs,
                    rhs,
                    grad,
                    &lhs_shape,
                    &rhs_shape,
                    &self.output_shape,
                    self.kind,
                    |value| value,
                );
                Ok(vec![
                    Some(make_f64_gradient(lhs_grad, lhs_meta.clone())?),
                    Some(make_f64_gradient(rhs_grad, rhs_meta.clone())?),
                ])
            }
            (Storage::C64(lhs), Storage::C64(rhs), Storage::C64(grad)) => {
                let (lhs_grad, rhs_grad) = binary_backward_values(
                    lhs,
                    rhs,
                    grad,
                    &lhs_shape,
                    &rhs_shape,
                    &self.output_shape,
                    self.kind,
                    |value: Complex64| value.conj(),
                );
                Ok(vec![
                    Some(make_c64_gradient(lhs_grad, lhs_meta.clone())?),
                    Some(make_c64_gradient(rhs_grad, rhs_meta.clone())?),
                ])
            }
            _ => nondifferentiable_dtype(lhs.dtype()),
        }
    }
}

/// 计算广播二元算子对两侧输入的梯度，并在被广播维度上自动累加。
#[allow(
    clippy::too_many_arguments,
    reason = "the kernel needs both input buffers and shapes, the output shape, operator, and gradient conversion"
)]
fn binary_backward_values<T>(
    lhs: &[T],
    rhs: &[T],
    grad_output: &[T],
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    kind: BinaryKind,
    conjugate: impl Fn(T) -> T + Sync + Send,
) -> (Vec<T>, Vec<T>)
where
    T: Copy
        + Default
        + std::ops::AddAssign
        + std::ops::Neg<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync,
{
    if lhs_shape == output_shape && rhs_shape == output_shape {
        return (
            zip_map_values(grad_output, rhs, |gradient, value| match kind {
                BinaryKind::Add | BinaryKind::Sub => *gradient,
                BinaryKind::Mul => *gradient * conjugate(*value),
            }),
            zip_map_values(grad_output, lhs, |gradient, value| match kind {
                BinaryKind::Add => *gradient,
                BinaryKind::Sub => -*gradient,
                BinaryKind::Mul => *gradient * conjugate(*value),
            }),
        );
    }

    #[cfg(feature = "parallel")]
    if grad_output.len() >= PARALLEL_MIN_ELEMENTS {
        return parallel_broadcast_backward_values(
            lhs,
            rhs,
            grad_output,
            lhs_shape,
            rhs_shape,
            output_shape,
            kind,
            conjugate,
        );
    }

    let mut lhs_grad = vec![T::default(); lhs.len()];
    let mut rhs_grad = vec![T::default(); rhs.len()];

    for (output_index, gradient) in grad_output.iter().copied().enumerate() {
        let lhs_index = broadcast_index(output_index, output_shape, lhs_shape);
        let rhs_index = broadcast_index(output_index, output_shape, rhs_shape);
        match kind {
            // 对于 y = a + b，Ga = Gy，Gb = Gy
            BinaryKind::Add => {
                lhs_grad[lhs_index] += gradient;
                rhs_grad[rhs_index] += gradient;
            }
            // 对于 y = a - b，Ga = Gy，Gb = -Gy
            BinaryKind::Sub => {
                lhs_grad[lhs_index] += gradient;
                rhs_grad[rhs_index] += -gradient;
            }
            // 对于 y = a * b，Ga = Gy * B，Gb = Gy * A。
            // C64 调用时 conjugate 返回共轭，因此使用共轭 Wirtinger VJP。
            BinaryKind::Mul => {
                lhs_grad[lhs_index] += gradient * conjugate(rhs[rhs_index]);
                rhs_grad[rhs_index] += gradient * conjugate(lhs[lhs_index]);
            }
        }
    }

    (lhs_grad, rhs_grad)
}

#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments)]
fn parallel_broadcast_backward_values<T>(
    lhs: &[T],
    rhs: &[T],
    grad_output: &[T],
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    kind: BinaryKind,
    conjugate: impl Fn(T) -> T + Sync + Send,
) -> (Vec<T>, Vec<T>)
where
    T: Copy
        + Default
        + std::ops::AddAssign
        + std::ops::Neg<Output = T>
        + std::ops::Mul<Output = T>
        + Send
        + Sync,
{
    let (lhs_partials, rhs_partials) = grad_output
        .par_chunks(PARALLEL_CHUNK_SIZE)
        .enumerate()
        .fold(
            || (HashMap::new(), HashMap::new()),
            |(mut lhs_partials, mut rhs_partials), (chunk_index, gradients)| {
                let start = chunk_index * PARALLEL_CHUNK_SIZE;
                for (offset, gradient) in gradients.iter().copied().enumerate() {
                    let output_index = start + offset;
                    let lhs_index = broadcast_index(output_index, output_shape, lhs_shape);
                    let rhs_index = broadcast_index(output_index, output_shape, rhs_shape);

                    match kind {
                        BinaryKind::Add => {
                            *lhs_partials.entry(lhs_index).or_default() += gradient;
                            *rhs_partials.entry(rhs_index).or_default() += gradient;
                        }
                        BinaryKind::Sub => {
                            *lhs_partials.entry(lhs_index).or_default() += gradient;
                            *rhs_partials.entry(rhs_index).or_default() += -gradient;
                        }
                        BinaryKind::Mul => {
                            *lhs_partials.entry(lhs_index).or_default() +=
                                gradient * conjugate(rhs[rhs_index]);
                            *rhs_partials.entry(rhs_index).or_default() +=
                                gradient * conjugate(lhs[lhs_index]);
                        }
                    }
                }
                (lhs_partials, rhs_partials)
            },
        )
        .reduce(
            || (HashMap::new(), HashMap::new()),
            |(mut lhs_accumulator, mut rhs_accumulator), (lhs_partials, rhs_partials)| {
                merge_gradient_partials(&mut lhs_accumulator, lhs_partials);
                merge_gradient_partials(&mut rhs_accumulator, rhs_partials);
                (lhs_accumulator, rhs_accumulator)
            },
        );

    let mut lhs_grad = vec![T::default(); lhs.len()];
    let mut rhs_grad = vec![T::default(); rhs.len()];
    for (index, value) in lhs_partials {
        lhs_grad[index] = value;
    }
    for (index, value) in rhs_partials {
        rhs_grad[index] = value;
    }
    (lhs_grad, rhs_grad)
}

#[cfg(feature = "parallel")]
fn merge_gradient_partials<T>(target: &mut HashMap<usize, T>, source: HashMap<usize, T>)
where
    T: Default + std::ops::AddAssign,
{
    for (index, value) in source {
        *target.entry(index).or_default() += value;
    }
}

/// 一元逐元素算子的反向规则。
#[derive(Debug)]
pub(crate) struct UnaryBackward {
    pub(crate) kind: UnaryKind,
}

impl BackwardFn for UnaryBackward {
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(ArcQmlError::AutogradError(
                "unary backward expects one parent".to_string(),
            ));
        }

        let input = &parents[0];
        let input_meta = input.meta().clone();
        let input_storage = input.storage();
        let grad_storage = grad_output.storage();

        match (&*input_storage, &*grad_storage, self.kind) {
            // 对于 y = x^2，Gx = 2x * Gy
            (Storage::F32(input), Storage::F32(grad), UnaryKind::Square) => {
                Ok(vec![Some(make_f32_gradient(
                    zip_map_values(input, grad, |x, g| 2.0 * x * g),
                    input_meta.clone(),
                )?)])
            }
            (Storage::F64(input), Storage::F64(grad), UnaryKind::Square) => {
                Ok(vec![Some(make_f64_gradient(
                    zip_map_values(input, grad, |x, g| 2.0 * x * g),
                    input_meta.clone(),
                )?)])
            }
            (Storage::C64(input), Storage::C64(grad), UnaryKind::Square) => {
                Ok(vec![Some(make_c64_gradient(
                    // 复数 VJP 取共轭
                    zip_map_values(input, grad, |x, g| *g * 2.0 * x.conj()),
                    input_meta.clone(),
                )?)])
            }
            // 对于 y = x^(1/2)，Gx = (1/(2x^(1/2))) * Gy
            (Storage::F32(input), Storage::F32(grad), UnaryKind::Sqrt) => {
                Ok(vec![Some(make_f32_gradient(
                    zip_map_values(input, grad, |x, g| {
                        if *x == 0.0 { 0.0 } else { g / (2.0 * x.sqrt()) }
                    }),
                    input_meta.clone(),
                )?)])
            }
            (Storage::F64(input), Storage::F64(grad), UnaryKind::Sqrt) => {
                Ok(vec![Some(make_f64_gradient(
                    zip_map_values(input, grad, |x, g| {
                        if *x == 0.0 { 0.0 } else { g / (2.0 * x.sqrt()) }
                    }),
                    input_meta.clone(),
                )?)])
            }
            (Storage::C64(input), Storage::C64(grad), UnaryKind::Sqrt) => {
                Ok(vec![Some(make_c64_gradient(
                    zip_map_values(input, grad, |x, g| {
                        if *x == Complex64::new(0.0, 0.0) {
                            Complex64::new(0.0, 0.0)
                        } else {
                            // y = x^(1/2) 时使用 conj(1/(2x^(1/2)))
                            *g / (2.0 * x.sqrt().conj())
                        }
                    }),
                    input_meta.clone(),
                )?)])
            }
            // 对于 y = x^(1/2)，Gx = Gy (x>0) & Gx = -Gy (x<0)
            (Storage::F32(input), Storage::F32(grad), UnaryKind::Abs) => {
                Ok(vec![Some(make_f32_gradient(
                    zip_map_values(
                        input,
                        grad,
                        |x, g| {
                            if *x == 0.0 { 0.0 } else { *g * x.signum() }
                        },
                    ),
                    input_meta.clone(),
                )?)])
            }
            (Storage::F64(input), Storage::F64(grad), UnaryKind::Abs) => {
                Ok(vec![Some(make_f64_gradient(
                    zip_map_values(
                        input,
                        grad,
                        |x, g| {
                            if *x == 0.0 { 0.0 } else { *g * x.signum() }
                        },
                    ),
                    input_meta.clone(),
                )?)])
            }
            // 对实值 |z| 使用 PyTorch 风格的共轭 Wirtinger 约定，零点处选择零子梯度
            (Storage::C64(input), Storage::F64(grad), UnaryKind::Abs) => {
                Ok(vec![Some(make_c64_gradient(
                    zip_map_values(input, grad, |z, g| {
                        if z.norm() == 0.0 {
                            Complex64::new(0.0, 0.0)
                        } else {
                            *z * (*g / z.norm())
                        }
                    }),
                    input_meta.clone(),
                )?)])
            }
            _ => nondifferentiable_dtype(input.dtype()),
        }
    }
}

/// sum 与 mean 的反向规则。
#[derive(Debug)]
pub(crate) struct ReductionBackward {
    pub(crate) kind: ReductionKind,
}

impl BackwardFn for ReductionBackward {
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(ArcQmlError::AutogradError(
                "reduction backward expects one parent".to_string(),
            ));
        }

        if grad_output.numel() != 1 {
            return Err(ArcQmlError::AutogradError(
                "reduction backward expects one scalar gradient".to_string(),
            ));
        }

        let input = &parents[0];
        let input_meta = input.meta().clone();
        let divisor = if matches!(self.kind, ReductionKind::Mean) {
            input.numel() as f64
        } else {
            1.0
        };
        let grad = grad_output.storage();

        match (&*grad, input.dtype()) {
            (Storage::F32(values), arcqml_core::DType::F32) => Ok(vec![Some(make_f32_gradient(
                vec![values[0] / divisor as f32; input.numel()],
                input_meta.clone(),
            )?)]),
            (Storage::F64(values), arcqml_core::DType::F64) => Ok(vec![Some(make_f64_gradient(
                vec![values[0] / divisor; input.numel()],
                input_meta.clone(),
            )?)]),
            (Storage::C64(values), arcqml_core::DType::C64) => Ok(vec![Some(make_c64_gradient(
                vec![values[0] / divisor; input.numel()],
                input.meta().clone(),
            )?)]),
            _ => nondifferentiable_dtype(input.dtype()),
        }
    }
}

/// max 与 min 的反向规则，重复极值位置平均分配上游梯度。
#[derive(Debug)]
pub(crate) struct ExtremumBackward {
    pub(crate) kind: ExtremumKind,
}

impl BackwardFn for ExtremumBackward {
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(ArcQmlError::AutogradError(
                "extremum backward expects one parent".to_string(),
            ));
        }

        if grad_output.numel() != 1 {
            return Err(ArcQmlError::AutogradError(
                "extremum backward expects one scalar gradient".to_string(),
            ));
        }

        let input = &parents[0];
        let input_meta = input.meta().clone();
        let input_storage = input.storage();
        let grad_storage = grad_output.storage();

        match (&*input_storage, &*grad_storage) {
            (Storage::F32(input), Storage::F32(grad)) => Ok(vec![Some(make_f32_gradient(
                extremum_gradient(input, grad[0], self.kind),
                input_meta.clone(),
            )?)]),
            (Storage::F64(input), Storage::F64(grad)) => Ok(vec![Some(make_f64_gradient(
                extremum_gradient(input, grad[0], self.kind),
                input_meta.clone(),
            )?)]),
            _ => nondifferentiable_dtype(input.dtype()),
        }
    }
}

/// 计算极值掩码，并把梯度均分给所有并列极值。
fn extremum_gradient<T>(values: &[T], gradient: T, kind: ExtremumKind) -> Vec<T>
where
    T: Copy + Default + PartialOrd + std::ops::Div<Output = T> + From<f32> + Send + Sync,
{
    let extremum = values.iter().copied().reduce(|current, value| match kind {
        ExtremumKind::Max if value > current => value,
        ExtremumKind::Min if value < current => value,
        _ => current,
    });
    let Some(extremum) = extremum else {
        return Vec::new();
    };
    let count = values.iter().filter(|&&value| value == extremum).count();
    let share = gradient / T::from(count as f32);

    map_values(values, |value| {
        if *value == extremum {
            share
        } else {
            T::default()
        }
    })
}

/// dot 的反向规则。
#[derive(Debug)]
pub(crate) struct DotBackward;

impl BackwardFn for DotBackward {
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 2 || grad_output.numel() != 1 {
            return Err(ArcQmlError::AutogradError(
                "dot backward expects two parents and scalar gradient".to_string(),
            ));
        }

        let lhs = &parents[0];
        let rhs = &parents[1];
        let lhs_meta = lhs.meta().clone();
        let rhs_meta = rhs.meta().clone();
        let lhs_values = lhs.storage();
        let rhs_values = rhs.storage();
        let grad = grad_output.storage();

        match (&*lhs_values, &*rhs_values, &*grad) {
            // 对于 y = A dot B，Ga = B * Gy，Gb = A * Gy
            (Storage::F32(lhs), Storage::F32(rhs), Storage::F32(grad)) => Ok(vec![
                Some(make_f32_gradient(
                    map_values(rhs, |x| *x * grad[0]),
                    lhs_meta.clone(),
                )?),
                Some(make_f32_gradient(
                    map_values(lhs, |x| *x * grad[0]),
                    rhs_meta.clone(),
                )?),
            ]),
            (Storage::F64(lhs), Storage::F64(rhs), Storage::F64(grad)) => Ok(vec![
                Some(make_f64_gradient(
                    map_values(rhs, |x| *x * grad[0]),
                    lhs_meta.clone(),
                )?),
                Some(make_f64_gradient(
                    map_values(lhs, |x| *x * grad[0]),
                    rhs_meta.clone(),
                )?),
            ]),
            (Storage::C64(lhs), Storage::C64(rhs), Storage::C64(grad)) => Ok(vec![
                Some(make_c64_gradient(
                    // 反传对另一侧系数取共轭
                    map_values(rhs, |x| grad[0] * x.conj()),
                    lhs_meta.clone(),
                )?),
                Some(make_c64_gradient(
                    map_values(lhs, |x| grad[0] * x.conj()),
                    rhs_meta.clone(),
                )?),
            ]),
            _ => nondifferentiable_dtype(lhs.dtype()),
        }
    }
}

/// 二维矩阵乘法的反向规则。
///
/// 实数分支使用 G @ B^T 与 A^T @ G，C64 分支使用共轭转置。
#[derive(Debug)]
pub(crate) struct MatmulBackward;

impl BackwardFn for MatmulBackward {
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 2 {
            return Err(ArcQmlError::AutogradError(
                "matmul backward expects two parents".to_string(),
            ));
        }

        let lhs = &parents[0];
        let rhs = &parents[1];
        let lhs_meta = lhs.meta().clone();
        let rhs_meta = rhs.meta().clone();
        let m = lhs.shape()[0];
        let k = lhs.shape()[1];
        let n = rhs.shape()[1];
        let lhs_values = lhs.storage();
        let rhs_values = rhs.storage();
        let grad = grad_output.storage();

        match (&*lhs_values, &*rhs_values, &*grad) {
            (Storage::F32(lhs), Storage::F32(rhs), Storage::F32(grad)) => Ok(vec![
                Some(make_f32_gradient(
                    matmul_lhs_gradient(rhs, grad, m, k, n),
                    lhs_meta.clone(),
                )?),
                Some(make_f32_gradient(
                    matmul_rhs_gradient(lhs, grad, m, k, n),
                    rhs_meta.clone(),
                )?),
            ]),
            (Storage::F64(lhs), Storage::F64(rhs), Storage::F64(grad)) => Ok(vec![
                Some(make_f64_gradient(
                    matmul_lhs_gradient(rhs, grad, m, k, n),
                    lhs_meta.clone(),
                )?),
                Some(make_f64_gradient(
                    matmul_rhs_gradient(lhs, grad, m, k, n),
                    rhs_meta.clone(),
                )?),
            ]),
            (Storage::C64(lhs), Storage::C64(rhs), Storage::C64(grad)) => Ok(vec![
                Some(make_c64_gradient(
                    matmul_lhs_gradient_c64(rhs, grad, m, k, n),
                    lhs_meta.clone(),
                )?),
                Some(make_c64_gradient(
                    matmul_rhs_gradient_c64(lhs, grad, m, k, n),
                    rhs_meta.clone(),
                )?),
            ]),
            _ => nondifferentiable_dtype(lhs.dtype()),
        }
    }
}

// 左侧梯度：Ga = Gy * B^T。
fn matmul_lhs_gradient<T>(rhs: &[T], grad: &[T], m: usize, k: usize, n: usize) -> Vec<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + Send + Sync,
{
    map_indices(m * k, |output_index| {
        let row = output_index / k;
        let inner = output_index % k;
        let mut value = T::default();
        for col in 0..n {
            value += grad[row * n + col] * rhs[inner * n + col];
        }
        value
    })
}

// 右侧梯度：Gb = A^T * Gy。
fn matmul_rhs_gradient<T>(lhs: &[T], grad: &[T], m: usize, k: usize, n: usize) -> Vec<T>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T> + Send + Sync,
{
    map_indices(k * n, |output_index| {
        let inner = output_index / n;
        let col = output_index % n;
        let mut value = T::default();
        for row in 0..m {
            value += lhs[row * k + inner] * grad[row * n + col];
        }
        value
    })
}

// Ga = Gy * B^H （共轭转置）
fn matmul_lhs_gradient_c64(
    rhs: &[Complex64],
    grad: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    map_indices(m * k, |output_index| {
        let row = output_index / k;
        let inner = output_index % k;
        let mut value = Complex64::new(0.0, 0.0);
        for col in 0..n {
            value += grad[row * n + col] * rhs[inner * n + col].conj();
        }
        value
    })
}

// 右侧梯度：Gb = A^H * Gy。
fn matmul_rhs_gradient_c64(
    lhs: &[Complex64],
    grad: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    map_indices(k * n, |output_index| {
        let inner = output_index / n;
        let col = output_index % n;
        let mut value = Complex64::new(0.0, 0.0);
        for row in 0..m {
            value += lhs[row * k + inner].conj() * grad[row * n + col];
        }
        value
    })
}

/// L2 范数的反向规则，零范数处返回零子梯度。
#[derive(Debug)]
pub(crate) struct L2NormBackward;

impl BackwardFn for L2NormBackward {
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 || grad_output.numel() != 1 {
            return Err(ArcQmlError::AutogradError(
                "l2 norm backward expects one parent and scalar gradient".to_string(),
            ));
        }

        let input = &parents[0];
        let values = input.storage();
        let grad = grad_output.storage();

        match (&*values, &*grad) {
            (Storage::F32(values), Storage::F64(grad)) => {
                let norm = sum_mapped_f64(values, |x| (*x as f64).powi(2)).sqrt();
                Ok(vec![Some(make_f32_gradient(
                    map_values(values, |x| {
                        if norm == 0.0 {
                            0.0
                        } else {
                            *x * (grad[0] / norm) as f32
                        }
                    }),
                    input.meta().clone(),
                )?)])
            }
            (Storage::F64(values), Storage::F64(grad)) => {
                let norm = sum_mapped_f64(values, |x| x * x).sqrt();
                Ok(vec![Some(make_f64_gradient(
                    map_values(values, |x| {
                        if norm == 0.0 {
                            0.0
                        } else {
                            *x * grad[0] / norm
                        }
                    }),
                    input.meta().clone(),
                )?)])
            }
            (Storage::C64(values), Storage::F64(grad)) => {
                let norm = sum_mapped_f64(values, Complex64::norm_sqr).sqrt();
                Ok(vec![Some(make_c64_gradient(
                    map_values(values, |z| {
                        if norm == 0.0 {
                            Complex64::new(0.0, 0.0)
                        } else {
                            *z * (grad[0] / norm)
                        }
                    }),
                    input.meta().clone(),
                )?)])
            }
            _ => nondifferentiable_dtype(input.dtype()),
        }
    }
}

#[cfg(feature = "parallel")]
const PARALLEL_MIN_ELEMENTS: usize = 8 * 1024;
#[cfg(feature = "parallel")]
const PARALLEL_CHUNK_SIZE: usize = 4 * 1024;

#[cfg(feature = "parallel")]
fn map_values<T, R>(values: &[T], map: impl Fn(&T) -> R + Sync + Send) -> Vec<R>
where
    T: Sync,
    R: Send,
{
    if values.len() < PARALLEL_MIN_ELEMENTS {
        return values.iter().map(map).collect();
    }

    values.par_iter().map(map).collect()
}

#[cfg(not(feature = "parallel"))]
fn map_values<T, R>(values: &[T], map: impl Fn(&T) -> R) -> Vec<R> {
    values.iter().map(map).collect()
}

#[cfg(feature = "parallel")]
fn zip_map_values<T, U, R>(lhs: &[T], rhs: &[U], map: impl Fn(&T, &U) -> R + Sync + Send) -> Vec<R>
where
    T: Sync,
    U: Sync,
    R: Send,
{
    if lhs.len() < PARALLEL_MIN_ELEMENTS {
        return lhs
            .iter()
            .zip(rhs)
            .map(|(left, right)| map(left, right))
            .collect();
    }

    lhs.par_iter()
        .zip(rhs.par_iter())
        .map(|(left, right)| map(left, right))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn zip_map_values<T, U, R>(lhs: &[T], rhs: &[U], map: impl Fn(&T, &U) -> R) -> Vec<R> {
    lhs.iter()
        .zip(rhs)
        .map(|(left, right)| map(left, right))
        .collect()
}

#[cfg(feature = "parallel")]
fn map_indices<T>(length: usize, map: impl Fn(usize) -> T + Sync + Send) -> Vec<T>
where
    T: Send,
{
    if length < PARALLEL_MIN_ELEMENTS {
        return (0..length).map(map).collect();
    }

    (0..length).into_par_iter().map(map).collect()
}

#[cfg(not(feature = "parallel"))]
fn map_indices<T>(length: usize, map: impl Fn(usize) -> T) -> Vec<T> {
    (0..length).map(map).collect()
}

#[cfg(feature = "parallel")]
fn sum_mapped_f64<T>(values: &[T], map: impl Fn(&T) -> f64 + Sync + Send) -> f64
where
    T: Sync,
{
    if values.len() < PARALLEL_MIN_ELEMENTS {
        return values.iter().map(map).sum();
    }

    values.par_iter().map(map).sum()
}

#[cfg(not(feature = "parallel"))]
fn sum_mapped_f64<T>(values: &[T], map: impl Fn(&T) -> f64) -> f64 {
    values.iter().map(map).sum()
}

fn make_f32_gradient(values: Vec<f32>, meta: arcqml_core::TensorMeta) -> Result<Tensor> {
    Tensor::from_storage_meta(Storage::F32(values), meta)
}
fn make_f64_gradient(values: Vec<f64>, meta: arcqml_core::TensorMeta) -> Result<Tensor> {
    Tensor::from_storage_meta(Storage::F64(values), meta)
}
fn make_c64_gradient(values: Vec<Complex64>, meta: arcqml_core::TensorMeta) -> Result<Tensor> {
    Tensor::from_storage_meta(Storage::C64(values), meta)
}

fn nondifferentiable_dtype<T>(dtype: arcqml_core::DType) -> Result<T> {
    Err(ArcQmlError::AutogradError(format!(
        "dtype {} does not support linalg backward",
        dtype
    )))
}
