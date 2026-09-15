use crate::statevector::shared::dimension::dimension;
use crate::{SimError::*, SimResult};
use arcqml_core::{DType, Device, Layout, Storage, Tensor, TensorData};
use num_complex::Complex64;

/// 创建 `batch_size` 份 `|0…0⟩` 状态，形状为 `[batch_size, 2^num_qubits]`。
pub(crate) fn zero_state_tensor(num_qubits: usize, batch_size: usize) -> SimResult<Tensor> {
    let dimension = dimension(num_qubits)?;
    validate_batch_size(batch_size)?;
    let len = dimension
        .checked_mul(batch_size)
        .ok_or(StateDimensionOverflowError { num_qubits })?;
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); len];
    for row in amplitudes.chunks_exact_mut(dimension) {
        row[0] = Complex64::new(1.0, 0.0);
    }
    state_tensor_from_amplitudes(num_qubits, batch_size, &amplitudes)
}

/// 校验一批已归一化的 C64 状态，并返回其行主序数据。
pub(crate) fn tensor_amplitudes(
    num_qubits: usize,
    tensor: &Tensor,
) -> SimResult<(usize, Vec<Complex64>)> {
    let batch_size = validate_state_tensor(num_qubits, tensor)?;
    match &*tensor.storage() {
        Storage::C64(values) => Ok((batch_size, values.clone())),
        _ => unreachable!("C64 metadata and storage must agree"),
    }
}

/// 校验 batch 状态 Tensor 的布局、形状和每一行归一化，但不复制振幅数据。
pub(crate) fn validate_state_tensor(num_qubits: usize, tensor: &Tensor) -> SimResult<usize> {
    let dimension = dimension(num_qubits)?;
    let batch_size = validate_tensor_layout(tensor, dimension)?;
    match &*tensor.storage() {
        Storage::C64(values) => validate_normalized(batch_size, dimension, values)?,
        _ => unreachable!("C64 metadata and storage must agree"),
    }
    Ok(batch_size)
}

/// 读取形状为 `[batch_size, 2^num_qubits]` 的 `C64` Tensor，不要求逐行归一化。
pub(crate) fn tensor_values(
    num_qubits: usize,
    tensor: &Tensor,
) -> SimResult<(usize, Vec<Complex64>)> {
    let dimension = dimension(num_qubits)?;
    let batch_size = validate_tensor_layout(tensor, dimension)?;

    match &*tensor.storage() {
        Storage::C64(values) => Ok((batch_size, values.clone())),
        _ => unreachable!("C64 metadata and storage must agree"),
    }
}

/// 根据行主序振幅构造逐行归一化的 `[batch_size, 2^num_qubits]` `C64` Tensor。
pub(crate) fn state_tensor_from_amplitudes(
    num_qubits: usize,
    batch_size: usize,
    amplitudes: &[Complex64],
) -> SimResult<Tensor> {
    let dimension = dimension(num_qubits)?;
    validate_batch_size(batch_size)?;
    let expected = dimension
        .checked_mul(batch_size)
        .ok_or(StateDimensionOverflowError { num_qubits })?;
    if amplitudes.len() != expected {
        return Err(BatchTensorStateShapeError {
            expected: dimension,
            actual: vec![amplitudes.len()],
        });
    }
    validate_normalized(batch_size, dimension, amplitudes)?;

    Tensor::new(TensorData::FlatC64 {
        data: amplitudes.to_vec(),
        shape: vec![batch_size, dimension],
    })
    .map_err(|error| TensorError {
        message: error.to_string(),
    })
}

/// 创建与父 batch 状态 Tensor 匹配的 C64 梯度 Tensor。
pub(crate) fn gradient_tensor_like(
    parent: &Tensor,
    values: Vec<Complex64>,
) -> arcqml_core::Result<Tensor> {
    Tensor::from_storage_meta(Storage::C64(values), parent.meta().clone())
}

/// 验证 batch 大小非零。
fn validate_batch_size(batch_size: usize) -> SimResult<()> {
    if batch_size == 0 {
        return Err(EmptyBatchError);
    }
    Ok(())
}

/// 验证 batch 状态 Tensor 的 dtype、设备、布局和二维形状。
fn validate_tensor_layout(tensor: &Tensor, expected_dimension: usize) -> SimResult<usize> {
    if tensor.dtype() != DType::C64 {
        return Err(TensorStateDTypeError {
            dtype: tensor.dtype().to_string(),
        });
    }
    if tensor.device() != &Device::Cpu || tensor.layout() != Layout::Dense {
        return Err(UnsupportedTensorStateLayoutError);
    }
    if !tensor.is_contiguous() {
        return Err(NonContiguousTensorStateError);
    }
    if tensor.shape().len() != 2 || tensor.shape()[1] != expected_dimension {
        return Err(BatchTensorStateShapeError {
            expected: expected_dimension,
            actual: tensor.shape().to_vec(),
        });
    }

    let batch_size = tensor.shape()[0];
    validate_batch_size(batch_size)?;
    Ok(batch_size)
}

/// 验证 batch 中每一行状态向量均已归一化。
fn validate_normalized(
    batch_size: usize,
    dimension: usize,
    amplitudes: &[Complex64],
) -> SimResult<()> {
    for (batch_index, row) in amplitudes.chunks_exact(dimension).enumerate() {
        let norm_sqr: f64 = row.iter().map(Complex64::norm_sqr).sum();
        if (norm_sqr - 1.0).abs() > 1e-10 {
            return Err(NonNormalizedBatchStateError {
                batch_index,
                norm_sqr,
            });
        }
    }
    debug_assert_eq!(batch_size, amplitudes.len() / dimension);
    Ok(())
}
