use crate::statevector::shared::dimension::dimension;
use crate::{SimError::*, SimResult};
use arcqml_core::{DType, Device, Layout, Storage, Tensor, TensorData};
use num_complex::Complex64;

/// 创建 `|0…0⟩` 计算基态。
pub(crate) fn zero_state_tensor(num_qubits: usize) -> SimResult<Tensor> {
    let dim = dimension(num_qubits)?;
    let mut amplitudes = vec![Complex64::new(0.0, 0.0); dim];
    amplitudes[0] = Complex64::new(1.0, 0.0);
    state_tensor_from_amplitudes(num_qubits, &amplitudes)
}

/// 校验 Tensor 状态，并复制出连续的 `C64` 振幅。
pub(crate) fn tensor_amplitudes(num_qubits: usize, tensor: &Tensor) -> SimResult<Vec<Complex64>> {
    validate_state_tensor(num_qubits, tensor)?;
    tensor_values(num_qubits, tensor)
}

/// 校验 Tensor 是否为连续、归一化的 `C64` 状态向量，但不复制振幅数据。
pub(crate) fn validate_state_tensor(num_qubits: usize, tensor: &Tensor) -> SimResult<()> {
    let expected = dimension(num_qubits)?;

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
    if tensor.shape() != [expected] {
        return Err(TensorStateShapeError {
            expected,
            actual: tensor.shape().to_vec(),
        });
    }
    match &*tensor.storage() {
        Storage::C64(values) => validate_normalized(values),
        _ => unreachable!("C64 metadata and storage must agree"),
    }
}

/// 读取 `C64` Tensor 的数值，不要求归一化；用于反向传播读取 `grad_output` 等场景。
pub(crate) fn tensor_values(num_qubits: usize, tensor: &Tensor) -> SimResult<Vec<Complex64>> {
    let expected = dimension(num_qubits)?;

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
    if tensor.shape() != [expected] {
        return Err(TensorStateShapeError {
            expected,
            actual: tensor.shape().to_vec(),
        });
    }

    match &*tensor.storage() {
        Storage::C64(values) => Ok(values.clone()),
        _ => unreachable!("C64 metadata and storage must agree"),
    }
}

/// 通过输入振幅构造 Tensor。
pub(crate) fn state_tensor_from_amplitudes(
    num_qubits: usize,
    amplitudes: &[Complex64],
) -> SimResult<Tensor> {
    let expected = dimension(num_qubits)?;
    if amplitudes.len() != expected {
        return Err(InvalidStateLengthError {
            num_qubits,
            expected,
            actual: amplitudes.len(),
        });
    }
    validate_normalized(amplitudes)?;

    Tensor::new(TensorData::FlatC64 {
        data: amplitudes.to_vec(),
        shape: vec![expected],
    })
    .map_err(|error| TensorError {
        message: error.to_string(),
    })
}

/// 根据已有 C64 Tensor 的元信息创建相同 shape 的 C64 梯度 Tensor。
pub(crate) fn gradient_tensor_like(
    parent: &Tensor,
    values: Vec<Complex64>,
) -> arcqml_core::Result<Tensor> {
    Tensor::from_storage_meta(Storage::C64(values), parent.meta().clone())
}

/// 检验振幅是否归一化。
fn validate_normalized(amplitudes: &[Complex64]) -> SimResult<()> {
    let norm_sqr: f64 = amplitudes.iter().map(Complex64::norm_sqr).sum();
    if (norm_sqr - 1.0).abs() > 1e-10 {
        return Err(NonNormalizedStateError { norm_sqr });
    }
    Ok(())
}
