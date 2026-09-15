use crate::{
    UnitaryError, UnitaryResult, dense::validate_unitary, dimension::qubits_from_dimension,
};
use arcqml_core::{DType, Tensor};
use num_complex::Complex64;

/// 保存行主序 `C64` 方阵及其量子比特元数据。
///
/// [`DenseUnitary::from_row_major`] 和 [`DenseUnitary::from_tensor`] 只校验表示格式与
/// 有限数值；需要保证矩阵满足酉性时，应继续调用 [`DenseUnitary::validate`]，或使用
/// crate 根部的 [`unitary_from_tensor`](crate::unitary_from_tensor)。
#[derive(Debug, Clone, PartialEq)]
pub struct DenseUnitary {
    num_qubits: usize,
    dimension: usize,
    data: Vec<Complex64>,
}

impl DenseUnitary {
    /// 从行主序复数方阵数据创建容器，不执行酉性校验。
    ///
    /// # Errors
    ///
    /// 当 `dimension` 为零或不是二的幂、元素数计算溢出、`data` 长度不等于
    /// `dimension²`，或任一复数分量不是有限数时返回错误。
    pub fn from_row_major(data: Vec<Complex64>, dimension: usize) -> UnitaryResult<Self> {
        let num_qubits = qubits_from_dimension(dimension)?;
        let expected = dimension
            .checked_mul(dimension)
            .ok_or(UnitaryError::DimensionOverflowError { num_qubits })?;
        if data.len() != expected {
            return Err(UnitaryError::InvalidMatrixLengthError {
                expected,
                actual: data.len(),
            });
        }
        if data
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(UnitaryError::NonFiniteMatrixError);
        }
        Ok(Self {
            num_qubits,
            dimension,
            data,
        })
    }

    /// 从 `C64` 二维方阵 Tensor 按逻辑行主序复制数据，不执行酉性校验。
    ///
    /// # Errors
    ///
    /// 当 Tensor dtype 不是 `C64`、形状不是边长为二的幂的方阵、无法连续化，
    /// 或复制出的矩阵含非有限数值时返回错误。
    pub fn from_tensor(tensor: &Tensor) -> UnitaryResult<Self> {
        if tensor.dtype() != DType::C64 {
            return Err(UnitaryError::InvalidMatrixDTypeError {
                actual: tensor.dtype().to_string(),
            });
        }
        if tensor.shape().len() != 2 || tensor.shape()[0] != tensor.shape()[1] {
            return Err(UnitaryError::InvalidMatrixShapeError {
                shape: tensor.shape().to_vec(),
            });
        }
        let contiguous = tensor
            .contiguous()
            .map_err(|error| UnitaryError::TensorError {
                message: error.to_string(),
            })?;
        let dimension = contiguous.shape()[0];
        let data = contiguous
            .storage()
            .as_c64_slice()
            .ok_or_else(|| UnitaryError::InvalidMatrixDTypeError {
                actual: contiguous.dtype().to_string(),
            })?
            .to_vec();
        Self::from_row_major(data, dimension)
    }

    /// 返回该酉矩阵作用的量子比特数。
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// 返回矩阵边长 `2^n`，其中 `n` 为 [`DenseUnitary::num_qubits`]。
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// 返回公开的行主序矩阵数据，索引为 `U[output, input]`。
    pub fn as_row_major(&self) -> &[Complex64] {
        &self.data
    }

    /// 检查矩阵是否满足 `U†U = I`。
    ///
    /// # Errors
    ///
    /// 当 `tolerance` 不是有限非负数，或 `U†U` 与单位矩阵的逐元素最大绝对误差
    /// 大于 `tolerance` 时返回错误。
    pub fn validate(&self, tolerance: f64) -> UnitaryResult<()> {
        validate_unitary(self, tolerance)
    }
}
