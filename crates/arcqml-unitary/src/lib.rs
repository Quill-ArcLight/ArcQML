//! 从电路构造、比较和拟合整体稠密酉矩阵。
//!
//! [`DenseUnitary`] 使用行主序 `C64` 缓冲区，矩阵索引约定为 `U[输出, 输入]`。
//! [`unitary_loss`] 返回可微标量，并以解析伴随方法把梯度累积到电路参数。
//!
//! # 示例
//!
//! ```
//! use arcqml_circuit::Circuit;
//! use arcqml_unitary::{unitary_fidelity, unitary_from_circuit};
//!
//! let circuit = Circuit::new(1)?;
//! let unitary = unitary_from_circuit(&circuit)?;
//! assert!((unitary_fidelity(&unitary, &circuit)? - 1.0).abs() < 1e-12);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! 稠密矩阵需要 `4^n` 个复数，适合小规模线路分析；维度溢出和分配失败会返回错误。

mod dense;
mod dimension;
/// 酉矩阵构造、验证和拟合错误。
pub mod error;
mod evolve;
/// 电路酉矩阵的保真度、损失和解析梯度。
pub mod fit;

use arcqml_circuit::Circuit;
use arcqml_core::Tensor;
use dense::transpose_square;
use evolve::circuit_state;

pub use dense::DenseUnitary;
pub use error::{UnitaryError, UnitaryResult};
pub use fit::{unitary_fidelity, unitary_loss, unitary_loss_value};

/// [`unitary_from_tensor`] 验证 `U†U = I` 时使用的最大逐元素绝对误差。
pub const DEFAULT_UNITARY_TOLERANCE: f64 = 1e-10;

/// 按 `U[输出, 输入]` 行主序约定构建电路的整体稠密酉矩阵。
///
/// # Errors
///
/// 当电路无效、参数无法绑定、门执行失败、矩阵维度溢出或内存预留失败时返回错误。
pub fn unitary_from_circuit(circuit: &Circuit) -> UnitaryResult<DenseUnitary> {
    let columns = circuit_state(circuit)?;
    DenseUnitary::from_row_major(
        transpose_square(&columns.data, columns.dimension),
        columns.dimension,
    )
}

/// 从任意 `C64` 方阵 Tensor 复制数据，并使用 [`DEFAULT_UNITARY_TOLERANCE`] 验证酉性。
///
/// # Errors
///
/// 当 Tensor 不是有限的二维 `C64` 方阵、边长不是二的幂、无法连续化，或其
/// `U†U` 与单位矩阵的逐元素最大绝对误差超过默认容差时返回错误。
pub fn unitary_from_tensor(tensor: &Tensor) -> UnitaryResult<DenseUnitary> {
    let unitary = DenseUnitary::from_tensor(tensor)?;
    unitary.validate(DEFAULT_UNITARY_TOLERANCE)?;
    Ok(unitary)
}

/// 常用类型与函数的预导入模块。
pub mod prelude {
    pub use crate::{
        DEFAULT_UNITARY_TOLERANCE, DenseUnitary, UnitaryError, UnitaryResult, unitary_fidelity,
        unitary_from_circuit, unitary_from_tensor, unitary_loss, unitary_loss_value,
    };
}
