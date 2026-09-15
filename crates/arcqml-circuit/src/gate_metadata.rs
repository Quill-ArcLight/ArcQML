use crate::{CircuitError::*, CircuitResult, Gate, Gateparam, ParameterId};
use arcqml_core::checked_power_of_two;
use num_complex::Complex64;

impl Gate {
    /// 返回量子门的稳定显示名称。
    pub fn name(&self) -> &str {
        match self {
            Self::I => "I",
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
            Self::H => "H",
            Self::S => "S",
            Self::Sdg => "Sdg",
            Self::T => "T",
            Self::Tdg => "Tdg",
            Self::Sx => "SX",
            Self::Sxdg => "SXdg",
            Self::Rx(_) => "Rx",
            Self::Ry(_) => "Ry",
            Self::Rz(_) => "Rz",
            Self::Phase(_) => "Phase",
            Self::U3(..) => "U3",
            Self::CNot => "CNot",
            Self::CY => "CY",
            Self::CZ => "CZ",
            Self::CH => "CH",
            Self::CS => "CS",
            Self::CT => "CT",
            Self::CPhase(_) => "CP",
            Self::CRx(_) => "CRx",
            Self::CRy(_) => "CRy",
            Self::CRz(_) => "CRz",
            Self::Swap => "Swap",
            Self::ISwap => "iSwap",
            Self::DCX => "DCX",
            Self::Ecr => "ECR",
            Self::Rxx(_) => "RXX",
            Self::Ryy(_) => "RYY",
            Self::Rzz(_) => "RZZ",
            Self::Rzx(_) => "RZX",
            Self::FSim(..) => "fSim",
            Self::Toffoli => "Toffoli",
            Self::CSwap => "CSwap",
            Self::MCX { .. } => "MCX",
            Self::CustomUnitary { name, .. } => name,
        }
    }

    /// 返回量子门的 arity，即一次操作必须提供的量子比特数量。
    pub fn arity(&self) -> usize {
        match self {
            Self::I
            | Self::X
            | Self::Y
            | Self::Z
            | Self::H
            | Self::S
            | Self::Sdg
            | Self::T
            | Self::Tdg
            | Self::Sx
            | Self::Sxdg
            | Self::Rx(_)
            | Self::Ry(_)
            | Self::Rz(_)
            | Self::Phase(_)
            | Self::U3(..) => 1,
            Self::CNot
            | Self::CY
            | Self::CZ
            | Self::CH
            | Self::CS
            | Self::CT
            | Self::CPhase(_)
            | Self::CRx(_)
            | Self::CRy(_)
            | Self::CRz(_)
            | Self::Swap
            | Self::ISwap
            | Self::DCX
            | Self::Ecr
            | Self::Rxx(_)
            | Self::Ryy(_)
            | Self::Rzz(_)
            | Self::Rzx(_)
            | Self::FSim(..) => 2,
            Self::Toffoli | Self::CSwap => 3,
            Self::MCX { controls } => controls + 1,
            Self::CustomUnitary { arity, .. } => *arity,
        }
    }

    /// 按门参数槽位顺序返回固定值或电路参数引用。
    ///
    /// `U3` 的顺序为 `(θ, φ, λ)`，`fSim` 的顺序为 `(θ, φ)`；非参数门返回空向量。
    pub fn parameters(&self) -> Vec<Gateparam> {
        match self {
            Self::Rx(p)
            | Self::Ry(p)
            | Self::Rz(p)
            | Self::Phase(p)
            | Self::CPhase(p)
            | Self::CRx(p)
            | Self::CRy(p)
            | Self::CRz(p)
            | Self::Rxx(p)
            | Self::Ryy(p)
            | Self::Rzz(p)
            | Self::Rzx(p) => vec![*p],
            Self::U3(theta, phi, lambda) => vec![*theta, *phi, *lambda],
            Self::FSim(theta, phi) => vec![*theta, *phi],
            _ => Vec::new(),
        }
    }

    /// 按参数槽位顺序返回门引用的电路参数标识，忽略固定参数。
    pub fn parameter_ids(&self) -> Vec<ParameterId> {
        self.parameters()
            .iter()
            .filter_map(Gateparam::as_parameter)
            .collect()
    }
    /// 判断门是否至少包含一个 [`Gateparam::Param`] 槽位。
    pub fn is_parameterized(&self) -> bool {
        !self.parameter_ids().is_empty()
    }

    /// 验证自定义酉门的维度与酉性。
    ///
    /// # Errors
    ///
    /// 仅 [`Gate::CustomUnitary`] 可能失败：当维度计算溢出、矩阵长度错误，
    /// 或矩阵含非有限值或在绝对误差 `1e-12` 下不满足酉性时返回错误。
    pub fn validate(&self) -> CircuitResult<()> {
        if let Self::CustomUnitary {
            name,
            arity,
            matrix,
        } = self
        {
            let dim = checked_power_of_two(*arity)
                .ok_or(UnitaryDimensionOverflowError { arity: *arity })?;
            let expected = dim
                .checked_mul(dim)
                .ok_or(UnitaryDimensionOverflowError { arity: *arity })?;
            if matrix.len() != expected {
                return Err(InvalidUnitaryShapeError {
                    name: name.clone(),
                    arity: *arity,
                    expected,
                    actual: matrix.len(),
                });
            }
            for row in 0..dim {
                for col in 0..dim {
                    let value = (0..dim)
                        .map(|index| matrix[index * dim + row].conj() * matrix[index * dim + col])
                        .sum::<Complex64>();
                    let expected = if row == col {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    if !value.re.is_finite()
                        || !value.im.is_finite()
                        || (value - expected).norm() > 1e-12
                    {
                        return Err(NonUnitaryMatrixError {
                            name: name.clone(),
                            arity: *arity,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}
