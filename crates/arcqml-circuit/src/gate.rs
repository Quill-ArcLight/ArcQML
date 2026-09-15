use crate::{CircuitResult, Gateparam};
use num_complex::Complex64;

/// 可加入 [`Circuit`](crate::Circuit) 的内置量子门或自定义酉门。
///
/// 所有角参数均以弧度表示。旋转门采用
/// `R_P(θ) = cos(θ/2) I - i sin(θ/2) P`，其中 `P` 为对应的 Pauli
/// 算符或 Pauli 张量积。多量子比特门的局部基索引采用小端序：传给
/// [`Operation`](crate::Operation) 的第一个量子比特对应局部索引的最低位。
/// 对受控门而言，第一个量子比特是控制位，第二个量子比特是目标位。
#[derive(Debug, Clone, PartialEq)]
pub enum Gate {
    /// 单量子比特恒等门。
    I,
    /// Pauli-X 门。
    X,
    /// Pauli-Y 门。
    Y,
    /// Pauli-Z 门。
    Z,
    /// Hadamard 门。
    H,
    /// 相位门 S。
    S,
    /// S 门的共轭转置。
    Sdg,
    /// π/4 相位门 T。
    T,
    /// T 门的共轭转置。
    Tdg,
    /// Pauli-X 的主平方根门，满足 `Sx · Sx = X`。
    Sx,
    /// `Sx` 的共轭转置。
    Sxdg,
    /// X 轴旋转 `Rx(θ) = exp(-i θ X / 2)`。
    Rx(Gateparam),
    /// Y 轴旋转 `Ry(θ) = exp(-i θ Y / 2)`。
    Ry(Gateparam),
    /// Z 轴旋转 `Rz(θ) = exp(-i θ Z / 2)`。
    Rz(Gateparam),
    /// 相位门 `diag(1, exp(iθ))`。
    Phase(Gateparam),
    /// 通用单量子比特门。
    ///
    /// 参数顺序为 `(θ, φ, λ)`，矩阵为
    /// `[[cos(θ/2), -exp(iλ)sin(θ/2)], [exp(iφ)sin(θ/2), exp(i(φ+λ))cos(θ/2)]]`。
    U3(Gateparam, Gateparam, Gateparam),
    /// 受控 Pauli-X 门，等价于 CNOT/CX。
    CNot,
    /// 受控 Pauli-Y 门。
    CY,
    /// 受控 Pauli-Z 门。
    CZ,
    /// 受控 Hadamard 门。
    CH,
    /// 受控 S 门。
    CS,
    /// 受控 T 门。
    CT,
    /// 受控相位门 `diag(1, 1, 1, exp(iθ))`。
    CPhase(Gateparam),
    /// 受控 X 轴旋转门。
    CRx(Gateparam),
    /// 受控 Y 轴旋转门。
    CRy(Gateparam),
    /// 受控 Z 轴旋转门。
    CRz(Gateparam),
    /// 交换两个量子比特的门。
    Swap,
    /// 带 `i` 相位的交换门。
    ISwap,
    /// Double-CNOT 双量子比特门。
    DCX,
    /// Echoed cross-resonance 双量子比特门。
    Ecr,
    /// 双量子比特旋转 `Rxx(θ) = exp(-i θ X⊗X / 2)`。
    Rxx(Gateparam),
    /// 双量子比特旋转 `Ryy(θ) = exp(-i θ Y⊗Y / 2)`。
    Ryy(Gateparam),
    /// 双量子比特旋转 `Rzz(θ) = exp(-i θ Z⊗Z / 2)`。
    Rzz(Gateparam),
    /// 双量子比特旋转 `Rzx(θ) = exp(-i θ Z⊗X / 2)`。
    ///
    /// `Z` 作用于第一个局部量子比特，`X` 作用于第二个局部量子比特。
    Rzx(Gateparam),
    /// 参数为 `(θ, φ)` 的 fSim 双量子比特门。
    ///
    /// 在局部基索引 `0, 1, 2, 3` 上，矩阵为
    /// `[[1,0,0,0], [0,cosθ,-i sinθ,0], [0,-i sinθ,cosθ,0], [0,0,0,exp(-iφ)]]`。
    FSim(Gateparam, Gateparam),
    /// 双控制 Pauli-X 门。
    Toffoli,
    /// 受控交换门，也称 Fredkin 门。
    CSwap,
    /// 任意控制数量的多控制 Pauli-X 门。
    MCX {
        /// 控制量子比特数量；操作总 arity 为 `controls + 1`。
        controls: usize,
    },
    /// 调用方提供的固定稠密酉矩阵。
    CustomUnitary {
        /// 用于显示和错误信息的门名称。
        name: String,
        /// 该门作用的量子比特数量。
        arity: usize,
        /// 长度为 `4^arity` 的行主序 `C64` 方阵。
        matrix: Vec<Complex64>,
    },
}

impl Gate {
    /// 返回单量子比特恒等门。
    pub fn i() -> Self {
        Self::I
    }
    /// 创建 Pauli-X 门。
    pub fn x() -> Self {
        Self::X
    }
    /// 创建 Pauli-Y 门。
    pub fn y() -> Self {
        Self::Y
    }
    /// 创建 Pauli-Z 门。
    pub fn z() -> Self {
        Self::Z
    }
    /// 创建 Hadamard 门。
    pub fn h() -> Self {
        Self::H
    }
    /// 创建 S 门。
    pub fn s() -> Self {
        Self::S
    }
    /// 创建 S 的逆门。
    pub fn sdg() -> Self {
        Self::Sdg
    }
    /// 创建 T 门。
    pub fn t() -> Self {
        Self::T
    }
    /// 创建 T 的逆门。
    pub fn tdg() -> Self {
        Self::Tdg
    }
    /// 创建 SX 门。
    pub fn sx() -> Self {
        Self::Sx
    }
    /// 创建 SX 的逆门。
    pub fn sxdg() -> Self {
        Self::Sxdg
    }
    /// 创建弧度参数为 `theta` 的 [`Gate::Rx`] 门。
    pub fn rx(theta: impl Into<Gateparam>) -> Self {
        Self::Rx(theta.into())
    }
    /// 创建弧度参数为 `theta` 的 [`Gate::Ry`] 门。
    pub fn ry(theta: impl Into<Gateparam>) -> Self {
        Self::Ry(theta.into())
    }
    /// 创建弧度参数为 `theta` 的 [`Gate::Rz`] 门。
    pub fn rz(theta: impl Into<Gateparam>) -> Self {
        Self::Rz(theta.into())
    }
    /// 创建弧度参数为 `theta` 的 [`Gate::Phase`] 门。
    pub fn phase(theta: impl Into<Gateparam>) -> Self {
        Self::Phase(theta.into())
    }
    /// 按 `(theta, phi, lambda)` 顺序创建 [`Gate::U3`] 门；三个参数均为弧度。
    pub fn u3(
        theta: impl Into<Gateparam>,
        phi: impl Into<Gateparam>,
        lambda: impl Into<Gateparam>,
    ) -> Self {
        Self::U3(theta.into(), phi.into(), lambda.into())
    }
    /// 创建 `U1(lambda) = Phase(lambda)` 门。
    pub fn u1(lambda: impl Into<Gateparam>) -> Self {
        Self::phase(lambda)
    }
    /// 创建 `U2(phi, lambda) = U3(π/2, phi, lambda)` 门。
    pub fn u2(phi: impl Into<Gateparam>, lambda: impl Into<Gateparam>) -> Self {
        Self::u3(std::f64::consts::FRAC_PI_2, phi, lambda)
    }
    /// 创建受控 X 门；操作中的第一个量子比特为控制位。
    pub fn cnot() -> Self {
        Self::CNot
    }
    /// 返回 [`Gate::CNot`]，作为 CX 的同义构造函数。
    pub fn cx() -> Self {
        Self::CNot
    }
    /// 创建 CY 门。
    pub fn cy() -> Self {
        Self::CY
    }
    /// 创建 CZ 门。
    pub fn cz() -> Self {
        Self::CZ
    }
    /// 创建 CH 门。
    pub fn ch() -> Self {
        Self::CH
    }
    /// 创建 CS 门。
    pub fn cs() -> Self {
        Self::CS
    }
    /// 创建 CT 门。
    pub fn ct() -> Self {
        Self::CT
    }
    /// 创建弧度参数为 `theta` 的受控相位门。
    pub fn cphase(theta: impl Into<Gateparam>) -> Self {
        Self::CPhase(theta.into())
    }
    /// 创建受控相位门，等价于 [`Gate::cphase`]。
    pub fn cp(theta: impl Into<Gateparam>) -> Self {
        Self::cphase(theta)
    }
    /// 创建弧度参数为 `theta` 的受控 RX 门。
    pub fn crx(theta: impl Into<Gateparam>) -> Self {
        Self::CRx(theta.into())
    }
    /// 创建弧度参数为 `theta` 的受控 RY 门。
    pub fn cry(theta: impl Into<Gateparam>) -> Self {
        Self::CRy(theta.into())
    }
    /// 创建弧度参数为 `theta` 的受控 RZ 门。
    pub fn crz(theta: impl Into<Gateparam>) -> Self {
        Self::CRz(theta.into())
    }
    /// 创建交换门。
    pub fn swap() -> Self {
        Self::Swap
    }
    /// 创建 iSWAP 门。
    pub fn iswap() -> Self {
        Self::ISwap
    }
    /// 创建 DCX 门。
    pub fn dcx() -> Self {
        Self::DCX
    }
    /// 创建 ECR 门。
    pub fn ecr() -> Self {
        Self::Ecr
    }
    /// 创建弧度参数为 `theta` 的 [`Gate::Rxx`] 门。
    pub fn rxx(theta: impl Into<Gateparam>) -> Self {
        Self::Rxx(theta.into())
    }
    /// 创建弧度参数为 `theta` 的 [`Gate::Ryy`] 门。
    pub fn ryy(theta: impl Into<Gateparam>) -> Self {
        Self::Ryy(theta.into())
    }
    /// 创建弧度参数为 `theta` 的 [`Gate::Rzz`] 门。
    pub fn rzz(theta: impl Into<Gateparam>) -> Self {
        Self::Rzz(theta.into())
    }
    /// 创建弧度参数为 `theta` 的 [`Gate::Rzx`] 门。
    pub fn rzx(theta: impl Into<Gateparam>) -> Self {
        Self::Rzx(theta.into())
    }
    /// 按 `(theta, phi)` 顺序创建 [`Gate::FSim`] 门；两个参数均为弧度。
    pub fn fsim(theta: impl Into<Gateparam>, phi: impl Into<Gateparam>) -> Self {
        Self::FSim(theta.into(), phi.into())
    }
    /// 创建 Toffoli 门。
    pub fn toffoli() -> Self {
        Self::Toffoli
    }
    /// 创建受控交换门。
    pub fn cswap() -> Self {
        Self::CSwap
    }
    /// 创建具有 `controls` 个控制位和一个目标位的多控制 X 门。
    pub fn mcx(controls: usize) -> Self {
        Self::MCX { controls }
    }
    /// 创建并验证自定义酉门。
    ///
    /// `matrix` 是维度为 `2^arity × 2^arity` 的行主序矩阵；局部基索引中的
    /// 第 `k` 位对应操作量子比特切片中的第 `k` 个量子比特。
    ///
    /// # Errors
    ///
    /// 当 `2^arity` 或矩阵元素数溢出、`matrix` 长度不等于 `4^arity`，或矩阵
    /// 含非有限值或在绝对误差 `1e-12` 下不满足酉性时返回错误。
    pub fn custom_unitary(
        name: impl Into<String>,
        arity: usize,
        matrix: Vec<Complex64>,
    ) -> CircuitResult<Self> {
        let gate = Self::CustomUnitary {
            name: name.into(),
            arity,
            matrix,
        };
        gate.validate()?;
        Ok(gate)
    }
}
