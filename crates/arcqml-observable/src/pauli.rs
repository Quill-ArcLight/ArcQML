use num_complex::Complex64;

/// 单量子比特 Pauli 算符，包括恒等算符。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Pauli {
    /// 恒等算符。
    I,
    /// Pauli-X 算符。
    X,
    /// Pauli-Y 算符。
    Y,
    /// Pauli-Z 算符。
    Z,
}

impl Pauli {
    /// 返回 Pauli 名称。
    pub fn name(&self) -> &'static str {
        match self {
            Pauli::I => "I",
            Pauli::X => "X",
            Pauli::Y => "Y",
            Pauli::Z => "Z",
        }
    }

    /// 判断是否是单位算符。
    pub fn is_identity(&self) -> bool {
        matches!(self, Pauli::I)
    }

    /// 返回对应的 `2 × 2` 复矩阵，元素按行主序排列。
    pub fn matrix(&self) -> [Complex64; 4] {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let neg_one = Complex64::new(-1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let neg_i = Complex64::new(0.0, -1.0);

        match self {
            Pauli::I => [one, zero, zero, one],
            Pauli::X => [zero, one, one, zero],
            Pauli::Y => [zero, neg_i, i, zero],
            Pauli::Z => [one, zero, zero, neg_one],
        }
    }

    /// 按 `self × rhs` 的顺序计算乘积，返回 `(复相位, Pauli 算符)`。
    pub fn multiply(self, rhs: Pauli) -> (Complex64, Pauli) {
        let one = Complex64::new(1.0, 0.0);
        let i = Complex64::new(0.0, 1.0);
        let neg_i = Complex64::new(0.0, -1.0);

        match (self, rhs) {
            (Pauli::I, p) | (p, Pauli::I) => (one, p),
            (Pauli::X, Pauli::X) | (Pauli::Y, Pauli::Y) | (Pauli::Z, Pauli::Z) => (one, Pauli::I),
            (Pauli::X, Pauli::Y) => (i, Pauli::Z),
            (Pauli::Y, Pauli::X) => (neg_i, Pauli::Z),
            (Pauli::Y, Pauli::Z) => (i, Pauli::X),
            (Pauli::Z, Pauli::Y) => (neg_i, Pauli::X),
            (Pauli::Z, Pauli::X) => (i, Pauli::Y),
            (Pauli::X, Pauli::Z) => (neg_i, Pauli::Y),
        }
    }
}
