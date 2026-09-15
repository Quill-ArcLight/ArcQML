use std::fmt;

/// Tensor 元素的数据类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    /// IEEE 754 单精度浮点数。
    F32,
    /// IEEE 754 双精度浮点数。
    F64,
    /// 由两个 `f64` 分量组成的双精度复数 [`num_complex::Complex64`]。
    C64,
    /// 64 位有符号整数。
    I64,
    /// 布尔值。
    Bool,
}

impl DType {
    /// 返回类型名称。
    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::C64 => "complex64",
            DType::I64 => "i64",
            DType::Bool => "bool",
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
