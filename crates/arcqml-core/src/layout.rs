use std::fmt;

/// Tensor 的存储布局类别。
///
/// 当前仅 [`Layout::Dense`] 具备存储和算子实现；[`Layout::Sparse`] 是预留标识。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// 由形状和步长描述的稠密存储。
    Dense,
    /// 稀疏存储占位；当前没有对应的存储或计算实现。
    Sparse,
}

impl Default for Layout {
    /// 返回当前默认布局 [`Layout::Dense`]。
    fn default() -> Self {
        Layout::Dense
    }
}

impl Layout {
    /// 返回存储布局名称。
    pub fn name(&self) -> &'static str {
        match self {
            Layout::Dense => "dense",
            Layout::Sparse => "sparse",
        }
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
