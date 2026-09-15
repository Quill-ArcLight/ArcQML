/// 电路参数表中的零起始索引。
///
/// 标识只在创建它的电路及由追加操作返回的映射范围内有效；构造本身不检查边界。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterId {
    index: usize,
}

impl ParameterId {
    /// 从零起始索引创建标识，不执行电路边界检查。
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// 返回零起始索引。
    pub fn index(self) -> usize {
        self.index
    }
}

/// 量子门的固定数值或对电路可训练参数的引用。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gateparam {
    /// 直接存储在门中的固定 `f64` 数值，不进入电路参数表。
    Fixed(f64),
    /// 通过 [`ParameterId`] 引用电路参数表中的可训练值。
    Param(ParameterId),
}

impl Gateparam {
    /// 创建直接嵌入量子门的固定数值。
    pub fn fixed(value: f64) -> Self {
        Self::Fixed(value)
    }

    /// 创建对电路参数表中 `id` 的引用。
    pub fn param(id: ParameterId) -> Self {
        Self::Param(id)
    }

    /// 返回参数是否是固定参数。
    pub fn is_fixed(&self) -> bool {
        matches!(self, Gateparam::Fixed(_))
    }

    /// 判断参数是否引用电路参数表。
    pub fn is_parameter(&self) -> bool {
        matches!(self, Gateparam::Param(_))
    }

    /// 返回嵌入的固定值；参数引用返回 `None`。
    pub fn as_fixed(&self) -> Option<f64> {
        match self {
            Gateparam::Fixed(value) => Some(*value),
            Gateparam::Param(_) => None,
        }
    }

    /// 返回引用的参数标识；固定值返回 `None`。
    pub fn as_parameter(&self) -> Option<ParameterId> {
        match self {
            Gateparam::Fixed(_) => None,
            Gateparam::Param(id) => Some(*id),
        }
    }
}

impl From<f64> for Gateparam {
    fn from(value: f64) -> Self {
        Gateparam::Fixed(value)
    }
}

impl From<ParameterId> for Gateparam {
    fn from(id: ParameterId) -> Self {
        Gateparam::Param(id)
    }
}
