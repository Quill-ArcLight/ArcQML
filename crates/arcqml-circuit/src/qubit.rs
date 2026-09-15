/// 零起始的逻辑量子比特索引。
///
/// 构造标识本身不检查边界；将其加入电路或传给执行内核时才根据系统规模校验。
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Qubit {
    index: usize,
}

impl Qubit {
    /// 从零起始下标创建量子比特标识，不执行边界检查。
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// 返回零起始下标。
    pub fn index(&self) -> usize {
        self.index
    }
}

impl From<usize> for Qubit {
    fn from(index: usize) -> Self {
        Qubit::new(index)
    }
}
