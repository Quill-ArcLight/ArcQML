use std::fmt;

/// Tensor 所在的计算设备。
///
/// 当前仅 [`Device::Cpu`] 具备存储和算子实现；构造 Tensor 元数据时使用
/// [`Device::Cuda`] 会返回未实现错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Device {
    /// 主机 CPU；当前唯一具备计算实现的设备。
    Cpu,
    /// 指定编号的 CUDA 设备；当前仅作为元数据占位，不提供 CUDA 内核。
    Cuda(usize),
}

impl Default for Device {
    /// 返回当前默认设备 [`Device::Cpu`]。
    fn default() -> Self {
        Device::Cpu
    }
}

impl Device {
    /// 返回设备名称。
    pub fn name(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(id) => format!("cuda:{}", id),
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
