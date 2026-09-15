mod api;
pub(crate) mod execution;
pub(crate) mod shared;
pub(crate) mod state;

pub use api::measurement::MeasurementCounts;

/// 单个纯态的状态向量模拟器。
pub mod single {
    pub use super::api::single::StateVectorSimulator;
}

/// 共享电路的纯态 batch 模拟器。
pub mod batch {
    pub use super::api::batch::BatchStateVectorSimulator;
}
