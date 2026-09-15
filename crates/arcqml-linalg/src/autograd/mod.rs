mod complex;
mod extra_elementwise;
mod functions;
/// 矩阵乘法的向量-雅可比积。
pub mod matmul_backward;
mod normalized;
/// 全量归约算子的向量-雅可比积。
pub mod reductions_backward;
mod reductions_dim;
/// reshape 视图的向量-雅可比积。
pub mod reshape_backward;
mod segment;
/// 二维转置的向量-雅可比积。
pub mod transpose_backward;

pub use matmul_backward::matmul_backward_shapes;
pub use reductions_backward::reduction_backward_shape;
pub use reshape_backward::reshape_backward_shape;
pub use transpose_backward::transpose_backward_shape;

pub(crate) use complex::ConjBackward;
pub(crate) use extra_elementwise::{
    ClampBackward, DivBackward, ExtraUnaryBackward, ExtraUnaryKind,
};
pub(crate) use functions::{
    BinaryBackward, BinaryKind, DotBackward, ExtremumBackward, ExtremumKind, L2NormBackward,
    MatmulBackward, ReductionBackward, ReductionKind, UnaryBackward, UnaryKind,
};
pub(crate) use normalized::{LogSoftmaxBackward, LogSumExpBackward, SoftmaxBackward};
pub(crate) use reductions_dim::{DimReductionBackward, DimReductionKind};
pub(crate) use segment::SegmentSumBackward;
