/// 从二元运算输入推导输出形状。
pub mod infer;
/// 连续行主序步长计算。
pub mod strides;
/// 算子使用的形状、轴和连续性校验。
pub mod validate;

pub use infer::{
    infer_broadcast_shape, infer_matmul_shape, infer_reduction_shape, infer_transpose_shape,
};
pub use strides::{contiguous_strides, is_contiguous, transposed_2d_strides};
pub use validate::{
    numel, validate_axis, validate_broadcast_shapes, validate_buffer_len, validate_matmul_shapes,
    validate_reshape_shape, validate_same_numel, validate_transpose_shape,
};
