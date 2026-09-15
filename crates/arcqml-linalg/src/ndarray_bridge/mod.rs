/// Tensor 与拥有所有权的 `ndarray::ArrayD` 相互转换。
pub mod convert;
/// 在 Tensor 存储锁生命周期内创建只读 ndarray 视图。
pub mod view;

pub use convert::{
    from_arrayd_bool, from_arrayd_c64, from_arrayd_f32, from_arrayd_f64, from_arrayd_i64,
    to_arrayd_bool, to_arrayd_c64, to_arrayd_f32, to_arrayd_f64, to_arrayd_i64,
};
pub use view::{
    with_array_view_bool, with_array_view_c64, with_array_view_f32, with_array_view_f64,
    with_array_view_i64,
};
