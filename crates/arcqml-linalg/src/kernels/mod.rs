/// 类型化逐元素连续缓冲区内核。
pub mod elementwise;
/// 通用矩阵-矩阵乘法内核。
pub mod gemm;
/// 通用矩阵-向量乘法内核。
pub mod gemv;
/// 全量归约内核。
pub mod reductions;

pub use elementwise::{
    abs_c64, abs_f32, abs_f64, add_c64, add_f32, add_f64, add_i64, mul_c64, mul_f32, mul_f64,
    mul_i64, sqrt_c64, sqrt_f32, sqrt_f64, square_c64, square_f32, square_f64, square_i64, sub_c64,
    sub_f32, sub_f64, sub_i64,
};
pub use gemm::{gemm_c64, gemm_f32, gemm_f64, gemm_i64};
pub use gemv::{gemv_c64, gemv_f32, gemv_f64, gemv_i64};
pub use reductions::{
    max_f32, max_f64, max_i64, mean_c64, mean_f32, mean_f64, mean_i64, min_f32, min_f64, min_i64,
    sum_c64, sum_f32, sum_f64, sum_i64,
};
