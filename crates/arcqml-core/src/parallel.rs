use crate::{ArcQmlError, Result};

/// 初始化 Rayon 的全局工作线程池。
#[cfg(feature = "parallel")]
///
/// # Errors
///
/// 当线程数为零，或 Rayon 全局线程池已经初始化且配置不兼容时返回错误。
pub fn init_rayon(num_threads: usize) -> Result<()> {
    if num_threads == 0 {
        return Err(ArcQmlError::InvalidOperationError(
            "Rayon thread count must be greater than zero".to_string(),
        ));
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .map_err(|error| {
            ArcQmlError::InvalidOperationError(format!(
                "failed to initialize Rayon global thread pool: {error}; initialize it before running parallel work"
            ))
        })
}

/// 返回 Rayon 全局池中的工作线程数量。
#[cfg(feature = "parallel")]
pub fn rayon_num_threads() -> usize {
    rayon::current_num_threads()
}

/// 在没有并行支持的情况下构建则返回错误。
#[cfg(not(feature = "parallel"))]
pub fn init_rayon(_: usize) -> Result<()> {
    Err(ArcQmlError::NotImplementedError(
        "Rayon support is disabled; enable arcqml-core's `parallel` feature".to_string(),
    ))
}

/// 非并行构建总是在调用线程上执行。
#[cfg(not(feature = "parallel"))]
pub fn rayon_num_threads() -> usize {
    1
}
