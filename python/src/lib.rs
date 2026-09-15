//! ArcQML 的 CPython 原生扩展模块。
//!
//! 本 crate 使用 PyO3 把稳定的通用 API 注册为 Python 模块 `arcqml`，包括 Tensor、
//! 电路、Pauli 可观测量、单态与 batch 模拟器、损失函数、Adam 和状态分析函数。
//! Rust 应用程序应依赖 workspace 中的 `arcqml` 门面 crate；本 crate 的 Rust 入口仅用于
//! 构建和初始化 Python 扩展。
//!
//! Python 对象会把 Rust 错误转换为对应的 `ValueError` 或 `RuntimeError`，并与 Rust
//! 实现共享 CPU 稠密状态向量、量子比特端序和自动微分约定。

mod analysis;
mod circuit;
mod error;
mod loss;
mod observable;
mod optimizer;
mod simulator;
mod tensor;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

/// 初始化进程范围的 `Rayon` 工作线程池；必须在首次并行计算前调用。
#[pyfunction]
fn init_rayon(num_threads: usize) -> PyResult<()> {
    arcqml_core::init_rayon(num_threads).map_err(error::runtime_error)
}

/// 返回全局 `Rayon` 工作线程池的线程数量。
#[pyfunction]
fn rayon_num_threads() -> usize {
    arcqml_core::rayon_num_threads()
}

#[doc(hidden)]
#[pymodule]
/// 初始化 `arcqml` 原生扩展模块，并注册全部稳定的通用 Python API。
pub fn arcqml(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<tensor::PyTensor>()?;
    module.add_class::<tensor::PyNoGrad>()?;
    module.add_class::<circuit::PyCircuit>()?;
    module.add_class::<observable::PyPauliSum>()?;
    module.add_class::<optimizer::PyAdam>()?;
    module.add_class::<simulator::PyStateVectorSimulator>()?;
    module.add_class::<simulator::PyBatchStateVectorSimulator>()?;
    module.add_function(wrap_pyfunction!(tensor::tensor, module)?)?;
    module.add_function(wrap_pyfunction!(tensor::no_grad, module)?)?;
    module.add_function(wrap_pyfunction!(loss::mse_loss, module)?)?;
    module.add_function(wrap_pyfunction!(
        loss::binary_cross_entropy_with_logits,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(init_rayon, module)?)?;
    module.add_function(wrap_pyfunction!(rayon_num_threads, module)?)?;
    analysis::add_to_module(module)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
