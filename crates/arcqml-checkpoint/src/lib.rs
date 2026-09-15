//! ArcQML 电路参数的 JSON 检查点。
//!
//! 检查点只保存参数名称、数据类型和值，不保存门拓扑、梯度、自动微分图或优化器状态。
//! 加载前会完整校验格式、量子比特数和参数集合；校验失败不会部分修改目标电路。
//!
//! # 示例
//!
//! ```no_run
//! use arcqml_checkpoint::{load_weights, save_weights};
//! use arcqml_circuit::Circuit;
//!
//! let mut circuit = Circuit::new(1)?;
//! circuit.ry(0.3, 0usize)?;
//! save_weights(&circuit, "weights.json")?;
//! load_weights(&circuit, "weights.json")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod error;
mod weights;

pub use error::{CheckpointError, CheckpointResult};
pub use weights::{FORMAT, ParameterRecord, WeightCheckpoint, load_weights, save_weights};
