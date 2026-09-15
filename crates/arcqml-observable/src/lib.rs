//! Pauli 字符串和稀疏 Pauli 可观测量。
//!
//! [`PauliString`] 表示固定量子比特数上的 Pauli 张量积；[`SparsePauliOp`] 表示带实系数
//! 的 Pauli 项之和。模拟器可直接计算这些对象的可微期望值。
//!
//! # 示例
//!
//! ```
//! use arcqml_observable::{Pauli, PauliString, SparsePauliOp};
//!
//! let z0 = PauliString::z(2, 0usize)?;
//! let hamiltonian = SparsePauliOp::from_pauli_string(0.5, z0)?;
//! assert_eq!(hamiltonian.num_qubits(), 2);
//! # Ok::<(), arcqml_observable::ObservableError>(())
//! ```

mod compiled_hamiltonian;
/// 可观测量构造和状态向量计算错误。
pub mod error;
/// 单量子比特 Pauli 算符。
pub mod pauli;
/// 指定量子比特上的 Pauli 操作与 Pauli 字符串。
pub mod pauli_string;
/// 实系数稀疏 Pauli 和及其便捷构造。
pub mod sparse_pauli_op;

pub use error::{ObservableError, ObservableResult};
pub use pauli::Pauli;
pub use pauli_string::{PauliOp, PauliString};
pub use sparse_pauli_op::{Hamiltonian, PauliObservable, PauliSum, PauliTerm, SparsePauliOp};

/// 可观测量常用类型的预导入集合。
pub mod prelude {
    pub use crate::{
        Hamiltonian, ObservableError, ObservableResult, Pauli, PauliObservable, PauliOp,
        PauliString, PauliSum, PauliTerm, SparsePauliOp,
    };
    pub use arcqml_circuit::Qubit;
}
