use crate::SimResult;
use crate::statevector::{
    execution::{adjoint, gatewise},
    shared::circuit_validation,
    state::batch as state,
};
use arcqml_circuit::{Circuit, Gate, Operation, Qubit};
use arcqml_core::Tensor;
use arcqml_observable::{Hamiltonian, PauliString, SparsePauliOp};
/// 使用形状为 `[batch_size, 2^num_qubits]` 的 `C64` Tensor 保存一批纯态的模拟器。
///
/// 每一行是独立、归一化的状态向量；`q0` 对应行内计算基索引的最低有效位。
#[derive(Debug, Clone)]
pub struct BatchStateVectorSimulator {
    num_qubits: usize,
    batch_size: usize,
    state: Tensor,
}

impl BatchStateVectorSimulator {
    /// 创建 `batch_size` 个相互独立的 `|0…0⟩` 状态。
    ///
    /// # Errors
    ///
    /// 当量子比特数或 `batch_size` 为零、状态元素数溢出，或无法构造 Tensor 时返回错误。
    pub fn new(num_qubits: usize, batch_size: usize) -> SimResult<Self> {
        Ok(Self {
            num_qubits,
            batch_size,
            state: state::zero_state_tensor(num_qubits, batch_size)?,
        })
    }

    /// 从连续、逐行归一化且形状为 `[batch_size, 2^num_qubits]` 的 `C64` Tensor 创建模拟器。
    ///
    /// Tensor 必须位于 CPU 并采用稠密布局；它会直接保存而不复制。
    ///
    /// # Errors
    ///
    /// 当量子比特数无效，或 Tensor 的 dtype、设备、布局、连续性、形状、批大小
    /// 或任一行的归一化不满足约束时返回错误。
    pub fn from_state_tensor(num_qubits: usize, state_tensor: Tensor) -> SimResult<Self> {
        let batch_size = state::validate_state_tensor(num_qubits, &state_tensor)?;
        Ok(Self {
            num_qubits,
            batch_size,
            state: state_tensor,
        })
    }

    /// 返回每个 batch 样本对应的量子比特数量。
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// 返回当前模拟器保存的 batch 大小。
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// 返回当前批量量子态的可微复数振幅 Tensor。
    ///
    /// 返回 Tensor 的形状为 `[batch_size, 2^num_qubits]`，dtype 为 `C64`；
    /// 返回的共享 Tensor 句柄保留每个样本对应量子态的计算图。
    ///
    /// # Errors
    ///
    /// 当内部状态的 dtype、布局、连续性、形状或逐行归一化被无效底层写入破坏时返回错误。
    pub fn amplitudes(&self) -> SimResult<Tensor> {
        state::validate_state_tensor(self.num_qubits, &self.state)?;
        Ok(self.state.clone())
    }

    /// 丢弃当前状态及其计算图，将所有样本重置为 `|0…0⟩`。
    ///
    /// # Errors
    ///
    /// 当状态元素数溢出或无法构造新的批状态 Tensor 时返回错误。
    pub fn reset(&mut self) -> SimResult<&mut Self> {
        self.state = state::zero_state_tensor(self.num_qubits, self.batch_size)?;
        Ok(self)
    }

    /// 对所有样本应用一个不含未绑定参数的操作。
    ///
    /// # Errors
    ///
    /// 当批状态无效，操作的量子比特越界或重复、arity 不匹配、门含未绑定参数，
    /// 或无法创建自动微分节点时返回错误。
    pub fn apply_operation(&mut self, operation: &Operation) -> SimResult<&mut Self> {
        self.state = gatewise::batch::apply_gate(
            self.num_qubits,
            self.state.clone(),
            operation.gate().clone(),
            operation.qubits().to_vec(),
            Vec::new(),
        )?;
        Ok(self)
    }

    /// 在按局部位序给出的 `qubits` 上，对所有样本应用一个不含未绑定参数的门。
    ///
    /// # Errors
    ///
    /// 当批状态无效，`qubits` 越界或重复、数量与门 arity 不符、门含未绑定
    /// 参数，或无法创建自动微分节点时返回错误。
    pub fn apply_gate(&mut self, gate: &Gate, qubits: &[Qubit]) -> SimResult<&mut Self> {
        self.state = gatewise::batch::apply_gate(
            self.num_qubits,
            self.state.clone(),
            gate.clone(),
            qubits.to_vec(),
            Vec::new(),
        )?;
        Ok(self)
    }

    /// 使用逐门执行路径，将电路参数当前值应用到所有样本。
    ///
    /// # Errors
    ///
    /// 当电路量子比特数不匹配、参数不是受支持的浮点标量、操作或批状态无效，
    /// 或无法创建任一自动微分节点时返回错误。
    pub fn apply_circuit(&mut self, circuit: &Circuit) -> SimResult<&mut Self> {
        circuit_validation::validate_circuit_qubits(self.num_qubits, circuit)?;
        for operation in circuit.operations() {
            let bound_gate = arcqml_kernel::circuit::bind_gate(circuit, operation.gate())?;
            self.state = gatewise::batch::apply_gate(
                self.num_qubits,
                self.state.clone(),
                bound_gate.gate,
                operation.qubits().to_vec(),
                bound_gate.parameters,
            )?;
        }
        Ok(self)
    }

    /// 计算每个样本对 Pauli 字符串的可微期望值，返回形状为 `[batch_size]` 的 `F64` Tensor。
    ///
    /// # Errors
    ///
    /// 当 Pauli 字符串或批状态无效、量子比特数不匹配，或无法构造结果 Tensor 时返回错误。
    pub fn expectation_pauli_string(&self, string: &PauliString) -> SimResult<Tensor> {
        let observable = SparsePauliOp::from_pauli_string(1.0, string.clone())?;
        self.expectation_observable(&observable)
    }

    /// 计算每个样本对可观测量的可微期望值，返回形状为 `[batch_size]` 的 `F64` Tensor。
    ///
    /// # Errors
    ///
    /// 当批状态无效、可观测量的量子比特数不匹配或其编译失败，或无法构造
    /// 结果 Tensor 时返回错误。
    pub fn expectation_observable(&self, observable: &SparsePauliOp) -> SimResult<Tensor> {
        gatewise::batch::expectation(self.num_qubits, self.state.clone(), observable)
    }

    /// 计算每个样本对给定 Hamiltonian 的可微期望值。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`BatchStateVectorSimulator::expectation_observable`] 相同。
    pub fn expectation_hamiltonian(&self, hamiltonian: &Hamiltonian) -> SimResult<Tensor> {
        self.expectation_observable(hamiltonian)
    }

    /// 将整个批处理电路作为一个 Tensor 自动微分节点，以行主序按 batch 分块执行。
    ///
    /// 门演化、可观测量、伴随扫描与参数导数均使用串行行内核，
    /// 仅在 batch 顶层进行并行；反向同时计算电路参数和初始批处理状态的梯度。
    ///
    /// # Errors
    ///
    /// 当初态、电路或可观测量无效，三者量子比特数不一致，参数无法绑定，门内核
    /// 执行失败，或无法构造结果 Tensor 时返回错误。
    pub fn run(&self, circuit: &Circuit, observable: &SparsePauliOp) -> SimResult<Tensor> {
        adjoint::batch::run(self.num_qubits, &self.state, circuit, observable)
    }
}
