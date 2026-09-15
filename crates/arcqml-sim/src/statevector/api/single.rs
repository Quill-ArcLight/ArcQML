use crate::SimResult;
use crate::statevector::{
    execution::{adjoint, gatewise},
    shared::circuit_validation,
    state::single as state,
};
use arcqml_circuit::{Circuit, Gate, Operation, Qubit};
use arcqml_core::Tensor;
use arcqml_observable::{Hamiltonian, PauliString, SparsePauliOp};

/// 使用一维 `C64` Tensor 保存纯态的稠密 CPU 态向量模拟器。
///
/// 振幅按计算基索引升序存储，`q0` 对应索引的最低有效位。克隆模拟器会克隆
/// [`Tensor`] 句柄，因此底层存储和已有自动微分图仍按 Tensor 的共享语义处理。
#[derive(Debug, Clone)]
pub struct StateVectorSimulator {
    num_qubits: usize,
    pub(super) state: Tensor,
}

impl StateVectorSimulator {
    /// 创建 `|0…0⟩` 初态。
    ///
    /// # Errors
    ///
    /// 当 `num_qubits` 为零、`2^num_qubits` 溢出，或无法构造状态 Tensor 时返回错误。
    pub fn new(num_qubits: usize) -> SimResult<Self> {
        Ok(Self {
            num_qubits,
            state: state::zero_state_tensor(num_qubits)?,
        })
    }

    /// 从归一化、连续的一维 `C64` Tensor 创建模拟器。
    ///
    /// `state` 必须位于 CPU、采用稠密布局，形状为 `[2^num_qubits]`，且模平方和
    /// 与 `1` 的绝对误差不超过 `1e-10`。Tensor 会直接保存而不复制。
    ///
    /// # Errors
    ///
    /// 当量子比特数无效，或 `state` 的 dtype、设备、布局、连续性、形状或归一化
    /// 不满足上述约束时返回错误。
    pub fn from_state_tensor(num_qubits: usize, state: Tensor) -> SimResult<Self> {
        state::validate_state_tensor(num_qubits, &state)?;
        Ok(Self { num_qubits, state })
    }

    /// 返回量子比特数。
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// 返回当前归一化状态向量的共享 Tensor 句柄。
    ///
    /// 返回 Tensor 的形状为 `[2^num_qubits]`，dtype 为 `C64`；若状态包含计算图，
    /// 返回值保留该图。
    ///
    /// # Errors
    ///
    /// 当内部状态的 dtype、布局、连续性、形状或归一化被无效的底层写入破坏时返回错误。
    pub fn amplitudes(&self) -> SimResult<Tensor> {
        state::validate_state_tensor(self.num_qubits, &self.state)?;
        Ok(self.state.clone())
    }

    /// 丢弃当前状态及其计算图，并重置为 `|0…0⟩`。
    ///
    /// # Errors
    ///
    /// 当状态维度溢出或无法构造新的状态 Tensor 时返回错误。
    pub fn reset(&mut self) -> SimResult<&mut Self> {
        self.state = state::zero_state_tensor(self.num_qubits)?;
        Ok(self)
    }

    /// 对当前量子态应用一个不含未绑定参数的操作。
    ///
    /// # Errors
    ///
    /// 当当前状态无效，操作的量子比特越界或重复、arity 不匹配、门含未绑定参数，
    /// 或无法创建自动微分节点时返回错误。
    pub fn apply_operation(&mut self, operation: &Operation) -> SimResult<&mut Self> {
        self.state = gatewise::single::apply_gate(
            self.num_qubits,
            self.state.clone(),
            operation.gate().clone(),
            operation.qubits().to_vec(),
            Vec::new(),
        )?;
        Ok(self)
    }

    /// 在按局部位序给出的 `qubits` 上应用一个不含未绑定参数的量子门。
    ///
    /// # Errors
    ///
    /// 当当前状态无效，`qubits` 越界或重复、数量与门 arity 不符、门含未绑定
    /// 参数，或无法创建自动微分节点时返回错误。
    pub fn apply_gate(&mut self, gate: &Gate, qubits: &[Qubit]) -> SimResult<&mut Self> {
        self.state = gatewise::single::apply_gate(
            self.num_qubits,
            self.state.clone(),
            gate.clone(),
            qubits.to_vec(),
            Vec::new(),
        )?;
        Ok(self)
    }

    /// 使用电路中各 [`arcqml_core::Parameter`] 的当前值，按操作顺序原地推进模拟器。
    ///
    /// # Errors
    ///
    /// 当电路量子比特数不匹配、参数不是受支持的浮点标量、操作或当前状态无效，
    /// 或无法创建任一自动微分节点时返回错误。
    pub fn apply_circuit(&mut self, circuit: &Circuit) -> SimResult<&mut Self> {
        circuit_validation::validate_circuit_qubits(self.num_qubits, circuit)?;
        for operation in circuit.operations() {
            let bound_gate = arcqml_kernel::circuit::bind_gate(circuit, operation.gate())?;
            self.state = gatewise::single::apply_gate(
                self.num_qubits,
                self.state.clone(),
                bound_gate.gate,
                operation.qubits().to_vec(),
                bound_gate.parameters,
            )?;
        }
        Ok(self)
    }

    /// 计算当前状态对 Pauli 字符串的实标量可微期望值。
    ///
    /// # Errors
    ///
    /// 当 Pauli 字符串或当前状态无效、量子比特数不匹配，或无法构造结果 Tensor 时返回错误。
    pub fn expectation_pauli_string(&self, string: &PauliString) -> SimResult<Tensor> {
        let observable = SparsePauliOp::from_pauli_string(1.0, string.clone())?;
        self.expectation_observable(&observable)
    }

    /// 计算当前状态对稀疏 Pauli 算符的实标量可微期望值。
    ///
    /// # Errors
    ///
    /// 当当前状态无效、可观测量的量子比特数不匹配或其编译失败，或无法构造
    /// 结果 Tensor 时返回错误。
    pub fn expectation_observable(&self, observable: &SparsePauliOp) -> SimResult<Tensor> {
        gatewise::single::expectation(self.num_qubits, self.state.clone(), observable)
    }

    /// 计算 Hamiltonian 的实标量可微期望值。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`StateVectorSimulator::expectation_observable`] 相同。
    pub fn expectation_hamiltonian(&self, hamiltonian: &Hamiltonian) -> SimResult<Tensor> {
        self.expectation_observable(hamiltonian)
    }

    /// 将整条电路作为一个 Tensor 自动微分节点执行。
    ///
    /// 返回 `F64` 标量 Tensor。其反向规则使用量子伴随算法，因此普通 Tensor 损失
    /// 可直接调用 `loss.backward()`；反向同时计算电路参数和模拟器初态的梯度。
    ///
    /// # Errors
    ///
    /// 当初态、电路或可观测量无效，三者量子比特数不一致，参数无法绑定，门内核
    /// 执行失败，或无法构造结果 Tensor 时返回错误。
    pub fn run(&self, circuit: &Circuit, observable: &SparsePauliOp) -> SimResult<Tensor> {
        adjoint::single::run(self.num_qubits, &self.state, circuit, observable)
    }
}
