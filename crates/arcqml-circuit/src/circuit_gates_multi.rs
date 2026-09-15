use crate::{Circuit, CircuitResult, Gate, Qubit};

impl Circuit {
    /// 添加 CNOT 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn cnot(
        &mut self,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::cnot(), vec![control.into(), target.into()])
    }
    /// 添加 CX 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn cx(
        &mut self,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::cx(), vec![control.into(), target.into()])
    }
    /// 添加 CY 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn cy(
        &mut self,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::cy(), vec![control.into(), target.into()])
    }
    /// 添加 CZ 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn cz(
        &mut self,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::cz(), vec![control.into(), target.into()])
    }
    /// 添加 CH 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn ch(
        &mut self,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::ch(), vec![control.into(), target.into()])
    }
    /// 添加 CS 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn cs(
        &mut self,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::cs(), vec![control.into(), target.into()])
    }
    /// 添加 CT 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn ct(
        &mut self,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::ct(), vec![control.into(), target.into()])
    }
    /// 添加 SWAP 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn swap(
        &mut self,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::swap(), vec![lhs.into(), rhs.into()])
    }
    /// 添加 iSWAP 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn iswap(
        &mut self,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::iswap(), vec![lhs.into(), rhs.into()])
    }
    /// 添加 DCX 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn dcx(
        &mut self,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::dcx(), vec![lhs.into(), rhs.into()])
    }
    /// 添加 ECR 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn ecr(
        &mut self,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::ecr(), vec![lhs.into(), rhs.into()])
    }

    /// 添加 Toffoli 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn toffoli(
        &mut self,
        control0: impl Into<Qubit>,
        control1: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(
            Gate::toffoli(),
            vec![control0.into(), control1.into(), target.into()],
        )
    }
    /// 添加受控交换门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn cswap(
        &mut self,
        control: impl Into<Qubit>,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::cswap(), vec![control.into(), lhs.into(), rhs.into()])
    }
    /// 添加多控制 X 门。
    ///
    /// # Errors
    ///
    /// 当任一量子比特越界，或同一量子比特在该操作中重复出现时返回错误。
    pub fn mcx<I, Q>(&mut self, controls: I, target: impl Into<Qubit>) -> CircuitResult<&mut Self>
    where
        I: IntoIterator<Item = Q>,
        Q: Into<Qubit>,
    {
        let mut qubits: Vec<Qubit> = controls.into_iter().map(Into::into).collect();
        qubits.push(target.into());
        self.add_gate(Gate::mcx(qubits.len() - 1), qubits)
    }
}
