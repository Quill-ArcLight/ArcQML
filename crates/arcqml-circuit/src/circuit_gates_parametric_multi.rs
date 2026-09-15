use crate::{Circuit, CircuitResult, Gate, Gateparam, ParameterId, Qubit};

impl Circuit {
    /// 添加带新参数的受控相位门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn cp(
        &mut self,
        theta: f64,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_binary_parameterized_gate("cp", theta, control.into(), target.into(), Gate::cp)
    }
    /// 添加带新参数的受控 RX 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn crx(
        &mut self,
        theta: f64,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_binary_parameterized_gate("crx", theta, control.into(), target.into(), Gate::crx)
    }
    /// 添加带新参数的受控 RY 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn cry(
        &mut self,
        theta: f64,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_binary_parameterized_gate("cry", theta, control.into(), target.into(), Gate::cry)
    }
    /// 添加带新参数的受控 RZ 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn crz(
        &mut self,
        theta: f64,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_binary_parameterized_gate("crz", theta, control.into(), target.into(), Gate::crz)
    }
    /// 添加带新参数的 RXX 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rxx(
        &mut self,
        theta: f64,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_binary_parameterized_gate("rxx", theta, lhs.into(), rhs.into(), Gate::rxx)
    }
    /// 添加带新参数的 RYY 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn ryy(
        &mut self,
        theta: f64,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_binary_parameterized_gate("ryy", theta, lhs.into(), rhs.into(), Gate::ryy)
    }
    /// 添加带新参数的 RZZ 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rzz(
        &mut self,
        theta: f64,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_binary_parameterized_gate("rzz", theta, lhs.into(), rhs.into(), Gate::rzz)
    }
    /// 添加带新参数的 RZX 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rzx(
        &mut self,
        theta: f64,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_binary_parameterized_gate("rzx", theta, lhs.into(), rhs.into(), Gate::rzx)
    }

    /// 添加带两个新参数的 fSim 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn fsim(
        &mut self,
        theta: f64,
        phi: f64,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        let lhs = lhs.into();
        let rhs = rhs.into();
        self.validate_qubits(&[lhs, rhs])?;
        let theta = self.add_gate_parameter("fsim", "theta", lhs, theta)?;
        let phi = self.add_gate_parameter("fsim", "phi", lhs, phi)?;
        self.add_gate(Gate::fsim(theta, phi), vec![lhs, rhs])
    }

    /// 使用既有参数添加受控相位门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn cp_param(
        &mut self,
        id: ParameterId,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_existing_binary_parameterized_gate(id, control.into(), target.into(), Gate::cp)
    }
    /// 使用既有参数添加受控 RX 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn crx_param(
        &mut self,
        id: ParameterId,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_existing_binary_parameterized_gate(id, control.into(), target.into(), Gate::crx)
    }
    /// 使用既有参数添加受控 RY 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn cry_param(
        &mut self,
        id: ParameterId,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_existing_binary_parameterized_gate(id, control.into(), target.into(), Gate::cry)
    }
    /// 使用既有参数添加受控 RZ 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn crz_param(
        &mut self,
        id: ParameterId,
        control: impl Into<Qubit>,
        target: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_existing_binary_parameterized_gate(id, control.into(), target.into(), Gate::crz)
    }
    /// 使用既有参数添加 RXX 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rxx_param(
        &mut self,
        id: ParameterId,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_existing_binary_parameterized_gate(id, lhs.into(), rhs.into(), Gate::rxx)
    }
    /// 使用既有参数添加 RYY 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn ryy_param(
        &mut self,
        id: ParameterId,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_existing_binary_parameterized_gate(id, lhs.into(), rhs.into(), Gate::ryy)
    }
    /// 使用既有参数添加 RZZ 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rzz_param(
        &mut self,
        id: ParameterId,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_existing_binary_parameterized_gate(id, lhs.into(), rhs.into(), Gate::rzz)
    }
    /// 使用既有参数添加 RZX 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rzx_param(
        &mut self,
        id: ParameterId,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_existing_binary_parameterized_gate(id, lhs.into(), rhs.into(), Gate::rzx)
    }
    /// 使用既有参数添加 fSim 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界或重复，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn fsim_params(
        &mut self,
        theta: ParameterId,
        phi: ParameterId,
        lhs: impl Into<Qubit>,
        rhs: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.parameter(theta)?;
        self.parameter(phi)?;
        self.add_gate(Gate::fsim(theta, phi), vec![lhs.into(), rhs.into()])
    }

    /// 添加一个带新参数的双比特参数化门。
    fn add_binary_parameterized_gate(
        &mut self,
        gate_name: &str,
        value: f64,
        lhs: Qubit,
        rhs: Qubit,
        make_gate: impl Fn(Gateparam) -> Gate,
    ) -> CircuitResult<&mut Self> {
        self.validate_qubits(&[lhs, rhs])?;
        let id = self.add_gate_parameter(gate_name, "theta", lhs, value)?;
        self.add_gate(make_gate(Gateparam::param(id)), vec![lhs, rhs])
    }
    /// 使用既有参数添加一个双比特参数化门。
    fn add_existing_binary_parameterized_gate(
        &mut self,
        id: ParameterId,
        lhs: Qubit,
        rhs: Qubit,
        make_gate: impl Fn(Gateparam) -> Gate,
    ) -> CircuitResult<&mut Self> {
        self.parameter(id)?;
        self.add_gate(make_gate(Gateparam::param(id)), vec![lhs, rhs])
    }
}
