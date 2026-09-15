use crate::{Circuit, CircuitResult, Gate, Gateparam, ParameterId, Qubit};

impl Circuit {
    /// 添加恒等门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn i(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::i(), vec![qubit.into()])
    }
    /// 添加 Pauli-X 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn x(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::x(), vec![qubit.into()])
    }
    /// 添加 Pauli-Y 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn y(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::y(), vec![qubit.into()])
    }
    /// 添加 Pauli-Z 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn z(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::z(), vec![qubit.into()])
    }
    /// 添加 Hadamard 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn h(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::h(), vec![qubit.into()])
    }
    /// 添加 S 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn s(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::s(), vec![qubit.into()])
    }
    /// 添加 S 的逆门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn sdg(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::sdg(), vec![qubit.into()])
    }
    /// 添加 T 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn t(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::t(), vec![qubit.into()])
    }
    /// 添加 T 的逆门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn tdg(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::tdg(), vec![qubit.into()])
    }
    /// 添加 SX 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn sx(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::sx(), vec![qubit.into()])
    }
    /// 添加 SX 的逆门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn sxdg(&mut self, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::sxdg(), vec![qubit.into()])
    }

    /// 添加带新参数的 RX 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rx(&mut self, value: f64, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        let qubit = qubit.into();
        self.validate_qubits(&[qubit])?;
        let id = self.add_gate_parameter("rx", "theta", qubit, value)?;
        self.rx_param(id, qubit)
    }
    /// 添加带新参数的 RY 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn ry(&mut self, value: f64, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        let qubit = qubit.into();
        self.validate_qubits(&[qubit])?;
        let id = self.add_gate_parameter("ry", "theta", qubit, value)?;
        self.ry_param(id, qubit)
    }
    /// 添加带新参数的 RZ 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rz(&mut self, value: f64, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        let qubit = qubit.into();
        self.validate_qubits(&[qubit])?;
        let id = self.add_gate_parameter("rz", "theta", qubit, value)?;
        self.rz_param(id, qubit)
    }
    /// 添加带新参数的相位门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn phase(&mut self, value: f64, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        let qubit = qubit.into();
        self.validate_qubits(&[qubit])?;
        let id = self.add_gate_parameter("phase", "phi", qubit, value)?;
        self.phase_param(id, qubit)
    }

    /// 使用既有参数添加 RX 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rx_param(
        &mut self,
        id: ParameterId,
        qubit: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.parameter(id)?;
        self.add_gate(Gate::rx(Gateparam::param(id)), vec![qubit.into()])
    }
    /// 使用既有参数添加 RY 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn ry_param(
        &mut self,
        id: ParameterId,
        qubit: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.parameter(id)?;
        self.add_gate(Gate::ry(Gateparam::param(id)), vec![qubit.into()])
    }
    /// 使用既有参数添加 RZ 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rz_param(
        &mut self,
        id: ParameterId,
        qubit: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.parameter(id)?;
        self.add_gate(Gate::rz(Gateparam::param(id)), vec![qubit.into()])
    }
    /// 使用既有参数添加相位门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn phase_param(
        &mut self,
        id: ParameterId,
        qubit: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.parameter(id)?;
        self.add_gate(Gate::phase(Gateparam::param(id)), vec![qubit.into()])
    }

    /// 添加固定角度的 RX 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rx_fixed(&mut self, value: f64, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::rx(value), vec![qubit.into()])
    }
    /// 添加固定角度的 RY 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn ry_fixed(&mut self, value: f64, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::ry(value), vec![qubit.into()])
    }
    /// 添加固定角度的 RZ 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn rz_fixed(&mut self, value: f64, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::rz(value), vec![qubit.into()])
    }
    /// 添加固定角度的相位门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn phase_fixed(&mut self, value: f64, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::phase(value), vec![qubit.into()])
    }

    /// 添加带三个新参数的 U3 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn u3(
        &mut self,
        theta: f64,
        phi: f64,
        lambda: f64,
        qubit: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        let qubit = qubit.into();
        self.validate_qubits(&[qubit])?;
        let theta = self.add_gate_parameter("u3", "theta", qubit, theta)?;
        let phi = self.add_gate_parameter("u3", "phi", qubit, phi)?;
        let lambda = self.add_gate_parameter("u3", "lambda", qubit, lambda)?;
        self.u3_params(theta, phi, lambda, qubit)
    }
    /// 使用既有参数添加 U3 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn u3_params(
        &mut self,
        theta: ParameterId,
        phi: ParameterId,
        lambda: ParameterId,
        qubit: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.parameter(theta)?;
        self.parameter(phi)?;
        self.parameter(lambda)?;
        self.add_gate(Gate::u3(theta, phi, lambda), vec![qubit.into()])
    }
    /// 添加固定角度的 U3 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn u3_fixed(
        &mut self,
        theta: f64,
        phi: f64,
        lambda: f64,
        qubit: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        self.add_gate(Gate::u3(theta, phi, lambda), vec![qubit.into()])
    }
    /// 添加带新参数的 U1 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn u1(&mut self, lambda: f64, qubit: impl Into<Qubit>) -> CircuitResult<&mut Self> {
        self.phase(lambda, qubit)
    }
    /// 添加带两个新参数的 U2 门。
    ///
    /// # Errors
    ///
    /// 当量子比特越界，或该方法使用的既有参数标识无效时返回错误；创建新参数时，底层标量 Tensor 构造失败也会返回错误。
    pub fn u2(
        &mut self,
        phi: f64,
        lambda: f64,
        qubit: impl Into<Qubit>,
    ) -> CircuitResult<&mut Self> {
        let qubit = qubit.into();
        self.validate_qubits(&[qubit])?;
        let phi = self.add_gate_parameter("u2", "phi", qubit, phi)?;
        let lambda = self.add_gate_parameter("u2", "lambda", qubit, lambda)?;
        self.add_gate(Gate::u2(phi, lambda), vec![qubit])
    }
}
