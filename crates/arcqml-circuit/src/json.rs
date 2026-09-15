use crate::{Circuit, CircuitError::*, CircuitResult, Gate, Gateparam, ParameterId, Qubit};
use arcqml_core::Tensor;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// 电路 JSON 数据传输对象，仅保存结构和参数名称，不保存参数运行时数值。
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CircuitJson {
    qubit_count: usize,
    parameters: Vec<ParameterJson>,
    operations: Vec<OperationJson>,
}

/// 参数声明的 JSON 表示，数组位置即参数标识。
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParameterJson {
    name: String,
}

/// 单个门操作的 JSON 表示。
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationJson {
    gate: GateJson,
    qubits: Vec<usize>,
}

/// 门参数的 JSON 表示，区分固定数值和电路参数引用。
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum GateparamJson {
    Fixed { value: f64 },
    Parameter { id: usize },
}

/// 自定义酉矩阵元素的 JSON 表示。
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComplexJson {
    re: f64,
    im: f64,
}

/// 内置门与自定义酉门的 JSON 表示。
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum GateJson {
    I,
    X,
    Y,
    Z,
    H,
    S,
    Sdg,
    T,
    Tdg,
    Sx,
    Sxdg,
    Rx {
        parameter: GateparamJson,
    },
    Ry {
        parameter: GateparamJson,
    },
    Rz {
        parameter: GateparamJson,
    },
    Phase {
        parameter: GateparamJson,
    },
    U3 {
        theta: GateparamJson,
        phi: GateparamJson,
        lambda: GateparamJson,
    },
    Cnot,
    Cy,
    Cz,
    Ch,
    Cs,
    Ct,
    Cphase {
        parameter: GateparamJson,
    },
    Crx {
        parameter: GateparamJson,
    },
    Cry {
        parameter: GateparamJson,
    },
    Crz {
        parameter: GateparamJson,
    },
    Swap,
    Iswap,
    Dcx,
    Ecr,
    Rxx {
        parameter: GateparamJson,
    },
    Ryy {
        parameter: GateparamJson,
    },
    Rzz {
        parameter: GateparamJson,
    },
    Rzx {
        parameter: GateparamJson,
    },
    Fsim {
        theta: GateparamJson,
        phi: GateparamJson,
    },
    Toffoli,
    Cswap,
    Mcx {
        controls: usize,
    },
    CustomUnitary {
        name: String,
        arity: usize,
        matrix: Vec<ComplexJson>,
    },
}

impl Circuit {
    /// 将有效电路编码为格式化 JSON，不保存可训练参数的当前数值或梯度。
    ///
    /// # Errors
    ///
    /// 当电路自身校验失败，或 JSON 序列化失败时返回错误。
    pub fn to_json(&self) -> CircuitResult<String> {
        self.validate()?;
        serde_json::to_string_pretty(&CircuitJson::from_circuit(self)).map_err(|error| {
            JsonSerializationError {
                message: error.to_string(),
            }
        })
    }

    /// 从 JSON 恢复电路结构，并以 `F64` 零标量初始化其中的可训练参数。
    ///
    /// # Errors
    ///
    /// 当 JSON 语法、字段或枚举标签无效，量子比特和参数引用不能构成有效电路，
    /// 自定义酉门无效，或无法构造参数 Tensor 时返回错误。
    pub fn from_json(input: &str) -> CircuitResult<Self> {
        let json = serde_json::from_str::<CircuitJson>(input).map_err(|error| {
            JsonDeserializationError {
                message: error.to_string(),
            }
        })?;
        json.into_circuit()
    }
}

impl CircuitJson {
    /// 从有效电路构建只含结构信息的 JSON 数据传输对象。
    fn from_circuit(circuit: &Circuit) -> Self {
        Self {
            qubit_count: circuit.num_qubits,
            parameters: circuit
                .parameters
                .iter()
                .map(|parameter| ParameterJson {
                    name: parameter.name().expect("已校验电路的参数必须具有名称"),
                })
                .collect(),
            operations: circuit
                .operations
                .iter()
                .map(OperationJson::from_operation)
                .collect(),
        }
    }

    /// 将 JSON 数据传输对象转换为经过完整校验的电路。
    fn into_circuit(self) -> CircuitResult<Circuit> {
        let mut circuit = Circuit::new(self.qubit_count)?;
        for parameter in self.parameters {
            let tensor = Tensor::new(0.0).map_err(|error| TensorCreateError {
                message: error.to_string(),
            })?;
            circuit.add_parameter_tensor_named(tensor, parameter.name)?;
        }
        for operation in self.operations {
            let gate = operation.gate.into_gate()?;
            let qubits = operation.qubits.into_iter().map(Qubit::new).collect();
            circuit.add_gate(gate, qubits)?;
        }
        circuit.validate()?;
        Ok(circuit)
    }
}

impl OperationJson {
    /// 从内部操作创建 JSON 操作表示。
    fn from_operation(operation: &crate::Operation) -> Self {
        Self {
            gate: GateJson::from_gate(operation.gate()),
            qubits: operation.qubits().iter().map(Qubit::index).collect(),
        }
    }
}

impl GateparamJson {
    /// 从内部门参数创建 JSON 参数表示。
    fn from_gateparam(parameter: Gateparam) -> Self {
        match parameter {
            Gateparam::Fixed(value) => Self::Fixed { value },
            Gateparam::Param(id) => Self::Parameter { id: id.index() },
        }
    }

    /// 将 JSON 参数表示转换为内部门参数。
    fn into_gateparam(self) -> Gateparam {
        match self {
            Self::Fixed { value } => Gateparam::fixed(value),
            Self::Parameter { id } => Gateparam::param(ParameterId::new(id)),
        }
    }
}

impl GateJson {
    /// 从内部量子门创建 JSON 门表示。
    fn from_gate(gate: &Gate) -> Self {
        match gate {
            Gate::I => Self::I,
            Gate::X => Self::X,
            Gate::Y => Self::Y,
            Gate::Z => Self::Z,
            Gate::H => Self::H,
            Gate::S => Self::S,
            Gate::Sdg => Self::Sdg,
            Gate::T => Self::T,
            Gate::Tdg => Self::Tdg,
            Gate::Sx => Self::Sx,
            Gate::Sxdg => Self::Sxdg,
            Gate::Rx(p) => Self::Rx {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::Ry(p) => Self::Ry {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::Rz(p) => Self::Rz {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::Phase(p) => Self::Phase {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::U3(a, b, c) => Self::U3 {
                theta: GateparamJson::from_gateparam(*a),
                phi: GateparamJson::from_gateparam(*b),
                lambda: GateparamJson::from_gateparam(*c),
            },
            Gate::CNot => Self::Cnot,
            Gate::CY => Self::Cy,
            Gate::CZ => Self::Cz,
            Gate::CH => Self::Ch,
            Gate::CS => Self::Cs,
            Gate::CT => Self::Ct,
            Gate::CPhase(p) => Self::Cphase {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::CRx(p) => Self::Crx {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::CRy(p) => Self::Cry {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::CRz(p) => Self::Crz {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::Swap => Self::Swap,
            Gate::ISwap => Self::Iswap,
            Gate::DCX => Self::Dcx,
            Gate::Ecr => Self::Ecr,
            Gate::Rxx(p) => Self::Rxx {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::Ryy(p) => Self::Ryy {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::Rzz(p) => Self::Rzz {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::Rzx(p) => Self::Rzx {
                parameter: GateparamJson::from_gateparam(*p),
            },
            Gate::FSim(a, b) => Self::Fsim {
                theta: GateparamJson::from_gateparam(*a),
                phi: GateparamJson::from_gateparam(*b),
            },
            Gate::Toffoli => Self::Toffoli,
            Gate::CSwap => Self::Cswap,
            Gate::MCX { controls } => Self::Mcx {
                controls: *controls,
            },
            Gate::CustomUnitary {
                name,
                arity,
                matrix,
            } => Self::CustomUnitary {
                name: name.clone(),
                arity: *arity,
                matrix: matrix
                    .iter()
                    .map(|v| ComplexJson { re: v.re, im: v.im })
                    .collect(),
            },
        }
    }

    /// 将 JSON 门表示转换为内部量子门，并验证自定义酉门。
    fn into_gate(self) -> CircuitResult<Gate> {
        Ok(match self {
            Self::I => Gate::I,
            Self::X => Gate::X,
            Self::Y => Gate::Y,
            Self::Z => Gate::Z,
            Self::H => Gate::H,
            Self::S => Gate::S,
            Self::Sdg => Gate::Sdg,
            Self::T => Gate::T,
            Self::Tdg => Gate::Tdg,
            Self::Sx => Gate::Sx,
            Self::Sxdg => Gate::Sxdg,
            Self::Rx { parameter } => Gate::rx(parameter.into_gateparam()),
            Self::Ry { parameter } => Gate::ry(parameter.into_gateparam()),
            Self::Rz { parameter } => Gate::rz(parameter.into_gateparam()),
            Self::Phase { parameter } => Gate::phase(parameter.into_gateparam()),
            Self::U3 { theta, phi, lambda } => Gate::u3(
                theta.into_gateparam(),
                phi.into_gateparam(),
                lambda.into_gateparam(),
            ),
            Self::Cnot => Gate::cnot(),
            Self::Cy => Gate::cy(),
            Self::Cz => Gate::cz(),
            Self::Ch => Gate::ch(),
            Self::Cs => Gate::cs(),
            Self::Ct => Gate::ct(),
            Self::Cphase { parameter } => Gate::cphase(parameter.into_gateparam()),
            Self::Crx { parameter } => Gate::crx(parameter.into_gateparam()),
            Self::Cry { parameter } => Gate::cry(parameter.into_gateparam()),
            Self::Crz { parameter } => Gate::crz(parameter.into_gateparam()),
            Self::Swap => Gate::swap(),
            Self::Iswap => Gate::iswap(),
            Self::Dcx => Gate::dcx(),
            Self::Ecr => Gate::ecr(),
            Self::Rxx { parameter } => Gate::rxx(parameter.into_gateparam()),
            Self::Ryy { parameter } => Gate::ryy(parameter.into_gateparam()),
            Self::Rzz { parameter } => Gate::rzz(parameter.into_gateparam()),
            Self::Rzx { parameter } => Gate::rzx(parameter.into_gateparam()),
            Self::Fsim { theta, phi } => Gate::fsim(theta.into_gateparam(), phi.into_gateparam()),
            Self::Toffoli => Gate::toffoli(),
            Self::Cswap => Gate::cswap(),
            Self::Mcx { controls } => Gate::mcx(controls),
            Self::CustomUnitary {
                name,
                arity,
                matrix,
            } => Gate::custom_unitary(
                name,
                arity,
                matrix
                    .into_iter()
                    .map(|v| Complex64::new(v.re, v.im))
                    .collect(),
            )?,
        })
    }
}
