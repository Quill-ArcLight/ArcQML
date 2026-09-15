use crate::{CheckpointError, CheckpointResult};
use arcqml_circuit::{Circuit, ParameterId};
use arcqml_core::{Parameter, Storage, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

/// 当前权重检查点 JSON 格式的标识符。
pub const FORMAT: &str = "arcqml/weights";

/// [`save_weights`] 写入的顶层 JSON 数据。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightCheckpoint {
    /// 文件格式标识；当前值必须等于 [`FORMAT`]。
    pub format: String,
    /// 保存参数时电路包含的量子比特数。
    pub num_qubits: usize,
    /// 按参数名称字典序排列的标量参数记录。
    pub parameters: Vec<ParameterRecord>,
}

/// 单个标量电路参数的可序列化记录。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParameterRecord {
    /// 电路中的唯一参数名称。
    pub name: String,
    /// 参数数据类型名称，当前支持 `f32` 和 `f64`。
    pub dtype: String,
    /// 转换为 `f64` 后保存的有限标量值。
    pub value: f64,
}

/// 将电路中全部具名标量浮点参数的当前值保存为格式化 JSON 文件。
///
/// 参数按名称排序以获得稳定输出。若目标文件已存在，创建文件时会先截断它。
///
/// # Errors
///
/// 当参数缺少名称、不是单元素 `F32`/`F64` Tensor，或无法创建、序列化或刷新
/// 目标文件时返回错误。
pub fn save_weights(circuit: &Circuit, path: impl AsRef<Path>) -> CheckpointResult<()> {
    let path = path.as_ref();
    let checkpoint = checkpoint_from_circuit(circuit)?;
    let file = File::create(path).map_err(|source| CheckpointError::WriteError {
        path: path.to_path_buf(),
        source,
    })?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &checkpoint).map_err(|source| {
        CheckpointError::SerializeError {
            path: path.to_path_buf(),
            source,
        }
    })?;
    writer
        .flush()
        .map_err(|source| CheckpointError::WriteError {
            path: path.to_path_buf(),
            source,
        })?;
    Ok(())
}

/// 将 JSON 权重检查点按参数名称加载到已构建完成的电路中。
///
/// 加载只替换参数值，不改变电路结构；成功后会清空全部已恢复参数的梯度。
///
/// # Errors
///
/// 当文件无法读取或解析、格式标识或量子比特数不匹配、参数集合或 dtype 不一致、
/// 记录含非有限值，或无法构造及写入替换 Tensor 时返回错误。
pub fn load_weights(circuit: &Circuit, path: impl AsRef<Path>) -> CheckpointResult<()> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|source| CheckpointError::ReadError {
        path: path.to_path_buf(),
        source,
    })?;
    let reader = BufReader::new(file);
    let checkpoint =
        serde_json::from_reader(reader).map_err(|source| CheckpointError::ParseError {
            path: path.to_path_buf(),
            source,
        })?;

    apply_checkpoint(circuit, checkpoint)
}

/// 从电路提取参数值，并构造可写入 JSON 的检查点数据。
pub(crate) fn checkpoint_from_circuit(circuit: &Circuit) -> CheckpointResult<WeightCheckpoint> {
    let mut parameters = Vec::with_capacity(circuit.num_parameters());
    for (index, parameter) in circuit.parameters().iter().enumerate() {
        let name = circuit
            .parameter_name(ParameterId::new(index))
            .map_err(|error| CheckpointError::TensorConstructionError {
                name: format!("parameter#{index}"),
                message: error.to_string(),
            })?;
        let (dtype, value) = scalar_value(&name, parameter)?;
        parameters.push(ParameterRecord { name, dtype, value });
    }
    parameters.sort_by(|left, right| left.name.cmp(&right.name));

    Ok(WeightCheckpoint {
        format: FORMAT.to_string(),
        num_qubits: circuit.num_qubits(),
        parameters,
    })
}

/// 校验检查点与目标电路兼容，并将检查点中的权重写入目标电路。
pub(crate) fn apply_checkpoint(
    circuit: &Circuit,
    checkpoint: WeightCheckpoint,
) -> CheckpointResult<()> {
    if checkpoint.format != FORMAT {
        return Err(CheckpointError::UnsupportedFormatError {
            expected: FORMAT,
            actual: checkpoint.format,
        });
    }
    if checkpoint.num_qubits != circuit.num_qubits() {
        return Err(CheckpointError::QubitCountMismatchError {
            expected: circuit.num_qubits(),
            actual: checkpoint.num_qubits,
        });
    }

    let target_parameters = named_parameters(circuit)?;
    let mut saved_parameters = BTreeMap::new();
    for record in checkpoint.parameters {
        let name = record.name.clone();
        if saved_parameters.insert(name.clone(), record).is_some() {
            return Err(CheckpointError::DuplicateParameterNameError { name });
        }
    }

    for name in target_parameters.keys() {
        if !saved_parameters.contains_key(name) {
            return Err(CheckpointError::MissingParameterError { name: name.clone() });
        }
    }
    for name in saved_parameters.keys() {
        if !target_parameters.contains_key(name) {
            return Err(CheckpointError::UnexpectedParameterError { name: name.clone() });
        }
    }

    // 先构造全部替换 Tensor，再修改参数；这样校验失败时不会留下部分恢复的电路。
    let mut replacements = Vec::with_capacity(target_parameters.len());
    for (name, parameter) in target_parameters {
        let record = saved_parameters
            .get(&name)
            .expect("parameter sets were checked above");
        let (expected_dtype, _) = scalar_value(&name, &parameter)?;
        if record.dtype != expected_dtype {
            return Err(CheckpointError::DtypeMismatchError {
                name,
                expected: dtype_name(&expected_dtype),
                actual: record.dtype.clone(),
            });
        }
        if !record.value.is_finite() {
            return Err(CheckpointError::NonFiniteValueError { name });
        }
        let tensor = tensor_from_record(record).map_err(|message| {
            CheckpointError::TensorConstructionError {
                name: record.name.clone(),
                message,
            }
        })?;
        replacements.push((parameter, tensor));
    }

    for (parameter, tensor) in replacements {
        parameter.try_set_tensor(tensor).map_err(|error| {
            CheckpointError::TensorConstructionError {
                name: parameter.name().unwrap_or_else(|| "<unnamed>".to_string()),
                message: error.to_string(),
            }
        })?;
        parameter.zero_grad();
    }
    Ok(())
}

/// 按参数名收集电路参数，便于稳定地校验和匹配检查点中的记录。
fn named_parameters(circuit: &Circuit) -> CheckpointResult<BTreeMap<String, Parameter>> {
    let mut named = BTreeMap::new();
    for (index, parameter) in circuit.parameters().iter().enumerate() {
        let name = circuit
            .parameter_name(ParameterId::new(index))
            .map_err(|error| CheckpointError::TensorConstructionError {
                name: format!("parameter#{index}"),
                message: error.to_string(),
            })?;
        if named.insert(name.clone(), parameter.clone()).is_some() {
            return Err(CheckpointError::DuplicateParameterNameError { name });
        }
    }
    Ok(named)
}

/// 读取参数的标量值及其数据类型，并校验该参数可以写入当前权重格式。
fn scalar_value(name: &str, parameter: &Parameter) -> CheckpointResult<(String, f64)> {
    let tensor = parameter.tensor();
    if tensor.numel() != 1 {
        return Err(CheckpointError::InvalidParameterShapeError {
            name: name.to_string(),
            numel: tensor.numel(),
        });
    }
    match &*tensor.storage() {
        Storage::F32(values) => Ok(("f32".to_string(), values[0] as f64)),
        Storage::F64(values) => Ok(("f64".to_string(), values[0])),
        storage => Err(CheckpointError::UnsupportedParameterDtypeError {
            name: name.to_string(),
            dtype: storage.dtype().to_string(),
        }),
    }
}

/// 根据 JSON 参数记录重建对应数据类型的标量 Tensor。
fn tensor_from_record(record: &ParameterRecord) -> Result<Tensor, String> {
    match record.dtype.as_str() {
        "f32" => Tensor::new(record.value as f32).map_err(|error| error.to_string()),
        "f64" => Tensor::new(record.value).map_err(|error| error.to_string()),
        dtype => Err(format!("unsupported dtype {dtype}")),
    }
}

/// 将检查点中的数据类型标识转换为错误信息使用的静态名称。
fn dtype_name(dtype: &str) -> &'static str {
    match dtype {
        "f32" => "f32",
        "f64" => "f64",
        _ => "supported floating-point dtype",
    }
}
