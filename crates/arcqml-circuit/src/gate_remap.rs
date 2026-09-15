use crate::{Gate, Gateparam, ParameterId};

impl Gate {
    /// 使用完整的参数映射表原地重写门内的参数标识。
    pub(crate) fn remap_parameter_ids(&mut self, mapping: &[ParameterId]) {
        match self {
            Self::Rx(parameter)
            | Self::Ry(parameter)
            | Self::Rz(parameter)
            | Self::Phase(parameter)
            | Self::CPhase(parameter)
            | Self::CRx(parameter)
            | Self::CRy(parameter)
            | Self::CRz(parameter)
            | Self::Rxx(parameter)
            | Self::Ryy(parameter)
            | Self::Rzz(parameter)
            | Self::Rzx(parameter) => remap_gate_parameter(parameter, mapping),
            Self::U3(theta, phi, lambda) => {
                remap_gate_parameter(theta, mapping);
                remap_gate_parameter(phi, mapping);
                remap_gate_parameter(lambda, mapping);
            }
            Self::FSim(theta, phi) => {
                remap_gate_parameter(theta, mapping);
                remap_gate_parameter(phi, mapping);
            }
            _ => {}
        }
    }
}

/// 将一个参数槽位中的局部参数标识替换为目标电路中的标识。
fn remap_gate_parameter(parameter: &mut Gateparam, mapping: &[ParameterId]) {
    let Gateparam::Param(source) = parameter else {
        return;
    };
    *source = mapping
        .get(source.index())
        .copied()
        .expect("append preflight must map every source parameter id");
}
