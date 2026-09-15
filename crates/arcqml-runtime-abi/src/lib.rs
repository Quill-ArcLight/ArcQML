//! ArcQML 门面与原生运行时共享的稳定、源码可见 C ABI。
//!
//! 本 crate 刻意只包含声明；量子数值实现位于单独构建的专有运行时中。

use num_complex::Complex64;

/// 当前源码版本支持的 ABI 修订号。
pub const ABI_VERSION: u32 = 1;

/// 已完成参数绑定的借用门描述符。
///
/// 复数值只通过指针传递。`Complex64` 采用 `repr(C)`，内存布局与 `[f64; 2]` 兼容。
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GateDescriptor {
    pub abi_version: u32,
    pub opcode: u32,
    pub arity: usize,
    pub parameters: [f64; 3],
    pub matrix: *const Complex64,
    pub matrix_len: usize,
    pub name: *const u8,
    pub name_len: usize,
}

/// 由运行时拥有的导数矩阵。
///
/// 该句柄只能交给运行时 ABI 函数使用，并且必须通过配套的销毁函数释放。
#[repr(C)]
#[derive(Debug)]
pub struct DerivativeMatrixHandle {
    pub data: *mut Complex64,
    pub len: usize,
    pub capacity: usize,
}

impl DerivativeMatrixHandle {
    pub const fn empty() -> Self {
        Self {
            data: std::ptr::null_mut(),
            len: 0,
            capacity: 0,
        }
    }
}

impl GateDescriptor {
    pub const fn builtin(opcode: u32, arity: usize, parameters: [f64; 3]) -> Self {
        Self {
            abi_version: ABI_VERSION,
            opcode,
            arity,
            parameters,
            matrix: std::ptr::null(),
            matrix_len: 0,
            name: std::ptr::null(),
            name_len: 0,
        }
    }

    pub const fn custom(
        arity: usize,
        matrix: *const Complex64,
        matrix_len: usize,
        name: *const u8,
        name_len: usize,
    ) -> Self {
        Self {
            abi_version: ABI_VERSION,
            opcode: opcode::CUSTOM_UNITARY,
            arity,
            parameters: [0.0; 3],
            matrix,
            matrix_len,
            name,
            name_len,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ParameterBindingDescriptor {
    pub parameter_slot: usize,
    pub parent_index: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OperationDescriptor {
    pub gate: GateDescriptor,
    pub qubits: *const usize,
    pub qubits_len: usize,
    pub parameters: *const ParameterBindingDescriptor,
    pub parameters_len: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct CircuitPlanHandle {
    pub opaque: *mut std::ffi::c_void,
}

impl CircuitPlanHandle {
    pub const fn empty() -> Self {
        Self {
            opaque: std::ptr::null_mut(),
        }
    }
}

/// 稳定的门操作码；现有数值绝不能重新编号。
pub mod opcode {
    pub const I: u32 = 0;
    pub const X: u32 = 1;
    pub const Y: u32 = 2;
    pub const Z: u32 = 3;
    pub const H: u32 = 4;
    pub const S: u32 = 5;
    pub const SDG: u32 = 6;
    pub const T: u32 = 7;
    pub const TDG: u32 = 8;
    pub const SX: u32 = 9;
    pub const SXDG: u32 = 10;
    pub const RX: u32 = 11;
    pub const RY: u32 = 12;
    pub const RZ: u32 = 13;
    pub const PHASE: u32 = 14;
    pub const U3: u32 = 15;
    pub const CNOT: u32 = 16;
    pub const CY: u32 = 17;
    pub const CZ: u32 = 18;
    pub const CH: u32 = 19;
    pub const CS: u32 = 20;
    pub const CT: u32 = 21;
    pub const CPHASE: u32 = 22;
    pub const CRX: u32 = 23;
    pub const CRY: u32 = 24;
    pub const CRZ: u32 = 25;
    pub const SWAP: u32 = 26;
    pub const ISWAP: u32 = 27;
    pub const DCX: u32 = 28;
    pub const ECR: u32 = 29;
    pub const RXX: u32 = 30;
    pub const RYY: u32 = 31;
    pub const RZZ: u32 = 32;
    pub const RZX: u32 = 33;
    pub const FSIM: u32 = 34;
    pub const TOFFOLI: u32 = 35;
    pub const CSWAP: u32 = 36;
    pub const MCX: u32 = 37;
    pub const CUSTOM_UNITARY: u32 = 38;
}
