use arcqml_core::{Tensor, TensorData};
use arcqml_linalg::prelude::*;
use num_complex::Complex64;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // [2, 3] 输入与 [3] 偏置相加，演示 NumPy 风格广播。
    let input = Tensor::new(TensorData::FlatF64 {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape: vec![2, 3],
    })?;
    let bias = Tensor::new(TensorData::FlatF64 {
        data: vec![10.0, 20.0, 30.0],
        shape: vec![3],
    })?;
    let shifted = add(&input, &bias)?;
    println!("broadcast add: {:?}", shifted.storage().as_f64_slice());

    // 矩阵乘法的形状为 [2, 3] @ [3, 2] -> [2, 2]。
    let weight = Tensor::new(TensorData::FlatF64 {
        data: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        shape: vec![3, 2],
    })?;
    let output = matmul(&shifted, &weight)?;
    println!("matmul: {:?}", output.storage().as_f64_slice());

    // 反向传播：loss = sum(parameter * coefficient)。
    let parameter = Tensor::new(TensorData::FlatF64 {
        data: vec![2.0, -1.0],
        shape: vec![2],
    })?;
    parameter.set_requires_grad(true);
    let coefficient = Tensor::new(TensorData::FlatF64 {
        data: vec![3.0, 4.0],
        shape: vec![2],
    })?;
    let loss = sum(&mul(&parameter, &coefficient)?)?;
    loss.backward()?;
    println!(
        "real gradient: {:?}",
        parameter.grad().unwrap().storage().as_f64_slice()
    );

    // C64 的 L2 norm 返回实值 loss，反向使用共轭 Wirtinger 约定。
    let state = Tensor::new(TensorData::FlatC64 {
        data: vec![Complex64::new(3.0, 4.0)],
        shape: vec![1],
    })?;
    state.set_requires_grad(true);
    let norm = l2_norm(&state)?;
    norm.backward()?;
    println!("c64 L2 norm: {:?}", norm.storage().as_f64_slice());
    println!(
        "c64 gradient: {:?}",
        state.grad().unwrap().storage().as_c64_slice()
    );

    Ok(())
}
