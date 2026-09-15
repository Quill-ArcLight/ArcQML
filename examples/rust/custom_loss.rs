//! 自定义损失：算子组合、量子电路训练，以及 CustomOp 专用反向。
//! 运行：cargo run -p arcqml --example custom_loss
//! 教程：docs/tutorial/custom_loss.md

use arcqml::prelude::*;
use num_complex::Complex64;

type AppResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// 本示例只接受有限、非空、完整连续存储的 F64 张量。
fn validate_input(tensor: &Tensor) -> AppResult<()> {
    if tensor.dtype() != DType::F64
        || tensor.device() != &Device::Cpu
        || tensor.layout() != Layout::Dense
        || tensor.numel() == 0
        || !tensor.is_contiguous()
        || tensor.storage().len() != tensor.numel()
    {
        return Err(
            "expected a nonempty, contiguous CPU Dense F64 tensor with exact storage length".into(),
        );
    }
    if !tensor
        .storage()
        .as_f64_slice()
        .unwrap()
        .iter()
        .all(|x| x.is_finite())
    {
        return Err("loss inputs must be finite".into());
    }
    Ok(())
}

/// L = mean(w * (prediction - target)^2) / 2；按元素数归约，不按权重和归约。
fn weighted_mse(prediction: &Tensor, target: &Tensor, weights: &Tensor) -> AppResult<Tensor> {
    for tensor in [prediction, target, weights] {
        validate_input(tensor)?;
    }
    if prediction.shape() != target.shape() || prediction.shape() != weights.shape() {
        return Err("prediction, target and weights must have the same shape".into());
    }
    if weights
        .storage()
        .as_f64_slice()
        .unwrap()
        .iter()
        .any(|&w| w < 0.0)
    {
        return Err("weights must be nonnegative".into());
    }
    let error = sub(prediction, target)?;
    let weighted_squared_error = mul(weights, &square(&error)?)?;
    Ok(mul(
        &mean(&weighted_squared_error)?,
        &Tensor::new(0.5_f64)?,
    )?)
}

/// 进阶示例：标量 log(cosh(r)) 的稳定前向与整体反向。
#[derive(Debug)]
struct LogCosh;

impl CustomOp for LogCosh {
    type Context = f64;

    fn name(&self) -> &'static str {
        "scalar_log_cosh"
    }

    fn forward(&self, inputs: &[Tensor]) -> arcqml::core::Result<(Tensor, f64)> {
        let [input] = inputs else {
            return Err(ArcQmlError::AutogradError(
                "log_cosh expects one input".into(),
            ));
        };
        validate_input(input).map_err(|e| ArcQmlError::AutogradError(e.to_string()))?;
        if !input.shape().is_empty() {
            return Err(ArcQmlError::AutogradError(
                "log_cosh expects a scalar".into(),
            ));
        }
        let r = input.value()?;
        let a = r.abs();
        let value = a + (-2.0 * a).exp().ln_1p() - std::f64::consts::LN_2;
        // 保存数值快照，反向不依赖调用方后来修改的存储。
        Ok((Tensor::new(value)?, r))
    }

    fn backward(&self, r: &f64, grad_output: &Tensor) -> arcqml::core::Result<Vec<Option<Tensor>>> {
        validate_input(grad_output).map_err(|e| ArcQmlError::AutogradError(e.to_string()))?;
        if !grad_output.shape().is_empty() {
            return Err(ArcQmlError::AutogradError(
                "log_cosh expects a scalar upstream gradient".into(),
            ));
        }
        Ok(vec![Some(Tensor::new(grad_output.value()? * r.tanh())?)])
    }
}

/// 同时用解析式和中心差分验证经典损失，再验证专用反向的上游缩放。
fn check_gradients() -> AppResult<()> {
    let prediction = Tensor::new(vec![0.8_f64, -0.4])?;
    prediction.set_requires_grad(true);
    let target = Tensor::new(vec![0.2_f64, -0.2])?;
    let weights = Tensor::new(vec![1.0_f64, 3.0])?;
    weighted_mse(&prediction, &target, &weights)?.backward()?;
    let gradient = prediction.grad().ok_or("missing prediction gradient")?;
    let values = [0.8_f64, -0.4];
    for (index, expected) in [0.3_f64, -0.3].into_iter().enumerate() {
        let h = 1e-6;
        let mut plus = values;
        let mut minus = values;
        plus[index] += h;
        minus[index] -= h;
        let numerical = (weighted_mse(&Tensor::new(plus.to_vec())?, &target, &weights)?.value()?
            - weighted_mse(&Tensor::new(minus.to_vec())?, &target, &weights)?.value()?)
            / (2.0 * h);
        let actual = gradient.storage().as_f64_slice().unwrap()[index];
        assert!((actual - expected).abs() < 1e-12);
        assert!((actual - numerical).abs() < 1e-8);
    }

    for r in [-1000.0_f64, -2.0, 0.0, 2.0, 1000.0] {
        let residual = Tensor::new(r)?;
        residual.set_requires_grad(true);
        let loss = apply_custom_op(LogCosh, std::slice::from_ref(&residual))?;
        assert!(loss.value()?.is_finite());
        if r == 0.0 {
            assert!(loss.value()?.abs() < 1e-12);
        }
        loss.backward_with_grad(Tensor::new(3.0_f64)?)?;
        assert!(
            (residual
                .grad()
                .ok_or("missing residual gradient")?
                .value()?
                - 3.0 * r.tanh())
            .abs()
                < 1e-12
        );
    }
    println!("Gradient checks passed (weighted MSE and custom log-cosh).");
    Ok(())
}

fn main() -> AppResult<()> {
    check_gradients()?;

    // 两条独立初态 |0>、|1>，共享同一个 RY(theta)。
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let states = Tensor::new(TensorData::FlatC64 {
        data: vec![one, zero, zero, one],
        shape: vec![2, 2],
    })?;
    let simulator = BatchStateVectorSimulator::from_state_tensor(1, states)?;
    let mut circuit = Circuit::new(1)?;
    circuit.ry(0.3, 0usize)?;
    let observable = SparsePauliOp::single(1, 0usize, Pauli::Z, 1.0)?;
    let target = Tensor::new(vec![0.2_f64, -0.2])?;
    let weights = Tensor::new(vec![1.0_f64, 3.0])?;
    let optimizer = Sgd::new(0.1, 0.0)?;

    for step in 0..80 {
        let prediction = simulator.run(&circuit, &observable)?;
        let loss = weighted_mse(&prediction, &target, &weights)?;
        loss.backward()?;

        if step == 0 {
            // 此电路 L = (cos(theta) - 0.2)^2。
            let expected = -2.0 * (0.3_f64.cos() - 0.2) * 0.3_f64.sin();
            let actual = circuit.parameters()[0]
                .tensor()
                .grad()
                .ok_or("missing circuit gradient")?
                .value()?;
            assert!((actual - expected).abs() < 1e-10);
            println!("Circuit gradient: {actual:.10} (analytic: {expected:.10}).");
        }
        if step % 20 == 0 {
            println!("step {step:02}: loss = {:.10}", loss.value()?);
        }
        optimizer.step(circuit.parameters())?;
        optimizer.zero_grad(circuit.parameters());
    }

    let _guard = no_grad();
    let prediction = simulator.run(&circuit, &observable)?;
    let final_loss = weighted_mse(&prediction, &target, &weights)?.value()?;
    println!(
        "final predictions: {:?}",
        prediction.storage().as_f64_slice().unwrap()
    );
    println!("final loss: {final_loss:.3e}");
    assert!(final_loss < 1e-10, "training did not converge");
    Ok(())
}
