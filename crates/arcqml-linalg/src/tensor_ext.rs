use crate::error::LinalgResult;
use crate::ops;
use arcqml_core::Tensor;

/// 在 [`Tensor`] 上提供可微线性代数与逐元素运算的便捷方法。
///
/// 每个方法都直接委托给 [`crate::ops`] 中同名的自由函数，并具有相同的数值语义和错误条件。
pub trait TensorLinalgExt {
    /// 执行支持 NumPy 风格广播的逐元素加法。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::add`] 相同。
    fn add(&self, rhs: &Tensor) -> LinalgResult<Tensor>;

    /// 执行支持 NumPy 风格广播的逐元素减法。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::sub`] 相同。
    fn sub(&self, rhs: &Tensor) -> LinalgResult<Tensor>;

    /// 执行支持 NumPy 风格广播的逐元素乘法。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::mul`] 相同。
    fn mul(&self, rhs: &Tensor) -> LinalgResult<Tensor>;

    /// 执行支持 NumPy 风格广播的逐元素除法。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::div`] 相同。
    fn div(&self, rhs: &Tensor) -> LinalgResult<Tensor>;

    /// 返回逐元素相反数。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::neg`] 相同。
    fn neg(&self) -> LinalgResult<Tensor>;

    /// 返回逐元素自然指数。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::exp`] 相同。
    fn exp(&self) -> LinalgResult<Tensor>;

    /// 返回逐元素自然对数。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::log`] 相同。
    fn log(&self) -> LinalgResult<Tensor>;

    /// 返回逐元素 sigmoid。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::sigmoid`] 相同。
    fn sigmoid(&self) -> LinalgResult<Tensor>;

    /// 返回逐元素双曲正切。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::tanh`] 相同。
    fn tanh(&self) -> LinalgResult<Tensor>;

    /// 返回 `C64` Tensor 的逐元素复共轭。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::conj`] 相同。
    fn conj(&self) -> LinalgResult<Tensor>;

    /// 将每个实数元素限制在闭区间 `[min, max]` 内。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::clamp`] 相同。
    fn clamp(&self, min: f64, max: f64) -> LinalgResult<Tensor>;

    /// 执行二维矩阵乘法。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::matmul()`] 相同。
    fn matmul(&self, rhs: &Tensor) -> LinalgResult<Tensor>;

    /// 执行一维向量点积；复数输入不取共轭。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::dot()`] 相同。
    fn dot(&self, rhs: &Tensor) -> LinalgResult<Tensor>;

    /// 按行主序逻辑元素顺序改变形状。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::reshape()`] 相同。
    fn reshape(&self, shape: Vec<usize>) -> LinalgResult<Tensor>;

    /// 返回二维转置视图。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::transpose()`] 相同。
    fn transpose(&self) -> LinalgResult<Tensor>;

    /// 对全部元素求和，返回标量 Tensor。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::sum`] 相同。
    fn sum(&self) -> LinalgResult<Tensor>;

    /// 对全部元素求算术平均，返回标量 Tensor。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::mean`] 相同。
    fn mean(&self) -> LinalgResult<Tensor>;

    /// 沿指定轴求和。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::sum_dim`] 相同。
    fn sum_dim(&self, axis: usize, keepdim: bool) -> LinalgResult<Tensor>;

    /// 沿指定轴求算术平均。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::mean_dim`] 相同。
    fn mean_dim(&self, axis: usize, keepdim: bool) -> LinalgResult<Tensor>;

    /// 沿指定轴计算 softmax。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::softmax`] 相同。
    fn softmax(&self, axis: usize) -> LinalgResult<Tensor>;

    /// 沿指定轴计算 log-softmax。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::log_softmax`] 相同。
    fn log_softmax(&self, axis: usize) -> LinalgResult<Tensor>;

    /// 沿指定轴计算数值稳定的 logsumexp。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::logsumexp`] 相同。
    fn logsumexp(&self, axis: usize, keepdim: bool) -> LinalgResult<Tensor>;

    /// 按固定分段索引汇总一维 Tensor 的元素。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::segment_sum`] 相同。
    fn segment_sum(&self, segment_ids: &[usize], num_segments: usize) -> LinalgResult<Tensor>;

    /// 计算所有元素的 L2 范数。
    ///
    /// # Errors
    ///
    /// 错误条件与 [`ops::l2_norm`] 相同。
    fn l2_norm(&self) -> LinalgResult<Tensor>;
}

impl TensorLinalgExt for Tensor {
    fn add(&self, rhs: &Tensor) -> LinalgResult<Tensor> {
        ops::add(self, rhs)
    }

    fn sub(&self, rhs: &Tensor) -> LinalgResult<Tensor> {
        ops::sub(self, rhs)
    }

    fn mul(&self, rhs: &Tensor) -> LinalgResult<Tensor> {
        ops::mul(self, rhs)
    }

    fn div(&self, rhs: &Tensor) -> LinalgResult<Tensor> {
        ops::div(self, rhs)
    }

    fn neg(&self) -> LinalgResult<Tensor> {
        ops::neg(self)
    }

    fn exp(&self) -> LinalgResult<Tensor> {
        ops::exp(self)
    }

    fn log(&self) -> LinalgResult<Tensor> {
        ops::log(self)
    }

    fn sigmoid(&self) -> LinalgResult<Tensor> {
        ops::sigmoid(self)
    }

    fn tanh(&self) -> LinalgResult<Tensor> {
        ops::tanh(self)
    }

    /// 调用逐元素复共轭基础算子。
    fn conj(&self) -> LinalgResult<Tensor> {
        ops::conj(self)
    }

    fn clamp(&self, min: f64, max: f64) -> LinalgResult<Tensor> {
        ops::clamp(self, min, max)
    }

    fn matmul(&self, rhs: &Tensor) -> LinalgResult<Tensor> {
        ops::matmul(self, rhs)
    }

    fn dot(&self, rhs: &Tensor) -> LinalgResult<Tensor> {
        ops::dot(self, rhs)
    }

    fn reshape(&self, shape: Vec<usize>) -> LinalgResult<Tensor> {
        ops::reshape(self, shape)
    }

    fn transpose(&self) -> LinalgResult<Tensor> {
        ops::transpose(self)
    }

    fn sum(&self) -> LinalgResult<Tensor> {
        ops::sum(self)
    }

    fn mean(&self) -> LinalgResult<Tensor> {
        ops::mean(self)
    }

    fn sum_dim(&self, axis: usize, keepdim: bool) -> LinalgResult<Tensor> {
        ops::sum_dim(self, axis, keepdim)
    }

    fn mean_dim(&self, axis: usize, keepdim: bool) -> LinalgResult<Tensor> {
        ops::mean_dim(self, axis, keepdim)
    }

    fn softmax(&self, axis: usize) -> LinalgResult<Tensor> {
        ops::softmax(self, axis)
    }

    fn log_softmax(&self, axis: usize) -> LinalgResult<Tensor> {
        ops::log_softmax(self, axis)
    }

    fn logsumexp(&self, axis: usize, keepdim: bool) -> LinalgResult<Tensor> {
        ops::logsumexp(self, axis, keepdim)
    }

    /// 调用固定分段求和基础算子。
    fn segment_sum(&self, segment_ids: &[usize], num_segments: usize) -> LinalgResult<Tensor> {
        ops::segment_sum(self, segment_ids, num_segments)
    }

    fn l2_norm(&self) -> LinalgResult<Tensor> {
        ops::l2_norm(self)
    }
}
