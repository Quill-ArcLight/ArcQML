use crate::{ArcQmlError, DType, Device, Layout, Result};

/// Tensor 的形状、步长、偏移、dtype、设备和布局元数据。
#[derive(Debug, Clone)]
pub struct TensorMeta {
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
    dtype: DType,
    device: Device,
    layout: Layout,
}

impl TensorMeta {
    /// 创建零偏移元数据，并根据 `shape` 计算行主序连续步长。
    ///
    /// # Errors
    ///
    /// 当形状、步长、偏移、设备或布局无效，或元素数量计算溢出时返回错误。
    pub fn new(shape: Vec<usize>, dtype: DType, device: Device, layout: Layout) -> Result<Self> {
        let strides = Self::default_strides(&shape)?;
        Self::from_parts(shape, strides, 0, dtype, device, layout)
    }

    /// 使用完整信息创建 TensorMeta。
    ///
    /// # Errors
    ///
    /// 当形状、步长、偏移、设备或布局无效，或元素数量计算溢出时返回错误。
    pub fn from_parts(
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
        dtype: DType,
        device: Device,
        layout: Layout,
    ) -> Result<Self> {
        Self::validate_supported_backend(&device, layout)?;
        Self::validate_parts(&shape, &strides, offset)?;

        Ok(Self {
            shape,
            strides,
            offset,
            dtype,
            device,
            layout,
        })
    }

    /// 返回 Tensor 形状。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 返回 Tensor 步长。
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// 返回首个逻辑元素在底层存储中的元素偏移。
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// 返回 Tensor 数值类型。
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// 返回 Tensor 所在设备。
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// 返回 Tensor 存储布局。
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// 返回逻辑元素数量。
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// 判断元数据是否描述零偏移、行主序连续视图。
    pub fn is_contiguous(&self) -> bool {
        self.offset == 0
            && Self::default_strides(&self.shape)
                .map(|strides| self.strides == strides)
                .unwrap_or(false)
    }

    /// 检查张量视图描述。
    fn validate_parts(shape: &[usize], strides: &[isize], offset: usize) -> Result<()> {
        if shape.len() != strides.len() {
            return Err(ArcQmlError::ShapeError(format!(
                "shape rank {} does not match stride rank {}",
                shape.len(),
                strides.len()
            )));
        }

        if strides.iter().any(|&stride| stride < 0) {
            return Err(ArcQmlError::ShapeError(
                "negative strides are not supported".to_owned(),
            ));
        }

        shape.iter().try_fold(1usize, |numel, &dim| {
            numel.checked_mul(dim).ok_or_else(|| {
                ArcQmlError::ShapeError("tensor element count overflows usize".to_owned())
            })
        })?;

        if shape.is_empty() {
            offset.checked_add(1).ok_or_else(|| {
                ArcQmlError::ShapeError("tensor storage range overflows usize".to_owned())
            })?;
            return Ok(());
        }

        if shape.contains(&0) {
            return Ok(());
        }

        let max_index =
            shape
                .iter()
                .zip(strides)
                .try_fold(offset, |max_index, (&dim, &stride)| {
                    let step = (stride as usize).checked_mul(dim - 1).ok_or_else(|| {
                        ArcQmlError::ShapeError("tensor storage range overflows usize".to_owned())
                    })?;
                    max_index.checked_add(step).ok_or_else(|| {
                        ArcQmlError::ShapeError("tensor storage range overflows usize".to_owned())
                    })
                })?;

        max_index.checked_add(1).ok_or_else(|| {
            ArcQmlError::ShapeError("tensor storage range overflows usize".to_owned())
        })?;

        Ok(())
    }

    /// 检查当前框架后端和内存布局。
    fn validate_supported_backend(device: &Device, layout: Layout) -> Result<()> {
        if !matches!(device, Device::Cpu) {
            return Err(ArcQmlError::NotImplementedError(format!(
                "device {device} is not supported"
            )));
        }

        if !matches!(layout, Layout::Dense) {
            return Err(ArcQmlError::NotImplementedError(format!(
                "layout {layout} is not supported"
            )));
        }

        Ok(())
    }

    /// 返回该视图至少需要的底层存储元素数。
    ///
    /// 根据 `最大下标 = offset + Σ strides[k] × (shape[k] - 1)` 计算；标量需要
    /// `offset + 1` 个元素，含零维度的空 Tensor 需要 `offset` 个元素。
    pub fn storage_len_required(&self) -> usize {
        if self.shape.is_empty() {
            return self.offset + 1;
        } // 标量

        if self.shape.contains(&0) {
            return self.offset;
        } // Tensor 中没有元素

        let mut max_index = self.offset;

        for (dim, stride) in self.shape.iter().zip(self.strides.iter()) {
            if *dim == 0 {
                return self.offset;
            }

            let step = stride.unsigned_abs() * (dim - 1);

            max_index += step;
        }

        max_index + 1
    }

    /// 根据 shape 计算默认连续 strides。
    ///
    /// 例如形状 `[2, 3, 4]` 的默认步长为 `[12, 4, 1]`。
    fn default_strides(shape: &[usize]) -> Result<Vec<isize>> {
        if shape.is_empty() {
            return Ok(Vec::new());
        }

        let mut strides = vec![1isize; shape.len()];
        let mut stride = 1usize;

        for index in (0..shape.len()).rev() {
            strides[index] = isize::try_from(stride)
                .map_err(|_| ArcQmlError::ShapeError("tensor stride overflows isize".to_owned()))?;
            stride = stride.checked_mul(shape[index]).ok_or_else(|| {
                ArcQmlError::ShapeError("tensor element count overflows usize".to_owned())
            })?;
        }

        Ok(strides)
    }
}
