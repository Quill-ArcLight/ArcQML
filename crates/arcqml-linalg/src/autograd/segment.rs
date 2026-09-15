use arcqml_core::{ArcQmlError, BackwardFn, Result, Storage, Tensor};

/// 固定分段求和算子的反向规则。
#[derive(Debug)]
pub(crate) struct SegmentSumBackward {
    pub(crate) segment_ids: Vec<usize>,
    pub(crate) num_segments: usize,
}

impl BackwardFn for SegmentSumBackward {
    /// 按前向分段索引将每个输出梯度映射回对应输入元素。
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(ArcQmlError::AutogradError(
                "segment_sum backward expects one parent".to_string(),
            ));
        }
        let input = &parents[0];
        if grad_output.shape() != [self.num_segments] {
            return Err(ArcQmlError::AutogradError(format!(
                "segment_sum backward expected upstream shape [{}], got {:?}",
                self.num_segments,
                grad_output.shape()
            )));
        }
        let storage = match (&*input.storage(), &*grad_output.storage()) {
            (Storage::F32(_), Storage::F32(upstream)) => Storage::F32(
                self.segment_ids
                    .iter()
                    .map(|segment| upstream[*segment])
                    .collect(),
            ),
            (Storage::F64(_), Storage::F64(upstream)) => Storage::F64(
                self.segment_ids
                    .iter()
                    .map(|segment| upstream[*segment])
                    .collect(),
            ),
            (Storage::C64(_), Storage::C64(upstream)) => Storage::C64(
                self.segment_ids
                    .iter()
                    .map(|segment| upstream[*segment])
                    .collect(),
            ),
            _ => {
                return Err(ArcQmlError::AutogradError(format!(
                    "segment_sum backward dtype mismatch: input {}, upstream {}",
                    input.dtype(),
                    grad_output.dtype()
                )));
            }
        };
        Ok(vec![Some(Tensor::from_storage_meta(
            storage,
            input.meta().clone(),
        )?)])
    }
}
