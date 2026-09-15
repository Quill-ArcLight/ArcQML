use arcqml_core::{ArcQmlError, BackwardFn, Result, Storage, Tensor};

/// C64 共轭算子的反向规则。
#[derive(Debug)]
pub(crate) struct ConjBackward;

impl BackwardFn for ConjBackward {
    /// 将 C64 上游梯度取共轭后传回输入。
    fn backward(&self, parents: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        if parents.len() != 1 {
            return Err(ArcQmlError::AutogradError(
                "conj backward expects one parent".to_string(),
            ));
        }
        let input = &parents[0];
        let Storage::C64(upstream) = &*grad_output.storage() else {
            return Err(ArcQmlError::AutogradError(format!(
                "conj backward requires C64 upstream gradient, got {}",
                grad_output.dtype()
            )));
        };
        Ok(vec![Some(Tensor::from_storage_meta(
            Storage::C64(upstream.iter().map(|value| value.conj()).collect()),
            input.meta().clone(),
        )?)])
    }
}
