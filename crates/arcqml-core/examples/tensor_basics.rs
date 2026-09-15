use arcqml_core::{DType, Result, Tensor, TensorData};

fn main() -> Result<()> {
    // Tensor::new 会从输入推断 dtype 和 shape
    let scalar = Tensor::new(0.5_f64)?;
    let vector = Tensor::new(vec![1.0_f64, 2.0, 3.0])?;
    let matrix = Tensor::new(vec![vec![1.0_f64, 2.0, 3.0], vec![4.0, 5.0, 6.0]])?;
    // 这个写法和上面是等价的

    // TensorData 也可以在无法推断形状的场景下手动指定形状
    let flat_matrix = Tensor::new(TensorData::FlatF32 {
        data: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![2, 2],
    })?;

    // 创建零 Tensor
    let zeros = Tensor::zeros_with_dtype(vec![2, 3], DType::F32)?;

    // 创建标准正态 Tensor
    let random = Tensor::randn_with_seed(vec![2, 2], 42)?;

    println!(
        "scalar: shape={:?}, dtype={}",
        scalar.shape(),
        scalar.dtype()
    );
    println!(
        "vector: shape={:?}, data={:?}",
        vector.shape(),
        vector.storage()
    );
    println!(
        "matrix: shape={:?}, strides={:?}, contiguous={}",
        matrix.shape(),
        matrix.strides(),
        matrix.is_contiguous()
    );
    println!(
        "flat matrix: shape={:?}, data={:?}",
        flat_matrix.shape(),
        flat_matrix.storage()
    );
    println!("zeros: shape={:?}, dtype={}", zeros.shape(), zeros.dtype());
    println!("random: {:?}", random.storage());

    assert_eq!(scalar.shape(), &[]);
    assert_eq!(vector.shape(), &[3]);
    assert_eq!(matrix.shape(), &[2, 3]);
    assert_eq!(matrix.strides(), &[3, 1]);
    assert_eq!(flat_matrix.dtype(), DType::F32);

    Ok(())
}
