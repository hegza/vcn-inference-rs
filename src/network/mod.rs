use super::*;

pub mod classic;
pub mod sepconv;

pub use classic::*;
pub use sepconv::*;
// For input-shape
pub use geometry::ImageGeometry;

pub fn mtxmul_relu<T>(input_buffer: &[T], dense: &DenseLayer<T>) -> Vec<T>
where
    T: CoeffFloat,
{
    let out = mtx_mul(
        dense.weights(),
        input_buffer,
        dense.num_out(),
        dense.num_in(),
        1,
    );
    relu(&out)
}

pub fn mtxmul_softmax<F>(input_buffer: &[F], dense: &DenseLayer<F>) -> Vec<F>
where
    F: CoeffFloat,
{
    let out = mtx_mul(
        dense.weights(),
        input_buffer,
        dense.num_out(),
        dense.num_in(),
        1,
    );
    softmax(&out, dense.num_out(), 1)
}
