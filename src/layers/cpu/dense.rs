use super::*;
use math::*;

pub trait CpuDense<T>: WeightedLayer<T>
where
    T: Coeff,
{
    fn mtx_mul(&self, input_buffer: &[T]) -> Vec<T> {
        mtx_mul(
            self.weights(),
            input_buffer,
            self.num_out(),
            self.num_in(),
            1,
        )
    }
}

impl<T> CpuDense<T> for DenseLayer<T>
where
    T: Coeff,
{
}
