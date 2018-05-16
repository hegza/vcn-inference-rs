use super::*;
use math::*;

impl<T> ComputeOnHost<T> for DenseLayer<T>
where
    T: Coeff,
{
    fn compute(&self, in_buf: &[T]) -> Vec<T> {
        mtx_mul(self.weights(), in_buf, self.num_out(), 1)
    }
}
