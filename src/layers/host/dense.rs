use super::*;
use math::*;
use num_traits::{Float, Num, NumAssign, PrimInt, Zero};
use std::ops::{AddAssign, Mul};

impl<T> ComputeOnHost<T> for DenseLayer<T>
where
    T: CoeffFloat,
{
    fn compute(&self, in_buf: &[T]) -> Vec<T> {
        let mut c = vec![Zero::zero(); self.num_out()];
        gemm_naive(
            1,
            self.num_out(),
            in_buf.len(),
            in_buf,
            self.weights(),
            &mut c,
        );
        c
    }
}
