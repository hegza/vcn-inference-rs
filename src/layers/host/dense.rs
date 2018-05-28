use super::*;
use math::*;
use num_traits::{Float, Num, NumAssign, PrimInt, Zero};
use std::ops::{AddAssign, Mul};

impl<T> ComputeOnHost<T> for DenseLayer<T>
where
    T: CoeffFloat,
{
    fn compute(&self, in_buf: &[T]) -> Vec<T> {
        mtx_mul(self.weights(), in_buf, self.num_out(), 1)
    }
}

impl ComputeOnHost<i8> for DenseLayer<i8> {
    fn compute(&self, in_buf: &[i8]) -> Vec<i8> {
        mtx_mul_normint(self.weights(), in_buf, self.num_out(), 1)
    }
}
