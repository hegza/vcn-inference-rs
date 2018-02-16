use std;
use util::*;
use num_traits::{Float, Num, NumAssign, Zero};
use std::ops::Mul;

/// Combines the max operations of integral and float types.
pub trait GenericOps {
    /// Returns the higher value of the two.
    fn generic_max(self, other: Self) -> Self;
    /// Returns the absolute value of the variable.
    fn generic_abs(&self) -> Self;
}

/// Convert negative values in source to zero
pub fn relu<T>(source: &[T]) -> Vec<T>
where
    T: Num + GenericOps + Copy,
{
    source
        .iter()
        .map(|&x| x.generic_max(Zero::zero()))
        .collect()
}

pub fn mtx_mul<T>(a: &[T], b: &[T], m_dim: usize, n_dim: usize, k_dim: usize) -> Vec<T>
where
    T: NumAssign + Mul + Zero + Copy,
{
    let mut c_mul = vec![Zero::zero(); m_dim * k_dim];
    for i in 0..m_dim {
        for j in 0..k_dim {
            for z in 0..n_dim {
                *c_mul.elem_mut(k_dim, i, j) += *a.elem(n_dim, i, z) * *b.elem(k_dim, z, j);
            }
        }
    }
    c_mul
}

pub fn softmax<T>(input_buf: &[T], m: usize, n: usize) -> Vec<T>
where
    T: Float + NumAssign,
{
    let mut sum = Zero::zero();
    for i in 0..m {
        for j in 0..n {
            sum += input_buf.elem(n, i, j).exp();
        }
    }
    input_buf
        .iter()
        .map(|&val| val.exp() / sum)
        .collect::<Vec<T>>()
}

impl GenericOps for f32 {
    fn generic_max(self, other: f32) -> f32 {
        self.max(other)
    }
    fn generic_abs(&self) -> Self {
        self.abs()
    }
}

impl GenericOps for i32 {
    fn generic_max(self, other: i32) -> i32 {
        self.max(other)
    }
    fn generic_abs(&self) -> Self {
        self.abs()
    }
}
