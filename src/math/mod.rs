#![allow(exceeding_bitshifts)]

mod quantize;
#[cfg(test)]
mod test;
// HACK: being pub might be hacky
mod convolve;
pub mod gemm;

pub use self::gemm::*;
pub use self::quantize::*;
use crate::util::*;
use num_traits::{Float, Num, NumAssign, PrimInt, Zero};
use std;
use std::cmp::Ordering;
use std::mem::size_of;
use std::ops::{AddAssign, Mul};

/// Naive matrix multiplication on the host
pub fn gemm_naive<T>(m: usize, n: usize, k: usize, a: &[T], b: &[T], c: &mut [T])
where
    T: NumAssign + Zero + Copy,
{
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for m_idx in 0..m {
        for n_idx in 0..n {
            let mut acc = T::zero();
            for k_idx in 0..k {
                acc += a[k_idx * m + m_idx] * b[n_idx * k + k_idx]
            }
            c[n_idx * m + m_idx] = acc;
        }
    }
}

pub fn gemm_naive_quantized_i8(m: usize, n: usize, k: usize, a: &[i8], b: &[i8], c: &mut [i8]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);

    for m_idx in 0..m {
        for n_idx in 0..n {
            let mut acc = 0;
            for k_idx in 0..k {
                acc += a[k_idx * m + m_idx] * b[n_idx * k + k_idx]
            }
            c[n_idx * m + m_idx] = (acc >> 24) as i8;
        }
    }
}

/// Combines the max operations of integral and float types.
pub trait GenericOps {
    /// Returns the higher value of the two.
    fn generic_max(self, other: &Self) -> Self;
    /// Returns the absolute value of the variable.
    fn generic_abs(self) -> Self;
    fn generic_partial_cmp(&self, other: &Self) -> Option<Ordering>;
    fn generic_exp(self) -> f32;
}

// TODO: &mut vs. consume?
/// Convert negative values in source to zero
pub fn relu<T>(source: Vec<T>) -> Vec<T>
where
    T: Zero + GenericOps + Copy,
{
    source
        .into_iter()
        .map(|x| x.generic_max(&T::zero()))
        .collect()
}

pub fn softmax<T>(input: Vec<T>) -> Vec<f32>
where
    T: GenericOps + Num + Copy,
{
    let m = input.len();
    let n = 1;
    let mut sum = 0f32;
    for i in 0..m {
        for j in 0..n {
            sum += input.elem(n, i, j).generic_exp();
        }
    }
    input
        .iter()
        .map(|&val| val.generic_exp() / sum)
        .collect::<Vec<f32>>()
}

impl GenericOps for f32 {
    fn generic_max(self, other: &Self) -> Self {
        self.max(*other)
    }
    fn generic_abs(self) -> Self {
        self.abs()
    }
    fn generic_partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
    fn generic_exp(self) -> f32 {
        self.exp()
    }
}

impl GenericOps for f64 {
    fn generic_max(self, other: &Self) -> Self {
        self.max(*other)
    }
    fn generic_abs(self) -> Self {
        self.abs()
    }
    fn generic_partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
    fn generic_exp(self) -> f32 {
        // HACK: maybe hacky
        self.exp() as f32
    }
}

impl GenericOps for i32 {
    fn generic_max(self, other: &i32) -> i32 {
        self.max(*other)
    }
    fn generic_abs(self) -> Self {
        self.abs()
    }
    fn generic_partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
    fn generic_exp(self) -> f32 {
        (self as f32).exp()
    }
}

impl GenericOps for i8 {
    fn generic_max(self, other: &Self) -> Self {
        self.max(*other)
    }
    fn generic_abs(self) -> Self {
        self.abs()
    }
    fn generic_partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
    fn generic_exp(self) -> f32 {
        f32::from(self).exp()
    }
}

impl GenericOps for u8 {
    fn generic_max(self, other: &Self) -> Self {
        self.max(*other)
    }
    fn generic_abs(self) -> Self {
        self
    }
    fn generic_partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
    fn generic_exp(self) -> f32 {
        f32::from(self).exp()
    }
}
