#[cfg(test)]
mod test;
mod quantize;
mod mtx_mul;

pub use self::quantize::*;
pub use self::mtx_mul::*;
use std;
use util::*;
use num_traits::{Float, Num, NumAssign, PrimInt, Zero};
use std::ops::{AddAssign, Mul};
use std::cmp::Ordering;
use std::mem::size_of;

/// Combines the max operations of integral and float types.
pub trait GenericOps {
    /// Returns the higher value of the two.
    fn generic_max(self, other: &Self) -> Self;
    /// Returns the absolute value of the variable.
    fn generic_abs(self) -> Self;
    fn generic_partial_cmp(&self, other: &Self) -> Option<Ordering>;
    fn generic_exp(self) -> f32;
}

/// Convert negative values in source to zero
pub fn relu<T>(source: &[T]) -> Vec<T>
where
    T: Zero + GenericOps + Copy,
{
    source.iter().map(|&x| x.generic_max(&T::zero())).collect()
}

pub fn softmax<T>(input: &[T]) -> Vec<f32>
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
    fn generic_max(self, other: &f32) -> f32 {
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
        (self as f32).exp()
    }
}
