#[cfg(test)]
mod test;

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
    fn generic_abs(&self) -> Self;
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

pub fn mtx_mul<'a, T>(v: &'a [T], b: &[T], m_dim: usize, k_dim: usize) -> Vec<T>
where
    T: NumAssign + Zero + Copy,
{
    let mut c_mul = vec![Zero::zero(); m_dim * k_dim];
    for i in 0..m_dim {
        for j in 0..k_dim {
            for z in 0..b.len() {
                *c_mul.elem_mut(k_dim, i, j) += *v.elem(b.len(), i, z) * *b.elem(k_dim, z, j);
            }
        }
    }
    c_mul
}

/// Variant of matrix multiplication where the integers are normalized in the way described by van Houcke et al. (2011)
pub fn mtx_mul_normint(v: &[i8], b: &[i8], m_dim: usize, k_dim: usize) -> Vec<i8> {
    let mut c_mul = vec![Zero::zero(); m_dim * k_dim];
    for i in 0..m_dim {
        for j in 0..k_dim {
            let mut accumulator: i32 = Zero::zero();
            for z in 0..b.len() {
                accumulator += *v.elem(b.len(), i, z) as i32 * *b.elem(k_dim, z, j) as i32;
            }
            *c_mul.elem_mut(k_dim, i, j) = (accumulator >> 24) as i8;
        }
    }
    c_mul
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
    fn generic_abs(&self) -> Self {
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
    fn generic_abs(&self) -> Self {
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
    fn generic_max(self, other: &i8) -> i8 {
        self.max(*other)
    }
    fn generic_abs(&self) -> Self {
        self.abs()
    }
    fn generic_partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
    fn generic_exp(self) -> f32 {
        (self as f32).exp()
    }
}

pub trait QuantizeInto<T> {
    /// Input must be bounded between min and max, it will then be scaled into the target type's range
    fn quantize(&self, min: Self, max: Self) -> T;
}

impl QuantizeInto<i8> for f32 {
    fn quantize(&self, min: f32, max: f32) -> i8 {
        debug_assert!(self >= &min && self <= &max);
        let p_range: f32 = max - min;
        let frac: f32 = (self - min) / p_range;
        let n_range: i32 = std::i8::MAX as i32 - std::i8::MIN as i32;
        ((frac * n_range as f32).round() as i32 + std::i8::MIN as i32) as i8
    }
}
