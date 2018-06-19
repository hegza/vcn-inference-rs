use std;
use util::*;
use num_traits::{Float, Num, NumAssign, PrimInt, Zero};
use std::ops::{AddAssign, Mul};
use std::cmp::Ordering;
use std::mem::size_of;

pub fn mtx_mul<'lhs, T>(lhs: &'lhs [T], rhs: &[T], m_dim: usize, k_dim: usize) -> Vec<T>
where
    T: NumAssign + Zero + Copy,
{
    let mut c_mul = vec![Zero::zero(); m_dim * k_dim];
    for i in 0..m_dim {
        for j in 0..k_dim {
            for z in 0..rhs.len() {
                *c_mul.elem_mut(k_dim, i, j) += *lhs.elem(rhs.len(), i, z) * *rhs.elem(k_dim, z, j);
            }
        }
    }
    c_mul
}

/// Variant of matrix multiplication where the integers are normalized in the way described by van Houcke et al. (2011)
pub fn mtx_mul_normint(lhs: &[i8], rhs: &[i8], m_dim: usize, k_dim: usize) -> Vec<i8> {
    let mut c_mul = vec![Zero::zero(); m_dim * k_dim];
    for i in 0..m_dim {
        for j in 0..k_dim {
            let mut accumulator: i32 = Zero::zero();
            for z in 0..rhs.len() {
                accumulator += *lhs.elem(rhs.len(), i, z) as i32 * *rhs.elem(k_dim, z, j) as i32;
            }
            *c_mul.elem_mut(k_dim, i, j) = (accumulator >> 24) as i8;
        }
    }
    c_mul
}
