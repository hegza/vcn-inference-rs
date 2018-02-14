use std;
use util::*;

// TODO: generalize from f32 to anything zeroable, while asserting performance match with benchmarks on GPU

/// Convert negative values in source to zero
pub fn relu(source: &[f32], row: usize, column: usize) -> Vec<f32> {
    let mut destination = unsafe { vec![std::mem::uninitialized(); source.len()] };
    for i in 0..row {
        for j in 0..column {
            // Convert negative values to zero
            let elem = source.elem(column, i, j).max(0f32);
            *destination.elem_mut(column, i, j) = elem;
            /*
            // TODO: Try this alternative in-place implementation that works without the return
            // value (allocation).
            let elem: &mut f32 = destination.elem_mut(column, i, j);
            *elem = elem.max(0f32);
            */
        }
    }
    destination
}

pub fn mtx_mul(a: &[f32], b: &[f32], m_dim: usize, n_dim: usize, k_dim: usize) -> Vec<f32> {
    let mut c_mul = vec![0f32; m_dim * k_dim];
    for i in 0..m_dim {
        for j in 0..k_dim {
            for z in 0..n_dim {
                *c_mul.elem_mut(k_dim, i, j) += a.elem(n_dim, i, z) * b.elem(k_dim, z, j);
            }
        }
    }
    c_mul
}

pub fn softmax(input_buf: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut sum = 0f32;
    for i in 0..m {
        for j in 0..n {
            sum += input_buf.elem(n, i, j).exp();
        }
    }
    input_buf
        .iter()
        .map(|&val| val.exp() / sum)
        .collect::<Vec<f32>>()
}
