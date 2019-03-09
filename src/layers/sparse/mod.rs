#[cfg(test)]
mod test;

use super::*;
use crate::util::*;
use ocl::SpatialDims;
use sprs::{CsMat, CsMatBase, CsMatI, CsVec, TriMatBase};
use std::ops::Deref;

const FLOAT_ZERO_THRESHOLD: f32 = 0.000_000_1f32;

/// A complete descriptor for a sparsified fully-connected layer
pub struct SparseLayer<T>
where
    T: Coeff,
{
    weights: CsMatI<T, usize>,
    num_in: usize,
    num_out: usize,
}

impl SparseLayer<f32> {
    /// Creates a descriptor of a fully-connected layer
    pub fn from_dense(
        input_dim: usize,
        output_dim: usize,
        dense_weights: Vec<f32>,
    ) -> SparseLayer<f32> {
        // Make sure that the weight count is correct
        debug_assert_eq!(
            input_dim * output_dim,
            dense_weights.len(),
            "\nlayer: sparse, num_weights: {}, expected: {}",
            dense_weights.len(),
            input_dim * output_dim
        );

        let num_nnz = dense_weights
            .iter()
            .filter(|&val| val.abs() >= FLOAT_ZERO_THRESHOLD)
            .count();

        let weights = {
            let (rows, cols) = (input_dim, output_dim);
            let mut sparse =
                TriMatBase::<Vec<usize>, Vec<f32>>::with_capacity((rows, cols), num_nnz);

            for row in 0..rows {
                for col in 0..cols {
                    let val = dense_weights[row * cols + col];
                    if val.abs() >= FLOAT_ZERO_THRESHOLD {
                        sparse.add_triplet(row, col, val);
                    }
                }
            }
            sparse.to_csc()
        };

        let layer = SparseLayer {
            weights,
            num_in: input_dim,
            num_out: output_dim,
        };
        debug!(
            "Create Sparse layer with input: {}, output: {}, weights: {}.",
            layer.num_in(),
            layer.num_out(),
            layer.num_weights()
        );

        layer
    }
}

impl<T> Layer for SparseLayer<T>
where
    T: Coeff,
{
    fn num_out(&self) -> usize {
        self.num_out
    }
    fn num_in(&self) -> usize {
        self.num_in
    }
    fn gws_hint(&self) -> SpatialDims {
        unimplemented!()
    }
    fn lws_hint(&self, _device_max_wgs: usize) -> SpatialDims {
        unimplemented!()
    }
    fn name(&self) -> &'static str {
        "sparse"
    }
}

impl<T> WeightedLayer<T> for SparseLayer<T>
where
    T: Coeff,
{
    fn weights(&self) -> &[T] {
        panic!("getting the weights of a sparse layer without indices makes no sense")
    }
}

use ndarray;
use ndarray::{arr2, Array, ArrayView, FixedInitializer};
use sprs;
use sprs::binop::mul_dense_mat_same_ordering;

impl<T> ComputeOnHost<T> for SparseLayer<T>
where
    T: CoeffFloat,
{
    // TODO: optimize
    // C = A * B (in * wgt); weights are CHWN
    fn compute(&self, in_buf: &[T]) -> Vec<T> {
        let mut out: Vec<T> = vec![T::zero(); self.num_out()];

        let mat = self.weights.view();

        for (col_ind, vec) in mat.outer_iterator().enumerate() {
            let mut acc = T::zero();
            for (row_ind, &value) in vec.iter() {
                acc += in_buf[row_ind] * value;
            }
            out[col_ind] = acc;
        }

        out
    }
}
