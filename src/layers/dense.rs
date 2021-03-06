use super::*;
use crate::util::*;
use ocl::SpatialDims;
use std::fmt;
use std::ops::Deref;

/// A complete descriptor for a fully-connected layer
#[derive(Clone)]
pub struct DenseLayer<T>
where
    T: Coeff,
{
    weights: Vec<T>,
    num_in: usize,
    num_out: usize,
}

impl<T> DenseLayer<T>
where
    T: Coeff,
{
    /// Creates a descriptor of a fully-connected layer
    pub fn new(input_dim: usize, output_dim: usize, weights: Vec<T>) -> DenseLayer<T> {
        // Make sure that the weight count is correct
        debug_assert_eq!(
            input_dim * output_dim,
            weights.len(),
            "layer: dense, weights: {}, expected: {}",
            weights.len(),
            input_dim * output_dim
        );
        let layer = DenseLayer {
            weights,
            num_in: input_dim,
            num_out: output_dim,
        };
        debug!(
            "Create dense layer with input: {}, output: {}, weights: {}.",
            layer.num_in(),
            layer.num_out(),
            layer.num_weights()
        );

        layer
    }
}

impl<T> Layer for DenseLayer<T>
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
        SpatialDims::One(self.num_out)
    }
    fn lws_hint(&self, _device_max_wgs: usize) -> SpatialDims {
        unimplemented!()
    }
    fn name(&self) -> &'static str {
        "dense"
    }
}

impl<T> WeightedLayer<T> for DenseLayer<T>
where
    T: Coeff,
{
    fn weights(&self) -> &[T] {
        &self.weights
    }
}

impl<T> fmt::Debug for DenseLayer<T>
where
    T: Coeff,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "DenseLayer {{ in: {}, out: {}}}",
            self.num_in, self.num_out
        )
    }
}
