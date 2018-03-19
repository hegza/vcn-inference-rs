use util::*;
use super::{Coeff, Layer, LayerData};
use std::ops::Deref;
use ocl::SpatialDims;

/// A complete descriptor for a fully-connected layer
pub struct DenseLayer<T>
where
    T: Coeff,
{
    layer_data: LayerData<T>,
    num_in: usize,
    num_out: usize,
}

impl<T> DenseLayer<T>
where
    T: Coeff,
{
    /// Creates a descriptor of a fully-connected layer
    pub fn new(input_dim: usize, output_dim: usize, weights: Vec<T>) -> DenseLayer<T> {
        trace!(
            "Create dense-layer with input-size: {}, output-size: {}, weights-size: {}.",
            input_dim,
            output_dim,
            weights.len()
        );
        // Make sure that the weight count is correct
        debug_assert_eq!(input_dim * output_dim, weights.len());
        DenseLayer {
            layer_data: LayerData::<T> { weights },
            num_in: input_dim,
            num_out: output_dim,
        }
    }
}

impl<T> Deref for DenseLayer<T>
where
    T: Coeff,
{
    type Target = LayerData<T>;

    fn deref(&self) -> &Self::Target {
        &self.layer_data
    }
}

impl<T> Layer<T> for DenseLayer<T>
where
    T: Coeff,
{
    fn weights(&self) -> &Vec<T> {
        &self.weights
    }
    fn num_out(&self) -> usize {
        self.num_out
    }
    fn num_in(&self) -> usize {
        self.num_in
    }
    fn gws(&self) -> SpatialDims {
        SpatialDims::One(self.num_out)
    }
}
