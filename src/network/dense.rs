use util::*;
use super::{Layer, LayerData};
use std::ops::Deref;
use ocl::SpatialDims;

/// A complete descriptor for a fully-connected layer
pub struct DenseLayer {
    layer_data: LayerData<f32>,
    num_in: usize,
    num_out: usize,
}

impl DenseLayer {
    /// Creates a descriptor of a fully-connected layer
    pub fn new(input_dim: usize, output_dim: usize, weights_file: &str) -> DenseLayer {
        trace!(
            "Create dense-layer with input-size: {}, output-size: {}, weights: {}.",
            input_dim,
            output_dim,
            weights_file
        );
        DenseLayer {
            layer_data: LayerData::<f32> {
                weights: read_file_as_f32s(weights_file),
            },
            num_in: input_dim,
            num_out: output_dim,
        }
    }
}

impl Deref for DenseLayer {
    type Target = LayerData<f32>;

    fn deref(&self) -> &Self::Target {
        &self.layer_data
    }
}

impl Layer<f32> for DenseLayer {
    fn weights(&self) -> &Vec<f32> {
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
