use util::*;
use super::LayerData;
use std::ops::Deref;

/// A blueprint or a descriptor for a fully-connected layer
pub struct DenseLayer(LayerData<f32>);

impl DenseLayer {
    /// Creates a descriptor of a fully-connected layer
    pub fn new(input_dim: usize, output_dim: usize, weights_file: &str) -> DenseLayer {
        debug!(
            "Create dense-layer with input-size: {}, output-size: {}, weights: {}.",
            input_dim, output_dim, weights_file
        );
        trace!(
            "\t↳ input: {0}, output: {1}, weights-size: {0}x{1} = {2}.",
            input_dim,
            output_dim,
            input_dim * output_dim
        );
        DenseLayer(LayerData::<f32> {
            num_in: input_dim,
            num_out: output_dim,
            weights: read_file_as_f32s_checked(weights_file, input_dim * output_dim).unwrap(),
        })
    }
}

impl Deref for DenseLayer {
    type Target = LayerData<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
