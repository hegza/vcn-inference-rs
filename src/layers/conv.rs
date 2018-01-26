use util::*;
use geometry::*;

/// A blueprint or a descriptor for a convolutional layer
pub struct ConvLayer {
    filter_shape: PaddedSquare,
    weights: Vec<f32>,
}

impl ConvLayer {
    /// Creates a descriptor of a convolutional layer with a square filter the
    /// side of which will be set to filter_side.
    pub fn new(
        filter_side: usize,
        filter_channels: usize,
        feature_map_count: usize,
        weights_file: &str,
    ) -> ConvLayer {
        let filter_shape = PaddedSquare::from_side(filter_side);
        ConvLayer {
            filter_shape: filter_shape,
            weights: read_file_as_f32s_checked(
                weights_file,
                filter_shape.num_elements() * filter_channels * feature_map_count,
            ).unwrap(),
        }
    }
    pub fn weights(&self) -> &Vec<f32> {
        &self.weights
    }
    pub fn num_weights(&self) -> usize {
        self.weights.len()
    }
    pub fn filter_shape(&self) -> &PaddedSquare {
        &self.filter_shape
    }
}
