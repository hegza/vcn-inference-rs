use util::*;
use geometry::*;
use super::LayerData;
use std::ops::Deref;

/// A blueprint or a descriptor for a convolutional layer
pub struct ConvLayer(LayerData<f32>);

impl ConvLayer {
    /// Creates a descriptor of a convolutional layer with a square filter the
    /// side of which will be set to filter_side.
    pub fn from_shapes(
        filter_shape: &PaddedSquare,
        input_shape: &ImageGeometry,
        output_shape: &ImageGeometry,
        weights_file: &str,
    ) -> ConvLayer {
        debug!(
            "Create conv-layer with filter-shape: {:?}, input-shape: {:?}, output-shape: {:?}, weights: {}.",
            filter_shape, input_shape, output_shape, weights_file
        );
        trace!(
            "\t↳ input (size: {0}, channels: {1}), output: (size: {2}, channels: {3}), filter-size: {4}, weights-size: {4}x{2}x{2} = {5}.",
            input_shape.num_elems(),
            input_shape.channels(),
            output_shape.num_elems(),
            output_shape.channels(),
            filter_shape.num_elems(),
            filter_shape.num_elems() * input_shape.channels() * output_shape.channels()
        );
        ConvLayer(LayerData::<f32> {
            num_in: input_shape.num_elems(),
            num_out: output_shape.num_elems(),
            weights: read_file_as_f32s_checked(
                weights_file,
                filter_shape.num_elems() * input_shape.channels() * output_shape.channels(),
            ).unwrap(),
        })
    }
}

impl Deref for ConvLayer {
    type Target = LayerData<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
