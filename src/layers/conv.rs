use util::*;
use geometry::*;
use super::{Layer, LayerData};
use std::ops::Deref;
use ocl::SpatialDims;

/// A blueprint or a descriptor for a convolutional layer
pub struct ConvLayer {
    layer_data: LayerData<f32>,
    input_shape: ImageGeometry,
    output_shape: ImageGeometry,
}

impl ConvLayer {
    /// Creates a descriptor of a convolutional layer with a square filter the
    /// side of which will be set to filter_side.
    pub fn from_shapes(
        num_filter_elems: usize,
        input_shape: &ImageGeometry,
        output_shape: &ImageGeometry,
        weights_file: &str,
    ) -> ConvLayer {
        debug!(
            "Create conv-layer with filter-elems: {:?}, input-shape: {:?}, output-shape: {:?}, weights: {}.",
            num_filter_elems, input_shape, output_shape, weights_file
        );
        ConvLayer {
            layer_data: LayerData::<f32> {
                weights: read_file_as_f32s_checked(
                    weights_file,
                    num_filter_elems * input_shape.channels() * output_shape.channels(),
                ).unwrap(),
            },
            input_shape: input_shape.clone(),
            output_shape: output_shape.clone(),
        }
    }

    // The global work-group-size of the matching kernel
    pub fn gws(&self) -> SpatialDims {
        SpatialDims::Three(
            self.output_shape.channels(),
            self.output_shape.side(),
            self.output_shape.side(),
        )
    }

    pub fn input_shape(&self) -> &ImageGeometry {
        &self.input_shape
    }

    pub fn output_shape(&self) -> &ImageGeometry {
        &self.output_shape
    }
}

impl Deref for ConvLayer {
    type Target = LayerData<f32>;

    fn deref(&self) -> &Self::Target {
        &self.layer_data
    }
}

impl Layer<f32> for ConvLayer {
    fn weights(&self) -> &Vec<f32> {
        &self.weights
    }
    fn num_out(&self) -> usize {
        self.output_shape.num_elems()
    }
    fn num_in(&self) -> usize {
        self.input_shape.num_elems()
    }
}
