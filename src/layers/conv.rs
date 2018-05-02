use util::*;
use geometry::*;
use super::*;
use std::ops::Deref;
use ocl::SpatialDims;

/// A complete descriptor for a convolutional layer
#[derive(Clone)]
pub struct ConvLayer<T>
where
    T: Coeff,
{
    weights: Vec<T>,
    input_shape: ImageGeometry,
    output_shape: ImageGeometry,
}

impl<T> ConvLayer<T>
where
    T: Coeff,
{
    /// Creates a descriptor of a convolutional layer with a square filter the
    /// side of which will be set to filter_side.
    pub fn from_shapes(
        num_filter_elems: usize,
        input_shape: &ImageGeometry,
        output_shape: &ImageGeometry,
        weights: Vec<T>,
    ) -> ConvLayer<T> {
        // Make sure that the weight count is correct
        debug_assert_eq!(
            num_filter_elems * input_shape.channels() * output_shape.channels(),
            weights.len(),
            "layer: conv, weights: {}, expected: {}",
            weights.len(),
            num_filter_elems * input_shape.channels() * output_shape.channels(),
        );
        let layer = ConvLayer {
            weights,
            input_shape: *input_shape,
            output_shape: *output_shape,
        };
        debug!(
            "Create convolution layer with input: {}, output: {}, weights: {}.",
            layer.num_in(),
            layer.num_out(),
            layer.num_weights()
        );
        layer
    }

    pub fn input_shape(&self) -> &ImageGeometry {
        &self.input_shape
    }
    pub fn output_shape(&self) -> &ImageGeometry {
        &self.output_shape
    }
}

impl<T> Layer for ConvLayer<T>
where
    T: Coeff,
{
    fn num_out(&self) -> usize {
        self.output_shape.num_elems()
    }
    fn num_in(&self) -> usize {
        self.input_shape.num_elems()
    }
    fn gws_hint(&self) -> SpatialDims {
        SpatialDims::Three(
            self.output_shape.channels(),
            self.output_shape.side(),
            self.output_shape.side(),
        )
    }
    fn lws_hint(&self, _device_max_wgs: usize) -> SpatialDims {
        unimplemented!()
    }
    fn name(&self) -> &'static str {
        "conv"
    }
}

impl<T> WeightedLayer<T> for ConvLayer<T>
where
    T: Coeff,
{
    fn weights(&self) -> &Vec<T> {
        &self.weights
    }
}
