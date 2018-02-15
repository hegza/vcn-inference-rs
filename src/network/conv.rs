use util::*;
use geometry::*;
use super::{Coeff, Layer, LayerData};
use std::ops::Deref;
use ocl::SpatialDims;

/// A complete descriptor for a convolutional layer
pub struct ConvLayer<T>
where
    T: Coeff,
{
    layer_data: LayerData<T>,
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
        weights_file: &str,
    ) -> ConvLayer<T> {
        trace!(
            "Create conv-layer with filter-elems: {:?}, input-shape: {:?}, output-shape: {:?}, weights-file: {:?}.",
            num_filter_elems,
            input_shape,
            output_shape,
            weights_file
        );
        ConvLayer {
            layer_data: LayerData::<T> {
                weights: T::read_bin_from_file(weights_file),
            },
            input_shape: input_shape.clone(),
            output_shape: output_shape.clone(),
        }
    }

    pub fn input_shape(&self) -> &ImageGeometry {
        &self.input_shape
    }
    pub fn output_shape(&self) -> &ImageGeometry {
        &self.output_shape
    }
}

impl<T> Deref for ConvLayer<T>
where
    T: Coeff,
{
    type Target = LayerData<T>;

    fn deref(&self) -> &Self::Target {
        &self.layer_data
    }
}

impl<T> Layer<T> for ConvLayer<T>
where
    T: Coeff,
{
    fn weights(&self) -> &Vec<T> {
        &self.weights
    }
    fn num_out(&self) -> usize {
        self.output_shape.num_elems()
    }
    fn num_in(&self) -> usize {
        self.input_shape.num_elems()
    }
    fn gws(&self) -> SpatialDims {
        SpatialDims::Three(
            self.output_shape.channels(),
            self.output_shape.side(),
            self.output_shape.side(),
        )
    }
}
