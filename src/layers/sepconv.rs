use util::*;
use geometry::*;
use super::*;
use std::ops::Deref;
use ocl::SpatialDims;

/// A complete descriptor for the vertical (column) component of a separable convolutional layer
pub struct VConvLayer<T>(SepconvLayer<T>)
where
    T: Coeff;

/// A complete descriptor for the horizontal (row) component of a separable convolutional layer
pub struct HConvLayer<T>(SepconvLayer<T>)
where
    T: Coeff;

impl<T> VConvLayer<T>
where
    T: Coeff,
{
    pub fn new(
        filter_len: usize,
        in_shape: ImageGeometry,
        num_out_channels: usize,
        weights: Vec<T>,
    ) -> VConvLayer<T> {
        trace!(
            "Create v-conv-layer with filter_len: {:?}, in_shape: {:?}, num_out_channels: {:?}, weights-size: {}.",
            filter_len,
            in_shape,
            num_out_channels,
            weights.len()
        );
        VConvLayer(SepconvLayer::new(
            filter_len,
            in_shape,
            num_out_channels,
            weights,
        ))
    }
}

impl<T> HConvLayer<T>
where
    T: Coeff,
{
    pub fn new(
        filter_len: usize,
        in_shape: ImageGeometry,
        num_out_channels: usize,
        weights: Vec<T>,
    ) -> HConvLayer<T> {
        trace!(
            "Create h-conv-layer with filter_len: {:?}, in_shape: {:?}, num_out_channels: {:?}, weights-size: {}.",
            filter_len,
            in_shape,
            num_out_channels,
            weights.len()
        );
        HConvLayer(SepconvLayer::new(
            filter_len,
            in_shape,
            num_out_channels,
            weights,
        ))
    }
}
impl<T> Deref for VConvLayer<T>
where
    T: Coeff,
{
    type Target = SepconvLayer<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Deref for HConvLayer<T>
where
    T: Coeff,
{
    type Target = SepconvLayer<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct SepconvLayer<T>
where
    T: Coeff,
{
    weights: Vec<T>,
    in_shape: ImageGeometry,
    out_shape: ImageGeometry,
}

impl<T> SepconvLayer<T>
where
    T: Coeff,
{
    /// Creates a descriptor of a convolutional layer with a square filter the
    /// side of which will be set to filter_side.
    fn new(
        filter_len: usize,
        in_shape: ImageGeometry,
        num_out_channels: usize,
        weights: Vec<T>,
    ) -> SepconvLayer<T> {
        // Make sure that the weight count is correct
        debug_assert_eq!(
            filter_len * in_shape.channels() * num_out_channels,
            weights.len()
        );
        let out_shape = ImageGeometry::new(in_shape.side(), num_out_channels);
        SepconvLayer {
            weights,
            in_shape,
            out_shape,
        }
    }

    pub fn input_shape(&self) -> &ImageGeometry {
        &self.in_shape
    }
    pub fn output_shape(&self) -> &ImageGeometry {
        &self.out_shape
    }
}

impl<T> Layer for SepconvLayer<T>
where
    T: Coeff,
{
    fn num_out(&self) -> usize {
        self.out_shape.num_elems()
    }
    fn num_in(&self) -> usize {
        self.in_shape.num_elems()
    }
    fn gws_hint(&self) -> SpatialDims {
        SpatialDims::Three(
            self.in_shape.side(),
            self.in_shape.side(),
            self.out_shape.channels(),
        )
    }
}

impl<T> WeightedLayer<T> for SepconvLayer<T>
where
    T: Coeff,
{
    fn weights(&self) -> &Vec<T> {
        &self.weights
    }
}
