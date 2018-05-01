use util::*;
use geometry::*;
use super::*;
use std::ops::Deref;
use ocl::SpatialDims;

/// A complete descriptor for the vertical (column) component of a separable convolutional layer
#[derive(Clone)]
pub struct VConvLayer<T>(pub SepconvLayer<T>)
where
    T: Coeff;

/// A complete descriptor for the horizontal (row) component of a separable convolutional layer
#[derive(Clone)]
pub struct HConvLayer<T>(pub SepconvLayer<T>)
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
        let layer = VConvLayer(SepconvLayer::new(
            filter_len,
            in_shape,
            num_out_channels,
            weights,
        ));
        debug!(
            "Create vertical (columns) convolution with input: {}, output: {}, weights: {}.",
            layer.num_in(),
            layer.num_out(),
            layer.num_weights()
        );
        layer
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
        let layer = HConvLayer(SepconvLayer::new(
            filter_len,
            in_shape,
            num_out_channels,
            weights,
        ));
        debug!(
            "Create horizontal (rows) convolution with input: {}, output: {}, weights: {}.",
            layer.num_in(),
            layer.num_out(),
            layer.num_weights()
        );
        layer
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

#[derive(Clone)]
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
    fn lws_hint(&self, _device_max_wgs: usize) -> SpatialDims {
        unimplemented!()
    }
    fn name(&self) -> &'static str {
        "sepconv"
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
