mod conv;
mod dense;

pub use self::conv::*;
pub use self::dense::*;
use geometry::*;
use std::ops::Deref;
use ocl::{OclPrm, SpatialDims};
use util::*;
use num_traits::{Float, NumAssign};
use math::GenericOps;

/// A layer of a convolutive neural network.
pub trait Layer<T> {
    fn weights(&self) -> &Vec<T>;
    fn num_weights(&self) -> usize {
        self.weights().len()
    }
    /// Gets the number of elements in the output shape
    fn num_out(&self) -> usize;
    /// Gets the number of elements in the input shape
    fn num_in(&self) -> usize;
    // The global work-group-size of the matching kernel
    fn gws(&self) -> SpatialDims;
}

/// The data contained in any given CNN-layer.
#[derive(Debug)]
pub struct LayerData<T> {
    weights: Vec<T>,
}

#[derive(Clone, Debug)]
pub struct ClassicHyperParams {
    pub source_side: usize,
    // channels for each rgb color
    pub num_source_channels: usize,
    // the size of the filter/kernels
    pub conv_1_filter_side: usize,
    pub conv_2_filter_side: usize,
    // the number of feature maps
    pub num_feature_maps: usize,
    pub stride: usize,
    // ???: what is this, what does it do? was originally magic in jani's code.
    pub fully_connected_const: usize,
    pub num_output_classes: usize,
}

pub struct NetworkParams {
    hyper_params: ClassicHyperParams,
    conv1_filter_shape: PaddedSquare,
    conv2_filter_shape: PaddedSquare,
    padded_input_shape: ImageGeometry,
    padded_fm1_shape: ImageGeometry,
    fm2_shape: ImageGeometry,
}

impl NetworkParams {
    pub fn new(hyper_params: ClassicHyperParams) -> NetworkParams {
        let conv1_filter_shape = PaddedSquare::from_side(hyper_params.conv_1_filter_side);
        let conv2_filter_shape = PaddedSquare::from_side(hyper_params.conv_2_filter_side);

        // Create descriptor for input geometry with the shape and properties of an image
        let input_shape =
            ImageGeometry::new(hyper_params.source_side, hyper_params.num_source_channels);
        let padded_input_shape = input_shape.with_filter_padding(&conv1_filter_shape);
        // Feature map 1 is a fraction of the side of initial image geometry due to stride
        let fm1_shape = ImageGeometry::new(
            input_shape.side() / hyper_params.stride,
            hyper_params.num_feature_maps,
        );
        let padded_fm1_shape = fm1_shape.with_filter_padding(&conv2_filter_shape);
        // Feature map 2 is a fraction of the side of the tier 1 feature map due to stride
        let fm2_shape = ImageGeometry::new(
            fm1_shape.side() / hyper_params.stride,
            hyper_params.num_feature_maps,
        );

        NetworkParams {
            hyper_params,
            conv1_filter_shape,
            conv2_filter_shape,
            padded_input_shape,
            padded_fm1_shape,
            fm2_shape,
        }
    }
    pub fn create_conv<T>(&self, idx: usize, weights: Vec<T>) -> ConvLayer<T>
    where
        T: Coeff,
    {
        // TODO: unwrap these into per-layer, not per-IO
        let (filter_elems, in_shape, out_shape) = match idx {
            1 => (
                self.conv1_filter_shape.num_elems(),
                self.padded_input_shape,
                self.padded_fm1_shape,
            ),
            2 => (
                self.conv2_filter_shape.num_elems(),
                self.padded_fm1_shape,
                self.fm2_shape,
            ),
            _ => panic!(format!("no conv layer for idx {}", idx)),
        };
        ConvLayer::from_shapes(filter_elems, &in_shape, &out_shape, weights)
    }
    pub fn create_dense<T>(&self, idx: usize, weights: Vec<T>) -> DenseLayer<T>
    where
        T: Coeff,
    {
        // TODO: unwrap these
        let (num_in, num_out) = match idx {
            3 => (self.fm2_shape.num_elems(), self.fully_connected_const),
            4 => (self.fully_connected_const, self.fully_connected_const),
            5 => (self.fully_connected_const, self.num_output_classes),
            _ => panic!(format!("no dense layer for idx {}", idx)),
        };
        DenseLayer::new(num_in, num_out, weights)
    }
}

pub struct Layers<T>
where
    T: Coeff,
{
    pub conv1: ConvLayer<T>,
    pub conv2: ConvLayer<T>,
    pub dense3: DenseLayer<T>,
    pub dense4: DenseLayer<T>,
    pub dense5: DenseLayer<T>,
}

impl Deref for NetworkParams {
    type Target = ClassicHyperParams;

    fn deref(&self) -> &Self::Target {
        &self.hyper_params
    }
}

pub trait Coeff: ReadBinFromFile + NumAssign + GenericOps + OclPrm {}
pub trait CoeffFloat: Float + Coeff {}

impl Coeff for f32 {}
impl CoeffFloat for f32 {}
