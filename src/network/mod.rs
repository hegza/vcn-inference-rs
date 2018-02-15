mod conv;
mod dense;

pub use self::conv::*;
pub use self::dense::*;
use geometry::*;
use std::ops::Deref;
use ocl::SpatialDims;

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
pub struct HyperParams {
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
    hyper_params: HyperParams,
    conv1_filter_shape: PaddedSquare,
    conv2_filter_shape: PaddedSquare,
    padded_input_shape: ImageGeometry,
    padded_fm1_shape: ImageGeometry,
    fm2_shape: ImageGeometry,
}

impl NetworkParams {
    pub fn new(hyper_params: HyperParams) -> NetworkParams {
        let conv1_filter_shape = PaddedSquare::from_side(hyper_params.conv_1_filter_side);
        let conv2_filter_shape = PaddedSquare::from_side(hyper_params.conv_2_filter_side);

        // Create descriptor for input geometry with the a shape and properties of an image
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
    pub fn create_conv1(&self, weights_file: &str) -> ConvLayer {
        ConvLayer::from_shapes(
            self.conv1_filter_shape.num_elems(),
            &self.padded_input_shape,
            &self.padded_fm1_shape,
            weights_file,
        )
    }
    pub fn create_conv2(&self, weights_file: &str) -> ConvLayer {
        ConvLayer::from_shapes(
            self.conv2_filter_shape.num_elems(),
            &self.padded_fm1_shape,
            &self.fm2_shape,
            weights_file,
        )
    }
    pub fn create_dense3(&self, weights_file: &str) -> DenseLayer {
        DenseLayer::new(
            self.fm2_shape.num_elems(),
            self.fully_connected_const,
            weights_file,
        )
    }
    pub fn create_dense4(&self, weights_file: &str) -> DenseLayer {
        DenseLayer::new(
            self.fully_connected_const,
            self.fully_connected_const,
            weights_file,
        )
    }
    pub fn create_dense5(&self, weights_file: &str) -> DenseLayer {
        DenseLayer::new(
            self.fully_connected_const,
            self.num_output_classes,
            weights_file,
        )
    }
}

impl Deref for NetworkParams {
    type Target = HyperParams;

    fn deref(&self) -> &Self::Target {
        &self.hyper_params
    }
}
