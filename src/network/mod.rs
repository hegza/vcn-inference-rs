mod conv;
mod dense;

pub use self::conv::*;
pub use self::dense::*;
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
