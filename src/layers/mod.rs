mod conv;
mod dense;
mod sepconv;
mod maxpool;
mod cl;
mod host;

pub use self::conv::*;
pub use self::dense::*;
pub use self::sepconv::*;
pub use self::maxpool::*;
pub use self::cl::*;
pub use self::host::*;
use geometry::*;
use std::ops::Deref;
use ocl::*;
use util::*;
use num_traits::{Float, NumAssign};
use math::GenericOps;
use cl_util;
use flags::DeviceType;

pub trait Coeff: NumAssign + GenericOps + OclPrm + cl_util::ClVecTypeName {}
pub trait CoeffFloat: Float + Coeff {}

/// Describes a layer of a convolutive neural network.
pub trait Layer {
    /// Gets the number of elements in the input shape
    fn num_in(&self) -> usize;
    /// Gets the number of elements in the output shape
    fn num_out(&self) -> usize;
    /// The probable optimal global work-group-size-shape of the matching kernel
    fn gws_hint(&self) -> SpatialDims;
    // The probable optimal local work-group-size-shape of the matching kernel
    fn lws_hint(&self, device_max_wgs: usize) -> SpatialDims;
    fn name(&self) -> &'static str;
}

pub trait WeightedLayer<T>: Layer {
    fn weights(&self) -> &Vec<T>;
    fn num_weights(&self) -> usize {
        self.weights().len()
    }
}

impl Coeff for f32 {}
impl CoeffFloat for f32 {}
impl Coeff for i8 {}
