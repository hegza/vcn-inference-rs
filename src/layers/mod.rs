mod cl;
mod conv;
mod dense;
mod host;
mod maxpool;
mod sepconv;
mod sparse;

pub use self::cl::*;
pub use self::conv::*;
pub use self::dense::*;
pub use self::host::*;
pub use self::maxpool::*;
pub use self::sepconv::*;
pub use self::sparse::*;
use crate::cl_util;
use crate::cl_util::ClVecTypeName;
use crate::flags::DeviceType;
use crate::geometry::*;
use crate::math::GenericOps;
use crate::util::*;
use num_traits::{Float, NumAssign, PrimInt};
use ocl::*;
use std::fmt::Display;
use std::ops::Deref;

pub trait Coeff: NumAssign + GenericOps + OclPrm + ClVecTypeName + Display {}
pub trait CoeffFloat: Coeff + Float {}

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
    fn weights(&self) -> &[T];
    fn num_weights(&self) -> usize {
        self.weights().len()
    }
}

impl Coeff for f32 {}
impl CoeffFloat for f32 {}
impl Coeff for f64 {}
impl CoeffFloat for f64 {}
impl Coeff for i8 {}
impl Coeff for u8 {}
