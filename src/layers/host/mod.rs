mod dense;
mod maxpool;

pub use self::dense::*;
pub use self::maxpool::*;
use super::*;

/// Implement layer computations on host with floating-point values.
pub trait ComputeOnHost<T>: Layer {
    fn compute(&self, in_buf: &[T]) -> Vec<T>;
}
