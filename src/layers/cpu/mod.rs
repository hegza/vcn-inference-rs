mod dense;
mod maxpool;

pub use self::dense::*;
pub use self::maxpool::*;
use super::*;

pub trait ComputeOnCpu<T>: Layer
where
    T: Coeff,
{
    fn compute(&self, in_buf: &[T]) -> Vec<T>;
}
