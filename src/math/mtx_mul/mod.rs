mod naive;
// HACK: being public is hack here
pub mod sgemm;

pub use self::naive::{mtx_mul, mtx_mul_normint};
pub use self::sgemm::GemmKernel;
