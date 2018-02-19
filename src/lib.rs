//! The main interface of the convolutive neural network. Intended for ease of benchmarking and
//! accuracy measurements.
#![allow(unused_imports)]
extern crate byteorder;
extern crate env_logger;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate num_traits;
extern crate ocl;

pub mod cl_util;
pub mod geometry;
pub mod network;
mod layers;
mod util;
mod math;
#[cfg(test)]
mod tests;

pub use util::*;
pub use layers::*;
pub use math::*;
pub use network::*;
use ocl::{flags, Buffer, Kernel, OclPrm, Program, Queue};
use cl_util as cl;
use num_traits::{Float, Num, NumAssign, Zero};
use std::ops::Mul;
use std::ops::Deref;
