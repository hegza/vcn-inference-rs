//! The main interface of the convolutive neural network. Intended for ease of benchmarking and
//! accuracy measurements.
#![allow(unused_imports)]
extern crate byteorder;
extern crate env_logger;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
#[macro_use(array, s)]
extern crate ndarray;
extern crate num_traits;
extern crate ocl;
extern crate rand;

/// Usage: println!("{}", format_result!(expression));
#[macro_use]
#[allow(unused_macros)]
macro_rules! format_result {
    // This macro takes an expression of type `expr` and prints
    // it as a string along with its result.
    // The `expr` designator is used for expressions.
    ($expression:expr) => (
        // `stringify!` will convert the expression *as it is* into a string.
        format!("{:?} = {:?}",
                 stringify!($expression),
                 $expression);
    )
}

pub mod cl_util;
pub mod geometry;
pub mod network;
mod layers;
mod util;
pub mod math;
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
