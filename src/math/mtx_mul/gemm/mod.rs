#![allow(dead_code)]

//! Matrix multiplication based on https://cnugteren.github.io/tutorial/
//! Matrices must be multiples of 32 to fit into OpenCL work-groups.
//! Matrices are all column-major.
//!
//! * C := A * B
//! * A is k by m
//! * B is n by k
//! * C is m by n
//!

mod naive_1_gemm;
#[cfg(test)]
mod test;
mod tiling_6_gemm;
mod transpose_5_gemm;
mod vectors_4_gemm;

pub use self::naive_1_gemm::*;
pub use self::tiling_6_gemm::*;
pub use self::transpose_5_gemm::*;
pub use self::vectors_4_gemm::*;
pub use ocl::flags::DeviceType;

use cl_util;
use layers::Coeff;
use ocl;
use ocl::{flags, Buffer, Context, Device, Kernel, OclPrm, Platform, Program, Queue, SpatialDims};

/// The idea here is to allow usage of module as drop-in where cross-platform high-perf matrix
/// multiplication is used. API calls are generally split into three categories: init, load and calculate,
/// where init-calls setup the OpenCL-context, load-calls move work to a device if necessary and calculate
/// calls do work.
///
/// # Examples
///
/// Run naïve Rust implementation on host
/// ```ignore
/// gemm_naive(m, n, k, &a, &b, &mut c)
/// ```
///
/// Run OpenCL implementation on a GPU
/// ```ignore
/// let device = DeviceType::GPU;
/// let gemm_kernel = OclGemm::from_slices(m, n, k, &a, &b, &mut c, device);
/// gemm_kernel.calculate(); // or calculate_wait for testing
/// ```
///
/// Run OpenCL implementation on CPU
/// ```ignore
/// let device = DeviceType::CPU;
/// let gemm_kernel = OclGemm::from_slices(m, n, k, &a, &b, &mut c, device);
/// gemm_kernel.calculate(); // or calculate_wait for testing
/// ```
///
/// Run an implementation repeatedly with varying inputs
/// ```ignore
/// let device = DeviceType::GPU;
/// let gemm_kernel = OclGemm::uninitialized(m, n, k, device);
/// loop {
///     gemm_kernel.set_buffers_from_slices(&a, &b, &mut c);
///     gemm_kernel.calculate();
/// }
/// ```
///
/// Run as part of another OpenCL implementation
/// ```ignore
/// let preexisting_out_buf: ocl::Buffer<f32> = ...;
/// let my_matrix = (0..32*32).map(|x| f32::into(x)).collect::<Vec<f32>>();
///
/// let device = DeviceType::GPU;
/// let gemm_kernel = OclGemm::attach(m, n, k, GemmInput::OclBuffer(&preexisting_out_buf), GemmInput::Slice(&my_matrix), GemmOutput::Slice(&mut c), device);
/// loop {
///     gemm_kernel.calculate();
/// }
/// ```
///
/// Run multiple chained kernels
/// ```ignore
/// let device_a = DeviceType::GPU;
/// let device_b = DeviceType::CPU;
///
///
/// network.calculate();
/// ```
/// Converting column-major into row-major would be by switching A and B and n and m
pub trait OclGemm<SuperKernel> {
    fn uninitialized(m: usize, n: usize, k: usize, device: DeviceType) -> SuperKernel;
    fn from_slices(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        device: DeviceType,
    ) -> SuperKernel;
    fn set_buffers_from_slices(&self, a: &[f32], b: &[f32]);
    // TODO: this should probably return some kind of a future or such to allow checking ready state; see ocl
    fn calculate(&self);
    fn queue(&self) -> &Queue;
    fn calculate_wait(&self) {
        self.calculate();
        self.queue().finish().unwrap();
    }
}

/*
// Represents a reference to an input matrix whether it's a slice on a host or data on a physical
// device represented by an ocl::Buffer.
pub enum GemmInput<'a> {
    Slice(&'a [f32]),
    OclBuffer(&'a ocl::Buffer<f32>),
}

// Represents a reference to an output matrix whether it's a slice on a host or data on a physical
// device represented by an ocl::Buffer.
pub enum GemmOutput<'a> {
    Slice(&'a mut [f32]),
    OclBuffer(&'a ocl::Buffer<f32>),
}
*/
