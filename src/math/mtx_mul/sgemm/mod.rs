#![allow(non_snake_case)]
#![allow(dead_code)]

//! Matrix multiplication based on https://cnugteren.github.io/tutorial/
//! Matrices must be multiples of 32 to fit into OpenCL work-groups.
//! Matrices are all column-major.
//!
//! * C := A * B
//! * A is K by M
//! * B is N by K
//! * C is M by N
//!

/// The idea here is to allow usage of module as drop-in where cross-platform high-perf matrix
/// multiplication is used. API calls are generally split into three categories: init, load and run,
/// where init-calls setup the OpenCL-context, load-calls move work to a device if necessary and run
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
/// let gemm_kernel = OclGemm::with_slices(M, N, K, &a, &b, &mut c, device)
/// gemm_kernel.run(); // or run_wait for testing
/// ```
///
/// Run OpenCL implementation on CPU
/// ```ignore
/// let device = DeviceType::CPU;
/// let gemm_kernel = OclGemm::with_slices(M, N, K, &a, &b, &mut c, device);
/// gemm_kernel.run(); // or run_wait for testing
/// ```
///
/// Run an implementation repeatedly with varying inputs
/// ```ignore
/// let device = DeviceType::GPU;
/// let gemm_kernel = OclGemm::uninitialized(M, N, K, device);
/// loop {
///     gemm_kernel.set_buffers_from_slices(&a, &b, &mut c);
///     gemm_kernel.run();
/// }
/// ```
///
/// Run as part of another OpenCL application
/// ```ignore
/// let preexisting_out_buf: ocl::Buffer<f32> = ...;
/// let my_matrix = (0..32*32).map(|x| f32::into(x)).collect::<Vec<f32>>();
///
/// let device = DeviceType::GPU;
/// let gemm_kernel = OclGemm::uninitialized(M, N, K, device).set_buffers(GemmInput::OclBuffer(&preexisting_out_buf), GemmInput::Slice(&my_matrix), GemmOutput::Slice(&mut c));
/// ```
///
/// Run multiple chained kernels
/// ```ignore
/// let device_a = DeviceType::GPU;
/// let device_b = DeviceType::CPU;
///
///
/// network.run();
/// ```
// Converting column-major into row-major would be by switching A and B and N and M

// HACK: being pub here is hack
pub mod algo;

// Re-export the used (or best) algorithm here
pub use self::algo::Tiling6GemmKernel as GemmKernel;

use layers::Coeff;
use ocl;
use ocl::{flags, Buffer, Context, Device, Kernel, OclPrm, Platform, Program, Queue, SpatialDims};

/*
pub use ocl::flags::DeviceType;

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

pub struct OclGemmLoader {}
pub struct OclGemm {}

// Non-performance critical init-implementation
impl OclGemmLoader {
    pub fn new(m: usize, n: usize, k: usize, device: DeviceType) -> OclGemmLoader {
        unimplemented!()
    }

    /// Sets input and output buffers from the slices a, b and c. Essentially free on CPU and
    /// devices with shared GPU/CPU memory.
    ///
    /// use_host_slice() guarantees performance where necessary.
    pub fn set_buffers_from_slices(self, a: &[f32], b: &[f32], c: &mut [f32]) -> OclGemm {
        unimplemented!()
    }

    pub fn set_buffers(self, a: GemmInput, b: GemmInput, c: GemmOutput) -> OclGemm {
        unimplemented!()
    }

    pub fn uninitialized(self, a: &[f32], b: &[f32], c: &mut [f32]) -> OclGemm {
        unimplemented!()
    }
}

// Performance critical implementation
impl OclGemm {
    pub fn set_buffers_from_slices(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        unimplemented!()
    }

    pub fn run(&self) {
        unimplemented!()
    }

    pub fn run_wait(&self) {
        self.run();
        //TODO: self.queue.finish().unwrap();
    }
}
*/
