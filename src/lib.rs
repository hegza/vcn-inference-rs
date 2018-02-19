//! The main interface of the convolutive neural network. Intended for ease of benchmarking and
//! accuracy measurements.
#![allow(unused_imports)]
#![feature(test)]
extern crate byteorder;
extern crate env_logger;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate num_traits;
extern crate ocl;
extern crate test;

pub mod cl_util;
pub mod geometry;
mod layers;
mod util;
mod math;
#[cfg(test)]
mod tests;

pub use util::*;
pub use layers::*;
pub use math::*;
use ocl::{flags, Buffer, Kernel, OclPrm, Program, Queue};
use cl_util as cl;
use num_traits::{Float, Num, NumAssign, Zero};
use std::ops::Mul;
use std::ops::Deref;

pub const HYPER_PARAMS: HyperParams = HyperParams {
    source_side: 96,
    num_source_channels: 3,
    conv_1_filter_side: 5,
    conv_2_filter_side: 5,
    num_feature_maps: 32,
    stride: 2,
    fully_connected_const: 100,
    num_output_classes: 4,
};
const WEIGHTS_DIR: &'static str = "input/weights";

pub struct Network<T>
where
    T: Coeff,
{
    pub layers: Layers<T>,
    pub conv_relu1: Kernel,
    pub conv_relu2: Kernel,
    pub dense3_kernel: Kernel,
    pub dense3_out_buf: Buffer<T>,
    // TODO: rewrap in_buffer and weights buffers as private (predict()-refactor)
    pub in_buf: Buffer<T>,
    pub conv1_wgts_buf: Buffer<T>,
    pub conv2_wgts_buf: Buffer<T>,
    pub dense3_wgts_buf: Buffer<T>,
}

impl<T> Network<T>
where
    T: CoeffFloat,
{
    /// Initializes the network, kernels and buffers. Returns only after all OpenCL-commands have
    /// finished running. Note that you must call upload_buffers before the network is run.
    pub fn new(program: &Program, queue: &Queue) -> ocl::Result<Network<T>> {
        // Create the network representation from network hyper-parameters
        let layers = create_layers(HYPER_PARAMS.clone());

        // Allocate read-only memory for the weights of the 1st three layers
        let conv1_wgts_buf =
            cl::create_buffer::<T>(layers.conv1.num_weights(), flags::MEM_READ_ONLY, &queue)?;
        let conv2_wgts_buf =
            cl::create_buffer::<T>(layers.conv2.num_weights(), flags::MEM_READ_ONLY, &queue)?;
        let dense3_wgts_buf =
            cl::create_buffer::<T>(layers.dense3.num_weights(), flags::MEM_READ_ONLY, &queue)?;

        // Allocate read-only memory for the input geometry on device with host-accessible pointer for
        // writing input from file
        let in_buf = cl::create_buffer::<T>(
            layers.conv1.num_in(),
            flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        )?;
        // Allocate read-write memory for the 1st feature map on device
        let fm1_buf =
            cl::create_buffer::<T>(layers.conv1.num_out(), flags::MEM_READ_WRITE, &queue)?;
        // Allocate read-write memory for the 2nd feature map on device
        let fm2_buf =
            cl::create_buffer::<T>(layers.conv2.num_out(), flags::MEM_READ_WRITE, &queue)?;
        // Allocate read-write memory for the dense (3rd) layer output on device with host pointer for reading
        let dense3_out_buf = cl::create_buffer::<T>(
            layers.dense3.num_out(),
            flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
            &queue,
        )?;

        // Create the kernel for the 1st layer (Convolution + ReLU)
        let conv_relu1 = Kernel::new("conv_relu_1", &program)?
            .queue(queue.clone())
            .gws(layers.conv1.gws())
            // Input
            .arg_buf(&in_buf)
            // Output
            .arg_buf(&fm1_buf)
            .arg_buf(&conv1_wgts_buf);

        // Create the kernel for the 2nd layer (Convolution + ReLU)
        let conv_relu2 = Kernel::new("conv_relu_2", &program)?
            .queue(queue.clone())
            .gws(layers.conv2.gws())
            // Input
            .arg_buf(&fm1_buf)
            // Output
            .arg_buf(&fm2_buf)
            .arg_buf(&conv2_wgts_buf);

        // Create the kernel for the 3rd layer (Dense layer matrix multiplication)
        let dense3_kernel = Kernel::new("mtx_mulf", &program)?
            .queue(queue.clone())
            .gws(layers.dense3.gws())
            // Input
            .arg_buf(&fm2_buf)
            // Output
            .arg_buf(&dense3_out_buf)
            .arg_buf(&dense3_wgts_buf);

        // Wait until all commands have finished running before returning.
        queue.finish()?;

        Ok(Network {
            layers: layers,
            conv_relu1,
            conv_relu2,
            dense3_kernel,
            dense3_out_buf,
            in_buf,
            conv1_wgts_buf,
            conv2_wgts_buf,
            dense3_wgts_buf,
        })
    }
    /// Writes weights to device memory, maps input buffer. Returns only after all commands have finished running.
    pub fn upload_buffers(&self, input_data: &[T], queue: &Queue) -> ocl::Result<()> {
        // Write the weights of the 1st three layers to the global memory of the device
        self.conv1_wgts_buf.write(self.conv1.weights()).enq()?;
        self.conv2_wgts_buf.write(self.conv2.weights()).enq()?;
        self.dense3_wgts_buf.write(self.dense3.weights()).enq()?;

        unsafe {
            cl::map_to_buf(&self.in_buf, &input_data)?;
        }
        queue.finish()
    }
    /// Runs the network with the currently loaded buffers, returning the result.
    pub fn run(&self, queue: &Queue) -> Vec<T> {
        unsafe {
            // Enqueue the kernel for the 1st layer (Convolution + ReLU)
            self.conv_relu1.cmd().queue(&queue).enq().unwrap();
            // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
            self.conv_relu2.cmd().queue(&queue).enq().unwrap();
            // Enqueue the 3rd layer (fully-connected)
            self.dense3_kernel.cmd().queue(&queue).enq().unwrap();
        }
        // Wait for all on-device calculations to finish
        queue.finish().unwrap();

        let dense3_out = relu(&unsafe { cl::read_buf(&self.dense3_out_buf).unwrap() });

        // Run the 4th layer (fully-connected)
        let dense4_out = mtxmul_relu(&dense3_out, &self.dense4);

        // Run the 5th layer (fully-connected)
        mtxmul_softmax(&dense4_out, &self.dense5)
    }
}

pub fn create_layers<T>(params: HyperParams) -> Layers<T>
where
    T: Coeff,
{
    let params = NetworkParams::new(params);
    // Create a representation of the 1st convolutional layer with weights from a file
    let conv1 = params.create_conv(
        1,
        T::read_bin_from_file(&format!("{}/conv1-f32-le.bin", WEIGHTS_DIR)),
    );
    // Create a representation of the 2nd convolutional layer with weights from a file
    let conv2 = params.create_conv(
        2,
        T::read_bin_from_file(&format!("{}/conv2-f32-le.bin", WEIGHTS_DIR)),
    );
    // Create the representations of the fully-connected layers
    let dense3 = params.create_dense(
        3,
        T::read_bin_from_file(&format!("{}/fc3-f32-le.bin", WEIGHTS_DIR)),
    );
    let dense4 = params.create_dense(
        4,
        T::read_bin_from_file(&format!("{}/fc4-f32-le.bin", WEIGHTS_DIR)),
    );
    let dense5 = params.create_dense(
        5,
        T::read_bin_from_file(&format!("{}/fc5-f32-le.bin", WEIGHTS_DIR)),
    );

    // Verify that I/O dimensions match between layers
    verify_network_dimensions(&[&conv1, &conv2, &dense3, &dense4, &dense5]);

    Layers {
        conv1,
        conv2,
        dense3,
        dense4,
        dense5,
    }
}

/// Runs the kernel but returns only after it has finished.
pub fn run_kernel_wait(kernel: &Kernel, queue: &Queue) -> ocl::Result<()> {
    unsafe {
        kernel.cmd().queue(&queue).enq()?;
    }
    queue.finish()
}

/// Creates a standalone kernel for benchmarking. Returns only after all commands have finished.
pub fn create_standalone_kernel<L: Layer<T>, T: Num + OclPrm>(
    layer: &L,
    kernel_func: &str,
    input_data: &[T],
) -> ocl::Result<(Kernel, Buffer<T>, Queue)> {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init()?;

    let wgts_buf = cl::create_buffer::<T>(layer.num_weights(), flags::MEM_READ_ONLY, &queue)?;
    let in_buf = cl::create_buffer::<T>(
        layer.num_in(),
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    )?;
    let out_buf = cl::create_buffer::<T>(
        layer.num_out(),
        flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    )?;

    let kernel = Kernel::new(kernel_func, &program)?
        .queue(queue.clone())
        .gws(layer.gws())
        // Input
        .arg_buf(&in_buf)
        // Output
        .arg_buf(&out_buf)
        .arg_buf(&wgts_buf);

    // Write the weights and input to the global memory of the device
    wgts_buf.write(layer.weights()).enq()?;
    unsafe {
        cl::map_to_buf(&in_buf, &input_data)?;
    }
    queue.finish()?;

    Ok((kernel, out_buf, queue))
}

pub fn mtxmul_relu<T>(input_buffer: &[T], dense: &DenseLayer<T>) -> Vec<T>
where
    T: CoeffFloat,
{
    let out = mtx_mul(
        dense.weights(),
        input_buffer,
        dense.num_out(),
        dense.num_in(),
        1,
    );
    relu(&out)
}

pub fn mtxmul_softmax<F>(input_buffer: &[F], dense: &DenseLayer<F>) -> Vec<F>
where
    F: CoeffFloat,
{
    let out = mtx_mul(
        dense.weights(),
        input_buffer,
        dense.num_out(),
        dense.num_in(),
        1,
    );
    softmax(&out, dense.num_out(), 1)
}

impl<T> Deref for Network<T>
where
    T: Coeff,
{
    type Target = Layers<T>;

    fn deref(&self) -> &Self::Target {
        &self.layers
    }
}
