#![allow(unused_imports)]
#![feature(test)]
extern crate byteorder;
extern crate env_logger;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate ocl;
extern crate test;

mod geometry;
mod network;
mod cl_util;
mod util;
mod math;
#[cfg(test)]
mod tests;

pub use util::*;
use network::*;
use math::*;
use ocl::{flags, Buffer, Kernel, Queue};
use cl_util as cl;

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

pub fn create_network(
    params: HyperParams,
) -> (ConvLayer, ConvLayer, DenseLayer, DenseLayer, DenseLayer) {
    let params = Network::new(params);
    // Create a representation of the 1st convolutional layer with weights from a file
    let conv1 = params.create_conv1(&format!("{}/conv1_update.bin", WEIGHTS_DIR));
    // Create a representation of the 2nd convolutional layer with weights from a file
    let conv2 = params.create_conv2(&format!("{}/conv2_update.bin", WEIGHTS_DIR));
    // Create the representations of the fully-connected layers
    let dense3 = params.create_dense3(&format!("{}/ip3.bin", WEIGHTS_DIR));
    let dense4 = params.create_dense4(&format!("{}/ip4.bin", WEIGHTS_DIR));
    let dense5 = params.create_dense5(&format!("{}/ip_last.bin", WEIGHTS_DIR));

    // Verify that I/O dimensions match between layers
    verify_network_dimensions(&[&conv1, &conv2, &dense3, &dense4, &dense5]);

    (conv1, conv2, dense3, dense4, dense5)
}

pub unsafe fn run_kernel<L: Layer<f32>>(
    kernel: &Kernel,
    layer: &L,
    queue: &Queue,
) -> ocl::Result<()> {
    kernel
            .cmd()
            .queue(&queue)
            // TODO: gws is unrequired here? already included in Kernel, verify with benchmarks on a GPU
            .gws(layer.gws())
            .enq()?;
    queue.finish()
}

pub fn create_kernel<L: Layer<f32>>(
    layer: &L,
    kernel_func: &str,
    input_data: &[f32],
) -> ocl::Result<(Kernel, Buffer<f32>, Queue)> {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init()?;

    let wgts_buf =
        cl::create_buffer::<f32>("weights", layer.num_weights(), flags::MEM_READ_ONLY, &queue)?;
    let in_buf = cl::create_buffer::<f32>(
        "input",
        layer.num_in(),
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    )?;
    let out_buf = cl::create_buffer::<f32>(
        "output",
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

    // Write the weights to the global memory of the device
    wgts_buf.write(layer.weights()).enq()?;

    unsafe {
        // Create a host-accessible input buffer for writing the image into device memory
        let mut mem_map = in_buf
            .map()
            .flags(flags::MAP_WRITE)
            .len(layer.num_in())
            .enq()?;

        // Write input data to device memory
        for (idx, f) in input_data.iter().enumerate() {
            mem_map[idx] = *f;
        }
        mem_map.unmap().enq()?;
    }

    Ok((kernel, out_buf, queue))
}

pub fn mtxmul_relu(input_buffer: &[f32], dense: &DenseLayer) -> Vec<f32> {
    let out = mtx_mul(
        dense.weights(),
        input_buffer,
        dense.num_out(),
        dense.num_in(),
        1,
    );
    relu(&out, dense.num_out(), 1)
}

pub fn mtxmul_softmax(input_buffer: &[f32], dense: &DenseLayer) -> Vec<f32> {
    let out = mtx_mul(
        dense.weights(),
        input_buffer,
        dense.num_out(),
        dense.num_in(),
        1,
    );
    softmax(&out, dense.num_out(), 1)
}
