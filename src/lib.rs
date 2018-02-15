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

pub mod cl_util;
mod geometry;
mod network;
mod util;
mod math;
#[cfg(test)]
mod tests;

pub use util::*;
pub use network::*;
pub use math::*;
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
    let params = NetworkParams::new(params);
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

pub fn init_network(
    input_file: &str,
) -> ocl::Result<
    (
        ConvLayer,
        ConvLayer,
        DenseLayer,
        DenseLayer,
        DenseLayer,
        Kernel,
        Kernel,
        Kernel,
        Buffer<f32>,
        Queue,
    ),
> {
    // Create the network representation from network hyper-parameters
    let (conv1, conv2, dense3, dense4, dense5) = create_network(HYPER_PARAMS.clone());

    // Initialize OpenCL
    let (queue, program, _context) = cl::init()?;

    // Allocate read-only memory for the weights of the 1st three layers
    let conv1_wgts_buf = cl::create_buffer::<f32>(
        "conv 1 weights",
        conv1.num_weights(),
        flags::MEM_READ_ONLY,
        &queue,
    )?;
    let conv2_wgts_buf = cl::create_buffer::<f32>(
        "conv 2 weights",
        conv2.num_weights(),
        flags::MEM_READ_ONLY,
        &queue,
    )?;
    let dense3_wgts_buf = cl::create_buffer::<f32>(
        "dense 3 weights",
        dense3.num_weights(),
        flags::MEM_READ_ONLY,
        &queue,
    )?;

    // Allocate read-only memory for the input geometry on device with host-accessible pointer for
    // writing input from file
    let input_buf = cl::create_buffer::<f32>(
        "input img",
        conv1.num_in(),
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    )?;
    // Allocate read-write memory for the 1st feature map on device
    let fm1_buf = cl::create_buffer::<f32>("FM 1", conv1.num_out(), flags::MEM_READ_WRITE, &queue)?;
    // Allocate read-write memory for the 2nd feature map on device
    let fm2_buf = cl::create_buffer::<f32>("FM 2", conv2.num_out(), flags::MEM_READ_WRITE, &queue)?;
    // Allocate read-write memory for the dense (3rd) layer output on device with host pointer for reading
    let dense3_out_buf = cl::create_buffer::<f32>(
        "dense 3 output",
        dense3.num_out(),
        flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    )?;

    // Write the weights of the 1st three layers to the global memory of the device
    conv1_wgts_buf.write(conv1.weights()).enq()?;
    conv2_wgts_buf.write(conv2.weights()).enq()?;
    dense3_wgts_buf.write(dense3.weights()).enq()?;
    // TODO: this might not make any difference anywhere (verify with benchmark)
    queue.finish()?;

    // Create the kernel for the 1st layer (Convolution + ReLU)
    let conv_relu1 = Kernel::new("conv_relu_1", &program)?
        .queue(queue.clone())
        .gws(conv1.gws())
        // Input
        .arg_buf(&input_buf)
        // Output
        .arg_buf(&fm1_buf)
        .arg_buf(&conv1_wgts_buf);

    // Create the kernel for the 2nd layer (Convolution + ReLU)
    let conv_relu2 = Kernel::new("conv_relu_2", &program)?
        .queue(queue.clone())
        .gws(conv2.gws())
        // Input
        .arg_buf(&fm1_buf)
        // Output
        .arg_buf(&fm2_buf)
        .arg_buf(&conv2_wgts_buf);

    // Create the kernel for the 3rd layer (Dense layer matrix multiplication)
    let dense3_kernel = Kernel::new("mtx_mulf", &program)?
        .queue(queue.clone())
        .gws(dense3.gws())
        // Input
        .arg_buf(&fm2_buf)
        // Output
        .arg_buf(&dense3_out_buf)
        .arg_buf(&dense3_wgts_buf);

    let input_data = read_image_with_padding(input_file, *conv1.input_shape());
    unsafe {
        cl::write_buf(&input_buf, &input_data)?;
    }

    Ok((
        conv1,
        conv2,
        dense3,
        dense4,
        dense5,
        conv_relu1,
        conv_relu2,
        dense3_kernel,
        dense3_out_buf,
        queue,
    ))
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
