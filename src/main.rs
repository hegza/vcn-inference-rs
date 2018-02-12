#![feature(test)]

extern crate byteorder;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate ocl;
extern crate test;

mod geometry;
mod network;
mod cl_util;
mod util;
mod math;

use cl_util as cl;
use util::*;
use network::*;
use geometry::*;
use math::*;
use ocl::{flags, Buffer, Kernel, Queue};
use std::time::Instant;

const HYPER_PARAMS: HyperParams = HyperParams {
    source_side: 96,
    num_source_channels: 3,
    conv_1_filter_side: 5,
    conv_2_filter_side: 5,
    num_feature_maps: 32,
    stride: 2,
    fully_connected_const: 100,
    num_output_classes: 4,
};
const IN_FILE: &'static str = "input/baseline/input1/in.bin";
const WEIGHTS_DIR: &'static str = "input/weights";
const OUT_FILE: &'static str = "output/out.f";

fn main() {
    env_logger::init();

    match run() {
        Ok(_) => info!("Exited great."),
        Err(err) => info!("Exited with error: {}.", err),
    }
}

fn create_network(
    params: HyperParams,
) -> (ConvLayer, ConvLayer, DenseLayer, DenseLayer, DenseLayer) {
    let conv1_filter_shape = PaddedSquare::from_side(params.conv_1_filter_side);
    let conv2_filter_shape = PaddedSquare::from_side(params.conv_2_filter_side);

    // Create descriptor for input geometry with the a shape and properties of an image
    let input_shape = ImageGeometry::new(params.source_side, params.num_source_channels);
    let padded_input_shape = input_shape.with_filter_padding(&conv1_filter_shape);
    // Feature map 1 is a fraction of the side of initial image geometry due to stride
    let fm1_shape = ImageGeometry::new(input_shape.side() / params.stride, params.num_feature_maps);
    let padded_fm1_shape = fm1_shape.with_filter_padding(&conv2_filter_shape);
    // Feature map 2 is a fraction of the side of the tier 1 feature map due to stride
    let fm2_shape = ImageGeometry::new(fm1_shape.side() / params.stride, params.num_feature_maps);

    // Create a representation of the 1st convolutional layer with weights from a file
    let conv1 = ConvLayer::from_shapes(
        conv1_filter_shape.num_elems(),
        &padded_input_shape,
        &padded_fm1_shape,
        &format!("{}/conv1_update.bin", WEIGHTS_DIR),
    );
    // Create a representation of the 2nd convolutional layer with weights from a file
    let conv2 = ConvLayer::from_shapes(
        conv2_filter_shape.num_elems(),
        &conv1.output_shape(),
        &fm2_shape,
        &format!("{}/conv2_update.bin", WEIGHTS_DIR),
    );
    // Create the representations of the fully-connected layers
    let dense3 = DenseLayer::new(
        conv2.num_out(),
        params.fully_connected_const,
        &format!("{}/ip3.bin", WEIGHTS_DIR),
    );
    let dense4 = DenseLayer::new(
        dense3.num_out(),
        params.fully_connected_const,
        &format!("{}/ip4.bin", WEIGHTS_DIR),
    );
    let dense5 = DenseLayer::new(
        dense4.num_out(),
        params.num_output_classes,
        &format!("{}/ip_last.bin", WEIGHTS_DIR),
    );

    // Verify that I/O dimensions match between layers
    verify_network_dimensions(&[&conv1, &conv2, &dense3, &dense4, &dense5]);

    (conv1, conv2, dense3, dense4, dense5)
}

fn run_conv_kernel(kernel: &Kernel, conv: &ConvLayer, queue: &Queue) -> ocl::Result<()> {
    unsafe {
        kernel
            .cmd()
            .queue(&queue)
            // TODO: gws is unrequired here? already included in Kernel
            .gws(conv.gws())
            .enq()?;
    }
    queue.finish()?;
    Ok(())
}

fn run_dense_kernel_into_vec(
    dense_kernel: &Kernel,
    kernel_output_buf: &Buffer<f32>,
    dense: &DenseLayer,
    queue: &Queue,
) -> ocl::Result<Vec<f32>> {
    unsafe {
        dense_kernel
            .cmd()
            .queue(&queue)
            // TODO: gws is unrequired here? already included in Kernel
            .gws(dense.gws())
            .enq()?;
    }
    queue.finish()?;

    let out = unsafe {
        // Create a host-accessible output buffer for reading the dense3 output to host memory
        let mut mem_map = kernel_output_buf
            .map()
            .flags(flags::MAP_READ)
            .len(dense.num_out())
            .enq()?;

        // Read the dense3 output into host memory
        let dense_output_host_with_relu = relu(&mem_map, dense.num_out(), 1);

        mem_map.unmap().enq()?;
        dense_output_host_with_relu
    };

    Ok(out)
}

fn mtxmul_relu(input_buffer: &[f32], dense: &DenseLayer) -> Vec<f32> {
    let out = mtx_mul(
        dense.weights(),
        input_buffer,
        dense.num_out(),
        dense.num_in(),
        1,
    );
    relu(&out, dense.num_out(), 1)
}

fn mtxmul_softmax(input_buffer: &[f32], dense: &DenseLayer) -> Vec<f32> {
    let out = mtx_mul(
        dense.weights(),
        input_buffer,
        dense.num_out(),
        dense.num_in(),
        1,
    );
    softmax(&out, dense.num_out(), 1)
}

fn run() -> ocl::Result<()> {
    // TODO: extract hyper-parameters
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
    unsafe {
        // TODO: the blockings here are probably not required
        conv1_wgts_buf.write(conv1.weights()).block(true).enq()?;
        conv2_wgts_buf.write(conv2.weights()).block(true).enq()?;
        dense3_wgts_buf.write(dense3.weights()).block(true).enq()?;
    }
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

    unsafe {
        // Create a host-accessible input buffer for writing the image into device memory
        let mut mem_map = input_buf
            .map()
            .flags(flags::MAP_WRITE)
            .len(conv1.num_in())
            .enq()?;

        // Read the input image into the input_buf as f32s
        let input_data = read_image_with_padding(IN_FILE, *conv1.input_shape());
        for (idx, f) in input_data.into_iter().enumerate() {
            // TODO: one could pack them into Float4s, for instance here
            mem_map[idx] = f;
        }
        mem_map.unmap().enq()?;
    }
    // TODO: I added this here just in case, might not make a difference
    queue.finish()?;

    let start_time = Instant::now();

    // Enqueue the kernel for the 1st layer (Convolution + ReLU)
    run_conv_kernel(&conv_relu1, &conv1, &queue)?;
    let conv1_done_time = Instant::now();

    // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
    run_conv_kernel(&conv_relu2, &conv2, &queue)?;
    let conv2_done_time = Instant::now();

    // Enqueue the 3rd layer (fully-connected)
    let dense3_out = run_dense_kernel_into_vec(&dense3_kernel, &dense3_out_buf, &dense3, &queue)?;
    let dense3_done_time = Instant::now();

    // Run the 4th layer (fully-connected)
    let dense4_out = mtxmul_relu(&dense3_out, &dense4);
    let dense4_done_time = Instant::now();

    // Run the 5th layer (fully-connected)
    let output = mtxmul_softmax(&dense4_out, &dense5);
    let end_time = Instant::now();

    info!(
        "Per-layer computation time, L1: {}, L2: {}, L3: {}, L4: {}, L5: {}",
        duration_between(start_time, conv1_done_time),
        duration_between(conv1_done_time, conv2_done_time),
        duration_between(conv2_done_time, dense3_done_time),
        duration_between(dense3_done_time, dense4_done_time),
        duration_between(dense4_done_time, end_time)
    );
    info!(
        "Total computation time: {}",
        duration_between(start_time, end_time)
    );
    write_file_f32s(OUT_FILE, &output);

    Ok(())
}
