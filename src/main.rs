#![feature(test)]

extern crate byteorder;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate ocl;
extern crate test;

mod geometry;
mod layers;
mod cl_util;
mod util;
mod math;

use cl_util as cl;
use util::*;
use layers::*;
use geometry::*;
use math::*;
use ocl::{flags, Buffer, Kernel, Queue, SpatialDims};
use std::time::Instant;

// TODO: Write these into a hyper-parameters struct
// Channels for each RGB color
const NUM_IMAGE_CHANNELS: usize = 3;
// The size of the filter/kernels
const CONV_1_FILTER_SIDE: usize = 5;
const CONV_2_FILTER_SIDE: usize = 5;
// The number of feature maps
const NUM_FEATURE_MAPS: usize = 32;
const STRIDE: usize = 2;
// ???: What is this, what does it do? Was MAGIC in Jani's code
const FULLY_CONNECTED_CONST: usize = 100;
const NUM_OUTPUT_CLASSES: usize = 4;
const IN_FILE: &'static str = "input/baseline/input1/in.bin";
const OUT_FILE: &'static str = "output/out.f";

fn main() {
    env_logger::init();

    match run() {
        Ok(_) => info!("Exited great."),
        Err(err) => info!("Exited with error: {}.", err),
    }
}

fn create_network() -> (ConvLayer, ConvLayer, DenseLayer, DenseLayer, DenseLayer) {
    let conv1_filter_shape = PaddedSquare::from_side(CONV_1_FILTER_SIDE);
    let conv2_filter_shape = PaddedSquare::from_side(CONV_2_FILTER_SIDE);

    // Create descriptor for input geometry with the a shape of an image, side: 96,
    // channels: 3 (RGB)
    let input_geom = ImageGeometry::new(96, NUM_IMAGE_CHANNELS);
    let padded_input_geom = input_geom.with_filter_padding(&conv1_filter_shape);
    // Feature map 1 is a fraction of the side of initial image geometry due to stride
    let fm1_geom = ImageGeometry::new(input_geom.side() / STRIDE, NUM_FEATURE_MAPS);
    let padded_fm1_geom = fm1_geom.with_filter_padding(&conv2_filter_shape);
    // Feature map 2 is a fraction of the side of the tier 1 feature map due to stride
    let fm2_geom = ImageGeometry::new(fm1_geom.side() / STRIDE, NUM_FEATURE_MAPS);

    // Create a representation of the 1st convolutional layer with weights from a file
    let conv1 = ConvLayer::from_shapes(
        conv1_filter_shape.num_elems(),
        &padded_input_geom,
        &padded_fm1_geom,
        "input/weights/conv1_update.bin",
    );
    // Create a representation of the 2nd convolutional layer with weights from a file
    let conv2 = ConvLayer::from_shapes(
        conv2_filter_shape.num_elems(),
        &padded_fm1_geom,
        &fm2_geom,
        "input/weights/conv2_update.bin",
    );
    // Create representations of the fully-connected layers
    let dense3 = DenseLayer::new(
        conv2.num_out(),
        FULLY_CONNECTED_CONST,
        "input/weights/ip3.bin",
    );
    let dense4 = DenseLayer::new(
        dense3.num_out(),
        FULLY_CONNECTED_CONST,
        "input/weights/ip4.bin",
    );
    let dense5 = DenseLayer::new(
        dense4.num_out(),
        NUM_OUTPUT_CLASSES,
        "input/weights/ip_last.bin",
    );

    // Verify that I/O dimensions match between layers
    verify_network_dimensions(&[&conv1, &conv2, &dense3, &dense4, &dense5]);

    (conv1, conv2, dense3, dense4, dense5)
}

fn run_conv(conv: &ConvLayer, kernel: &Kernel, queue: &Queue) -> ocl::Result<()> {
    debug!(
        "Enqueuing conv kernel with global-workgroup-size {:?} = {}.",
        conv.gws().to_lens()?,
        conv.gws().to_len()
    );
    unsafe {
        kernel
            .cmd()
            .queue(&queue)
            // TODO: gws is unrequired?
            .gws(conv.gws())
            .enq()?;
    }
    queue.finish()?;
    Ok(())
}

fn run_dense3(
    dense3: &DenseLayer,
    dense3_kernel: &Kernel,
    dense3_out: &Buffer<f32>,
    kernel3_global_size: SpatialDims,
    queue: &Queue,
) -> ocl::Result<Vec<f32>> {
    debug!(
        "Enqueuing kernel dense3 with global-workgroup-size {:?} = {}.",
        kernel3_global_size.to_lens()?,
        kernel3_global_size.to_len()
    );
    unsafe {
        dense3_kernel
            .cmd()
            .queue(&queue)
            //TODO: gws is unrequired?
            .gws(kernel3_global_size)
            .enq()?;
    }
    queue.finish()?;

    let fifo_relu_out1 = unsafe {
        // Create a host-accessible output buffer for reading the dense3 output to host memory
        let mut mem_map = dense3_out
            .map()
            .flags(flags::MAP_READ)
            .len(dense3.num_out())
            .enq()?;
        trace!("dense3_out mem_map: {:?}", &mem_map as &[f32]);

        // Read the dense3 output into host memory
        debug!(
            "Running lide_c_image_relu for mem_map. row: {}, column: {}.",
            dense3.num_out(),
            1
        );
        let dense3_output_host_with_relu = lide_c_image_relu(&mem_map, dense3.num_out(), 1);

        mem_map.unmap().enq()?;
        dense3_output_host_with_relu
    };
    trace!("fifo_relu_out1: {:?}", &fifo_relu_out1);
    Ok(fifo_relu_out1)
}

fn run_dense4(weights: &Vec<f32>, fifo: &[f32]) -> Vec<f32> {
    // TODO: why are all the consts here; replace with the actual source
    debug!(
        "Running lide_c_mtx_mulf for dense4.weights()*fifo_relu_out1. m_dim: {}, n_dim: {}, k_dim: {}.",
        FULLY_CONNECTED_CONST,
        FULLY_CONNECTED_CONST,
        1
    );
    let fifo_multi_out2 = lide_c_mtx_mulf(
        weights,
        fifo,
        FULLY_CONNECTED_CONST,
        FULLY_CONNECTED_CONST,
        1,
    );
    trace!("fifo_multi_out2: {:?}", &fifo_multi_out2);
    debug!(
        "Running lide_c_image_relu for fifo_multi_out2. row: {}, column: {}.",
        FULLY_CONNECTED_CONST, 1
    );
    lide_c_image_relu(&fifo_multi_out2, FULLY_CONNECTED_CONST, 1)
}

fn run_dense5(weights: &Vec<f32>, fifo: &[f32]) -> Vec<f32> {
    debug!(
        "Running lide_c_mtx_mulf for dense5.weights()*fifo_relu_out2. m_dim: {}, n_dim: {}, k_dim: {}.",
        NUM_OUTPUT_CLASSES,
        FULLY_CONNECTED_CONST,
        1
    );
    let fifo_multi_out3 =
        lide_c_mtx_mulf(weights, &fifo, NUM_OUTPUT_CLASSES, FULLY_CONNECTED_CONST, 1);
    trace!("fifo_multi_out3: {:?}", &fifo_multi_out3);
    lide_c_softmax(&fifo_multi_out3, 4, 1)
}

fn run() -> ocl::Result<()> {
    // TODO: extract hyper-parameters
    // Create the network representation from network hyper-parameters
    let (conv1, conv2, dense3, dense4, dense5) = create_network();

    // Initialize OpenCL
    let (queue, program, _context) = cl::init()?;

    // Allocate read-only memory for the weights of the 1st three layers
    let conv1_wgts_buf =
        cl::create_buffer::<f32>(conv1.num_weights(), flags::MEM_READ_ONLY, &queue)?;
    let conv2_wgts_buf =
        cl::create_buffer::<f32>(conv2.num_weights(), flags::MEM_READ_ONLY, &queue)?;
    let dense3_wgts_buf =
        cl::create_buffer::<f32>(dense3.num_weights(), flags::MEM_READ_ONLY, &queue)?;

    // Allocate read-only memory for the input geometry on device with host-accessible pointer for
    // writing input from file
    let input_buf = cl::create_buffer::<f32>(
        conv1.num_in(),
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    )?;
    // Allocate read-write memory for the 1st feature map on device
    let fm1_buf = cl::create_buffer::<f32>(conv1.num_out(), flags::MEM_READ_WRITE, &queue)?;
    // Allocate read-write memory for the 2nd feature map on device
    let fm2_buf = cl::create_buffer::<f32>(conv2.num_out(), flags::MEM_READ_WRITE, &queue)?;
    // Allocate read-write memory for the dense (3rd) layer output on device with host pointer for reading
    let dense3_out_buf = cl::create_buffer::<f32>(dense3.num_out(), flags::MEM_WRITE_ONLY, &queue)?;

    // Write the weights of the 1st three layers to the global memory of the device
    unsafe {
        // TODO: the blockings here are probably not required
        conv1_wgts_buf.write(conv1.weights()).block(true).enq()?;
        conv2_wgts_buf.write(conv2.weights()).block(true).enq()?;
        dense3_wgts_buf.write(dense3.weights()).block(true).enq()?;
    }
    // TODO: this might not make any difference anywhere (verify with benchmark)
    queue.finish()?;

    info!(
        "dense3.num_out() = {}, fm2_geom.side() = {}, fm2_geom.side() = {}",
        dense3.num_out(),
        conv2.output_shape().side(),
        conv2.output_shape().side()
    );
    let kernel3_global_size = SpatialDims::Three(
        dense3.num_out(),
        conv2.output_shape().side(),
        conv2.output_shape().side(),
    );

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
        .gws(kernel3_global_size)
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
    run_conv(&conv1, &conv_relu1, &queue)?;

    let conv1_done_time = Instant::now();

    // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
    run_conv(&conv2, &conv_relu2, &queue)?;

    let conv2_done_time = Instant::now();

    // Enqueue the 3rd layer (fully-connected)
    let dense3_out = run_dense3(
        &dense3,
        &dense3_kernel,
        &dense3_out_buf,
        kernel3_global_size,
        &queue,
    )?;
    let dense3_done_time = Instant::now();

    // Run the 4th layer (fully-connected)
    let dense4_out = run_dense4(&dense4.weights(), &dense3_out);
    let dense4_done_time = Instant::now();

    // Run the 5th layer (fully-connected)
    let output = run_dense5(&dense5.weights(), &dense4_out);
    let end_time = Instant::now();

    info!(
        "Total computation time: {}",
        duration_between(start_time, end_time)
    );
    info!(
        "Per-layer computation time, L1: {}, L2: {}, L3: {}, L4: {}, L5: {}",
        duration_between(start_time, conv1_done_time),
        duration_between(conv1_done_time, conv2_done_time),
        duration_between(conv2_done_time, dense3_done_time),
        duration_between(dense3_done_time, dense4_done_time),
        duration_between(dense4_done_time, end_time)
    );
    write_file_f32s(OUT_FILE, &output);

    Ok(())
}

fn duration_between(start: Instant, end: Instant) -> f64 {
    let duration = end.duration_since(start);
    duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 0.000000001f64
}

/// Reads a file into a Vec of f32s and adds the given amount of padding.
pub fn read_image_with_padding(filename: &str, padded_image_shape: ImageGeometry) -> Vec<f32> {
    let image = read_file_as_f32s(filename);
    let image_shape = padded_image_shape.unpadded();

    debug_assert_eq!(image.len(), image_shape.num_elems());
    let padding = (padded_image_shape.side() - image_shape.side()) / 2;

    // TODO: There's some room to optimize here :)
    let mut v: Vec<f32> =
        unsafe { vec![std::mem::uninitialized(); padded_image_shape.num_elems()] };
    {
        let channels = v.chunks_mut(padded_image_shape.num_elems() / padded_image_shape.channels());
        for (c, channel) in channels.enumerate() {
            let mut rows: Vec<&mut [f32]> = channel.chunks_mut(padded_image_shape.side()).collect();
            let (first_rows, other_rows) = rows.split_at_mut(padding);
            let (n_rows, last_rows) = other_rows.split_at_mut(image_shape.side());

            // Set the first row elements as 0's
            first_rows
                .iter_mut()
                .for_each(|row| row.iter_mut().for_each(|elem| *elem = 0f32));
            for (row_idx, row) in n_rows.iter_mut().enumerate() {
                let (mut pad_left, mut right) = row.split_at_mut(padding);
                let (mut im_middle, mut pad_right) = right.split_at_mut(image_shape.side());
                // Pad left side of image with 0's
                pad_left.iter_mut().for_each(|x| *x = 0f32);
                // Fill image center with contents
                for (col_idx, elem) in im_middle.iter_mut().enumerate() {
                    *elem = image[c * (image_shape.num_elems() / padded_image_shape.channels())
                                      + row_idx * image_shape.side()
                                      + col_idx];
                }
                // Pad right side of image with 0's
                pad_right.iter_mut().for_each(|x| *x = 0f32);
            }
            // Set the last rows elements as 0's
            last_rows
                .iter_mut()
                .for_each(|row| row.iter_mut().for_each(|elem| *elem = 0f32));
        }
    }
    debug_assert_eq!(v.len(), padded_image_shape.num_elems());
    v
}

#[allow(dead_code)]
fn write_buf_to_file(filename: &str, buf: ocl::Buffer<f32>) -> ocl::Result<()> {
    unsafe {
        let mut mem_map = buf.map().flags(flags::MAP_READ).len(buf.len()).enq()?;
        write_file_f32s(filename, &mem_map);

        mem_map.unmap().enq()?;
    };
    Ok(())
}
