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
#[cfg(test)]
mod tests;

use util::*;
use network::*;
use geometry::*;
use math::*;
use ocl::{flags, Buffer, Kernel, Queue};

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
const IN_FILE: &'static str = "input/baseline/input1/in.bin";
const WEIGHTS_DIR: &'static str = "input/weights";

pub fn create_network(
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

pub fn run_conv_kernel(kernel: &Kernel, conv: &ConvLayer, queue: &Queue) -> ocl::Result<()> {
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

pub fn run_dense_kernel_into_vec(
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
