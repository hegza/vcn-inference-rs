extern crate byteorder;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate ocl;

mod geometry;
mod layers;
mod ocl_util;
mod util;

use ocl_util::*;
use util::*;
use layers::*;
use geometry::*;
use ocl::{flags, Buffer, Kernel, SpatialDims};

// Channels for each RGB color
const IMAGE_CHANNELS: usize = 3;
// The size of the filter/kernels
const CONV_1_FILTER_SIDE: usize = 5;
const CONV_2_FILTER_SIDE: usize = 5;
// The number of feature maps
const FM_COUNT: usize = 32;
// ???: What is this, what does it do?
const PAD_NUM: usize = 2;

fn main() {
    env_logger::init();

    match run() {
        Ok(_) => println!("Exited in a fine fashion."),
        Err(err) => println!("Exited with error: {}.", err),
    }
}

fn run() -> ocl::Result<()> {
    let (queue, program, _context) = init()?;

    // Create a representation of the 1st convolutional layer with weights from a file
    let conv1 = ConvLayer::new(
        CONV_1_FILTER_SIDE,
        IMAGE_CHANNELS,
        FM_COUNT,
        "data/conv1_update.bin",
    );
    // Create a representation of the 2nd convolutional layer with weights from a file
    let conv2 = ConvLayer::new(
        CONV_2_FILTER_SIDE,
        FM_COUNT,
        FM_COUNT,
        "data/conv2_update.bin",
    );
    // TODO: Create a representation of the 3rd layer (dense) with weights from a file
    // TODO: Create a representation of the 4th layer (dense) with weights from a file
    // TODO: Create a representation of the 5th layer (dense) with weights from a file

    // Create descriptor for input geometry with the a shape of an image, 96 of side, and 3 RGB channels
    let input_geometry = ImageGeometry::new(96, 0, IMAGE_CHANNELS);
    let padded_input_geometry = input_geometry.with_filter_padding(conv2.filter_shape());
    // Feature map 1 is half the side of initial image geometry due to stride of 2
    let feature_map1 = ImageGeometry::new(input_geometry.side() / 2, 0, FM_COUNT)
        .with_filter_padding(&conv2.filter_shape());
    // Feature map 2 is half the side of the tier 1 feature map due to stride of 2
    let feature_map2 = ImageGeometry::new(feature_map1.side() / 2, 0, FM_COUNT);

    // Allocate read-only memory for the input geometry on device
    let input_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR)
        .dims(padded_input_geometry.num_elements())
        .build()?;
    // Allocate read-only memory for the weights of the first layer on device
    let conv1_weights_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY)
        .dims(conv1.num_weights())
        .build()?;
    // Allocate read-write memory for the 1st feature map on device
    let middle_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .dims(feature_map1.num_elements())
        .build()?;
    // Allocate read-write memory for the 2nd feature map on device
    let conv2_output_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_WRITE)
        .dims(feature_map2.num_elements())
        .build()?;
    // Allocate read-only memory for the weights of the 2nd layer on device
    let conv2_weights_buffer = Buffer::<f32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY)
        .dims(conv2.num_weights())
        .build()?;

    // Assign the weights of the 1st and 2nd layer to the global memory of the device
    unsafe {
        conv1_weights_buffer
            .write(conv1.weights())
            .block(true)
            .enq()?;
        conv2_weights_buffer
            .write(conv2.weights())
            .block(true)
            .enq()?;
    }

    // Create the kernel for the 1st layer (Convolution + ReLU)
    let conv_relu_1 = Kernel::new("conv_relu_1", &program)?
        .queue(queue.clone())
        .gws(conv1.num_weights())
        // Input
        .arg_buf(&input_buffer)
        // Output
        .arg_buf(&middle_buffer)
        .arg_buf(&conv1_weights_buffer);

    // Create the kernel for the 2nd layer (Convolution + ReLU)
    let conv_relu_2 = Kernel::new("conv_relu_2", &program)?
        .queue(queue.clone())
        .gws(conv2.num_weights())
        // Input
        .arg_buf(&middle_buffer)
        // Output
        .arg_buf(&conv2_output_buffer)
        .arg_buf(&conv2_weights_buffer);

    unsafe {
        let mut mem_map = input_buffer
            .map()
            .flags(flags::MAP_WRITE)
            .len(padded_input_geometry.num_elements())
            .enq()?;

        // Read the input image into the input_buffer as f32s
        let input_data =
            read_file_as_f32s_checked("data/c.bin", input_geometry.num_elements()).unwrap();
        println!("Reading {} f32s from \"data/c.bin\" to global memory of the device at offset: 0 f32s, size: {} f32s.", input_data.len(), mem_map.len());
        for (idx, f) in input_data.into_iter().enumerate() {
            // TODO: one could pack them into Float4s, for instance here
            mem_map[idx] = f;
        }
        mem_map.unmap().enq()?;
    }

    let global_size = SpatialDims::Three(
        FM_COUNT,
        feature_map1.side() + 2 * PAD_NUM,
        feature_map1.side() + 2 * PAD_NUM,
    );
    println!(
        "Enqueuing kernel conv_relu_1 with global-workgroup-size {:?} = {}.",
        global_size.to_lens().unwrap(),
        global_size.to_len()
    );

    // Enqueue the kernel for the 1st layer (Convolution + ReLU)
    unsafe {
        conv_relu_1.cmd().queue(&queue).gws(global_size).enq()?;
    }
    queue.finish().unwrap();

    let global_size = SpatialDims::Three(
        FM_COUNT,
        feature_map1.side() / PAD_NUM,
        feature_map1.side() / PAD_NUM,
    );
    println!(
        "Enqueuing kernel conv_relu_2 with global-workgroup-size {:?} = {}.",
        global_size.to_lens().unwrap(),
        global_size.to_len()
    );

    // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
    unsafe {
        conv_relu_2.cmd().queue(&queue).gws(global_size).enq()?;
    }
    queue.finish().unwrap();

    // TODO: Enqueue the 3rd layer (fully-connected)

    // TODO: Enqueue the 4th layer (fully-connected)

    // TODO: Enqueue the 5th layer (fully-connected)

    Ok(())
}
