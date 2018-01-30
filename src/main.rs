extern crate byteorder;
extern crate env_logger;
#[macro_use]
extern crate log;
extern crate ocl;

mod geometry;
mod layers;
mod cl_util;
mod util;

use cl_util as cl;
use util::*;
use layers::*;
use geometry::*;
use ocl::{flags, Kernel, SpatialDims};

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
const OUTPUT_FILE: &'static str = "output/predictions.f";

fn main() {
    env_logger::init();

    match run() {
        Ok(_) => info!("Exited great."),
        Err(err) => info!("Exited with error: {}.", err),
    }
}

fn run() -> ocl::Result<()> {
    let conv1_filter_shape = PaddedSquare::from_side(CONV_1_FILTER_SIDE);
    let conv2_filter_shape = PaddedSquare::from_side(CONV_2_FILTER_SIDE);

    // Create descriptor for input geometry with the a shape of an image, side: 96, channels: 3 (RGB)
    let input_geom = ImageGeometry::new(96, NUM_IMAGE_CHANNELS);
    let padded_input_geom = input_geom.with_filter_padding(&conv2_filter_shape);
    // Feature map 1 is a fraction of the side of initial image geometry due to stride
    let fm1_geom = ImageGeometry::new(input_geom.side() / STRIDE, NUM_FEATURE_MAPS)
        .with_filter_padding(&conv2_filter_shape);
    // Feature map 2 is a fraction of the side of the tier 1 feature map due to stride
    let fm2_geom = ImageGeometry::new(input_geom.side() / STRIDE / STRIDE, NUM_FEATURE_MAPS);

    // Create a representation of the 1st convolutional layer with weights from a file
    let conv1 = ConvLayer::from_shapes(
        &conv1_filter_shape,
        &input_geom,
        &fm1_geom,
        "data/conv1_update.bin",
    );
    // Create a representation of the 2nd convolutional layer with weights from a file
    let conv2 = ConvLayer::from_shapes(
        &conv2_filter_shape,
        &fm1_geom,
        &fm2_geom,
        "data/conv2_update.bin",
    );
    // Create representations of the fully-connected layers
    let dense3 = DenseLayer::new(conv2.num_out(), FULLY_CONNECTED_CONST, "data/ip3.bin");
    let dense4 = DenseLayer::new(dense3.num_out(), FULLY_CONNECTED_CONST, "data/ip4.bin");
    let dense5 = DenseLayer::new(dense4.num_out(), NUM_OUTPUT_CLASSES, "data/ip_last.bin");

    // Verify that I/O dimensions match between layers
    verify_network_dimensions(&[&conv1, &conv2, &dense3, &dense4, &dense5]);

    // Initialize OpenCL
    let (queue, program, _context) = cl::init()?;

    // Allocate read-only memory for the weights of the 1st three layers
    debug!("Create conv1_wgts_buf with conv1.num_weights() elements.");
    let conv1_wgts_buf =
        cl::create_buffer::<f32>(conv1.num_weights(), flags::MEM_READ_ONLY, &queue)?;
    debug!("Create conv2_wgts_buf with conv2.num_weights() elements.");
    let conv2_wgts_buf =
        cl::create_buffer::<f32>(conv2.num_weights(), flags::MEM_READ_ONLY, &queue)?;
    debug!("Create dense3_wgts_buf with dense3.num_weights() elements.");
    let dense3_wgts_buf =
        cl::create_buffer::<f32>(dense3.num_weights(), flags::MEM_READ_ONLY, &queue)?;

    // Allocate read-only memory for the input geometry on device with host-accessible pointer for writing input from file
    debug!("Create input_buf with padded_input_geom.num_elems() elements.");
    let input_buf = cl::create_buffer::<f32>(
        padded_input_geom.num_elems(),
        /*flags::MEM_READ_ONLY*/ flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    )?;
    // Allocate read-write memory for the 1st feature map on device
    debug!("Create fm1_buf with fm1_geom.num_elems() elements.");
    let fm1_buf = cl::create_buffer::<f32>(fm1_geom.num_elems(), flags::MEM_READ_WRITE, &queue)?;
    // Allocate read-write memory for the 2nd feature map on device
    debug!("Create fm2_buf with fm2_geom.num_elems() elements.");
    let fm2_buf = cl::create_buffer::<f32>(fm2_geom.num_elems(), flags::MEM_READ_WRITE, &queue)?;
    // Allocate read-write memory for the dense layer output on device with host pointer for reading
    debug!("Create dense3_out with dense3.num_out() elements.");
    let dense3_out = cl::create_buffer::<f32>(
        dense3.num_out(),
        flags::MEM_READ_WRITE | flags::MEM_ALLOC_HOST_PTR,
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
    queue.finish().unwrap();

    let kernel1_global_size =
        SpatialDims::Three(fm1_geom.channels(), fm1_geom.side(), fm1_geom.side());
    let kernel2_global_size =
        SpatialDims::Three(fm2_geom.channels(), fm2_geom.side(), fm2_geom.side());
    let kernel3_global_size =
        SpatialDims::Three(dense3.num_out(), fm2_geom.side(), fm2_geom.side());

    // Create the kernel for the 1st layer (Convolution + ReLU)
    let conv_relu1 = Kernel::new("conv_relu_1", &program)?
        .queue(queue.clone())
        .gws(kernel1_global_size)//conv1.num_weights())
        // Input
        .arg_buf(&input_buf)
        // Output
        .arg_buf(&fm1_buf)
        .arg_buf(&conv1_wgts_buf);

    // Create the kernel for the 2nd layer (Convolution + ReLU)
    let conv_relu2 = Kernel::new("conv_relu_2", &program)?
        .queue(queue.clone())
        .gws(kernel2_global_size)//conv2.num_weights())
        // Input
        .arg_buf(&fm1_buf)
        // Output
        .arg_buf(&fm2_buf)
        .arg_buf(&conv2_wgts_buf);

    // Create the kernel for the 3rd layer (Dense layer matrix multiplication)
    let dense3_kernel = Kernel::new("mtx_mulf", &program)?
        .queue(queue.clone())
        .gws(kernel3_global_size)//dense3.num_weights())
        // Input
        .arg_buf(&fm2_buf)
        // Output
        .arg_buf(&dense3_out)
        .arg_buf(&dense3_wgts_buf);

    unsafe {
        // Create a host-accessible input buffer for writing the image into device memory
        let mut mem_map = input_buf
            .map()
            .flags(flags::MAP_WRITE)
            .len(padded_input_geom.num_elems())
            .enq()?;

        // Read the input image into the input_buf as f32s
        let input_data =
            read_image_with_filter_padding("data/c.bin", input_geom, conv1_filter_shape);
        for (idx, f) in input_data.into_iter().enumerate() {
            // TODO: one could pack them into Float4s, for instance here
            mem_map[idx] = f;
        }
        mem_map.unmap().enq()?;
    }

    // TODO: start profiling here, or at start of kernels

    // Enqueue the kernel for the 1st layer (Convolution + ReLU)
    debug!(
        "Enqueuing kernel conv_relu1 with global-workgroup-size {:?} = {}.",
        kernel1_global_size.to_lens().unwrap(),
        kernel1_global_size.to_len()
    );
    unsafe {
        conv_relu1
            .cmd()
            .queue(&queue)
            // TODO: gws is unrequired?
            .gws(kernel1_global_size)
            .enq()?;
    }
    queue.finish().unwrap();

    // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
    debug!(
        "Enqueuing kernel conv_relu2 with global-workgroup-size {:?} = {}.",
        kernel2_global_size.to_lens().unwrap(),
        kernel2_global_size.to_len()
    );
    unsafe {
        conv_relu2
            .cmd()
            .queue(&queue)
            // TODO: gws is unrequired?
            .gws(kernel2_global_size)
            .enq()?;
    }
    queue.finish().unwrap();

    // Enqueue the 3rd layer (fully-connected)
    debug!(
        "Enqueuing kernel dense3 with global-workgroup-size {:?} = {}.",
        kernel3_global_size.to_lens().unwrap(),
        kernel3_global_size.to_len()
    );
    unsafe {
        dense3_kernel
            .cmd()
            .queue(&queue)
            // TODO: gws is unrequired?
            .gws(kernel3_global_size)
            .enq()?;
    }
    queue.finish().unwrap();

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

    // TODO: why are all the consts here; replace with the actual source
    // Run the 4th layer (fully-connected)
    debug!(
        "Running lide_c_mtx_mulf for dense4.weights()*fifo_relu_out1. m_dim: {}, n_dim: {}, k_dim: {}.",
        FULLY_CONNECTED_CONST,
        FULLY_CONNECTED_CONST,
        1
    );
    let fifo_multi_out2 = lide_c_mtx_mulf(
        dense4.weights(),
        &fifo_relu_out1,
        FULLY_CONNECTED_CONST,
        FULLY_CONNECTED_CONST,
        1,
    );
    trace!("fifo_multi_out2: {:?}", &fifo_multi_out2);
    debug!(
        "Running lide_c_image_relu for fifo_multi_out2. row: {}, column: {}.",
        FULLY_CONNECTED_CONST, 1
    );
    let fifo_relu_out2 = lide_c_image_relu(&fifo_multi_out2, FULLY_CONNECTED_CONST, 1);
    trace!("fifo_relu_out2: {:?}", &fifo_relu_out2);

    // Run the 5th layer (fully-connected)
    debug!(
        "Running lide_c_mtx_mulf for dense5.weights()*fifo_relu_out2. m_dim: {}, n_dim: {}, k_dim: {}.",
        NUM_OUTPUT_CLASSES,
        FULLY_CONNECTED_CONST,
        1
    );
    let fifo_multi_out3 = lide_c_mtx_mulf(
        dense5.weights(),
        &fifo_relu_out2,
        NUM_OUTPUT_CLASSES,
        FULLY_CONNECTED_CONST,
        1,
    );
    trace!("fifo_multi_out3: {:?}", &fifo_multi_out3);
    debug!(
        "Writing fifo_multi_out3 to file {}. m: {}, n: {}",
        OUTPUT_FILE, 4, 1
    );
    lide_c_softmax_write(OUTPUT_FILE, &fifo_multi_out3, 4, 1);

    Ok(())
}

trait IndexMatrix<T> {
    fn elem(&self, length: usize, row: usize, column: usize) -> &T;
    fn elem_mut(&mut self, length: usize, row: usize, column: usize) -> &mut T;
}

impl<T> IndexMatrix<T> for [T] {
    fn elem(&self, length: usize, row: usize, column: usize) -> &T {
        &self[row * length + column]
    }
    fn elem_mut(&mut self, length: usize, row: usize, column: usize) -> &mut T {
        &mut self[row * length + column]
    }
}

/// Convert negative values in source to zero
fn lide_c_image_relu(source: &[f32], row: usize, column: usize) -> Vec<f32> {
    let mut destination = unsafe { vec![std::mem::uninitialized(); source.len()] };
    for i in 0..row {
        for j in 0..column {
            // Convert negative values to zero
            let elem = source.elem(column, i, j).max(0f32);
            *destination.elem_mut(column, i, j) = elem;
            /*
            // TODO: Try this alternative in-place implementation that works without the return value (allocation).
            let elem: &mut f32 = destination.elem_mut(column, i, j);
            *elem = elem.max(0f32);
            */
        }
    }
    destination
}

fn lide_c_mtx_mulf(a: &[f32], b: &[f32], m_dim: usize, n_dim: usize, k_dim: usize) -> Vec<f32> {
    let mut c_mul = unsafe { vec![std::mem::uninitialized(); m_dim * k_dim] };
    for i in 0..m_dim {
        for j in 0..k_dim {
            for z in 0..n_dim {
                *c_mul.elem_mut(k_dim, i, j) += a.elem(n_dim, i, z) * b.elem(k_dim, z, j);
            }
        }
    }
    c_mul
}

// FIXME: this function is weird with the types and trusting on undefined behavior of C. The results would be just a bit different if the sum wasn't put into an int. Ask Mir about it.
fn lide_c_softmax(input_array: &[f32], m: usize, n: usize) -> Vec<f32> {
    let mut sum = 0f32;
    for i in 0..m {
        for j in 0..n {
            sum += input_array.elem(n, i, j).exp();
        }
    }
    input_array
        .iter()
        .map(|&val| val.exp() / sum)
        .collect::<Vec<f32>>()
}

fn lide_c_softmax_write(filename: &str, input_array: &[f32], m: usize, n: usize) {
    let softmaxed = lide_c_softmax(input_array, m, n);
    write_file_f32s(filename, &softmaxed);
}

/// Reads a file into a Vec of f32s and verifies that the byte-count of the
/// input file matches with the expected amount of f32s.
pub fn read_image_with_filter_padding(
    filename: &str,
    src_image_shape: ImageGeometry,
    filter_shape: PaddedSquare,
) -> Vec<f32> {
    let padded_image_shape = src_image_shape.with_filter_padding(&filter_shape);
    let image = read_file_as_f32s(filename);

    debug_assert_eq!(image.len(), src_image_shape.num_elems());
    let padding = (padded_image_shape.side() - src_image_shape.side()) / 2;

    // TODO: There's some room to optimize here :)
    let mut v: Vec<f32> =
        unsafe { vec![std::mem::uninitialized(); padded_image_shape.num_elems()] };
    {
        let channels = v.chunks_mut(padded_image_shape.num_elems() / src_image_shape.channels());
        for (c, channel) in channels.enumerate() {
            let mut rows: Vec<&mut [f32]> = channel.chunks_mut(padded_image_shape.side()).collect();
            let (first_rows, other_rows) = rows.split_at_mut(padding);
            let (n_rows, last_rows) = other_rows.split_at_mut(src_image_shape.side());

            // Set the first row elements as 0's
            first_rows
                .iter_mut()
                .for_each(|row| row.iter_mut().for_each(|elem| *elem = 0f32));
            for (row_idx, row) in n_rows.iter_mut().enumerate() {
                let (mut pad_left, mut right) = row.split_at_mut(padding);
                let (mut im_middle, mut pad_right) = right.split_at_mut(src_image_shape.side());
                // Pad left side of image with 0's
                pad_left.iter_mut().for_each(|x| *x = 0f32);
                // Fill image center with contents
                for (col_idx, elem) in im_middle.iter_mut().enumerate() {
                    *elem = image[c * (src_image_shape.num_elems() / src_image_shape.channels())
                                      + row_idx * src_image_shape.side()
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
    // HACK: write input_buf into file
    unsafe {
        let mut mem_map = buf.map().flags(flags::MAP_READ).len(buf.len()).enq()?;
        write_file_f32s(filename, &mem_map);

        mem_map.unmap().enq()?;
    };
    Ok(())
}
