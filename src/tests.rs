use cl_util as cl;
use util::*;
use network::*;
use ocl;
use ocl::{flags, Kernel};
use std::time::Instant;
use env_logger;
use super::*;

#[test]
fn test_network() {
    env_logger::init();

    // HACK: this is hacky, should put it into a file probably
    let result = run_network();
    let correct = vec![0.000000f32, 0.000019f32, 0.999980f32, 0.000000f32];
    match result {
        Ok(v) => assert!(is_within_margin(&v, &correct, 0.000001f32)),
        Err(err) => info!("Exited with error: {}", err),
    }
    assert!(run_network().is_ok());
}

fn is_within_margin(a: &[f32], b: &[f32], margin: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (idx, item) in a.iter().enumerate() {
        if (b[idx] - item).abs() > margin {
            return false;
        }
    }
    true
}

fn run_network() -> ocl::Result<Vec<f32>> {
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

    Ok(output)
}
