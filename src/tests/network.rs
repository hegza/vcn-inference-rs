use super::*;

#[test]
fn test_network() {
    env_logger::init();

    let output = run_network().unwrap();
    let correct = f32::read_from_file(&format!("{}/out5.f", BASELINE_DIR));
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

fn run_network() -> ocl::Result<Vec<f32>> {
    let (net, conv_relu1, conv_relu2, dense3_kernel, dense3_out_buf, queue) =
        init_network(&format!("{}/in.bin", BASELINE_DIR))?;

    let start_time = Instant::now();

    // Enqueue the kernel for the 1st layer (Convolution + ReLU)
    unsafe {
        run_kernel(&conv_relu1, &queue)?;
    }
    let conv1_done_time = Instant::now();

    // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
    unsafe {
        run_kernel(&conv_relu2, &queue)?;
    }
    let conv2_done_time = Instant::now();

    // Enqueue the 3rd layer (fully-connected)
    unsafe {
        run_kernel(&dense3_kernel, &queue)?;
    }
    let dense3_out = relu(
        &unsafe { cl::read_buf(&dense3_out_buf)? },
        net.dense3.num_out(),
        1,
    );

    let dense3_done_time = Instant::now();

    // Run the 4th layer (fully-connected)
    let dense4_out = mtxmul_relu(&dense3_out, &net.dense4);
    let dense4_done_time = Instant::now();

    // Run the 5th layer (fully-connected)
    let output = mtxmul_softmax(&dense4_out, &net.dense5);
    let end_time = Instant::now();

    debug!(
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
