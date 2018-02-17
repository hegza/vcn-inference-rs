use super::*;

#[test]
fn test_l1() {
    let output = run_l1(&TEST_NETWORK).unwrap();
    let correct = f32::read_from_file(&format!("{}/fm1.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l2() {
    let output = run_l2(&TEST_NETWORK).unwrap();
    let correct = f32::read_from_file(&format!("{}/fm2.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l3() {
    let output = run_l3(&TEST_NETWORK).unwrap();
    let correct = f32::read_from_file(&format!("{}/fc3.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l4() {
    let output = run_l4(&TEST_NETWORK);
    let correct = f32::read_from_file(&format!("{}/fc4.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l5() {
    let output = run_l5(&TEST_NETWORK);
    let correct = f32::read_from_file(&format!("{}/out5.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

fn run_l1(params: &NetworkParams) -> ocl::Result<Vec<f32>> {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = params.create_conv(
        1,
        f32::read_bin_from_file(&format!("{}/conv1-f32-le.bin", WEIGHTS_DIR)),
    );

    let input_data = read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", BASELINE_DIR),
        *layer.input_shape(),
    );
    let (kernel, out_buf, queue) = create_standalone_kernel(&layer, "conv_relu_1", &input_data)?;
    // Enqueue the kernel for the 1st layer (Convolution + ReLU)
    run_kernel_wait(&kernel, &queue)?;
    unsafe { cl::read_buf(&out_buf) }
}

fn run_l2(params: &NetworkParams) -> ocl::Result<Vec<f32>> {
    // Create the representation of the 2nd convolutional layer with weights from a file
    let layer = params.create_conv(
        2,
        f32::read_bin_from_file(&format!("{}/conv2-f32-le.bin", WEIGHTS_DIR)),
    );

    let input_data = f32::read_from_file(&format!("{}/fm1.f", BASELINE_DIR));
    let (kernel, out_buf, queue) = create_standalone_kernel(&layer, "conv_relu_2", &input_data)?;
    // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
    run_kernel_wait(&kernel, &queue)?;
    unsafe { cl::read_buf(&out_buf) }
}

fn run_l3(params: &NetworkParams) -> ocl::Result<Vec<f32>> {
    // Create the representation of the fully-connected layer
    let layer = params.create_dense(
        3,
        f32::read_bin_from_file(&format!("{}/fc3-f32-le.bin", WEIGHTS_DIR)),
    );

    let input_data = f32::read_from_file(&format!("{}/fm2.f", BASELINE_DIR));
    let (kernel, out_buf, queue) = create_standalone_kernel(&layer, "mtx_mulf", &input_data)?;
    // Enqueue the kernel for the 3rd layer (Convolution)
    run_kernel_wait(&kernel, &queue)?;
    // Run relu on CPU
    Ok(relu(&unsafe { cl::read_buf(&out_buf)? }))
}

fn run_l4(params: &NetworkParams) -> Vec<f32> {
    // Create the representation of the fully-connected layer
    let layer = params.create_dense(
        4,
        f32::read_bin_from_file(&format!("{}/fc4-f32-le.bin", WEIGHTS_DIR)),
    );

    let input_data = f32::read_from_file(&format!("{}/fc3.f", BASELINE_DIR));
    mtxmul_relu(&input_data, &layer)
}

fn run_l5(params: &NetworkParams) -> Vec<f32> {
    // Create the representation of the fully-connected layer
    let layer = params.create_dense(
        5,
        f32::read_bin_from_file(&format!("{}/fc5-f32-le.bin", WEIGHTS_DIR)),
    );

    let input_data = f32::read_from_file(&format!("{}/fc4.f", BASELINE_DIR));
    mtxmul_softmax(&input_data, &layer)
}
