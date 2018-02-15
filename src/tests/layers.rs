use super::*;

#[test]
fn test_l1() {
    let output = run_l1(&TEST_NETWORK).unwrap();
    let correct = read_file_f32s(&format!("{}/fm1.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l2() {
    let output = run_l2(&TEST_NETWORK).unwrap();
    let correct = read_file_f32s(&format!("{}/fm2.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l3() {
    let output = run_l3(&TEST_NETWORK).unwrap();
    let correct = read_file_f32s(&format!("{}/fc3.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l4() {
    let output = run_l4(&TEST_NETWORK);
    let correct = read_file_f32s(&format!("{}/fc4.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l5() {
    let output = run_l5(&TEST_NETWORK);
    let correct = read_file_f32s(&format!("{}/out5.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

fn run_l1(params: &NetworkParams) -> ocl::Result<Vec<f32>> {
    // Create the representation of the 1st convolutional layer with weights from a file
    let conv = params.create_conv1(&format!("{}/conv1_update.bin", WEIGHTS_DIR));

    let input_data =
        read_image_with_padding(&format!("{}/in.bin", BASELINE_DIR), *conv.input_shape());
    let (kernel, out_buf, queue) = create_kernel(&conv, "conv_relu_1", &input_data)?;
    // Enqueue the kernel for the 1st layer (Convolution + ReLU)
    unsafe {
        run_kernel(&kernel, &conv, &queue)?;
    }
    unsafe { cl::read_buf(&out_buf) }
}

fn run_l2(params: &NetworkParams) -> ocl::Result<Vec<f32>> {
    // Create the representation of the 2nd convolutional layer with weights from a file
    let conv = params.create_conv2(&format!("{}/conv2_update.bin", WEIGHTS_DIR));

    let input_data = read_file_f32s(&format!("{}/fm1.f", BASELINE_DIR));
    let (kernel, out_buf, queue) = create_kernel(&conv, "conv_relu_2", &input_data)?;
    // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
    unsafe {
        run_kernel(&kernel, &conv, &queue)?;
    }
    unsafe { cl::read_buf(&out_buf) }
}

fn run_l3(params: &NetworkParams) -> ocl::Result<Vec<f32>> {
    // Create the representation of the fully-connected layer
    let dense = params.create_dense3(&format!("{}/ip3.bin", WEIGHTS_DIR));

    let input_data = read_file_f32s(&format!("{}/fm2.f", BASELINE_DIR));
    let (kernel, out_buf, queue) = create_kernel(&dense, "mtx_mulf", &input_data)?;
    // Enqueue the kernel for the 3rd layer (Convolution)
    unsafe {
        run_kernel(&kernel, &dense, &queue)?;
    }
    // Run relu on CPU
    Ok(relu(
        &unsafe { cl::read_buf(&out_buf)? },
        dense.num_out(),
        1,
    ))
}

fn run_l4(params: &NetworkParams) -> Vec<f32> {
    // Create the representation of the fully-connected layer
    let dense = params.create_dense4(&format!("{}/ip4.bin", WEIGHTS_DIR));

    let input_data = read_file_f32s(&format!("{}/fc3.f", BASELINE_DIR));
    mtxmul_relu(&input_data, &dense)
}

fn run_l5(params: &NetworkParams) -> Vec<f32> {
    // Create the representation of the fully-connected layer
    let dense = params.create_dense5(&format!("{}/ip_last.bin", WEIGHTS_DIR));

    let input_data = read_file_f32s(&format!("{}/fc4.f", BASELINE_DIR));
    mtxmul_softmax(&input_data, &dense)
}
