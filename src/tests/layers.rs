use super::*;
use geometry::*;
use ocl::{Kernel, SpatialDims};

#[test]
fn test_l1() {
    let output = run_l1(&TEST_NETWORK).unwrap();
    let correct = f32::read_lines_from_file(&format!("{}/fm1.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l2() {
    let output = run_l2(&TEST_NETWORK).unwrap();
    let correct = f32::read_lines_from_file(&format!("{}/fm2.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l3() {
    let output = run_l3(&TEST_NETWORK).unwrap();
    let correct = f32::read_lines_from_file(&format!("{}/fc3.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l4() {
    let output = run_l4(&TEST_NETWORK);
    let correct = f32::read_lines_from_file(&format!("{}/fc4.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l5() {
    let output = run_l5(&TEST_NETWORK);
    let correct = f32::read_lines_from_file(&format!("{}/out5.f", BASELINE_DIR));
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

    let input_data = f32::read_lines_from_file(&format!("{}/fm1.f", BASELINE_DIR));
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

    let input_data = f32::read_lines_from_file(&format!("{}/fm2.f", BASELINE_DIR));
    let (kernel, out_buf, queue) = create_standalone_kernel(&layer, "mtx_mul_f32", &input_data)?;
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

    let input_data = f32::read_lines_from_file(&format!("{}/fc3.f", BASELINE_DIR));
    relu(&layer.mtx_mul(&input_data))
}

fn run_l5(params: &NetworkParams) -> Vec<f32> {
    // Create the representation of the fully-connected layer
    let layer = params.create_dense(
        5,
        f32::read_bin_from_file(&format!("{}/fc5-f32-le.bin", WEIGHTS_DIR)),
    );

    let input_data = f32::read_lines_from_file(&format!("{}/fc4.f", BASELINE_DIR));
    softmax(&layer.mtx_mul(&input_data))
}

// Test that Maxpool + ReLU produces the correct output
#[test]
fn test_mxp() {
    let in_img = f32::read_csv("src/tests/in/img-4x4_mono-norm.csv");
    const SIDE: usize = 4;
    let in_shape = ImageGeometry::new(SIDE, 1);

    // Make sure the input image matches with the assumed input shape
    assert_eq!(in_shape.num_elems(), in_img.len());

    let mxp = MaxpoolLayer::new(in_shape, 2);

    // Run mxp on GPU
    let (queue, program, _context) = cl::init(&["test/mxp.cl"], &[]).unwrap();

    let (in_buf, out_buf) = mxp.create_io_bufs(
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    );
    let mxp_krn = Kernel::builder()
        .program(&program)
        .name("max_pool")
        .queue(queue.clone())
        .global_work_size(SpatialDims::Three(SIDE, SIDE, 1))
        .local_work_size(SpatialDims::Three(SIDE, SIDE, 1))
        .arg(&in_buf)
        .arg(&out_buf)
        .build()
        .unwrap();
    let gpu_out = unsafe {
        cl::map_to_buf(&in_buf, &in_img).unwrap();
        mxp_krn.cmd().queue(&queue).enq().unwrap();
        queue.finish().unwrap();
        cl::read_buf(&out_buf).unwrap()
    };

    // Run maxpool on CPU
    let cpu_out = mxp.compute(&in_img);

    // Verify match between confirmed correct output and GPU and CPU outputs
    let correct = f32::read_csv("src/tests/out/img-4x4_mono-norm-mxp2.csv");
    assert!(is_within_margin(&gpu_out, &correct, RESULT_MARGIN));
    assert!(is_within_margin(&cpu_out, &correct, RESULT_MARGIN));
}
