use super::*;

#[test]
fn test_l1() {
    let output = run_l1(&HYPER_PARAMS).unwrap();
    let correct = read_file_f32s(&format!("{}/fm1.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l2() {
    let output = run_l2(&HYPER_PARAMS).unwrap();
    let correct = read_file_f32s(&format!("{}/fm2.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l3() {
    let output = run_l3(&HYPER_PARAMS).unwrap();
    let correct = read_file_f32s(&format!("{}/fc3.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l4() {
    let output = run_l4(&HYPER_PARAMS);
    let correct = read_file_f32s(&format!("{}/fc4.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

#[test]
fn test_l5() {
    let output = run_l5(&HYPER_PARAMS);
    let correct = read_file_f32s(&format!("{}/out5.f", BASELINE_DIR));
    assert_eq!(output.len(), correct.len());
    assert!(is_within_margin(&output, &correct, RESULT_MARGIN));
}

fn create_kernel<L: Layer<f32>>(
    layer: &L,
    kernel_func: &str,
    input_data: &[f32],
) -> ocl::Result<(Kernel, Buffer<f32>, Queue)> {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init()?;

    let wgts_buf =
        cl::create_buffer::<f32>("weights", layer.num_weights(), flags::MEM_READ_ONLY, &queue)?;
    let in_buf = cl::create_buffer::<f32>(
        "input",
        layer.num_in(),
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    )?;
    let out_buf = cl::create_buffer::<f32>(
        "output",
        layer.num_out(),
        flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    )?;

    let kernel = Kernel::new(kernel_func, &program)?
        .queue(queue.clone())
        .gws(layer.gws())
        // Input
        .arg_buf(&in_buf)
        // Output
        .arg_buf(&out_buf)
        .arg_buf(&wgts_buf);

    // Write the weights to the global memory of the device
    wgts_buf.write(layer.weights()).enq()?;

    unsafe {
        // Create a host-accessible input buffer for writing the image into device memory
        let mut mem_map = in_buf
            .map()
            .flags(flags::MAP_WRITE)
            .len(layer.num_in())
            .enq()?;

        // Write input data to device memory
        for (idx, f) in input_data.iter().enumerate() {
            mem_map[idx] = *f;
        }
        mem_map.unmap().enq()?;
    }

    Ok((kernel, out_buf, queue))
}

fn run_l1(params: &HyperParams) -> ocl::Result<Vec<f32>> {
    let conv1_filter_shape = PaddedSquare::from_side(params.conv_1_filter_side);
    let conv2_filter_shape = PaddedSquare::from_side(params.conv_2_filter_side);

    // Create descriptor for input geometry with the a shape and properties of an image
    let input_shape = ImageGeometry::new(params.source_side, params.num_source_channels);
    let padded_input_shape = input_shape.with_filter_padding(&conv1_filter_shape);

    // Feature map 1 is a fraction of the side of initial image geometry due to stride
    let fm1_shape = ImageGeometry::new(input_shape.side() / params.stride, params.num_feature_maps);
    let padded_fm1_shape = fm1_shape.with_filter_padding(&conv2_filter_shape);

    run_conv(
        params.conv_1_filter_side,
        padded_input_shape,
        padded_fm1_shape,
        &format!("{}/conv1_update.bin", WEIGHTS_DIR),
        "conv_relu_1",
        &read_image_with_padding(&format!("{}/in.bin", BASELINE_DIR), padded_input_shape),
    )
}

fn run_l2(params: &HyperParams) -> ocl::Result<Vec<f32>> {
    let conv2_filter_shape = PaddedSquare::from_side(params.conv_2_filter_side);

    // Create descriptor for input geometry with the a shape and properties of an image
    let input_shape = ImageGeometry::new(params.source_side, params.num_source_channels);
    // Feature map 1 is a fraction of the side of initial image geometry due to stride
    let fm1_shape = ImageGeometry::new(input_shape.side() / params.stride, params.num_feature_maps);
    let padded_fm1_shape = fm1_shape.with_filter_padding(&conv2_filter_shape);
    // Feature map 2 is a fraction of the side of the tier 1 feature map due to stride
    let fm2_shape = ImageGeometry::new(fm1_shape.side() / params.stride, params.num_feature_maps);

    run_conv(
        params.conv_2_filter_side,
        padded_fm1_shape,
        fm2_shape,
        &format!("{}/conv2_update.bin", WEIGHTS_DIR),
        "conv_relu_2",
        &read_file_f32s(&format!("{}/fm1.f", BASELINE_DIR)),
    )
}

fn run_l3(params: &HyperParams) -> ocl::Result<Vec<f32>> {
    // Create descriptor for input geometry with the a shape and properties of an image
    let input_shape = ImageGeometry::new(params.source_side, params.num_source_channels);
    // Feature map 1 is a fraction of the side of initial image geometry due to stride
    let fm1_shape = ImageGeometry::new(input_shape.side() / params.stride, params.num_feature_maps);
    // Feature map 2 is a fraction of the side of the tier 1 feature map due to stride
    let fm2_shape = ImageGeometry::new(fm1_shape.side() / params.stride, params.num_feature_maps);

    // Create the representation of the fully-connected layer
    let dense = DenseLayer::new(
        fm2_shape.num_elems(),
        params.fully_connected_const,
        &format!("{}/ip3.bin", WEIGHTS_DIR),
    );

    let input_data = read_file_f32s(&format!("{}/fm2.f", BASELINE_DIR));
    let (kernel, out_buf, queue) = create_kernel(&dense, "mtx_mulf", &input_data)?;
    run_kernel_relu(&kernel, &out_buf, &dense, &queue)
}

fn run_l4(params: &HyperParams) -> Vec<f32> {
    // Create the representation of the fully-connected layer
    let dense = DenseLayer::new(
        params.fully_connected_const,
        params.fully_connected_const,
        &format!("{}/ip4.bin", WEIGHTS_DIR),
    );

    let input_data = read_file_f32s(&format!("{}/fc3.f", BASELINE_DIR));
    mtxmul_relu(&input_data, &dense)
}

fn run_l5(params: &HyperParams) -> Vec<f32> {
    // Create the representation of the fully-connected layer
    let dense = DenseLayer::new(
        params.fully_connected_const,
        params.num_output_classes,
        &format!("{}/ip_last.bin", WEIGHTS_DIR),
    );

    let input_data = read_file_f32s(&format!("{}/fc4.f", BASELINE_DIR));
    mtxmul_softmax(&input_data, &dense)
}

fn run_conv(
    filter_side: usize,
    input_shape: ImageGeometry,
    output_shape: ImageGeometry,
    weights_file: &str,
    kernel_func: &str,
    input_data: &[f32],
) -> ocl::Result<Vec<f32>> {
    let filter = PaddedSquare::from_side(filter_side);

    // Create the representation of the 1st convolutional layer with weights from a file
    let conv = ConvLayer::from_shapes(
        filter.num_elems(),
        &input_shape,
        &output_shape,
        weights_file,
    );

    let (kernel, out_buf, queue) = create_kernel(&conv, kernel_func, &input_data)?;
    // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
    run_kernel(&kernel, &conv, &queue)?;
    unsafe { cl::read_buf(&out_buf) }
}
