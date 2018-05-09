use super::*;
use geometry::*;
use ocl::{flags, Context, Device, Kernel, Platform, Program, SpatialDims};

#[test]
fn test_l1() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = CLASSIC_PARAMS.create_conv(
        1,
        f32::read_bin_from_file(&format!("{}/conv1-f32-le.bin", WEIGHTS_DIR)),
    );
    let input_data = read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", CLASSIC_BASELINE),
        *layer.input_shape(),
    );

    let output = run_single_layer("conv_relu_1", &layer, &input_data);
    let correct = f32::read_lines_from_file(&format!("{}/fm1.f", CLASSIC_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn test_l2() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = CLASSIC_PARAMS.create_conv(
        2,
        f32::read_bin_from_file(&format!("{}/conv2-f32-le.bin", WEIGHTS_DIR)),
    );
    let input_data = f32::read_lines_from_file(&format!("{}/fm1.f", CLASSIC_BASELINE));

    let output = run_single_layer("conv_relu_2", &layer, &input_data);
    let correct = f32::read_lines_from_file(&format!("{}/fm2.f", CLASSIC_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn test_l3() {
    // Create the representation of the fully-connected layer
    let layer = CLASSIC_PARAMS.create_dense(
        3,
        f32::read_bin_from_file(&format!("{}/fc3-f32-le.bin", WEIGHTS_DIR)),
    );
    let input_data = f32::read_lines_from_file(&format!("{}/fm2.f", CLASSIC_BASELINE));

    let output = relu(&run_single_layer("mtx_mul", &layer, &input_data));
    let correct = f32::read_lines_from_file(&format!("{}/fc3.f", CLASSIC_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn test_l4() {
    // Create the representation of the fully-connected layer
    let layer = CLASSIC_PARAMS.create_dense(
        4,
        f32::read_bin_from_file(&format!("{}/fc4-f32-le.bin", WEIGHTS_DIR)),
    );
    let input_data = f32::read_lines_from_file(&format!("{}/fc3.f", CLASSIC_BASELINE));

    let output = relu(&layer.mtx_mul(&input_data));
    let correct = f32::read_lines_from_file(&format!("{}/fc4.f", CLASSIC_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn test_l5() {
    // Create the representation of the fully-connected layer
    let layer = CLASSIC_PARAMS.create_dense(
        5,
        f32::read_bin_from_file(&format!("{}/fc5-f32-le.bin", WEIGHTS_DIR)),
    );
    let input_data = f32::read_lines_from_file(&format!("{}/fc4.f", CLASSIC_BASELINE));

    let output = softmax(&layer.mtx_mul(&input_data));
    let correct = f32::read_lines_from_file(&format!("{}/out5.f", CLASSIC_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

fn run_single_layer<L, T>(kernel_func: &str, layer: &L, input: &[T]) -> Vec<T>
where
    L: ClWeightedLayer<T>,
    T: Coeff,
{
    let (kernel, out_buf, queue) = create_standalone_kernel(layer, kernel_func, input).unwrap();
    // Enqueue kernel and wait for it to end
    run_kernel_wait(&kernel, &queue).unwrap();
    unsafe { cl::read_buf(&out_buf).unwrap() }
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
    let (queue, program, _context) = cl::init::<f32>(&["test/mxp.cl"], &[]).unwrap();

    let (in_buf, out_buf) = mxp.create_io_bufs(
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    );
    let max_wgs = cl_util::max_wgs(None);
    let mxp_krn = Kernel::builder()
        .program(&program)
        .name("max_pool")
        .queue(queue.clone())
        .global_work_size(mxp.gws_hint())
        .local_work_size(mxp.lws_hint(max_wgs))
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

#[test]
fn test_dense3_cl_cpu_vec4() {
    // Create the representation of the fully-connected layer
    let dense3 = CLASSIC_PARAMS.create_dense(
        3,
        f32::read_bin_from_file(&format!("{}/fc3-f32-le.bin", WEIGHTS_DIR)),
    );

    // Custom-initialize OpenCL
    let platform = Platform::default();
    let device = Device::list(platform, Some(flags::DeviceType::CPU))
        .unwrap()
        .first()
        .unwrap()
        .clone();

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()
        .unwrap();

    let mut program_b = Program::builder();
    cl::configure_program::<f32>(&mut program_b, &device);

    // Input the kernel source files
    program_b.src_file("src/cl/mtx_mul.cl");

    let program = program_b.build(&context).unwrap();

    let queue = Queue::new(&context, device, None).unwrap();

    let wgts_buf =
        cl::create_buffer::<f32>(dense3.num_weights(), flags::MEM_READ_ONLY, &queue).unwrap();
    wgts_buf.write(dense3.weights()).enq().unwrap();

    let in_buf = cl::create_buffer::<f32>(
        dense3.num_in(),
        flags::MEM_READ_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    ).unwrap();
    let out_buf = cl::create_buffer::<f32>(
        dense3.num_out(),
        flags::MEM_WRITE_ONLY | flags::MEM_ALLOC_HOST_PTR,
        &queue,
    ).unwrap();

    let kernel = Kernel::builder().program(&program).name("mtx_mul_vec4")
        .queue(queue.clone())
        .global_work_size(dense3.gws_hint())
        // Input
        .arg(&in_buf)
        // Output
        .arg(&out_buf)
        .arg(&wgts_buf).build().unwrap();

    let input_data = f32::read_lines_from_file(&format!("{}/fm2.f", CLASSIC_BASELINE));
    unsafe {
        cl::map_to_buf(&in_buf, &input_data).unwrap();
    }
    queue.finish().unwrap();

    unsafe {
        kernel.cmd().queue(&queue).enq().unwrap();
    }
    queue.finish().unwrap();

    let output = relu(unsafe { &cl::read_buf(&out_buf).unwrap() });
    let correct = f32::read_lines_from_file(&format!("{}/fc3.f", CLASSIC_BASELINE));

    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, COARSE_RESULT_MARGIN);
}
