use super::*;
use geometry::*;
use ocl::flags::DeviceType;
use ocl::{flags, Context, Device, Kernel, Platform, Program, SpatialDims};

#[test]
fn l1_returns_baseline() {
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
fn l2_returns_baseline() {
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
fn l3_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = CLASSIC_PARAMS.create_dense(
        3,
        f32::read_bin_from_file(&format!("{}/fc3-f32-le.bin", WEIGHTS_DIR)),
    );
    let input_data = f32::read_lines_from_file(&format!("{}/fm2.f", CLASSIC_BASELINE));

    let output = &run_single_layer("mtx_mul", &layer, &input_data);
    let correct = f32::read_lines_from_file(&format!("{}/fc3.f", CLASSIC_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn l4_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = CLASSIC_PARAMS.create_dense(
        4,
        f32::read_bin_from_file(&format!("{}/fc4-f32-le.bin", WEIGHTS_DIR)),
    );
    let input_data = f32::read_lines_from_file(&format!("{}/fc3.f", CLASSIC_BASELINE));

    let output = relu(layer.compute(&input_data));
    let correct = f32::read_lines_from_file(&format!("{}/fc4.f", CLASSIC_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn l5_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = CLASSIC_PARAMS.create_dense(
        5,
        f32::read_bin_from_file(&format!("{}/fc5-f32-le.bin", WEIGHTS_DIR)),
    );
    let input_data = f32::read_lines_from_file(&format!("{}/fc4.f", CLASSIC_BASELINE));

    let output = softmax(&layer.compute(&input_data));
    let correct = f32::read_lines_from_file(&format!("{}/out5.f", CLASSIC_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

fn run_single_layer<L, T>(kernel_func: &str, layer: &L, input: &[T]) -> Vec<T>
where
    L: ClWeightedLayer<T>,
    T: Coeff,
{
    let cl_layer = layer.impl_standalone(
        &["src/cl/conv_mxp_relu.cl", "src/cl/mtx_mul.cl"],
        kernel_func,
        &[],
        None,
        LocalWorkSizePolicy::UseDefault,
    );

    // Enqueue kernel and wait for it to end, return the result
    cl_layer.run_with_input(&input)
}

// Test that Maxpool + ReLU produces the correct output
#[test]
fn mxp_returns_baseline() {
    let in_img = f32::read_csv("src/tests/in/img-4x4_mono-norm.csv");
    const SIDE: usize = 4;
    let in_shape = ImageGeometry::new(SIDE, 1);

    // Make sure the input image matches with the assumed input shape
    assert_eq!(in_shape.num_elems(), in_img.len());

    let mxp = MaxpoolLayer::new(in_shape, 2);

    // Implement mxp on GPU if possible
    let dev_max_wgs = cl_util::max_wgs(None);
    let cl_impl = mxp.impl_standalone(
        &["src/cl/test/mxp.cl"],
        "max_pool",
        &[],
        None,
        LocalWorkSizePolicy::Infer { dev_max_wgs },
    );

    let gpu_out = cl_impl.run_with_input(&in_img);

    // Run maxpool on CPU
    let cpu_out = mxp.compute(&in_img);

    // Verify match between confirmed correct output and GPU and CPU outputs
    let correct = f32::read_csv("src/tests/out/img-4x4_mono-norm-mxp2.csv");
    assert!(is_within_margin(&gpu_out, &correct, RESULT_MARGIN));
    assert!(is_within_margin(&cpu_out, &correct, RESULT_MARGIN));
}

#[test]
fn dense3_cl_cpu_vec4_returns_baseline() {
    // Create the representation of the fully-connected layer
    let dense3 = CLASSIC_PARAMS.create_dense(
        3,
        f32::read_bin_from_file(&format!("{}/fc3-f32-le.bin", WEIGHTS_DIR)),
    );

    let cl_impl = dense3.impl_standalone(
        &["src/cl/mtx_mul.cl"],
        "mtx_mul",
        &["-D VECN=4"],
        Some(DeviceType::CPU),
        LocalWorkSizePolicy::UseDefault,
    );

    let input_data = f32::read_lines_from_file(&format!("{}/fm2.f", CLASSIC_BASELINE));
    let output = &cl_impl.run_with_input(&input_data);

    let correct = f32::read_lines_from_file(&format!("{}/fc3.f", CLASSIC_BASELINE));

    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, COARSE_RESULT_MARGIN);
}
