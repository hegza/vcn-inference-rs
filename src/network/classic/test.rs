use super::*;
use geometry::{ImageGeometry, PaddedSquare};
use ndarray::Array;
use tests::{CLASSIC_BASELINE, COARSE_RESULT_MARGIN, F32_GEMM_MAX_EPSILON, RESULT_MARGIN};
use util::verify;

lazy_static! {
    static ref LAYERS: Layers<f32> = { Layers::<f32>::new(Weights::default()) };
}

#[test]
fn classic_predicts() {
    let network = ClNetwork::<f32>::new(Weights::default());

    let input = {
        let padded_raw = Array::from_shape_vec(
            (3, 100, 100),
            read_image_with_padding_from_bin_in_channels::<f32>(
                &format!("{}/in.bin", CLASSIC_BASELINE),
                network.input_shape(),
            ).into_iter()
            .map(|x| f32::from(x))
            .collect::<Vec<f32>>(),
        ).unwrap()
        .permuted_axes((0, 1, 2))
        .iter()
        .cloned()
        .collect::<Vec<f32>>();
        padded_raw
    };

    let result = network.predict(&input);

    let correct = f32::read_lines_from_file(&format!("{}/out5.f", CLASSIC_BASELINE)).unwrap();

    verify(&result, &correct, COARSE_RESULT_MARGIN);
}

#[cfg_attr(not(feature = "test_classic"), ignore)]
#[test]
fn l1_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.conv1;

    // Load image with padding and in (channels, height, width)-order
    let padded_input = {
        let input_shape = ImageGeometry::new(96, 3);
        let filter_shape = PaddedSquare::from_side(5);
        let padded_input_shape = input_shape.with_filter_padding(&filter_shape);
        let padded_raw = Array::from_shape_vec(
            (3, 100, 100),
            read_image_with_padding_from_bin_in_channels::<f32>(
                &format!("{}/in.bin", CLASSIC_BASELINE),
                &padded_input_shape,
            ),
        ).unwrap()
        .permuted_axes((0, 2, 1))
        .iter()
        .cloned()
        .collect::<Vec<f32>>();
        padded_raw
    };

    // Out is in same order as in
    let output = run_single_layer("conv_relu_1", layer, &padded_input);

    // Load model outputs in (channels, height, width)-order
    let correct = Array::from_shape_vec(
        (32, 52, 52),
        f32::read_lines_from_file(&format!("input/baseline/orig-f32-all-layers/fm1.f")).unwrap(),
    ).unwrap()
    .permuted_axes((0, 2, 1))
    .into_iter()
    .cloned()
    .collect::<Vec<f32>>();

    verify(&output, &correct, RESULT_MARGIN);
}

#[cfg_attr(not(feature = "test_classic"), ignore)]
#[test]
fn l2_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.conv2;
    let input_data = f32::read_lines_from_file(&format!("{}/fm1.f", CLASSIC_BASELINE)).unwrap();

    let output = run_single_layer("conv_relu_2", layer, &input_data);
    let correct = f32::read_lines_from_file(&format!("{}/fm2.f", CLASSIC_BASELINE)).unwrap();
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn l3_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = &LAYERS.dense3;
    let input_data = f32::read_lines_from_file(&format!("{}/fm2.f", CLASSIC_BASELINE)).unwrap();

    let output = run_single_layer("mtx_mul", layer, &input_data);
    let correct = f32::read_lines_from_file(&format!("{}/fc3.f", CLASSIC_BASELINE)).unwrap();
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn l4_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = &LAYERS.dense4;
    let input_data = f32::read_lines_from_file(&format!("{}/fc3.f", CLASSIC_BASELINE)).unwrap();

    let output = relu(layer.compute(&input_data));
    let correct = f32::read_lines_from_file(&format!("{}/fc4.f", CLASSIC_BASELINE)).unwrap();
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn l5_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = &LAYERS.dense5;
    let input_data = f32::read_lines_from_file(&format!("{}/fc4.f", CLASSIC_BASELINE)).unwrap();

    let output = softmax(layer.compute(&input_data));
    let correct = f32::read_lines_from_file(&format!("{}/out5.f", CLASSIC_BASELINE)).unwrap();
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn dense3_cl_cpu_vec4_returns_baseline() {
    // Create the representation of the fully-connected layer
    let dense3 = &LAYERS.dense3;

    let cl_impl = dense3.impl_standalone(
        &["src/cl/mtx_mul.cl"],
        "mtx_mul",
        &["-D VECN=4"],
        Some(DeviceType::CPU),
        LocalWorkSizePolicy::UseDefault,
    );

    let input_data = f32::read_lines_from_file(&format!("{}/fm2.f", CLASSIC_BASELINE)).unwrap();
    let output = &cl_impl.run_with_input(&input_data);

    let correct = f32::read_lines_from_file(&format!("{}/fc3.f", CLASSIC_BASELINE)).unwrap();

    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, COARSE_RESULT_MARGIN);
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
