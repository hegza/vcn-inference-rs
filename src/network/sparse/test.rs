use super::*;
use crate::geometry::{ImageGeometry, PaddedSquare};
use crate::tests::{COARSE_RESULT_MARGIN, F32_GEMM_MAX_EPSILON, RESULT_MARGIN};
use crate::util::verify;
use ndarray::Array;

lazy_static! {
    static ref LAYERS: Layers<f32> = { Layers::<f32>::new(Weights::default()) };
}

pub const SPARSE_BASELINE: &'static str = "input/baseline/sparse-f32";

#[test]
fn l1_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.conv1;

    // Load image with padding and in (channels, height, width)-order
    let padded_input: Vec<f32> = {
        let raw_input: Vec<f32> = load_jpeg_chw("input/baseline/sparse-f32/in.jpg");
        let mut padded = Array::zeros((3, 100, 100));
        padded
            .slice_mut(s![.., 2..-2, 2..-2])
            .assign(&Array::from_shape_vec((3, 96, 96), raw_input).unwrap());

        padded.into_iter().cloned().collect::<Vec<f32>>()
    };

    // Output is in (channels, height, width)-order
    let output = run_single_layer("conv_relu_1", layer, &padded_input);

    // Load model outputs in (channels, height, width)-order
    let correct = Array::from_shape_vec(
        (32, 52, 52),
        f32::read_csv(&format!("{}/fm1_mxp-cwh.csv", SPARSE_BASELINE)),
    )
    .unwrap()
    .permuted_axes((0, 2, 1))
    .into_iter()
    .cloned()
    .collect::<Vec<f32>>();

    verify(&output, &correct, F32_GEMM_MAX_EPSILON);
}

use itertools::Itertools;
#[test]
fn l2_returns_baseline() {
    // Create the representation of the 2nd convolutional layer with weights from a file
    let layer = &LAYERS.conv2;

    // Load input in (channels, height, width)-order
    let input_data = Array::from_shape_vec(
        (32, 52, 52),
        f32::read_csv(&format!("{}/fm1_mxp-cwh.csv", SPARSE_BASELINE)),
    )
    .unwrap()
    .permuted_axes((0, 2, 1))
    .into_iter()
    .cloned()
    .collect::<Vec<f32>>();

    // Output is in (channels, height, width)-order
    let output = run_single_layer("conv_relu_2", layer, &input_data);

    // Load model output in (channels, height, width)-order
    let correct = Array::from_shape_vec(
        (24, 24, 32),
        f32::read_csv(&format!("{}/fm2_mxp-hwc.csv", SPARSE_BASELINE)),
    )
    .unwrap()
    .permuted_axes((2, 0, 1))
    .into_iter()
    .cloned()
    .collect::<Vec<f32>>();

    verify(&output, &correct, COARSE_RESULT_MARGIN);
}

#[test]
fn l3_returns_baseline() {
    let layer = &LAYERS.sparse3;

    // Load input in (channels, height, width)-order
    let input_data = {
        let raw = f32::read_csv(&format!("{}/fm2_mxp-hwc.csv", SPARSE_BASELINE));
        let chw = Array::from_shape_vec((24, 24, 32), raw)
            .unwrap()
            .permuted_axes((2, 0, 1))
            .into_iter()
            .cloned()
            .collect::<Vec<f32>>();
        chw
    };

    // Output is in (fc-const)-order
    let output = layer.compute(&input_data);

    // Load model output in (fc-const)-order
    let correct = f32::read_csv(&format!("{}/hidden1.csv", SPARSE_BASELINE));

    verify(&output, &correct, COARSE_RESULT_MARGIN);
}

#[test]
fn l4_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = &LAYERS.dense4;
    let input_data = f32::read_csv(&format!("{}/hidden1.csv", SPARSE_BASELINE));

    let output = relu(layer.compute(&input_data));
    let correct = f32::read_csv(&format!("{}/hidden2.csv", SPARSE_BASELINE));
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn l5_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = &LAYERS.dense5;
    let input_data = f32::read_csv(&format!("{}/hidden2.csv", SPARSE_BASELINE));

    let output = softmax(layer.compute(&input_data));
    let correct = f32::read_csv(&format!("{}/out.csv", SPARSE_BASELINE));
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

#[test]
fn sparse_predicts() {
    let network = ClNetwork::<f32>::new(Weights::default());

    // Load image with padding and in (channels, height, width)-order
    let padded_input: Vec<f32> = {
        let raw_input: Vec<f32> = load_jpeg_chw("input/baseline/sparse-f32/in.jpg");
        let mut padded = Array::zeros((3, 100, 100));
        padded
            .slice_mut(s![.., 2..-2, 2..-2])
            .assign(&Array::from_shape_vec((3, 96, 96), raw_input).unwrap());

        padded.into_iter().cloned().collect::<Vec<f32>>()
    };

    let result = network.predict(&padded_input);

    let correct = f32::read_csv(&format!("{}/out.csv", SPARSE_BASELINE));

    verify(&result, &correct, RESULT_MARGIN);
}
