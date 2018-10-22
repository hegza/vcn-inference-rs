use super::*;
use geometry::{ImageGeometry, PaddedSquare};
use ndarray::Array;
use tests::{COARSE_RESULT_MARGIN, F32_GEMM_MAX_EPSILON, RESULT_MARGIN};
use util::verify;

lazy_static! {
    static ref LAYERS: Layers<f32> = { Layers::<f32>::new(Weights::default()) };
}

pub const SPARSE_BASELINE: &'static str = "input/baseline/sparse-f32";

#[test]
fn l1_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.conv1;
    let input_data =
        load_jpeg_with_filter_padding("input/baseline/sparse-f32/in.jpg", (96, 96, 3), 1);

    // FIXME: conv_relu_1 algo assumes padding (48x48->52x52) while tf produced version assumes 48x48;
    let output = run_single_layer("conv_relu_1", layer, &input_data);
    let correct = f32::read_csv(&format!("{}/fm1_mxp.csv", SPARSE_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn l2_returns_baseline() {
    // Create the representation of the 1st convolutional layer with weights from a file
    let layer = &LAYERS.conv2;
    let input_data = f32::read_csv(&format!("{}/fm1_mxp.csv", SPARSE_BASELINE));

    let output = run_single_layer("conv_relu_2", layer, &input_data);
    let correct = f32::read_csv(&format!("{}/fm2_mxp.csv", SPARSE_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
}

#[test]
fn l3_returns_baseline() {
    // Create the representation of the fully-connected layer
    let layer = &LAYERS.sparse3;
    let input_data = f32::read_csv(&format!("{}/fm2_mxp.csv", SPARSE_BASELINE));

    let output = layer.compute(&input_data);
    let correct = f32::read_csv(&format!("{}/hidden1.csv", SPARSE_BASELINE));
    assert_eq!(output.len(), correct.len());
    verify(&output, &correct, RESULT_MARGIN);
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
