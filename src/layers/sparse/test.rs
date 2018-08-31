use super::*;
use classic;
use geometry::{ImageGeometry, PaddedSquare, Square};
use math::relu;
use tests::CLASSIC_BASELINE;
use util::*;

#[test]
fn sparse_layer_works_for_dense() {
    // Hyper-parameters
    const INPUT_CHANNELS: usize = 3;
    const INPUT_SIDE: usize = 96;
    const CONV1_STRIDE: usize = 2;
    const NUM_FEATURE_MAPS: usize = 32;
    const CONV2_STRIDE: usize = 2;
    const LAYER3_SIZE: usize = 100;

    let input_shape = ImageGeometry::new(INPUT_SIDE, INPUT_CHANNELS);
    // Feature map 1 is a fraction of the side of initial image geometry due to stride
    let fm1_shape = ImageGeometry::new(input_shape.side() / CONV1_STRIDE, NUM_FEATURE_MAPS);
    // Feature map 2 is a fraction of the side of the tier 1 feature map due to stride
    let fm2_shape = ImageGeometry::new(fm1_shape.side() / CONV2_STRIDE, NUM_FEATURE_MAPS);

    let weights = f32::read_csv("input/weights/fc3-f32-chwn.csv");
    let layer = SparseLayer::from_dense(fm2_shape.num_elems(), LAYER3_SIZE, weights);

    let in_buf = f32::read_lines_from_file(&format!("{}/fm2.f", CLASSIC_BASELINE));
    let out = relu(layer.compute(&in_buf));
    let correct = f32::read_lines_from_file(&format!("{}/fc3.f", CLASSIC_BASELINE));
    verify(&out, &correct, 0.0001f32);
}

#[test]
fn sparse_and_dense_are_same_for_small() {
    let input = vec![1f32, 2f32];
    let w = vec![1f32, 2f32, 3f32, 4f32];
    let w_t = vec![1f32, 3f32, 2f32, 4f32];

    let len_c = w.len() / input.len();

    let layer = SparseLayer::from_dense(input.len(), len_c, w);
    let correct = DenseLayer::new(input.len(), len_c, w_t).compute(&input);
    let out = layer.compute(&input);

    verify(&out, &correct, 0.0001f32);
}
