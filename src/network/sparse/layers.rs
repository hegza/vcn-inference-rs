use super::*;
use layers::*;

pub struct Layers<T>
where
    T: Coeff,
{
    pub conv1: ConvLayer<T>,
    pub conv2: ConvLayer<T>,
    pub sparse3: SparseLayer<T>,
    pub dense4: DenseLayer<T>,
    pub dense5: DenseLayer<T>,
}

impl Layers<f32> {
    pub fn new(weights: Weights<f32>) -> Layers<f32> {
        // Hyper-parameters
        const INPUT_CHANNELS: usize = 3;
        const INPUT_SIDE: usize = 96;
        const CONV1_FILTER_SIDE: usize = 5;
        const CONV1_STRIDE: usize = 2;
        const NUM_FEATURE_MAPS: usize = 32;
        const CONV2_FILTER_SIDE: usize = 5;
        const CONV2_STRIDE: usize = 2;
        const LAYER3_SIZE: usize = 100;
        const NUM_CLASSES: usize = 4;

        // TODO: refactor by making the layers use a tensor-based API (ndarray?)
        let conv1_filter_shape = PaddedSquare::from_side(CONV1_FILTER_SIDE);
        let conv2_filter_shape = PaddedSquare::from_side(CONV2_FILTER_SIDE);
        let input_shape = ImageGeometry::new(INPUT_SIDE, INPUT_CHANNELS);
        let padded_input_shape = input_shape.with_filter_padding(&conv1_filter_shape);
        // Feature map 1 is a fraction of the side of initial image geometry due to stride
        let fm1_shape = ImageGeometry::new(input_shape.side() / CONV1_STRIDE, NUM_FEATURE_MAPS);
        let padded_fm1_shape = fm1_shape.with_filter_padding(&conv2_filter_shape);
        // Feature map 2 is a fraction of the side of the tier 1 feature map due to stride
        let fm2_shape = ImageGeometry::new(fm1_shape.side() / CONV2_STRIDE, NUM_FEATURE_MAPS);

        // Render network configuration
        let conv1 = ConvLayer::from_shapes(
            conv1_filter_shape.num_elems(),
            &padded_input_shape,
            &padded_fm1_shape,
            weights.0,
        );
        let conv2 = ConvLayer::from_shapes(
            conv2_filter_shape.num_elems(),
            &padded_fm1_shape,
            &fm2_shape,
            weights.1,
        );

        // Create the representations of the fully-connected layers
        let sparse3 = SparseLayer::from_dense(fm2_shape.num_elems(), LAYER3_SIZE, weights.2);
        let dense4 = DenseLayer::new(LAYER3_SIZE, LAYER3_SIZE, weights.3);
        let dense5 = DenseLayer::new(LAYER3_SIZE, NUM_CLASSES, weights.4);
        Layers {
            conv1,
            conv2,
            sparse3,
            dense4,
            dense5,
        }
    }
}
