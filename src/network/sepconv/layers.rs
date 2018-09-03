use super::*;
use layers::*;

pub struct Layers<T>
where
    T: Coeff,
{
    pub vconv1: VConvLayer<T>,
    pub hconv1: HConvLayer<T>,
    pub mxp1: MaxpoolLayer,
    pub vconv2: VConvLayer<T>,
    pub hconv2: HConvLayer<T>,
    pub mxp2: MaxpoolLayer,
    pub dense3: DenseLayer<T>,
    pub dense4: DenseLayer<T>,
    pub dense5: DenseLayer<T>,
}

impl<T> Layers<T>
where
    T: Coeff,
{
    pub fn new(weights: Weights<T>) -> Layers<T> {
        // Hyper-parameters
        const INPUT_CHANNELS: usize = 3;
        const INPUT_SIDE: usize = 96;
        const CONV_KERNEL_SPLIT: usize = 7;
        const KERNEL_LEN: usize = 5;
        const NUM_FEATURE_MAPS: usize = 32;
        const LAYER3_SIZE: usize = 100;
        const NUM_CLASSES: usize = 4;

        // TODO: refactor by making the layers use a tensor-based API

        let in_shape = ImageGeometry::new(INPUT_SIDE, INPUT_CHANNELS);
        let vconv1 = VConvLayer::new(KERNEL_LEN, &in_shape, CONV_KERNEL_SPLIT, weights.0);
        let hconv1 = HConvLayer::new(
            KERNEL_LEN,
            vconv1.output_shape(),
            NUM_FEATURE_MAPS,
            weights.1,
        );
        let mxp1 = MaxpoolLayer::new(hconv1.output_shape(), 2);
        let vconv2 = VConvLayer::new(
            KERNEL_LEN,
            mxp1.output_shape(),
            CONV_KERNEL_SPLIT,
            weights.2,
        );
        let hconv2 = HConvLayer::new(
            KERNEL_LEN,
            vconv2.output_shape(),
            NUM_FEATURE_MAPS,
            weights.3,
        );
        let mxp2 = MaxpoolLayer::new(hconv2.output_shape(), 2);

        // Create the representations of the fully-connected layers
        let dense3 = DenseLayer::new(mxp2.num_out(), LAYER3_SIZE, weights.4);
        let dense4 = DenseLayer::new(dense3.num_out(), LAYER3_SIZE, weights.5);
        let dense5 = DenseLayer::new(dense4.num_out(), NUM_CLASSES, weights.6);

        Layers {
            vconv1,
            hconv1,
            mxp1,
            vconv2,
            hconv2,
            mxp2,
            dense3,
            dense4,
            dense5,
        }
    }
}
