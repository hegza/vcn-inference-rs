use crate::layers::Coeff;
use crate::util::{reorder, ReadBinFromFile, ReadCsv};

pub const WEIGHTS_F32_DIR: &str = "input/weights/sepconv";
pub const WEIGHTS_I8_DIR: &str = "input/weights/sepconv-96-97/i8-converted";

#[derive(Clone)]
pub struct Weights<T>(
    pub Vec<T>,
    pub Vec<T>,
    pub Vec<T>,
    pub Vec<T>,
    pub Vec<T>,
    pub Vec<T>,
    pub Vec<T>,
);

impl Default for Weights<f32> {
    fn default() -> Weights<f32> {
        use ndarray::Array;
        let load_fun = |f: &'static str| f32::read_csv(&format!("{}/{}", WEIGHTS_F32_DIR, f));
        Weights(
            // Load the weights for the convolutional layers
            reorder(load_fun("vcr1-f32.csv"), (7, 3, 1, 5), (0, 1, 2, 3)),
            reorder(load_fun("hcr1-f32.csv"), (32, 7, 5, 1), (0, 1, 2, 3)),
            reorder(load_fun("vcr2-f32.csv"), (7, 32, 1, 5), (0, 1, 2, 3)),
            reorder(load_fun("hcr2-f32.csv"), (32, 7, 5, 1), (0, 1, 2, 3)),
            // Load in CHWN
            reorder(
                f32::read_csv(&format!("{}/archive/fc3-f32-nchw.csv", WEIGHTS_F32_DIR)),
                // Read as: n, hwc
                (100, 24, 24, 32),
                // n, chw
                (0, 3, 1, 2),
            ),
            f32::read_csv(&format!("{}/fc4-f32.csv", WEIGHTS_F32_DIR)),
            f32::read_csv(&format!("{}/fc5-f32.csv", WEIGHTS_F32_DIR)),
        )
    }
}

impl Default for Weights<i8> {
    fn default() -> Weights<i8> {
        Weights(
            // Load the weights for the convolutional layers
            i8::read_csv(&format!("{}/vcr1-i8.csv", WEIGHTS_I8_DIR)),
            i8::read_csv(&format!("{}/hcr1-i8.csv", WEIGHTS_I8_DIR)),
            i8::read_csv(&format!("{}/vcr2-i8.csv", WEIGHTS_I8_DIR)),
            i8::read_csv(&format!("{}/hcr2-i8.csv", WEIGHTS_I8_DIR)),
            // Load the weights for the dense layers
            i8::read_csv(&format!("{}/fc3-i8-nchw.csv", WEIGHTS_I8_DIR)),
            i8::read_csv(&format!("{}/fc4-i8.csv", WEIGHTS_I8_DIR)),
            i8::read_csv(&format!("{}/fc5-i8.csv", WEIGHTS_I8_DIR)),
        )
    }
}
