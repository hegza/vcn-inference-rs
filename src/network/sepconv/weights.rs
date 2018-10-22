use layers::Coeff;
use util::{reorder, ReadBinFromFile, ReadCsv};

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
        Weights(
            // Load the weights for the convolutional layers
            f32::read_csv(&format!("{}/vcr1-f32.csv", WEIGHTS_F32_DIR)),
            f32::read_csv(&format!("{}/hcr1-f32.csv", WEIGHTS_F32_DIR)),
            f32::read_csv(&format!("{}/vcr2-f32.csv", WEIGHTS_F32_DIR)),
            f32::read_csv(&format!("{}/hcr2-f32.csv", WEIGHTS_F32_DIR)),
            // Load the weights for the dense layers
            f32::read_csv(&format!("{}/fc3-f32-nchw.csv", WEIGHTS_F32_DIR)),
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
