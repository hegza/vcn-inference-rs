use layers::Coeff;
use util::{reorder, ReadBinFromFile, ReadCsv};

pub const WEIGHTS_DIR: &str = "input/weights";

#[derive(Clone)]
pub struct Weights<T>(pub Vec<T>, pub Vec<T>, pub Vec<T>, pub Vec<T>, pub Vec<T>);

impl Default for Weights<f32> {
    fn default() -> Weights<f32> {
        let load_fun = |f: &'static str| f32::read_bin_from_file(&format!("{}/{}", WEIGHTS_DIR, f));
        Weights(
            reorder(load_fun("conv1-f32-le.bin"), (32, 3, 5, 5), (0, 1, 2, 3)),
            load_fun("conv2-f32-le.bin"),
            load_fun("fc3-f32-le.bin"),
            load_fun("fc4-f32-le.bin"),
            load_fun("fc5-f32-le.bin"),
        )
    }
}
