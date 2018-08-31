use layers::Coeff;
use util::{ReadBinFromFile, ReadCsv};

#[derive(Clone)]
pub struct Weights<T>(pub Vec<T>, pub Vec<T>, pub Vec<T>, pub Vec<T>, pub Vec<T>);

impl<T> Default for Weights<T>
where
    T: Coeff + ReadBinFromFile,
{
    fn default() -> Weights<T> {
        const WEIGHTS_DIR: &str = "input/weights";
        Weights(
            T::read_bin_from_file(&format!("{}/conv1-f32-le.bin", WEIGHTS_DIR)),
            T::read_bin_from_file(&format!("{}/conv2-f32-le.bin", WEIGHTS_DIR)),
            T::read_bin_from_file(&format!("{}/fc3-f32-le.bin", WEIGHTS_DIR)),
            T::read_bin_from_file(&format!("{}/fc4-f32-le.bin", WEIGHTS_DIR)),
            T::read_bin_from_file(&format!("{}/fc5-f32-le.bin", WEIGHTS_DIR)),
        )
    }
}
