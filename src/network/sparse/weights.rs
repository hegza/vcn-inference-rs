use layers::Coeff;
use util::{ReadBinFromFile, ReadCsv};

#[derive(Clone)]
pub struct Weights<T>(pub Vec<T>, pub Vec<T>, pub Vec<T>, pub Vec<T>, pub Vec<T>);

impl<T> Default for Weights<T>
where
    T: Coeff + ReadCsv,
{
    fn default() -> Weights<T> {
        const SPARSE_DIR: &str = "input/weights/sparse-classic-95-96";
        Weights(
            T::read_csv(&format!("{}/conv1-f32.csv", SPARSE_DIR)),
            T::read_csv(&format!("{}/conv2-f32.csv", SPARSE_DIR)),
            T::read_csv(&format!("{}/fc3-f32-chwn.csv", SPARSE_DIR)),
            T::read_csv(&format!("{}/fc4-f32.csv", SPARSE_DIR)),
            T::read_csv(&format!("{}/fc5-f32.csv", SPARSE_DIR)),
        )
    }
}
