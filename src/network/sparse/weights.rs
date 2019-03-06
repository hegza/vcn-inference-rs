use crate::layers::Coeff;
use crate::util::{reorder, ReadBinFromFile, ReadCsv};

pub const WEIGHTS_DIR: &str = "input/weights/sparse";

#[derive(Clone)]
pub struct Weights<T>(pub Vec<T>, pub Vec<T>, pub Vec<T>, pub Vec<T>, pub Vec<T>);

impl<T> Default for Weights<T>
where
    T: Coeff + ReadCsv,
{
    fn default() -> Weights<T> {
        use ndarray::Array;
        let load_fun = |f: &'static str| T::read_csv(&format!("{}/{}", WEIGHTS_DIR, f));
        Weights(
            // TODO: reorder these in file already
            // Reverse filter X and Y (major axis), I can either to this or reverse the input and output images
            reorder(load_fun("conv1-f32-dcwh.csv"), (32, 3, 5, 5), (0, 1, 3, 2)),
            reorder(load_fun("conv2-f32-dcwh.csv"), (32, 32, 5, 5), (0, 1, 3, 2)),
            // Load weights in CHWN
            reorder(
                load_fun("archive/fc3-f32-nhwc.csv"),
                (100, 24, 24, 32),
                (3, 1, 2, 0),
            ),
            T::read_csv(&format!("{}/fc4-f32.csv", WEIGHTS_DIR)),
            T::read_csv(&format!("{}/fc5-f32.csv", WEIGHTS_DIR)),
        )
    }
}
