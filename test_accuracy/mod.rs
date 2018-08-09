extern crate env_logger;
extern crate image;
#[macro_use]
extern crate log;
extern crate noisy_float;
extern crate num_traits;
extern crate ocl;
extern crate rusty_cnn;

mod class;
mod util;

use class::Class;
use geometry::*;
use rusty_cnn::*;
use util::*;

const INPUT_IMG_DIR: &str = "input/images";

const TEST_CLASSIC: bool = true;
const TEST_SEPCONV_F32: bool = true;
//const TEST_SEPCONV_I8: bool = true;
const CLASSIC_SINGLE_SHOT: bool = false;
const SEPCONV_F32_SINGLE_SHOT: bool = false;
//const SEPCONV_I8_SINGLE_SHOT: bool = false;

pub fn main() {
    env_logger::init();

    // Figure out the existing classes for the network based on directory names
    let class_dir_names = list_dirs(INPUT_IMG_DIR).unwrap();

    if TEST_CLASSIC {
        debug!("Loading input images for original network...");

        let load_fun = |file: &String| -> Vec<f32> { load_jpeg_as_f32_with_padding(file) };
        let mut test_data = load_test_data(INPUT_IMG_DIR, &class_dir_names, load_fun);
        if CLASSIC_SINGLE_SHOT {
            test_data = test_data
                .into_iter()
                .take(1)
                .collect::<Vec<(Vec<f32>, Class)>>();
        }

        // Initialize OpenCL and the network
        let net = ClassicNetwork::<f32>::new();

        // Make classifications and measure accuracy using the original network
        let (correct, total) = measure_accuracy(&net, &test_data);
        let accuracy = correct as f32 / total as f32;
        println!("original network accuracy:");
        println!("{} ({}/{})", accuracy, correct, total);
    }

    debug!("Loading input images for sepconv networks...");

    let load_fun = |file: &String| -> Vec<f32> { load_jpeg_as_f32(file) };
    let test_data = load_test_data(INPUT_IMG_DIR, &class_dir_names, load_fun);

    if TEST_SEPCONV_F32 {
        let test_data = match SEPCONV_F32_SINGLE_SHOT {
            true => test_data
                .iter()
                .cloned()
                .take(1)
                .collect::<Vec<(Vec<f32>, Class)>>(),
            false => test_data.iter().cloned().collect(),
        };

        // Initialize OpenCL and the sep-conv network
        let net = SepconvNetwork::<f32>::new(sepconv::Weights::default());

        // Make classifications and measure accuracy using the sep-conv network
        let (correct, total) = measure_accuracy(&net, &test_data);
        let accuracy = correct as f32 / total as f32;
        println!("sepconv-f32 network accuracy:");
        println!("{} ({}/{})", accuracy, correct, total);
    }

    /*
    let load_fun = |file: &String| -> Vec<u8> { load_jpeg_as_u8_lossless(file) };
    let test_data = load_test_data(INPUT_IMG_DIR, &class_dir_names, load_fun);

    if TEST_SEPCONV_I8 {
        let test_data = match SEPCONV_I8_SINGLE_SHOT {
            true => test_data
                .iter()
                .cloned()
                .take(1)
                .collect::<Vec<(Vec<u8>, Class)>>(),
            false => test_data.iter().cloned().collect(),
        };
        let test_data = test_data
            .into_iter()
            .map(|(vec, c)| (math::quantize_vec_i8(&vec), c))
            .collect::<Vec<(Vec<u8>, Class)>>();

        use sepconv::Weights;
        let weights = Weights::<i8>::default();

        // Initialize OpenCL and the sep-conv network
        let net = SepconvNetwork::<i8>::new(weights);

        // Make classifications and measure accuracy using the sep-conv network
        let (correct, total) = measure_accuracy(&net, &test_data);
        let accuracy = correct as f32 / total as f32;
        println!("sepconv-i8 network accuracy:");
        println!("{} ({}/{})", accuracy, correct, total);
    }
    */
}

/// Returns (num_correct, num_total)
fn measure_accuracy<F, P, C>(predictor: &P, test_data: &[(Vec<F>, C)]) -> (usize, usize)
where
    F: Coeff,
    P: Predict<F>,
    C: AsRef<Class>,
{
    let mut num_correct = 0;
    let mut num_total = 0;
    for &(ref input_image, ref correct) in test_data.iter() {
        let result = predictor.predict(input_image);
        //println!("{:?}", &result);
        let idx_of_correct = result
            .iter()
            .enumerate()
            // Find the largest number
            .max_by(|x, y|
                match x.1.partial_cmp(y.1) {
                    Some(ord) => ord,
                    // Basically don't care about NaN comparisons
                    None => std::cmp::Ordering::Greater
                })
            .unwrap()
            // Take the index of the largest number
            .0;
        let prediction = idx_to_class(idx_of_correct);

        num_total += 1;
        if prediction == *correct.as_ref() {
            num_correct += 1;
        }
    }

    // Measure accuracy
    (num_correct, num_total)
}

fn idx_to_class(idx: usize) -> Class {
    use class::Class::*;
    match idx {
        0 => Bus,
        1 => NormalCar,
        2 => Truck,
        3 => Van,
        _ => panic!(),
    }
}

fn load_jpeg_as_f32_with_padding(file: &str) -> Vec<f32> {
    let input_shape = ImageGeometry::new(
        CLASSIC_HYPER_PARAMS.source_side,
        CLASSIC_HYPER_PARAMS.num_source_channels,
    );

    let conv1_filter_shape = PaddedSquare::from_side(CLASSIC_HYPER_PARAMS.conv_1_filter_side);
    let padded_image_shape = input_shape.with_filter_padding(&conv1_filter_shape);
    let padding = padded_image_shape.padding();

    // Load input as a vector of floats in the network format
    with_edge_padding_by_channel(&load_jpeg_as_f32(file), &input_shape, padding)
}
