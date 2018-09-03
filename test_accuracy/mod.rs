extern crate env_logger;
extern crate image;
extern crate ndarray;
#[macro_use]
extern crate log;
extern crate noisy_float;
extern crate num_traits;
extern crate ocl;
extern crate rusty_cnn;

mod class;
mod util;

use class::Class;
use rusty_cnn::*;
use std::time::Instant;
use util::*;

const INPUT_IMG_DIR: &str = "input/images";

const TEST_CLASSIC: bool = true;
const TEST_SPARSE: bool = true;
const TEST_SEPCONV_F32: bool = true;
//const TEST_SEPCONV_I8: bool = true;
const CLASSIC_SINGLE_SHOT: bool = false;
const SEPCONV_F32_SINGLE_SHOT: bool = false;
//const SEPCONV_I8_SINGLE_SHOT: bool = false;
const PRINT_RESULTS: bool = false;

pub fn main() {
    env_logger::init();

    // Figure out the existing classes for the network based on directory names
    let class_dir_names = list_dirs(INPUT_IMG_DIR).unwrap();

    debug!("Loading input images for classic networks...");

    let load_fun =
        |file: &String| -> Vec<f32> { load_jpeg_as_f32_with_filter_padding(file, (96, 96, 3), 5) };
    let mut test_data = load_test_data(INPUT_IMG_DIR, &class_dir_names, load_fun);
    if CLASSIC_SINGLE_SHOT {
        test_data = test_data
            .into_iter()
            .take(1)
            .collect::<Vec<(Vec<f32>, Class)>>();
    }

    if TEST_CLASSIC {
        let timer = Instant::now();

        // Initialize OpenCL and the network
        let net = classic::ClNetwork::<f32>::new(classic::Weights::default());
        let init_duration = timer.elapsed();

        // Make classifications and measure accuracy using the original network
        let (correct_inputs, total_inputs) = measure_accuracy(&net, &test_data);
        let total_duration = timer.elapsed();
        let accuracy = correct_inputs as f32 / total_inputs as f32;
        println!("classic network\t\t\t({} images)", total_inputs);
        println!(
            "\ttime: {:?}\t(+ {:?} init)",
            total_duration - init_duration,
            init_duration,
        );
        println!(
            "\taccu: {:.5}\t\t({}/{})",
            accuracy, correct_inputs, total_inputs
        );
    }

    if TEST_SPARSE {
        let timer = Instant::now();

        // Initialize OpenCL and the network
        let net = sparse::ClNetwork::<f32>::new(sparse::Weights::default());
        let init_duration = timer.elapsed();

        // Make classifications and measure accuracy using the original network
        let (correct_inputs, total_inputs) = measure_accuracy(&net, &test_data);
        let total_duration = timer.elapsed();
        let accuracy = correct_inputs as f32 / total_inputs as f32;
        println!("sparse network\t\t\t({} images)", total_inputs);
        println!(
            "\ttime: {:?}\t(+ {:?} init)",
            total_duration - init_duration,
            init_duration,
        );
        println!(
            "\taccu: {:.5}\t\t({}/{})",
            accuracy, correct_inputs, total_inputs
        );
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

        let timer = Instant::now();

        // Initialize OpenCL and the sep-conv network
        let net = sepconv::ClNetwork::<f32>::new(sepconv::Weights::default());
        let init_duration = timer.elapsed();

        // Make classifications and measure accuracy using the sep-conv network
        let (correct_inputs, total_inputs) = measure_accuracy(&net, &test_data);
        let total_duration = timer.elapsed();
        let accuracy = correct_inputs as f32 / total_inputs as f32;
        println!("sepconv-f32 network\t\t({} images)", total_inputs);
        println!(
            "\ttime: {:?}\t(+ {:?} init)",
            total_duration - init_duration,
            init_duration,
        );
        println!(
            "\taccu: {:.5}\t\t({}/{})",
            accuracy, correct_inputs, total_inputs
        );
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
        let net = sepconv::ClNetwork::<i8>::new(weights);

        // Make classifications and measure accuracy using the sep-conv network
        let (correct_inputs, total_inputs) = measure_accuracy(&net, &test_data);
        let accuracy = correct_inputs as f32 / total_inputs as f32;
        println!("sepconv-i8 network accuracy:");
        println!("{} ({}/{})", accuracy, correct_inputs, total_inputs);
    }
    */
}

/// Returns (num_correct_inputs, num_total_inputs)
fn measure_accuracy<F, P, C>(predictor: &P, test_data: &[(Vec<F>, C)]) -> (usize, usize)
where
    F: Coeff,
    P: Predict<F>,
    C: AsRef<Class>,
{
    let mut num_correct_inputs = 0;
    let mut num_total_inputs = 0;
    for &(ref input_image, ref correct_inputs) in test_data.iter() {
        let result = predictor.predict(input_image);
        if PRINT_RESULTS {
            println!(
                "\t\t{:?}",
                result
                    .iter()
                    .map(|f| format!("{:.3}", f))
                    .collect::<Vec<String>>()
            );
        }
        let idx_of_correct_inputs = result
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
        let prediction = idx_to_class(idx_of_correct_inputs);

        num_total_inputs += 1;
        if prediction == *correct_inputs.as_ref() {
            num_correct_inputs += 1;
        }
    }

    // Measure accuracy
    (num_correct_inputs, num_total_inputs)
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
