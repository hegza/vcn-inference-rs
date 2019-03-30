extern crate env_logger;
extern crate image;
#[macro_use]
extern crate ndarray;
#[macro_use]
extern crate log;
extern crate noisy_float;
extern crate num_traits;
extern crate ocl;
extern crate rusty_cnn;

mod class;
mod util;

use crate::class::Class;
use crate::util::*;
use rusty_cnn::*;
use rusty_cnn::{VCN_SEPCONV_F32_WEIGHTS_DIR, VCN_SPARSE_WEIGHTS_DIR, VCN_WEIGHTS_DIR};
use std::time::{Duration, Instant};

const INPUT_IMG_DIR: &str = "input/images";

const TEST_CLASSIC: bool = true;
const TEST_SPARSE: bool = true;
const TEST_SEPCONV_F32: bool = true;
const CLASSIC_SINGLE_SHOT: bool = false;
const SEPCONV_F32_SINGLE_SHOT: bool = false;
const PRINT_PREDICTIONS: bool = false;

pub fn main() {
    env_logger::init();

    // Figure out the existing classes for the network based on directory names
    let class_dir_names = list_dirs(INPUT_IMG_DIR).unwrap();

    debug!("Loading input images for classic network...");

    let load_fun = |file: &String| -> Vec<f32> {
        let raw_input: Vec<f32> = load_jpeg_chw(file);
        let mut padded = ndarray::Array::zeros((3, 100, 100));
        padded
            .slice_mut(s![.., 2..-2, 2..-2])
            .assign(&ndarray::Array::from_shape_vec((3, 96, 96), raw_input).unwrap());

        padded
            .permuted_axes((0, 1, 2))
            .into_iter()
            .cloned()
            .collect::<Vec<f32>>()
    };
    let mut test_data = load_test_data(INPUT_IMG_DIR, &class_dir_names, load_fun);
    if CLASSIC_SINGLE_SHOT {
        test_data = test_data
            .into_iter()
            .take(1)
            .collect::<Vec<(Vec<f32>, Class)>>();
    }

    if TEST_CLASSIC {
        debug!("Starting to test classic network.");
        let timer = Instant::now();

        // Initialize OpenCL and the network
        let net = classic::ClNetwork::<f32>::new(classic::Weights::default());
        let init_duration = timer.elapsed();

        // Make classifications and measure accuracy using the original network
        let (correct_inputs, total_inputs) = measure_accuracy(&net, &test_data);
        let total_duration = timer.elapsed();
        report(
            "classic network",
            init_duration,
            total_duration,
            correct_inputs,
            total_inputs,
            Some(format!("{}/tf_accuracy.f", VCN_WEIGHTS_DIR)),
        );
    }

    let load_fun = |file: &String| -> Vec<f32> {
        let raw_input: Vec<f32> = load_jpeg_chw(file);
        let mut padded = ndarray::Array::zeros((3, 100, 100));
        padded
            .slice_mut(s![.., 2..-2, 2..-2])
            .assign(&ndarray::Array::from_shape_vec((3, 96, 96), raw_input).unwrap());

        padded.into_iter().cloned().collect::<Vec<f32>>()
    };
    let test_data = load_test_data(INPUT_IMG_DIR, &class_dir_names, load_fun);

    if TEST_SPARSE {
        debug!("Starting to test sparse network.");
        let timer = Instant::now();

        // Initialize OpenCL and the network
        let net = sparse::ClNetwork::<f32>::new(sparse::Weights::default());
        let init_duration = timer.elapsed();

        // Make classifications and measure accuracy using the original network
        let (correct_inputs, total_inputs) = measure_accuracy(&net, &test_data);
        let total_duration = timer.elapsed();
        report(
            "sparse network",
            init_duration,
            total_duration,
            correct_inputs,
            total_inputs,
            Some(format!("{}/tf_accuracy.f", VCN_SPARSE_WEIGHTS_DIR)),
        );
    }

    debug!("Loading input images for sepconv network...");

    let load_fun = |file: &String| -> Vec<f32> { load_jpeg_chw(file) };
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

        debug!("Starting to test sepconv network.");
        let timer = Instant::now();

        // Initialize OpenCL and the sep-conv network
        let net = sepconv::ClNetwork::<f32>::new(sepconv::Weights::default());
        let init_duration = timer.elapsed();

        // Make classifications and measure accuracy using the sep-conv network
        let (correct_inputs, total_inputs) = measure_accuracy(&net, &test_data);
        let total_duration = timer.elapsed();
        report(
            "sepconv-f32 network",
            init_duration,
            total_duration,
            correct_inputs,
            total_inputs,
            Some(format!("{}/tf_accuracy.f", VCN_SEPCONV_F32_WEIGHTS_DIR)),
        );
    }
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
        if PRINT_PREDICTIONS {
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
            .max_by(|x, y| match x.1.partial_cmp(y.1) {
                Some(ord) => ord,
                // Basically don't care about NaN comparisons
                None => std::cmp::Ordering::Greater,
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
    use crate::class::Class::*;
    match idx {
        0 => Bus,
        1 => NormalCar,
        2 => Truck,
        3 => Van,
        _ => panic!(),
    }
}

fn report(
    case_name: &str,
    init_duration: Duration,
    total_duration: Duration,
    correct_inputs: usize,
    total_inputs: usize,
    baseline_accuracy_filename: Option<String>,
) {
    let accuracy = correct_inputs as f32 / total_inputs as f32;
    let acc_mismatch_msg = {
        if let Some(baseline_acc) =
            baseline_accuracy_filename.and_then(|f| f32::read_lines_from_file(&f).ok())
        {
            const EPSILON: f32 = 0.0001f32;
            let acc = baseline_acc.first().unwrap();
            if (acc - accuracy).abs() >= EPSILON {
                format!(" ACCURACY MISMATCH ({})", acc)
            } else {
                "".to_owned()
            }
        } else {
            " NO BASELINE ACCURACY PROVIDED".to_owned()
        }
    };

    println!("{0: <31} ({1} images)", case_name, total_inputs);
    println!(
        "    time: {:?}\t(+ {:?} init)",
        total_duration - init_duration,
        init_duration,
    );
    println!(
        "    accu: {:.5}\t\t({}/{}){}",
        accuracy, correct_inputs, total_inputs, acc_mismatch_msg
    );
}
