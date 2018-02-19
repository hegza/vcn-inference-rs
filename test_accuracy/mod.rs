extern crate env_logger;
extern crate image;
#[macro_use]
extern crate log;
extern crate noisy_float;
extern crate num_traits;
extern crate rusty_cnn;

mod class;
mod util;

use rusty_cnn::*;
use cl_util as cl;
use noisy_float::prelude::*;
use rusty_cnn::geometry::*;
use util::*;
use class::Class;

const INPUT_IMG_DIR: &str = "input/images";

pub fn main() {
    env_logger::init();

    let input_shape = ImageGeometry::new(
        CLASSIC_HYPER_PARAMS.source_side,
        CLASSIC_HYPER_PARAMS.num_source_channels,
    );

    // Load input images (x, y)
    debug!("Loading input images...");
    let class_dir_names = list_dirs(INPUT_IMG_DIR).unwrap();
    let load_fun = |file: &String| -> Vec<f32> { load_jpeg(file) };
    let test_data = load_test_data(INPUT_IMG_DIR, &class_dir_names, load_fun, &input_shape);

    debug!("Initializing network...");

    // Initialize OpenCL
    let (queue, program, _context) = cl::init("original_kernels.cl").unwrap();

    // Initialize the network
    let net = ClassicNetwork::<f32>::new(&program, &queue).unwrap();

    // Make classifications using the network
    let mut num_correct = 0;
    let mut num_total = 0;
    for &(ref input_image, ref correct) in test_data.iter() {
        let result = net.predict(&input_image, &queue).unwrap();
        let result = result.into_iter().map(|f| r32(f)).collect::<Vec<R32>>();
        let idx_of_correct = result
            .iter()
            .enumerate()
            .max_by_key(|&(_, item)| item)
            .unwrap()
            .0;
        let prediction = idx_to_class(idx_of_correct);

        num_total += 1;
        if prediction == *correct {
            num_correct += 1;
        }
    }

    // Measure accuracy
    let accuracy = num_correct as f32 / num_total as f32;
    println!("{}", accuracy);
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
