mod network;
mod layers;

use super::*;
use util::*;
use layers::*;
use ocl;
use ocl::{flags, Kernel};
use std::time::Instant;
use env_logger;
use num_traits::{Float, Num, NumAssignOps, NumRef};

const RESULT_MARGIN: f32 = 0.000002f32;
const COARSE_RESULT_MARGIN: f32 = 0.0035f32;
const CLASSIC_BASELINE: &'static str = "input/baseline/orig-f32-all-layers";
const SEPCONV_BASELINE: &'static str = "input/baseline/sepconv-f32-xcorr";
lazy_static!{
    static ref CLASSIC_PARAMS: NetworkParams = NetworkParams::new(CLASSIC_HYPER_PARAMS);
}

fn is_within_margin<T>(a: &[T], b: &[T], margin: T) -> bool
where
    T: Num + GenericOps + PartialOrd + Copy,
{
    if a.len() != b.len() {
        return false;
    }

    for (idx, item) in a.iter().enumerate() {
        if (b[idx] - *item).generic_abs() > margin {
            return false;
        }
    }
    true
}

// Wrap is_within_margin within an assert!()
fn verify(output: &[f32], correct: &[f32], margin: f32) {
    assert!(
        is_within_margin(output, correct, margin),
        "output is not within margin of correct: {:?} != {:?}",
        output,
        correct
    );
}
