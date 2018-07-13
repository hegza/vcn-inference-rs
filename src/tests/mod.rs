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

pub const RESULT_MARGIN: f32 = 0.000002f32;
pub const COARSE_RESULT_MARGIN: f32 = 0.0035f32;
pub const CLASSIC_BASELINE: &'static str = "input/baseline/orig-f32-all-layers";
pub const SEPCONV_BASELINE: &'static str = "input/baseline/sepconv-f32-xcorr";
const VEC_DISPLAY_ELEMENTS_MAX: usize = 6;
lazy_static!{
    static ref CLASSIC_PARAMS: NetworkParams = NetworkParams::new(CLASSIC_HYPER_PARAMS);
}

pub fn is_within_margin<T>(a: &[T], b: &[T], margin: T) -> bool
where
    T: Num + GenericOps + PartialOrd + Copy,
{
    // Assume that the inputs are equally long.
    debug_assert_eq!(a.len(), b.len());

    for (idx, item) in a.iter().enumerate() {
        if (b[idx] - *item).generic_abs() > margin {
            return false;
        }
    }
    true
}

// Wrap is_within_margin within an assert!()
pub fn verify(output: &[f32], correct: &[f32], margin: f32) {
    assert_eq!(output.len(), correct.len());

    if is_within_margin(output, correct, margin) {
        return;
    }

    // Contains NaN?
    let display_nan_msg = if output.iter().any(|&x| x.is_nan()) {
        ", vector contains NaN"
    } else {
        ""
    };

    let display = if output.len() <= VEC_DISPLAY_ELEMENTS_MAX {
        format!(
            "{:?}{}\n!=\n{:?}",
            &output[..],
            display_nan_msg,
            &correct[..]
        )
    } else {
        format!(
            "{:?}...{:?}{}\n!=\n{:?}...{:?}",
            &output[0..VEC_DISPLAY_ELEMENTS_MAX / 2],
            &output[output.len() - VEC_DISPLAY_ELEMENTS_MAX / 2..output.len()],
            display_nan_msg,
            &correct[0..VEC_DISPLAY_ELEMENTS_MAX / 2],
            &correct[correct.len() - VEC_DISPLAY_ELEMENTS_MAX / 2..correct.len()],
        )
    };

    assert!(
        false,
        "output is not within margin of correct:\n{}",
        display
    );
}
