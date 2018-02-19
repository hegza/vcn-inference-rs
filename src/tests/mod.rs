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
const BASELINE_DIR: &'static str = "input/baseline/orig-f32-all-layers";
lazy_static!{
    static ref TEST_NETWORK: NetworkParams = NetworkParams::new(CLASSIC_HYPER_PARAMS);
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
