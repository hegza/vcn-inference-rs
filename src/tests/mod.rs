mod layers;
mod network;

use super::*;
use env_logger;
use layers::*;
use num_traits::{Float, Num, NumAssignOps, NumRef};
use ocl;
use ocl::{flags, Kernel};
use std::time::Instant;
use util::*;

pub const RESULT_MARGIN: f32 = 0.000002f32;
pub const COARSE_RESULT_MARGIN: f32 = 0.0035f32;
pub const CLASSIC_BASELINE: &'static str = "input/baseline/orig-f32-all-layers";
pub const SEPCONV_BASELINE: &'static str = "input/baseline/sepconv-f32-xcorr";
lazy_static! {
    static ref CLASSIC_PARAMS: NetworkParams = NetworkParams::new(CLASSIC_HYPER_PARAMS);
}
