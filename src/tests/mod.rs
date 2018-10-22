mod layers;
mod quantized;

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
pub const F32_GEMM_MAX_EPSILON: f32 = 6.93f32;
pub const CLASSIC_BASELINE: &'static str = "input/baseline/orig-f32-all-layers";
pub const SEPCONV_BASELINE: &'static str = "input/baseline/sepconv-f32-xcorr/case a";
