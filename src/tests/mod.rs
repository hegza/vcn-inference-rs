mod layers;
mod quantized;

use super::*;
use crate::layers::*;
use crate::util::*;
use env_logger;
use num_traits::{Float, Num, NumAssignOps, NumRef};
use ocl;
use ocl::{flags, Kernel};
use std::time::Instant;

pub const RESULT_MARGIN: f32 = 0.000_002f32;
pub const COARSE_RESULT_MARGIN: f32 = 0.003_5f32;
pub const F32_GEMM_MAX_EPSILON: f32 = 6.93f32;
pub const CLASSIC_BASELINE: &str = "input/baseline/orig-f32-all-layers";
pub const SEPCONV_BASELINE: &str = "input/baseline/sepconv-f32-xcorr/case a";
