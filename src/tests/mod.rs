mod network;
mod layers;

use super::*;
use util::*;
use network::*;
use ocl;
use ocl::{flags, Kernel};
use std::time::Instant;
use env_logger;

const RESULT_MARGIN: f32 = 0.000002f32;
const BASELINE_DIR: &'static str = "input/baseline/input1";

fn is_within_margin(a: &[f32], b: &[f32], margin: f32) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (idx, item) in a.iter().enumerate() {
        if (b[idx] - item).abs() > margin {
            return false;
        }
    }
    true
}
