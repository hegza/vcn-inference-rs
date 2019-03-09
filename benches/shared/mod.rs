#![allow(dead_code)]

pub mod conv;
pub mod dense;
pub mod gemm;

use criterion::{black_box, Bencher};
use rand;
use rand::Rng;

pub fn create_random_vec(len: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..len).map(|_| rng.gen_range(-1f32, 1f32)).collect()
}
