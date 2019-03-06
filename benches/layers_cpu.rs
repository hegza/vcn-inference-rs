//! The purpose of this benchmark is to give an idea of the execution time of the CPU-layers in the
//! vehicle classifier.

#[macro_use]
extern crate criterion;
#[macro_use]
extern crate lazy_static;
extern crate matrixmultiply;
extern crate ndarray;
extern crate num_traits;
extern crate ocl;
extern crate rand;
extern crate rusty_cnn;

mod shared;

use criterion::{Benchmark, Criterion};
use crate::shared::dense::*;

const SAMPLE_SIZE: usize = 150;
const NOISE_THRESHOLD: f64 = 0.1;

/// Benchmark each layer separately.
fn per_layer_benchmark(c: &mut Criterion) {
    let (dense3_cl_id, dense3_cl) = bench_dense3_cl_cpu();
    let (dense3_matrixmultiply_id, dense3_matrixmultiply) = bench_dense_3_bluss_matrixmultiply();
    let (dense4_cl_id, dense4_cl) = bench_dense4();
    let (dense5_cl_id, dense5_cl) = bench_dense5();
    let bench = Benchmark::new(dense3_cl_id, dense3_cl)
        .with_function(dense3_matrixmultiply_id, dense3_matrixmultiply)
        .with_function(dense4_cl_id, dense4_cl)
        .with_function(dense5_cl_id, dense5_cl);
    c.bench("layers (CPU)", bench);
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = per_layer_benchmark
}
criterion_main!(benches);
