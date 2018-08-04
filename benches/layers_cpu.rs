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

use criterion::Criterion;
use shared::dense::*;

// Sample size of 150 puts the max-min of the benches at around 30 us at worst, and exec time at
// 1 min max per bench.
const SAMPLE_SIZE: usize = 150;
const NOISE_THRESHOLD: f64 = 0.1;

/// Benchmark each layer separately.
fn per_layer_benchmark(c: &mut Criterion) {
    bench_dense3_cl_cpu("layer 3 - cl mtxmul", c);
    bench_dense3_host_ndarray("layer 3 - host ndarray mtxmul", c);
    bench_dense4("layer 4 - host mtxmul", c);
    bench_dense5("layer 5 - host mtxmul", c);
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = per_layer_benchmark
}
criterion_main!(benches);
