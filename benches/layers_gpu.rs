//! The purpose of this benchmark is to give an idea of the execution time of the GPU layers in the
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
use shared::conv::*;

// Sample size of 100 puts the max-min of the benches at around 10 us at worst.
const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.06;

/// Benchmark each layer separately.
fn per_layer_benchmark(c: &mut Criterion) {
    bench_sepconv1("layer 1 - cl sepconv v+h+mxp", c);
    bench_sepconv2("layer 2 - cl sepconv v+h+mxp", c);
    bench_sepconv1and2("layers 1 + 2 - cl sepconv v+h+mxp", c);

    bench_conv1("layer 1 - cl conv", c);
    bench_conv2("layer 2 - cl conv", c);
    bench_conv1and2("layers 1 + 2 - cl conv", c);
    //bench_dense3_cl_gpu(&dense3, c);
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = per_layer_benchmark
}
criterion_main!(benches);
