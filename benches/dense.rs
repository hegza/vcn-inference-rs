//! The purpose of this benchmark is to test and compare the different ways of implementing a dense
//! layer in a CNN. Test case is based on the vehicle classifier layer 3.

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

use crate::shared::dense::*;
use criterion::{AxisScale, Benchmark, Criterion, PlotConfiguration};

const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.06;

fn bench_dense_layer_variants(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let (host_id, host) = bench_dense3_host();
    let (cl_id, cl) = bench_dense3_cl_cpu();
    let (matrixmultiply_id, matrixmultiply) = bench_dense_3_bluss_matrixmultiply();
    let (cnugteren_10_id, cnugteren_10) = bench_dense_3_cnugteren_10();
    let (sparse_id, sparse) = bench_sparse3();

    let bench = Benchmark::new(host_id, host)
        .with_function(cl_id, cl)
        .with_function(matrixmultiply_id, matrixmultiply)
        .with_function(cnugteren_10_id, cnugteren_10)
        .with_function(sparse_id, sparse);

    c.bench("layer-3-f32", bench.plot_config(plot_config));
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = bench_dense_layer_variants
}
criterion_main!(benches);
