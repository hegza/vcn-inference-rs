#[macro_use]
extern crate criterion;
extern crate ndarray;
extern crate ocl;
extern crate rand;
extern crate rusty_cnn;

mod common;

use common::*;
use criterion::Criterion;
use ocl::flags::*;
use rusty_cnn::*;

// Sample size of 150 puts the max-min of the benches at around 30 us at worst, and exec time at
// 1 min max per bench.
const SAMPLE_SIZE: usize = 150;
const NOISE_THRESHOLD: f64 = 0.1;

/// Benchmark each layer separately.
fn per_layer_benchmark(c: &mut Criterion) {
    let net = ClassicNetwork::create_layers(&CLASSIC_HYPER_PARAMS);
    // Create shorthands (and move)
    let (_, _, dense3, dense4, dense5) = net;

    // TODO: cl_[c|g]pu_vec16
    bench_dense3_cl_cpu(dense3.clone(), c);
    bench_dense3_host_ndarray(dense3.clone(), c);
    bench_dense4(dense4, c);
    bench_dense5(dense5, c);
}

fn bench_dense3_cl_cpu(dense3: DenseLayer<f32>, c: &mut Criterion) {
    let cl_layer = dense3.impl_standalone(
        &["src/cl/mtx_mul.cl"],
        "mtx_mul",
        &[],
        Some(DeviceType::CPU),
        LocalWorkSizePolicy::UseDefault,
    );

    c.bench_function("layer 3 - cl cpu mtxmul", move |b| {
        b.iter(|| cl_layer.dry_run())
    });
}

fn bench_dense3_host_ndarray(dense3: DenseLayer<f32>, cr: &mut Criterion) {
    use ndarray::*;
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/fm2.f",
        CLASSIC_BASELINE
    )));
    let m = dense3.num_out();
    let n = dense3.num_in();
    let a = Array2::<f32>::from_shape_vec((m, n), dense3.weights().clone()).unwrap();
    let k = 1;
    let b = Array2::<f32>::from_shape_vec((n, k), input_data).unwrap();
    cr.bench_function("layer 3 - host cpu ndarray mtxmul", move |be| {
        be.iter(|| a.dot(&b))
    });
}

fn bench_dense4(dense4: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/fc3.f",
        CLASSIC_BASELINE
    )));
    c.bench_function("layer 4 - host cpu mtxmul", move |b| {
        b.iter(|| relu(&dense4.compute(&input_data)))
    });
}

fn bench_dense5(dense5: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/fc4.f",
        CLASSIC_BASELINE
    )));
    c.bench_function("layer 5 - host cpu mtxmul", move |b| {
        b.iter(|| softmax(&dense5.compute(&input_data)))
    });
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = per_layer_benchmark
}
criterion_main!(benches);
