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

use criterion::{black_box, Bencher, Benchmark, Criterion};
use rusty_cnn::geometry::*;
use rusty_cnn::{classic, read_image_with_padding_from_bin_in_channels, sparse, Predict};
use crate::shared::CLASSIC_BASELINE;

// Sample size of 100 puts the max-min of the benches at around 10 us at worst.
const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.06;

pub fn bench_classic() -> (&'static str, impl FnMut(&mut Bencher)) {
    let net = classic::ClNetwork::<f32>::new(classic::Weights::default());
    let input_data = black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", CLASSIC_BASELINE),
        net.input_shape(),
    ));

    ("classic-f32", move |b: &mut Bencher| {
        b.iter(|| net.predict(&input_data))
    })
}

pub fn bench_sparse() -> (&'static str, impl FnMut(&mut Bencher)) {
    let net = sparse::ClNetwork::<f32>::new(sparse::Weights::default());

    let input_shape = ImageGeometry::new(96, 3);
    let filter_shape = PaddedSquare::from_side(5);
    let padded_input_shape = input_shape.with_filter_padding(&filter_shape);
    let input = black_box(read_image_with_padding_from_bin_in_channels::<f32>(
        &format!("{}/in.bin", CLASSIC_BASELINE),
        &padded_input_shape,
    ));

    ("classic-sparse3-f32", move |b: &mut Bencher| {
        b.iter(|| net.predict(&input))
    })
}

/// Benchmark each layer separately.
fn per_layer_benchmark(c: &mut Criterion) {
    // 1-2 GPU, 3 CPU, 4-5 host
    let (classic_id, classic) = bench_classic();
    // 1-2 GPU, 3 CPU + sparse, 4-5 host
    let (sparse_id, sparse) = bench_sparse();

    let bench = Benchmark::new(classic_id, classic).with_function(sparse_id, sparse);
    c.bench("classic variants", bench);
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = per_layer_benchmark
}
criterion_main!(benches);
