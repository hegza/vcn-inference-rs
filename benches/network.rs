#![allow(dead_code)]

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
use num_traits::bounds::Bounded;
use rand::distributions::uniform::SampleUniform;
use rand::Rng;
use rusty_cnn::*;

const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.05;

/// Benchmark full computations of original implementation.
fn bench_classic() -> (&'static str, impl FnMut(&mut Bencher)) {
    let net = classic::ClNetwork::<f32>::new(classic::Weights::default());
    let input_data = black_box(read_image_with_padding_from_bin_in_channels(
        TEST_IMAGE_BIN_PATH,
        net.input_shape(),
    ));

    ("classic-f32", move |b| {
        b.iter(|| net.predict(&input_data))
    })
}

/// Benchmark full computations of sepconv implementation.
fn bench_sepconv() -> (&'static str, impl FnMut(&mut Bencher)) {
    let net = sepconv::ClNetwork::<f32>::new(sepconv::Weights::default());
    let input_data = black_box(f32::read_bin_from_file(TEST_IMAGE_BIN_PATH));

    ("sepconv-f32", move |b| {
        b.iter(|| net.predict(&input_data))
    })
}

/// Benchmark full computations of sparse implementation.
fn bench_sparse() -> (&'static str, impl FnMut(&mut Bencher)) {
    let net = sparse::ClNetwork::<f32>::new(sparse::Weights::default());
    let input_data = black_box(load_jpeg_chw(TEST_IMAGE_JPEG_PATH));
    // OR
    /*
    let input_shape = ImageGeometry::new(96, 3);
    let filter_shape = PaddedSquare::from_side(5);
    let padded_input_shape = input_shape.with_filter_padding(&filter_shape);
    let input = black_box(read_image_with_padding_from_bin_in_channels::<f32>(
        TEST_IMAGE_BIN_PATH,
        &padded_input_shape,
    ));
    */


    ("sparse-f32", move |b| {
        b.iter(|| net.predict(&input_data))
    })
}

fn rng_vec<T>(len: usize) -> Vec<T>
where
    T: Bounded + PartialOrd + SampleUniform,
{
    let mut rng = rand::thread_rng();
    (0..len)
        .map(|_| rng.gen_range(T::min_value(), T::max_value()))
        .collect()
}

/// Benchmark each layer separately.
fn per_network_benchmark(c: &mut Criterion) {
    // 1-2 GPU, 3 CPU, 4-5 host
    let (classic_id, classic) = bench_classic();
    // 1-2 ?, 3 ?, 4-5 ?
    let (sepconv_id, sepconv) = bench_sepconv();
    // 1-2 GPU, 3 CPU/sparse, 4-5 host
    let (sparse_id, sparse) = bench_sparse();

    let bench = Benchmark::new(classic_id, classic).with_function(sepconv_id, sepconv).with_function(sparse_id, sparse);
    c.bench("full networks", bench);
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = per_network_benchmark
}
criterion_main!(benches);
