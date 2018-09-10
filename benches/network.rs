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

use criterion::Criterion;
use num_traits::bounds::Bounded;
use rand::distributions::range::SampleRange;
use rand::Rng;
use rusty_cnn::*;
use shared::*;

const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.05;

/// Benchmark full computations of original implementation.
fn classic_full(c: &mut Criterion) {
    let net = classic::ClNetwork::<f32>::new(classic::Weights::default());
    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", CLASSIC_BASELINE),
        net.input_shape(),
    ));

    c.bench_function("classic-f32 full", move |b| {
        b.iter(|| net.predict(&input_data))
    });
}

/// Benchmark full computations of sepconv implementation.
fn sepconv_f32_full(c: &mut Criterion) {
    let net = sepconv::ClNetwork::<f32>::new(sepconv::Weights::default());
    let input_data = criterion::black_box(f32::read_bin_from_file(
        "input/baseline/sepconv-f32-xcorr/in.bin",
    ));

    c.bench_function("sepconv-f32 full", move |b| {
        b.iter(|| net.predict(&input_data))
    });
}

/// Benchmark full computations of sparse implementation.
fn sparse_f32_full(c: &mut Criterion) {
    let net = sparse::ClNetwork::<f32>::new(sparse::Weights::default());
    let input_data = criterion::black_box(load_jpeg("input/baseline/sparse-f32/in.jpg"));

    c.bench_function("sparse-f32 full", move |b| {
        b.iter(|| net.predict(&input_data))
    });
}

fn rng_vec<T>(len: usize) -> Vec<T>
where
    T: Bounded + PartialOrd + SampleRange,
{
    let mut rng = rand::thread_rng();
    (0..len)
        .map(|_| rng.gen_range(T::min_value(), T::max_value()))
        .collect()
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = classic_full, sepconv_f32_full, sparse_f32_full /*sepconv_i8_full, */
}
criterion_main!(benches);
