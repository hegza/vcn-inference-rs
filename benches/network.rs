#[macro_use]
extern crate criterion;
extern crate num_traits;
extern crate rand;
extern crate rusty_cnn;

mod common;

use common::*;
use criterion::Criterion;
use rusty_cnn::*;
use rusty_cnn::cl_util as cl;
use rand::Rng;
use num_traits::bounds::Bounded;
use rand::distributions::range::SampleRange;

const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.05;

/// Benchmark writing of input to device memory.
fn net_map_input(c: &mut Criterion) {
    let net = ClassicNetwork::<f32>::new();

    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", CLASSIC_BASELINE),
        *net.input_shape(),
    ));
    c.bench_function("network map input", move |b| {
        b.iter(|| unsafe { cl::map_to_buf(&net.in_buf, &input_data).unwrap() })
    });
}

/// Benchmark full computations of original implementation.
fn classic_full(c: &mut Criterion) {
    let net = ClassicNetwork::<f32>::new();
    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", CLASSIC_BASELINE),
        *net.input_shape(),
    ));

    c.bench_function("classic-f32 full", move |b| {
        b.iter(|| net.predict(&input_data))
    });
}

/// Benchmark full computations of sepconv implementation.
fn sepconv_f32_full(c: &mut Criterion) {
    let net = SepconvNetwork::<f32>::new(sepconv::Weights::default());
    let input_data = criterion::black_box(f32::read_bin_from_file(
        "input/baseline/sepconv-f32-xcorr/in.bin",
    ));

    c.bench_function("sepconv-f32 full", move |b| {
        b.iter(|| net.predict(&input_data))
    });
}

/// Benchmark full computations of sepconv implementation.
fn sepconv_i8_full(c: &mut Criterion) {
    // HACK: Random-generate weights for now
    let wgts = sepconv::Weights(
        // H/V convs
        rng_vec(5 * 1 * 3 * 7),
        rng_vec(5 * 1 * 3 * 32),
        rng_vec(5 * 1 * 32 * 7),
        rng_vec(1 * 5 * 7 * 32),
        // Dense layers
        rng_vec(100 * 24 * 24 * 32),
        rng_vec(100 * 100),
        rng_vec(100 * 4),
    );

    let net = SepconvNetwork::<i8>::new(wgts);
    // TODO: load real input data
    //let input_data = i8::read_bin_from_file("input/baseline/sepconv-f32-xcorr/in.bin");
    let input_data: Vec<i8> = criterion::black_box(rng_vec(96 * 96 * 3));

    c.bench_function("sepconv-i8 full", move |b| {
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
    targets = classic_full, sepconv_f32_full, sepconv_i8_full, net_map_input
}
criterion_main!(benches);