#[macro_use]
extern crate criterion;
extern crate rand;
extern crate rusty_cnn;

use criterion::Criterion;
use rusty_cnn::*;
use rusty_cnn::cl_util as cl;
use rand::Rng;

const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.05;
const BASELINE_DIR: &'static str = "input/baseline/orig-f32-all-layers";

/// Benchmark writing of input to device memory.
fn net_map_input(c: &mut Criterion) {
    let net = ClassicNetwork::<f32>::new();

    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", BASELINE_DIR),
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
        &format!("{}/in.bin", BASELINE_DIR),
        *net.input_shape(),
    ));

    c.bench_function("classic-f32 full", move |b| {
        b.iter(|| net.predict(&input_data))
    });
}

/// Benchmark full computations of sepconv implementation.
fn sepconv_f32_full(c: &mut Criterion) {
    let net = SepconvNetwork::<f32>::new(Weights::default());
    let input_data = criterion::black_box(f32::read_bin_from_file(
        "input/baseline/sepconv-f32-xcorr/in.bin",
    ));

    c.bench_function("sepconv-f32 full", move |b| {
        b.iter(|| net.predict(&input_data))
    });
}

/// Benchmark full computations of sepconv implementation.
fn sepconv_i8_full(c: &mut Criterion) {
    use std::i8;
    let mut rng = rand::thread_rng();

    // HACK: Random-generate weights for now
    let wgts = sepconv::Weights(
        // H/V convs
        (0..5 * 1 * 3 * 7)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        (0..1 * 5 * 7 * 32)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        (0..5 * 1 * 32 * 7)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        (0..1 * 5 * 7 * 32)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        // Dense layers
        (0..100 * 24 * 24 * 32)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        (0..100 * 100)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
        (0..100 * 4)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
    );

    let net = SepconvNetwork::<i8>::new(wgts);
    // TODO: load real input data
    //let input_data = i8::read_bin_from_file("input/baseline/sepconv-f32-xcorr/in.bin");
    let input_data: Vec<i8> = criterion::black_box(
        (0..96 * 96 * 3)
            .map(|_| rng.gen_range(i8::MIN, i8::MAX))
            .collect(),
    );

    c.bench_function("sepconv-i8 full", move |b| {
        b.iter(|| net.predict(&input_data))
    });
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = classic_full, sepconv_f32_full, sepconv_i8_full, net_map_input
}
criterion_main!(benches);
