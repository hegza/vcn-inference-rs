#[macro_use]
extern crate criterion;
extern crate rusty_cnn;

use criterion::Criterion;
use rusty_cnn::*;
use rusty_cnn::cl_util as cl;

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

    c.bench_function("classic full", move |b| b.iter(|| net.predict(&input_data)));
}

/// Benchmark full computations of sepconv implementation.
fn sepconv_full(c: &mut Criterion) {
    let net = SepconvNetwork::<f32>::new(Weights::default());
    let input_data = criterion::black_box(f32::read_bin_from_file(
        "input/baseline/sepconv-f32-xcorr/in.bin",
    ));

    c.bench_function("sepconv full", move |b| b.iter(|| net.predict(&input_data)));
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = classic_full, sepconv_full, net_map_input
}
criterion_main!(benches);
