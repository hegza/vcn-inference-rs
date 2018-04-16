#[macro_use]
extern crate criterion;
extern crate rusty_cnn;

use criterion::Criterion;
use rusty_cnn::*;
use rusty_cnn::geometry::{ImageGeometry, PaddedSquare};

const SAMPLE_SIZE: usize = 10;
const NOISE_THRESHOLD: f64 = 0.03;
const BASELINE_DIR: &'static str = "input/baseline/orig-f32-all-layers";

/// Benchmark full-network with initialization included (excluding file I/O).
fn net_wall_benchmark(c: &mut Criterion) {
    // Create descriptor for input geometry for determining the size and padding of the image loaded
    // from disk.
    let conv1_filter_shape = PaddedSquare::from_side(CLASSIC_HYPER_PARAMS.conv_1_filter_side);
    let input_shape = ImageGeometry::new(
        CLASSIC_HYPER_PARAMS.source_side,
        CLASSIC_HYPER_PARAMS.num_source_channels,
    );
    let padded_input_shape = input_shape.with_filter_padding(&conv1_filter_shape);

    // Load input image with padding from disk
    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", BASELINE_DIR),
        padded_input_shape,
    ));

    c.bench_function("network wall", move |b| {
        b.iter(|| {
            let net = ClassicNetwork::<f32>::new();
            net.predict(&input_data)
        })
    });
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = net_wall_benchmark
}
criterion_main!(benches);
