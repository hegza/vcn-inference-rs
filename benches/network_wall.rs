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
use rusty_cnn::*;
use crate::shared::*;

const SAMPLE_SIZE: usize = 10;
const NOISE_THRESHOLD: f64 = 0.03;

/// Benchmark full-network with initialization included (excluding file I/O).
fn net_wall_benchmark(c: &mut Criterion) {
    let weights = classic::Weights::default();
    let layers = classic::Layers::<f32>::new(weights.clone());

    // Create descriptor for input geometry for determining the size and padding of the image loaded
    // from disk.
    let input_shape = layers.conv1.input_shape();

    // Load input image with padding from disk
    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", CLASSIC_BASELINE),
        input_shape,
    ));

    c.bench_function("classic-f32 wall", move |b| {
        b.iter(|| {
            let net = classic::ClNetwork::<f32>::new(weights.clone());
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
