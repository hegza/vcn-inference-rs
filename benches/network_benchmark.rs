#[macro_use]
extern crate criterion;
extern crate rusty_cnn;

use criterion::Criterion;
use rusty_cnn::*;
use rusty_cnn::cl_util as cl;

const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.05;
const BASELINE_DIR: &'static str = "input/baseline/orig-f32-all-layers";

/// Benchmark writing of input and weights to device memory.
fn net_buf_write_benchmark(c: &mut Criterion) {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init().unwrap();

    let net = Network::<f32>::new(&program, &queue).unwrap();

    let input_data = read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", BASELINE_DIR),
        *net.conv1.input_shape(),
    );
    c.bench_function("network write bufs", move |b| {
        b.iter(|| net.upload_buffers(&input_data, &queue).unwrap())
    });
}

/// Benchmark full-network computations.
fn net_comp_benchmark(c: &mut Criterion) {
    // Initialize OpenCL
    let (queue, program, _context) = cl::init().unwrap();

    let net = Network::<f32>::new(&program, &queue).unwrap();
    let input_data = read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", BASELINE_DIR),
        *net.conv1.input_shape(),
    );
    net.upload_buffers(&input_data, &queue).unwrap();

    c.bench_function("network comp", move |b| b.iter(|| net.run(&queue)));
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = net_comp_benchmark, net_buf_write_benchmark
}
criterion_main!(benches);
