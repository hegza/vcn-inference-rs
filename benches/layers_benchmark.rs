#[macro_use]
extern crate criterion;
extern crate rusty_cnn;

use criterion::Criterion;
use rusty_cnn::*;

const SAMPLE_SIZE: usize = 150;
const NOISE_THRESHOLD: f64 = 0.08;
const BASELINE_DIR: &'static str = "input/baseline/orig-f32-all-layers";

/// Benchmark each layer separately.
fn per_layer_benchmark(c: &mut Criterion) {
    let net = create_layers(CLASSIC_HYPER_PARAMS.clone());

    bench_layer1(net.conv1, c);
    bench_layer2(net.conv2, c);
    bench_layer3(net.dense3, c);
    bench_layer4(net.dense4, c);
    bench_layer5(net.dense5, c);
}

fn bench_layer1(conv1: ConvLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(read_image_with_padding_from_bin_in_channels(
        &format!("{}/in.bin", BASELINE_DIR),
        *conv1.input_shape(),
    ));
    let (kernel, _, queue) = create_standalone_kernel(&conv1, "conv_relu_1", &input_data).unwrap();
    c.bench_function("layer 1 kernel comp", move |b| {
        b.iter(|| run_kernel_wait(&kernel, &queue).unwrap())
    });
}

fn bench_layer2(conv2: ConvLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/fm1.f",
        BASELINE_DIR
    )));
    let (kernel, _, queue) = create_standalone_kernel(&conv2, "conv_relu_2", &input_data).unwrap();
    c.bench_function("layer 2 kernel comp", move |b| {
        b.iter(|| run_kernel_wait(&kernel, &queue).unwrap())
    });
}

fn bench_layer3(dense3: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/fm2.f",
        BASELINE_DIR
    )));
    let (kernel, _, queue) = create_standalone_kernel(&dense3, "mtx_mulf", &input_data).unwrap();
    c.bench_function("layer 3 kernel comp", move |b| {
        b.iter(|| run_kernel_wait(&kernel, &queue).unwrap())
    });
}

fn bench_layer4(dense4: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/fc3.f",
        BASELINE_DIR
    )));
    c.bench_function("layer 4 comp", move |b| {
        b.iter(|| mtxmul_relu(&input_data, &dense4))
    });
}

fn bench_layer5(dense5: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = criterion::black_box(f32::read_lines_from_file(&format!(
        "{}/fc4.f",
        BASELINE_DIR
    )));
    c.bench_function("layer 5 comp", move |b| {
        b.iter(|| mtxmul_softmax(&input_data, &dense5))
    });
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = per_layer_benchmark
}
criterion_main!(benches);
