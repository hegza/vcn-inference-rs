#[macro_use]
extern crate criterion;
extern crate rusty_cnn;

use criterion::Criterion;
use rusty_cnn::*;
use rusty_cnn::cl_util as cl;

const SAMPLE_SIZE: usize = 150;
const NOISE_THRESHOLD: f64 = 0.05;
const BASELINE_DIR: &'static str = "input/baseline/input1";

/// Benchmark each layer separately
fn per_layer_benchmark(c: &mut Criterion) {
    let net = create_network(HYPER_PARAMS.clone());

    bench_layer1(net.conv1, c);
    bench_layer2(net.conv2, c);
    bench_layer3(net.dense3, c);
    bench_layer4(net.dense4, c);
    bench_layer5(net.dense5, c);
}

fn bench_layer1(conv1: ConvLayer<f32>, c: &mut Criterion) {
    let input_data =
        read_image_with_padding(&format!("{}/in.bin", BASELINE_DIR), *conv1.input_shape());
    let (kernel, _, queue) = create_kernel(&conv1, "conv_relu_1", &input_data).unwrap();
    c.bench_function("layer 1 kernel comp", move |b| {
        b.iter(|| unsafe { run_kernel(&kernel, &queue).unwrap() })
    });
}

fn bench_layer2(conv2: ConvLayer<f32>, c: &mut Criterion) {
    let input_data = f32::read_from_file(&format!("{}/fm1.f", BASELINE_DIR));
    let (kernel, _, queue) = create_kernel(&conv2, "conv_relu_2", &input_data).unwrap();
    c.bench_function("layer 2 kernel comp", move |b| {
        b.iter(|| unsafe { run_kernel(&kernel, &queue).unwrap() })
    });
}

fn bench_layer3(dense3: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = f32::read_from_file(&format!("{}/fm2.f", BASELINE_DIR));
    let (kernel, _, queue) = create_kernel(&dense3, "mtx_mulf", &input_data).unwrap();
    c.bench_function("layer 3 kernel cop", move |b| {
        b.iter(|| unsafe { run_kernel(&kernel, &queue).unwrap() })
    });
}

fn bench_layer4(dense4: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = f32::read_from_file(&format!("{}/fc3.f", BASELINE_DIR));
    c.bench_function("layer 4 comp", move |b| {
        b.iter(|| mtxmul_relu(&input_data, &dense4))
    });
}

fn bench_layer5(dense5: DenseLayer<f32>, c: &mut Criterion) {
    let input_data = f32::read_from_file(&format!("{}/fc4.f", BASELINE_DIR));
    c.bench_function("layer 5 comp", move |b| {
        b.iter(|| mtxmul_softmax(&input_data, &dense5))
    });
}

/// Benchmark the entire network
fn full_network_benchmark(c: &mut Criterion) {
    let (net, conv_relu1, conv_relu2, dense3_kernel, dense3_out_buf, queue) =
        init_network::<f32>(&format!("{}/in.bin", BASELINE_DIR)).unwrap();

    c.bench_function("network comp", move |b| {
        b.iter(|| {
            // Enqueue the kernel for the 1st layer (Convolution + ReLU)
            unsafe {
                run_kernel(&conv_relu1, &queue).unwrap();
            }

            // Enqueue the kernel for the 2nd layer (Convolution + ReLU)
            unsafe {
                run_kernel(&conv_relu2, &queue).unwrap();
            }

            // Enqueue the 3rd layer (fully-connected)
            unsafe {
                run_kernel(&dense3_kernel, &queue).unwrap();
            }
            let dense3_out = relu(&unsafe { cl::read_buf(&dense3_out_buf).unwrap() });

            // Run the 4th layer (fully-connected)
            let dense4_out = mtxmul_relu(&dense3_out, &net.dense4);

            // Run the 5th layer (fully-connected)
            mtxmul_softmax(&dense4_out, &net.dense5)
        })
    });
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = full_network_benchmark, per_layer_benchmark
}
criterion_main!(benches);
