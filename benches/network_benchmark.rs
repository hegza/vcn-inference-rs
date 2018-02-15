#[macro_use]
extern crate criterion;
extern crate rusty_cnn;

use criterion::Criterion;
use rusty_cnn::*;

const BASELINE_DIR: &'static str = "input/baseline/input1";

fn criterion_benchmark(c: &mut Criterion) {
    //c.bench_function("fib 1a", |b| b.iter(|| fibonacci(1))); // 20 is original
    //c.bench_function("fib 1b", |b| b.iter(|| fibonacci(1))); // 20 is original

    // Create the network representation from network hyper-parameters
    let (conv1, conv2, dense3, dense4, dense5) = create_network(HYPER_PARAMS.clone());

    // Benchmark the 1st layer
    let input_data =
        read_image_with_padding(&format!("{}/in.bin", BASELINE_DIR), *conv1.input_shape());
    let (kernel, _, queue) = create_kernel(&conv1, "conv_relu_1", &input_data).unwrap();
    c.bench_function("layer 1 kernel comp", move |b| {
        b.iter(|| unsafe { run_kernel(&kernel, &conv1, &queue).unwrap() })
    });

    // Benchmark the 2nd layer
    let input_data = read_file_f32s(&format!("{}/fm1.f", BASELINE_DIR));
    let (kernel, _, queue) = create_kernel(&conv2, "conv_relu_2", &input_data).unwrap();
    c.bench_function("layer 2 kernel comp", move |b| {
        b.iter(|| unsafe { run_kernel(&kernel, &conv2, &queue).unwrap() })
    });

    // Benchmark the 3rd layer
    let input_data = read_file_f32s(&format!("{}/fm2.f", BASELINE_DIR));
    let (kernel, _, queue) = create_kernel(&dense3, "mtx_mulf", &input_data).unwrap();
    c.bench_function("layer 3 kernel comp", move |b| {
        b.iter(|| unsafe { run_kernel(&kernel, &dense3, &queue).unwrap() })
    });

    // Benchmark the 4th layer
    let input_data = read_file_f32s(&format!("{}/fc3.f", BASELINE_DIR));
    c.bench_function("layer 4 comp", move |b| {
        b.iter(|| mtxmul_relu(&input_data, &dense4))
    });

    // Benchmark the 5th layer
    let input_data = read_file_f32s(&format!("{}/fc4.f", BASELINE_DIR));
    c.bench_function("layer 5 comp", move |b| {
        b.iter(|| mtxmul_softmax(&input_data, &dense5))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
