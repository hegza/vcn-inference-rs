#[macro_use]
extern crate criterion;
extern crate matrixmultiply;
extern crate ndarray;
extern crate num_traits;
extern crate rand;
extern crate rusty_cnn;
#[macro_use]
extern crate enclose;

use criterion::{Benchmark, Criterion, Throughput};
use rusty_cnn::math::mtx_mul::sgemm::algo::*;
use rusty_cnn::verify;

const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.06;
const COARSE_RESULT_MARGIN: f32 = 0.0035f32;

const D: usize = 64;

/// On notation:
/// host =  compiled Rust.
/// GPU =   OpenCL / GPU.
/// CPU =   OpenCL / CPU.
///
/// ## Test setup
/// ### Initial setup
/// An OpenCL kernel is compiled for each of the OpenCL benchmarks. One instance of the input
/// matrices and the model output is allocated.
/// Optionally: output is verified against the correct output.
///
/// ### Once before each Benchmark
/// An instance of the input matrix is cloned using enclose! if necessary. The possible kernel and
/// an instance of an output buffer are moved inside the closure.
///
/// ### Before each benchmark-thread
/// Optionally: clone the inner input matrix with .clone().
///
/// ### After each Benchmark
/// Cloned input matrices are dropped.
///

// Benchmark each layer separately.
fn bench_sgemm_variants(c: &mut Criterion) {
    // Allocate input matrices and model output
    let input_a = criterion::black_box(
        String::from_utf8(include_bytes!("../src/tests/in/A_64x64.csv").to_vec())
            .unwrap()
            .split(',')
            .map(|word| word.trim().parse::<f32>())
            .filter_map(|res| res.ok())
            .collect::<Vec<f32>>(),
    );
    let input_b = criterion::black_box(
        String::from_utf8(include_bytes!("../src/tests/in/B_64x64.csv").to_vec())
            .unwrap()
            .split(',')
            .map(|word| word.trim().parse::<f32>())
            .filter_map(|res| res.ok())
            .collect::<Vec<f32>>(),
    );
    let correct_c = String::from_utf8(include_bytes!("../src/tests/out/C_64x64.csv").to_vec())
        .unwrap()
        .split(',')
        .map(|word| word.trim().parse::<f32>())
        .filter_map(|res| res.ok())
        .collect::<Vec<f32>>();

    // TODO: ParameterizedBenchmark to compare vector size
    // TODO: parameterize over device
    // TODO: bench upload + execute separately

    const GROUP: &str = "sgemm-f32 (64x64)";

    // Setup
    let mut naive_out = vec![0f32; D * D];

    // Verify result
    gemm_naive(D, D, D, &input_a, &input_b, &mut naive_out);
    verify(&naive_out, &correct_c, COARSE_RESULT_MARGIN);

    // Create benchmark-closure
    let bench = Benchmark::new(
        "naive (host)",
        enclose!((input_a, input_b) move |be| {
            // Measure
            be.iter(|| gemm_naive(D, D, D, &input_a, &input_b, &mut naive_out))
        }),
    );

    // Setup
    let mut matrixmultiply_out = vec![0f32; D * D];

    // Verify: matrixmultiply::sgemm uses row major instead of column-major and verification would
    // be troublesome. Calculation takes the same amount of time with the incorrect matrices, however.

    // Create benchmark-closure
    let bench = bench.with_function(
        // This is the implementation used by ndarray
        "bluss_matrixmultiply (host)",
        enclose!((input_a, input_b) move |be| {
        be.iter(|| unsafe {
            matrixmultiply::sgemm(
                D,
                D,
                D,
                1f32,
                input_a.as_ptr(),
                1,
                1,
                input_b.as_ptr(),
                1,
                1,
                1f32,
                matrixmultiply_out.as_mut_ptr(),
                1,
                1)
            }
        )}),
    );

    // Setup
    let mut sgemm_1_gpu_out = vec![0f32; D * D];
    let sgemm_1 = Naive1GemmKernel::new(D, D, D, &input_a, &input_b, &mut sgemm_1_gpu_out);

    // Verify Result
    sgemm_1.write(&input_a, &input_b);
    sgemm_1.run_wait();
    verify(&sgemm_1_gpu_out, &correct_c, COARSE_RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function(
        "cnugteren_1_naive (GPU)",
        enclose!((input_a, input_b) move |be| {
        be.iter_with_setup(|| sgemm_1.write(&input_a, &input_b), |_| sgemm_1.run_wait())
    }),
    );

    //  Setup
    let mut sgemm_4_gpu_out = vec![0f32; D * D];
    let sgemm_4 = Vectors4GemmKernel::new(D, D, D, &input_a, &input_b, &mut sgemm_4_gpu_out);

    // Verify Result
    sgemm_4.write(&input_a, &input_b);
    sgemm_4.run_wait();
    verify(&sgemm_4_gpu_out, &correct_c, COARSE_RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function(
        "cnugteren_4_vectors (GPU)",
        enclose!((input_a, input_b) move |be| {
        be.iter_with_setup(|| sgemm_4.write(&input_a, &input_b), |_| sgemm_4.run_wait())
    }),
    );

    // Setup
    let mut sgemm_5_gpu_out = vec![0f32; D * D];
    let sgemm_5 = Transpose5GemmKernel::new(D, D, D, &input_a, &input_b, &mut sgemm_5_gpu_out);

    // Verify Result
    sgemm_5.write(&input_a, &input_b);
    sgemm_5.run_wait();
    verify(&sgemm_5_gpu_out, &correct_c, COARSE_RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function(
        "cnugteren_5_transpose (GPU)",
        enclose!((input_a, input_b) move |be| {
        be.iter_with_setup(|| sgemm_5.write(&input_a, &input_b), |_| sgemm_5.run_wait())
    }),
    );

    // Setup
    let mut sgemm_6_gpu_out = vec![0f32; D * D];
    let sgemm_6 = Tiling6GemmKernel::new(D, D, D, &input_a, &input_b, &mut sgemm_6_gpu_out);

    // Verify Result
    sgemm_6.write(&input_a, &input_b);
    sgemm_6.run_wait();
    verify(&sgemm_6_gpu_out, &correct_c, COARSE_RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function(
        "cnugteren_6_tiling (GPU)",
        enclose!((input_a, input_b) move |be| {
        be.iter_with_setup(|| sgemm_6.write(&input_a, &input_b), |_| sgemm_6.run_wait())
    }),
    );

    c.bench(
        GROUP,
        bench.throughput(Throughput::Bytes(
            std::mem::size_of::<f32>() as u32 * 64 * 64,
        )),
    );
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = bench_sgemm_variants
}
criterion_main!(benches);
