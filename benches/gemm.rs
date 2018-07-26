#[macro_use]
extern crate criterion;
extern crate matrixmultiply;
extern crate ndarray;
extern crate num_traits;
extern crate rand;
extern crate rusty_cnn;

mod common;

use common::*;
use criterion::{AxisScale, Criterion, ParameterizedBenchmark, PlotConfiguration, Throughput};
use rusty_cnn::math::gemm_naive;
use rusty_cnn::math::mtx_mul::gemm::*;
use rusty_cnn::verify;
use std::collections::HashMap;

const SAMPLE_SIZE: usize = 20;
const NOISE_THRESHOLD: f64 = 0.06;
const RESULT_MARGIN: f32 = 0.00002f32;

const D: usize = 64;

const BASELINE: &str = "baseline 24-07-2018";

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
fn bench_gemm_variants(c: &mut Criterion) {
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

    // TODO: bench upload + execute separately

    const GROUP: &str = "gemm-f32";
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
    let input_sizes: Vec<usize> = vec![1024, 512, 256, 128, 64, 32];

    let mut naive_out = vec![0f32; D * D];

    // Verify result
    gemm_naive(D, D, D, &input_a, &input_b, &mut naive_out);
    verify(&naive_out, &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = ParameterizedBenchmark::new(
        "naive (host)",
        move |be, &ds| {
            // Measure
            be.iter_with_setup(
                || {
                    (
                        create_random_vec(ds * ds),
                        create_random_vec(ds * ds),
                        vec![0f32; ds * ds],
                    )
                },
                |(a, b, mut c)| gemm_naive(ds, ds, ds, &a, &b, &mut c),
            )
        },
        input_sizes.clone(),
    );

    // matrixmultiply::gemm uses row major instead of column-major and verification would be
    // troublesome. Calculation takes the same amount of time with the incorrect matrices, however.

    // Create benchmark-closure
    let bench = bench.with_function(
        // This is the implementation used by ndarray
        "bluss_matrixmultiply (host)",
        move |be, &ds| {
            be.iter_with_setup(
                || {
                    (
                        create_random_vec(ds * ds),
                        create_random_vec(ds * ds),
                        vec![0f32; ds * ds],
                    )
                },
                |(a, b, mut c)| unsafe {
                    matrixmultiply::sgemm(
                        ds,
                        ds,
                        ds,
                        1f32,
                        a.as_ptr(),
                        1,
                        1,
                        b.as_ptr(),
                        1,
                        1,
                        1f32,
                        c.as_mut_ptr(),
                        1,
                        1,
                    )
                },
            )
        },
    );

    // Setup
    let mut gemm_1_gpu_out = input_sizes
        .iter()
        .map(|&ds| (ds, vec![0f32; ds * ds]))
        .collect::<HashMap<usize, Vec<f32>>>();
    let gemm_1 = input_sizes
        .iter()
        .map(|&ds| {
            (
                ds,
                Naive1GemmKernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_1_gpu_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Naive1GemmKernel>>();

    // Verify result by setting the known matrices
    gemm_1[&D].set_buffers_from_slices(&input_a, &input_b);
    gemm_1[&D].calculate_wait();
    verify(&gemm_1_gpu_out[&D], &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren_1_naive (GPU)", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_1[&ds].set_buffers_from_slices(&a, &b);
            },
            |()| gemm_1[&ds].calculate_wait(),
        )
    });

    //  Setup
    let mut gemm_4_gpu_out = input_sizes
        .iter()
        .map(|&ds| (ds, vec![0f32; ds * ds]))
        .collect::<HashMap<usize, Vec<f32>>>();
    let gemm_4 = input_sizes
        .iter()
        .map(|&ds| {
            (
                ds,
                Vectors4GemmKernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_4_gpu_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Vectors4GemmKernel>>();

    // Verify Result
    gemm_4[&D].set_buffers_from_slices(&input_a, &input_b);
    gemm_4[&D].calculate_wait();
    verify(&gemm_4_gpu_out[&D], &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren_4_vectors (GPU)", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_4[&ds].set_buffers_from_slices(&a, &b);
            },
            |_| gemm_4[&ds].calculate_wait(),
        )
    });

    // Setup
    let mut gemm_5_gpu_out = input_sizes
        .iter()
        .map(|&ds| (ds, vec![0f32; ds * ds]))
        .collect::<HashMap<usize, Vec<f32>>>();
    let gemm_5 = input_sizes
        .iter()
        .map(|&ds| {
            (
                ds,
                Transpose5GemmKernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_5_gpu_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Transpose5GemmKernel>>();

    // Verify Result
    gemm_5[&D].set_buffers_from_slices(&input_a, &input_b);
    gemm_5[&D].calculate_wait();
    verify(&gemm_5_gpu_out[&D], &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren_5_transpose (GPU)", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_5[&ds].set_buffers_from_slices(&a, &b);
            },
            |_| gemm_5[&ds].calculate_wait(),
        )
    });

    // Setup
    let mut gemm_6_gpu_out = input_sizes
        .iter()
        .map(|&ds| (ds, vec![0f32; ds * ds]))
        .collect::<HashMap<usize, Vec<f32>>>();
    let gemm_6_gpu = input_sizes
        .iter()
        .map(|&ds| {
            (
                ds,
                Tiling6GemmKernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_6_gpu_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Tiling6GemmKernel>>();

    // Verify Result
    gemm_6_gpu[&D].set_buffers_from_slices(&input_a, &input_b);
    gemm_6_gpu[&D].calculate_wait();
    verify(&gemm_6_gpu_out[&D], &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren_6_tiling (GPU)", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_6_gpu[&ds].set_buffers_from_slices(&a, &b);
            },
            |_| gemm_6_gpu[&ds].calculate_wait(),
        )
    });

    c.bench(
        GROUP,
        bench
            .throughput(|&ds| {
                // Output throughput is size of matrix times size of data-type
                Throughput::Bytes(std::mem::size_of::<f32>() as u32 * ds as u32 * ds as u32)
            })
            .plot_config(plot_config),
    );
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD).retain_baseline(BASELINE.to_string());
    targets = bench_gemm_variants
}
criterion_main!(benches);
