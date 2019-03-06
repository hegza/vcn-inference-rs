//! The purpose of this benchmark is to give an overview of the performance of different GEMM
//! algorithms on smallish input sizes.

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

use crate::shared::gemm::*;
use crate::shared::*;
use criterion::{AxisScale, Criterion, ParameterizedBenchmark, PlotConfiguration, Throughput};
use rusty_cnn::math::gemm::*;
use rusty_cnn::math::gemm_naive;
use rusty_cnn::verify;
use std::collections::HashMap;

const SAMPLE_SIZE: usize = 20;
const NOISE_THRESHOLD: f64 = 0.06;
const RESULT_MARGIN: f32 = 0.00002f32;

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

    let input_sizes: Vec<usize> = vec![256, 128, 64, 32];

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

    let bench = bench.with_function("bluss_matrixmultiply (host)", bench_bluss_matrixmultiply());

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
                Gemm1Kernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_1_gpu_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Gemm1Kernel>>();

    // Verify result by setting the known matrices
    gemm_1[&D].set_buffers_from_slices(&input_a, &input_b);
    gemm_1[&D].calculate_wait();
    verify(&gemm_1_gpu_out[&D], &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren 1 naive (GPU)", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_1[&ds].set_buffers_from_slices(&a, &b);
            },
            |()| gemm_1[&ds].calculate_wait(),
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
                Gemm6WithBTransposeKernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_6_gpu_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Gemm6WithBTransposeKernel>>();

    // Verify Result
    gemm_6_gpu[&D].set_buffers_from_slices(&input_a, &input_b);
    gemm_6_gpu[&D].calculate_wait();
    verify(&gemm_6_gpu_out[&D], &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren 6 tiling (GPU)", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_6_gpu[&ds].set_buffers_from_slices(&a, &b);
            },
            |_| gemm_6_gpu[&ds].calculate_wait(),
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
                Gemm6Kernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_6_gpu_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Gemm6Kernel>>();

    // TODO: Verify result with cnugteren_6 with pretransposed B

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren 6 pretransposed (GPU)", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_6_gpu[&ds].set_buffers_from_slices(&a, &b);
            },
            |_| gemm_6_gpu[&ds].calculate_wait(),
        )
    });

    // Setup
    let mut out = input_sizes
        .iter()
        .map(|&ds| (ds, vec![0f32; ds * ds]))
        .collect::<HashMap<usize, Vec<f32>>>();
    let gemm_10 = input_sizes
        .iter()
        .map(|&ds| {
            (
                ds,
                Gemm10Kernel::uninitialized(ds, ds, ds, out.get_mut(&ds).unwrap(), DeviceType::ALL),
            )
        })
        .collect::<HashMap<usize, Gemm10Kernel>>();

    // TODO: Verify result with cnugteren_10 with pretransposed B

    let bench = bench.with_function(
        "cnugteren 10 pretransposed (GPU)",
        // Create benchmark-closure
        move |be, &ds| {
            be.iter_with_setup(
                || {
                    let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                    gemm_10[&ds].set_buffers_from_slices(&a, &b);
                },
                |_| gemm_10[&ds].calculate_wait(),
            )
        },
    );

    const GROUP: &str = "gemm-f32";
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);
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

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = bench_gemm_variants
}
criterion_main!(benches);
