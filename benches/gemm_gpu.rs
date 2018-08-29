//! The purpose of this benchmark is to compare the performance of different GEMM algorithms on the
//! GPU (most importantly, on large inputs).

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

use criterion::{AxisScale, Criterion, ParameterizedBenchmark, PlotConfiguration, Throughput};
use rusty_cnn::math::gemm::*;
use rusty_cnn::verify;
use shared::*;
use std::collections::HashMap;

const SAMPLE_SIZE: usize = 20;
const NOISE_THRESHOLD: f64 = 0.06;
const RESULT_MARGIN: f32 = 0.00002f32;

const D: usize = 64;

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

    let input_sizes: Vec<usize> = vec![1024, 512, 256, 128, 64, 32];

    // Setup
    let mut gemm_1_out = input_sizes
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
                    gemm_1_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Gemm1Kernel>>();

    // Verify result by setting the known matrices
    gemm_1[&D].set_buffers_from_slices(&input_a, &input_b);
    gemm_1[&D].calculate_wait();
    verify(&gemm_1_out[&D], &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = ParameterizedBenchmark::new(
        "cnugteren 1 naive",
        move |be, &ds| {
            be.iter_with_setup(
                || {
                    let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                    gemm_1[&ds].set_buffers_from_slices(&a, &b);
                },
                |()| gemm_1[&ds].calculate_wait(),
            )
        },
        input_sizes.clone(),
    );

    //  Setup
    let mut gemm_4_out = input_sizes
        .iter()
        .map(|&ds| (ds, vec![0f32; ds * ds]))
        .collect::<HashMap<usize, Vec<f32>>>();
    let gemm_4 = input_sizes
        .iter()
        .map(|&ds| {
            (
                ds,
                Gemm4Kernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_4_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Gemm4Kernel>>();

    // Verify Result
    gemm_4[&D].set_buffers_from_slices(&input_a, &input_b);
    gemm_4[&D].calculate_wait();
    verify(&gemm_4_out[&D], &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren 4 vectors", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_4[&ds].set_buffers_from_slices(&a, &b);
            },
            |_| gemm_4[&ds].calculate_wait(),
        )
    });

    // Setup
    let mut gemm_5_out = input_sizes
        .iter()
        .map(|&ds| (ds, vec![0f32; ds * ds]))
        .collect::<HashMap<usize, Vec<f32>>>();
    let gemm_5 = input_sizes
        .iter()
        .map(|&ds| {
            (
                ds,
                Gemm5Kernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_5_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Gemm5Kernel>>();

    // Verify Result
    gemm_5[&D].set_buffers_from_slices(&input_a, &input_b);
    gemm_5[&D].calculate_wait();
    verify(&gemm_5_out[&D], &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren 5 transpose", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_5[&ds].set_buffers_from_slices(&a, &b);
            },
            |_| gemm_5[&ds].calculate_wait(),
        )
    });

    // Setup
    let mut gemm_6_out = input_sizes
        .iter()
        .map(|&ds| (ds, vec![0f32; ds * ds]))
        .collect::<HashMap<usize, Vec<f32>>>();
    let gemm_6 = input_sizes
        .iter()
        .map(|&ds| {
            (
                ds,
                Gemm6WithBTransposeKernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_6_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Gemm6WithBTransposeKernel>>();

    // Verify Result
    gemm_6[&D].set_buffers_from_slices(&input_a, &input_b);
    gemm_6[&D].calculate_wait();
    verify(&gemm_6_out[&D], &correct_c, RESULT_MARGIN);

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren 6 tiling", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_6[&ds].set_buffers_from_slices(&a, &b);
            },
            |_| gemm_6[&ds].calculate_wait(),
        )
    });

    // Setup
    let mut gemm_6_out = input_sizes
        .iter()
        .map(|&ds| (ds, vec![0f32; ds * ds]))
        .collect::<HashMap<usize, Vec<f32>>>();
    let gemm_6 = input_sizes
        .iter()
        .map(|&ds| {
            (
                ds,
                Gemm6Kernel::uninitialized(
                    ds,
                    ds,
                    ds,
                    gemm_6_out.get_mut(&ds).unwrap(),
                    DeviceType::ALL,
                ),
            )
        })
        .collect::<HashMap<usize, Gemm6Kernel>>();

    // TODO: Verify result with cnugteren_6 with pretransposed B

    // Create benchmark-closure
    let bench = bench.with_function("cnugteren 6 pretransposed", move |be, &ds| {
        be.iter_with_setup(
            || {
                let (a, b) = (create_random_vec(ds * ds), create_random_vec(ds * ds));
                gemm_6[&ds].set_buffers_from_slices(&a, &b);
            },
            |_| gemm_6[&ds].calculate_wait(),
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
        "cnugteren 10 pretransposed",
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

    const GROUP: &str = "gemm-f32 (GPU)";
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

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = bench_gemm_variants
}
criterion_main!(benches);
