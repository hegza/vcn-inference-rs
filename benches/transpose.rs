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

const SAMPLE_SIZE: usize = 20;
const NOISE_THRESHOLD: f64 = 0.06;

use criterion::{AxisScale, Criterion, ParameterizedBenchmark, PlotConfiguration, Throughput};
use rusty_cnn::math::mtx_mul::gemm::*;
use shared::*;
use std::collections::HashMap;

fn bench_transpose_gpu(c: &mut Criterion) {
    bench_transpose(
        c,
        vec![4096, 2048, 1024, 512, 256, 128, 64, 32, 16],
        DeviceType::ALL,
        "transpose (GPU)",
    );
}

fn bench_transpose_cpu(c: &mut Criterion) {
    bench_transpose(
        c,
        vec![512, 256, 128, 64, 32, 16],
        DeviceType::CPU,
        "transpose (CPU)",
    );
}

/// Benches the transpose for local work-sizes of 8 and 16 parametrized over device and input size
fn bench_transpose(c: &mut Criterion, params: Vec<usize>, device: DeviceType, group_name: &str) {
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let test_data = params
        .iter()
        .map(|&ds| {
            (ds, {
                let transposex = 16;
                let transposey = 16;
                let (queue, program, _context) = rusty_cnn::cl_util::init_from_file::<f32>(
                    &["src/math/mtx_mul/gemm/cl/transpose.cl"],
                    &[
                        &format!("-D TRANSPOSEX={}", transposex),
                        &format!("-D TRANSPOSEY={}", transposey),
                    ],
                    Some(device),
                );

                let buffer = ocl::Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(ocl::flags::MEM_READ_ONLY)
                    .len(ds * ds)
                    .build()
                    .unwrap();
                let output = ocl::Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(ocl::flags::MEM_READ_WRITE)
                    .len(ds * ds)
                    .build()
                    .unwrap();

                let kernel = ocl::Kernel::builder()
                    .program(&program)
                    .name("transpose")
                    .queue(queue.clone())
                    .local_work_size(ocl::SpatialDims::Two(transposex, transposey))
                    .global_work_size(ocl::SpatialDims::Two(ds, ds))
                    .arg(ds as i32)
                    .arg(ds as i32)
                    .arg(&buffer)
                    .arg(&output)
                    .build()
                    .unwrap();
                (kernel, buffer, queue)
            })
        })
        .collect::<HashMap<usize, (ocl::Kernel, ocl::Buffer<f32>, ocl::Queue)>>();

    // Create benchmark-closure
    let bench = ParameterizedBenchmark::new(
        "transpose 16",
        move |be, &ds| {
            // Measure
            be.iter_with_setup(
                || {
                    let input = create_random_vec(ds * ds);
                    test_data[&ds].1.write(&input).enq().unwrap();
                    test_data[&ds].2.finish().unwrap();
                },
                |()| unsafe {
                    test_data[&ds]
                        .0
                        .cmd()
                        .queue(&test_data[&ds].2)
                        .enq()
                        .unwrap();
                    test_data[&ds].2.finish().unwrap();
                },
            )
        },
        params.clone(),
    );

    let test_data = params
        .iter()
        .map(|&ds| {
            (ds, {
                let transposex = 8;
                let transposey = 8;
                let (queue, program, _context) = rusty_cnn::cl_util::init_from_file::<f32>(
                    &["src/math/mtx_mul/gemm/cl/transpose.cl"],
                    &[
                        &format!("-D TRANSPOSEX={}", transposex),
                        &format!("-D TRANSPOSEY={}", transposey),
                    ],
                    Some(device),
                );

                let buffer = ocl::Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(ocl::flags::MEM_READ_ONLY)
                    .len(ds * ds)
                    .build()
                    .unwrap();
                let output = ocl::Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(ocl::flags::MEM_READ_WRITE)
                    .len(ds * ds)
                    .build()
                    .unwrap();

                let kernel = ocl::Kernel::builder()
                    .program(&program)
                    .name("transpose")
                    .queue(queue.clone())
                    .local_work_size(ocl::SpatialDims::Two(transposex, transposey))
                    .global_work_size(ocl::SpatialDims::Two(ds, ds))
                    .arg(ds as i32)
                    .arg(ds as i32)
                    .arg(&buffer)
                    .arg(&output)
                    .build()
                    .unwrap();
                (kernel, buffer, queue)
            })
        })
        .collect::<HashMap<usize, (ocl::Kernel, ocl::Buffer<f32>, ocl::Queue)>>();

    let bench = bench.with_function("transpose 8", move |be, &ds| {
        // Measure
        be.iter_with_setup(
            || {
                let input = create_random_vec(ds * ds);
                test_data[&ds].1.write(&input).enq().unwrap();
                test_data[&ds].2.finish().unwrap();
            },
            |()| unsafe {
                test_data[&ds]
                    .0
                    .cmd()
                    .queue(&test_data[&ds].2)
                    .enq()
                    .unwrap();
                test_data[&ds].2.finish().unwrap();
            },
        )
    });

    let test_data = params
        .iter()
        .map(|&ds| {
            (ds, {
                let transposex = 1;
                let transposey = 1;
                let (queue, program, _context) = rusty_cnn::cl_util::init_from_file::<f32>(
                    &["src/math/mtx_mul/gemm/cl/transpose.cl"],
                    &[
                        &format!("-D TRANSPOSEX={}", transposex),
                        &format!("-D TRANSPOSEY={}", transposey),
                    ],
                    Some(device),
                );

                let buffer = ocl::Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(ocl::flags::MEM_READ_ONLY)
                    .len(ds * ds)
                    .build()
                    .unwrap();
                let output = ocl::Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(ocl::flags::MEM_READ_WRITE)
                    .len(ds * ds)
                    .build()
                    .unwrap();

                let kernel = ocl::Kernel::builder()
                    .program(&program)
                    .name("transpose")
                    .queue(queue.clone())
                    .local_work_size(ocl::SpatialDims::Two(transposex, transposey))
                    .global_work_size(ocl::SpatialDims::Two(ds, ds))
                    .arg(ds as i32)
                    .arg(ds as i32)
                    .arg(&buffer)
                    .arg(&output)
                    .build()
                    .unwrap();
                (kernel, buffer, queue)
            })
        })
        .collect::<HashMap<usize, (ocl::Kernel, ocl::Buffer<f32>, ocl::Queue)>>();

    let bench = bench.with_function("transpose 1", move |be, &ds| {
        // Measure
        be.iter_with_setup(
            || {
                let input = create_random_vec(ds * ds);
                test_data[&ds].1.write(&input).enq().unwrap();
                test_data[&ds].2.finish().unwrap();
            },
            |()| unsafe {
                test_data[&ds]
                    .0
                    .cmd()
                    .queue(&test_data[&ds].2)
                    .enq()
                    .unwrap();
                test_data[&ds].2.finish().unwrap();
            },
        )
    });

    c.bench(
        group_name,
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
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD).retain_baseline("baseline 26-07-2018 fresh computer".to_string());
    targets = bench_transpose_gpu, bench_transpose_cpu
}
criterion_main!(benches);
