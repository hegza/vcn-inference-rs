#[macro_use]
extern crate criterion;
extern crate matrixmultiply;
extern crate ndarray;
extern crate num_traits;
extern crate rand;
extern crate rusty_cnn;

use criterion::{Criterion, Fun};
use rusty_cnn::math::mtx_mul::sgemm::algo::*;

const SAMPLE_SIZE: usize = 100;
const NOISE_THRESHOLD: f64 = 0.06;

const D: usize = 64;

/// On notation:
/// host =  compiled Rust.
/// GPU =   OpenCL / GPU.
/// CPU =   OpenCL / CPU.

// Benchmark each layer separately.
fn bench_sgemm_variants(c: &mut Criterion) {
    let (a_large, b_large) = criterion::black_box(ab_matrices());

    let mut out = vec![0f32; D * D];
    let a = a_large.clone();
    let b = b_large.clone();
    let naive = Fun::new("naive (host)", move |be, ()| {
        be.iter(|| {
            gemm_naive(D, D, D, &a, &b, &mut out);
        })
    });

    use matrixmultiply::sgemm;
    let mut out = vec![0f32; D * D];
    let a = a_large.clone();
    let b = b_large.clone();
    let matrixmultiply = Fun::new("bluss_matrixmultiply (host)", move |be, ()| {
        be.iter(|| unsafe {
            sgemm(
                D,
                D,
                D,
                1f32,
                a.as_ptr(),
                1,
                1,
                b.as_ptr(),
                1,
                1,
                1f32,
                out.as_mut_ptr(),
                1,
                1,
            );
        })
    });

    use ndarray::*;

    let a = Array2::<f32>::from_shape_vec((D, D), a_large.clone()).unwrap();
    let b = Array2::<f32>::from_shape_vec((D, D), b_large.clone()).unwrap();
    let ndarray = Fun::new("ndarray_dot (host)", move |be, ()| {
        be.iter(|| a.dot(&b))
    });

    let mut out = vec![0f32; D * D];
    let sgemm_1_kernel = Naive1GemmKernel::new(D, D, D, &a_large, &b_large, &mut out);
    let sgemm_1_gpu = Fun::new("cnugteren_1_naive (GPU)", move |b, _| {
        b.iter(|| sgemm_1_kernel.run_wait())
    });

    let mut out = vec![0f32; D * D];
    let sgemm_4_kernel = Vectors4GemmKernel::new(D, D, D, &a_large, &b_large, &mut out);
    let sgemm_4_gpu = Fun::new("cnugteren_4_vectors (GPU)", move |b, _| {
        b.iter(|| sgemm_4_kernel.run_wait())
    });

    let mut out = vec![0f32; D * D];
    let sgemm_5_kernel = Transpose5GemmKernel::new(D, D, D, &a_large, &b_large, &mut out);
    let sgemm_5_gpu = Fun::new("cnugteren_5_transpose (GPU)", move |b, _| {
        b.iter(|| sgemm_5_kernel.run_wait())
    });

    let mut out = vec![0f32; D * D];
    let sgemm_6_kernel = Tiling6GemmKernel::new(D, D, D, &a_large, &b_large, &mut out);
    let sgemm_6_gpu = Fun::new("cnugteren_6_tiling (GPU)", move |b, _| {
        b.iter(|| sgemm_6_kernel.run_wait())
    });

    c.bench_functions(
        "sgemm-f32 (64x64)",
        vec![
            naive,
            matrixmultiply,
            ndarray,
            sgemm_1_gpu,
            sgemm_4_gpu,
            sgemm_5_gpu,
            sgemm_6_gpu,
        ],
        (),
    );
}

fn ab_matrices() -> (Vec<f32>, Vec<f32>) {
    (
        String::from_utf8(include_bytes!("../src/tests/in/A_64x64.csv").to_vec())
            .unwrap()
            .split(',')
            .map(|word| word.trim().parse::<f32>())
            .filter_map(|res| res.ok())
            .collect::<Vec<f32>>(),
        String::from_utf8(include_bytes!("../src/tests/in/B_64x64.csv").to_vec())
            .unwrap()
            .split(',')
            .map(|word| word.trim().parse::<f32>())
            .filter_map(|res| res.ok())
            .collect::<Vec<f32>>(),
    )
}

criterion_group!{
    name = benches;
    config = Criterion::default().sample_size(SAMPLE_SIZE).noise_threshold(NOISE_THRESHOLD);
    targets = bench_sgemm_variants
}
criterion_main!(benches);
