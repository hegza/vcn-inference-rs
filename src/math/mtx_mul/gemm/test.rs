use super::*;
use tests::COARSE_RESULT_MARGIN;
use verify;

// Cubic-root of the dimension of the test matrices in the files
const D: usize = 64;

lazy_static! {
    static ref A: Vec<f32> = String::from_utf8(
        include_bytes!("../../../tests/in/A_64x64.csv").to_vec()
    ).unwrap()
        .split(',')
        .map(|word| word.trim().parse::<f32>())
        .filter_map(|res| res.ok())
        .collect::<Vec<f32>>();
    static ref B: Vec<f32> = String::from_utf8(
        include_bytes!("../../../tests/in/B_64x64.csv").to_vec(),
    ).unwrap()
        .split(',')
        .map(|word| word.trim().parse::<f32>())
        .filter_map(|res| res.ok())
        .collect::<Vec<f32>>();
    static ref C: Vec<f32> = String::from_utf8(
        include_bytes!("../../../tests/out/C_64x64.csv").to_vec(),
    ).unwrap()
        .split(',')
        .map(|word| word.trim().parse::<f32>())
        .filter_map(|res| res.ok())
        .collect::<Vec<f32>>();
}

#[test]
fn mtx_mul_1_naive_is_correct() {
    let mut out = vec![0f32; D * D];

    let kernel = Naive1GemmKernel::from_slices(D, D, D, &A, &B, &mut out, DeviceType::ALL);
    kernel.calculate_wait();
    verify(&out, &C, COARSE_RESULT_MARGIN);
}

#[test]
fn mtx_mul_4_vector_data_types_cl_is_correct() {
    let mut out = vec![0f32; D * D];

    let kernel = Vectors4GemmKernel::from_slices(D, D, D, &A, &B, &mut out, DeviceType::ALL);
    kernel.calculate_wait();
    verify(&out, &C, COARSE_RESULT_MARGIN);
}

#[test]
fn mtx_mul_5_transpose_cl_is_correct() {
    let mut out = vec![0f32; D * D];

    let kernel = Transpose5GemmKernel::from_slices(D, D, D, &A, &B, &mut out, DeviceType::ALL);
    kernel.calculate_wait();
    verify(&out, &C, COARSE_RESULT_MARGIN);
}

#[test]
fn mtx_mul_6_register_tiling_cl_is_correct() {
    let mut out = vec![0f32; D * D];

    let kernel = Tiling6GemmKernel::from_slices(D, D, D, &A, &B, &mut out, DeviceType::ALL);
    kernel.calculate_wait();
    verify(&out, &C, COARSE_RESULT_MARGIN);
}

fn test_mtx_mul<F>(mtx_mul_impl: F)
where
    F: Fn(usize, usize, usize, &[f32], &[f32], &mut [f32]),
{
    let mut out = vec![0f32; D * D];

    mtx_mul_impl(D, D, D, &A, &B, &mut out);
    verify(&out, &C, COARSE_RESULT_MARGIN);
}

#[test]
fn gemm_naive_is_correct() {
    test_mtx_mul(super::super::gemm_naive);
}
