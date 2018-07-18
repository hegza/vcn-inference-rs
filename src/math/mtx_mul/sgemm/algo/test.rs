#![allow(non_snake_case)]

use super::*;
use tests::{verify, COARSE_RESULT_MARGIN};

const M: usize = 2;
const N: usize = 1;
const K: usize = 3;
static A_SMALL: [f32; M * K] = [11f32, 21f32, 12f32, 22f32, 13f32, 23f32];
static B_SMALL: [f32; K * N] = [11f32, 21f32, 31f32];
static C_SMALL: [f32; M * N] = [
    11f32 * 11f32 + 12f32 * 21f32 + 13f32 * 31f32,
    21f32 * 11f32 + 22f32 * 21f32 + 23f32 * 31f32,
];

// Dimension of the matrices in the files
const D: usize = 64;

fn test_mtx_mul<F>(mtx_mul_impl: F)
where
    F: Fn(usize, usize, usize, &[f32], &[f32], &mut [f32]),
{
    let mut out = vec![0f32; D * D];
    let (a_large, b_large, c_large_correct) = abc_matrices();

    mtx_mul_impl(D, D, D, &a_large, &b_large, &mut out);
    verify(&out, &c_large_correct, COARSE_RESULT_MARGIN);
}

fn abc_matrices() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        String::from_utf8(include_bytes!("../../../../tests/in/A_64x64.csv").to_vec())
            .unwrap()
            .split(',')
            .map(|word| word.trim().parse::<f32>())
            .filter_map(|res| res.ok())
            .collect::<Vec<f32>>(),
        String::from_utf8(include_bytes!("../../../../tests/in/B_64x64.csv").to_vec())
            .unwrap()
            .split(',')
            .map(|word| word.trim().parse::<f32>())
            .filter_map(|res| res.ok())
            .collect::<Vec<f32>>(),
        String::from_utf8(include_bytes!("../../../../tests/out/C_64x64.csv").to_vec())
            .unwrap()
            .split(',')
            .map(|word| word.trim().parse::<f32>())
            .filter_map(|res| res.ok())
            .collect::<Vec<f32>>(),
    )
}

#[test]
fn gemm_naive_small_is_correct() {
    let mut out = vec![0f32; M * N];
    gemm_naive(M, N, K, &A_SMALL, &B_SMALL, &mut out);
    assert_eq!(&C_SMALL, &out[..]);
}

#[test]
fn gemm_naive_is_correct() {
    test_mtx_mul(gemm_naive);
}

#[test]
fn mtx_mul_1_naive_is_correct() {
    let mut out = vec![0f32; D * D];
    let (a_large, b_large, c_large_correct) = abc_matrices();

    let kernel = Naive1GemmKernel::new(D, D, D, &a_large, &b_large, &mut out);
    kernel.run_wait();
    verify(&out, &c_large_correct, COARSE_RESULT_MARGIN);
}

#[test]
fn mtx_mul_4_vector_data_types_cl_is_correct() {
    let mut out = vec![0f32; D * D];
    let (a_large, b_large, c_large_correct) = abc_matrices();

    let kernel = Vectors4GemmKernel::new(D, D, D, &a_large, &b_large, &mut out);
    kernel.run_wait();
    verify(&out, &c_large_correct, COARSE_RESULT_MARGIN);
}

#[test]
fn mtx_mul_5_transpose_cl_is_correct() {
    let mut out = vec![0f32; D * D];
    let (a_large, b_large, c_large_correct) = abc_matrices();

    let kernel = Transpose5GemmKernel::new(D, D, D, &a_large, &b_large, &mut out);
    kernel.run_wait();
    verify(&out, &c_large_correct, COARSE_RESULT_MARGIN);
}

#[test]
fn mtx_mul_6_register_tiling_cl_is_correct() {
    let mut out = vec![0f32; D * D];
    let (a_large, b_large, c_large_correct) = abc_matrices();

    let kernel = Tiling6GemmKernel::new(D, D, D, &a_large, &b_large, &mut out);
    kernel.run_wait();
    verify(&out, &c_large_correct, COARSE_RESULT_MARGIN);
}
