#![allow(non_snake_case)]

use super::*;

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
    let A_large = String::from_utf8(include_bytes!("A_64x64.csv").to_vec())
        .unwrap()
        .split(',')
        .map(|word| word.trim().parse::<f32>())
        .filter_map(|res| res.ok())
        .collect::<Vec<f32>>();
    let B_large = String::from_utf8(include_bytes!("B_64x64.csv").to_vec())
        .unwrap()
        .split(',')
        .map(|word| word.trim().parse::<f32>())
        .filter_map(|res| res.ok())
        .collect::<Vec<f32>>();
    let C_large_correct = String::from_utf8(include_bytes!("C_64x64.csv").to_vec())
        .unwrap()
        .split(',')
        .map(|word| word.trim().parse::<f32>())
        .filter_map(|res| res.ok())
        .collect::<Vec<f32>>();

    mtx_mul_impl(D, D, D, &A_large, &B_large, &mut out);
    assert_eq!(&C_large_correct, &out);
}

#[test]
fn mtx_mul_1_naive_host_small_is_correct() {
    let mut out = vec![0f32; M * N];
    mtx_mul_1_naive_host(M, N, K, &A_SMALL, &B_SMALL, &mut out);
    assert_eq!(&C_SMALL, &out[..]);
}

#[test]
fn mtx_mul_1_naive_host_is_correct() {
    test_mtx_mul(mtx_mul_1_naive_host);
}

#[test]
fn mtx_mul_1_naive_cl_is_correct() {
    test_mtx_mul(mtx_mul_1_naive_cl);
}
