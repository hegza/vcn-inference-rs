//! Re-exports of the matrix-multiplication algorithms in-use.

mod naive;
// HACK: being public might be hack here
pub mod gemm;

// TODO: consolidate self::naive::mtx_mul and self::gemm_naive
pub use self::gemm::Gemm6WithBTransposeKernel as GemmKernel;
pub use self::naive::{mtx_mul, mtx_mul_normint};

/// Naive matrix multiplication on the host
pub fn gemm_naive(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), k * m);
    debug_assert_eq!(b.len(), n * k);
    debug_assert_eq!(c.len(), m * n);

    for m_idx in 0..m {
        for n_idx in 0..n {
            let mut acc = 0f32;
            for k_idx in 0..k {
                acc += a[k_idx * m + m_idx] * b[n_idx * k + k_idx]
            }
            c[n_idx * m + m_idx] = acc;
        }
    }
}
