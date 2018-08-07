use super::*;

use matrixmultiply;

/// This is also the implementation used by ndarray.
pub fn bench_bluss_matrixmultiply() -> impl FnMut(&mut Bencher, &usize) {
    // matrixmultiply::gemm uses row major instead of column-major and verification would be
    // troublesome. Calculation takes the same amount of time with the incorrect matrices, however.

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
    }
}
